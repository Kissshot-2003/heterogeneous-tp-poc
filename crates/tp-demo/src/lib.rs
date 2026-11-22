//! tp-demo 核心逻辑。
//!
//! 抽成 lib 是为了让 `main.rs` 和集成测试共用同一份端到端流程，
//! 避免"测试通过但 demo 行为不同"的偏差。
//!
//! **P0：异步重叠** —— GPU submit 后立即返回，CPU matmul 在 GPU 执行期间
//! 并行运行，最后统一取回。关键等式（HeteGen）：`T_CPU = T_GPU + T_COM`。

pub mod dense;
pub mod matmul_cpu;

use std::time::Instant;

use het_comm::{as_gpu, CommDomain, TensorBuf, VulkanComm};

use crate::dense::{dense_matmul, max_abs_error};
use crate::matmul_cpu::{matmul_cpu, warmup_matmul_cpu};

/// 默认形状：M=64, K=256, N=128（列切分后每半 64 列）。
pub const DEFAULT_M: usize = 64;
pub const DEFAULT_K: usize = 256;
pub const DEFAULT_N: usize = 128;

/// P2c：GPU 承担的列比例。
///
/// 理论最优约 0.62（按算力比例），但实测发现 wgpu/Vulkan 的 D2H 存在
/// **16KB staging buffer 快路径阈值**：C1 超过 16KB（gpu_cols > 64）时
/// map 耗时从 ~40µs 暴涨到 ~300µs。因此本机保持 0.5（C1=16KB）反而最快。
/// 详细数据与尝试见 HANDOVER.md 第 9 节 P2c。
pub const GPU_COL_FRACTION: f64 = 0.5;

/// 一次 PoC 运行的完整结果，供打印与测试断言共用。
#[derive(Debug, Clone)]
pub struct PocResult {
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    /// M2a：H2D→D2H 回环逐元素相等
    pub h2d_roundtrip_ok: bool,
    /// M2b：smoke compute pass（data[i] += 1）结果正确
    pub smoke_compute_ok: bool,
    /// M4：与 dense 参考的最大绝对误差
    pub max_abs_error: f32,
    /// H2D 提交（CPU 侧 write_buffer）耗时
    pub h2d_submit_us: u128,
    /// CPU matmul 耗时（与 GPU 执行重叠）
    pub cpu_matmul_us: u128,
    /// GPU 阶段墙钟：从 submit 到 collect 完成（期间 CPU 并行计算）
    pub gpu_wall_us: u128,
    /// P2：timestamp query 测得的纯 GPU compute 时间（µs）；不支持时为 None
    pub gpu_compute_us: Option<u128>,
    /// D2H 读回耗时（含等待 GPU compute 完成的 poll 时间）
    pub d2h_transfer_us: u128,
    /// 端到端墙钟
    pub total_us: u128,
    /// 重叠节省的时间：CPU 计算被 GPU 等待覆盖的部分
    pub overlap_saving_us: u128,
    /// P2c：非均匀切分的列数（CPU / GPU）
    pub cpu_cols: usize,
    pub gpu_cols: usize,
}

/// 已提交到 GPU、等待取回的 matmul 任务。
///
/// 只需持有 C1 的 GPU buffer：其余 buffer / pipeline / bind group 在 submit 后
/// 由 wgpu 内部保持存活直到命令完成，Rust 侧 drop 不会取消已提交的工作。
pub struct PendingGpuMatmul {
    /// P2b：compute 完成后，C1 已被同一次 submit  copy 到这里，collect 只需 map。
    c1_staging: wgpu::Buffer,
    len: usize,
    /// P2 timestamp 测量的待读取句柄；设备不支持时为 None
    timestamp: Option<TimestampGuard>,
}

/// P2：保存 timestamp query 的 resolve buffer 与时间戳周期，供 submit 后读取。
pub struct TimestampGuard {
    _query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    period: f32,
}

/// 提交阶段：创建资源 + H2D + submit compute，**立即返回，不等待 GPU**。
///
/// 返回 `(pending, h2d_submit_us)`。CPU 可在此之后立即开始计算 C0，
/// 与 GPU 执行重叠；最后调用 `gpu_matmul_collect` 取回结果。
pub fn gpu_matmul_submit(
    comm: &VulkanComm,
    a: &[f32],
    b1: &[f32],
    m: usize,
    k: usize,
    n2: usize,
) -> (PendingGpuMatmul, u128) {
    // --- 初始化（不计入 H2D 计时）---
    let module = comm.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
    });
    let pipeline = comm.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul pipeline"),
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    // --- P2：创建 timestamp query 资源（若设备支持）---
    let timestamp = if comm
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        let query_set = comm.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("matmul timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buf = comm.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: 16, // 2 × QUERY_SIZE(8)
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Some(TimestampGuard {
            _query_set: query_set,
            resolve_buf,
            period: comm.queue.get_timestamp_period(),
        })
    } else {
        None
    };

    let mut a_buf = het_comm::make_gpu_buffer(&comm.device, a.len());
    let mut b_buf = het_comm::make_gpu_buffer(&comm.device, b1.len());
    let c_buf = het_comm::make_gpu_buffer(&comm.device, m * n2);
    let dims_buf = comm.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matmul dims"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    comm.queue.write_buffer(
        &dims_buf,
        0,
        bytemuck::cast_slice(&[m as u32, k as u32, n2 as u32, 0u32]),
    );

    // --- H2D：上传 A 与 B1（CPU 侧 write_buffer）---
    let t = Instant::now();
    comm.copy(&TensorBuf::Cpu(a.to_vec()), &mut a_buf, a.len());
    comm.copy(&TensorBuf::Cpu(b1.to_vec()), &mut b_buf, b1.len());
    let h2d_us = t.elapsed().as_micros();

    let bind_group = comm.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: as_gpu(&a_buf).expect("A 必须是 GPU buffer"),
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: as_gpu(&b_buf).expect("B1 必须是 GPU buffer"),
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: as_gpu(&c_buf).expect("C1 必须是 GPU buffer"),
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &dims_buf,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    // --- 提交 compute（立即返回；不调用 barrier，GPU 后台执行）---
    let mut encoder = comm.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul encoder"),
    });
    {
        // P2：把 pass 起止时间戳写入 query set
        let timestamp_writes = timestamp.as_ref().map(|g| wgpu::ComputePassTimestampWrites {
            query_set: &g._query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // P1：@workgroup_size(16,16) + 16×16 shared memory tile。
        // 实测 TILE=16 优于 TILE=8（barrier 次数减半），故固定为 16。
        pass.dispatch_workgroups(((m + 15) / 16) as u32, ((n2 + 15) / 16) as u32, 1);
    }

    // P2：把 query 结果 resolve 到 buffer（与 compute 同一次 submit，顺序保证）
    if let Some(g) = &timestamp {
        encoder.resolve_query_set(&g._query_set, 0..2, &g.resolve_buf, 0);
    }

    // P2b：把 D2H staging copy 也塞进同一次 submit。
    // GPU 顺序执行 compute → copy，CPU 在 submit 后并行算 C0；
    // collect 时只需 map staging，省掉一次 submit 和一次 barrier poll。
    let c1_staging = comm.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c1 d2h staging"),
        size: (m * n2 * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        as_gpu(&c_buf).expect("C1 必须是 GPU buffer"),
        0,
        &c1_staging,
        0,
        (m * n2 * 4) as u64,
    );
    comm.queue.submit(Some(encoder.finish()));

    (
        PendingGpuMatmul {
            c1_staging,
            len: m * n2,
            timestamp,
        },
        h2d_us,
    )
}

/// 取回阶段：从已 copy 好的 staging buffer 直接 map 读回 C1。
///
/// P2b：staging copy 已在 submit 时与 compute 合并提交，这里只做 map+poll+read，
/// 不再 submit、不再 create_buffer——省掉一次 submit 和一次显式 barrier。
pub fn gpu_matmul_collect(
    comm: &VulkanComm,
    pending: PendingGpuMatmul,
) -> (Vec<f32>, u128, Option<TimestampGuard>) {
    let t = Instant::now();

    pending
        .c1_staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    let _ = comm.device.poll(wgpu::PollType::wait_indefinitely());

    let view = pending
        .c1_staging
        .get_mapped_range(..)
        .expect("C1 staging 映射失败");
    let c1 = bytemuck::cast_slice::<u8, f32>(&view[..])[..pending.len].to_vec();
    drop(view);
    pending.c1_staging.unmap();

    let d2h_us = t.elapsed().as_micros();
    (c1, d2h_us, pending.timestamp)
}

/// P2：读取 timestamp query，返回纯 GPU compute 时间（µs）。
///
/// 在 `gpu_wall_us` 测完后再调用，避免把 timestamp 读回开销算进 GPU 墙钟。
pub fn read_gpu_timestamps(comm: &VulkanComm, guard: Option<TimestampGuard>) -> Option<u128> {
    let guard = guard?;

    let staging = comm.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = comm.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("timestamp copy"),
    });
    encoder.copy_buffer_to_buffer(&guard.resolve_buf, 0, &staging, 0, 16);
    comm.queue.submit(Some(encoder.finish()));

    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    let _ = comm.device.poll(wgpu::PollType::wait_indefinitely());

    let view = staging
        .get_mapped_range(..)
        .expect("timestamp readback 映射失败");
    let ticks: &[u64] = bytemuck::cast_slice(&view[..]);
    let elapsed_ns = ticks[1].wrapping_sub(ticks[0]) as f64 * guard.period as f64;
    drop(view);
    staging.unmap();

    Some((elapsed_ns / 1000.0) as u128)
}

/// M2b：验证 shader 编译 / pipeline / bind group / dispatch 全链路。
fn smoke_compute(comm: &VulkanComm) -> bool {
    const SMOKE_WGSL: &str = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            data[gid.x] = data[gid.x] + 1.0;
        }
    "#;
    let module = comm.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("smoke shader"),
        source: wgpu::ShaderSource::Wgsl(SMOKE_WGSL.into()),
    });
    let pipeline = comm.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("smoke pipeline"),
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    let n = 16usize;
    let mut data_buf = het_comm::make_gpu_buffer(&comm.device, n);
    let input: Vec<f32> = (0..n).map(|x| x as f32).collect();
    comm.copy(&TensorBuf::Cpu(input), &mut data_buf, n);

    let bind_group = comm.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: as_gpu(&data_buf).expect("data_buf 必须是 GPU buffer"),
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut encoder = comm.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("smoke pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // 1 × 64 threads，覆盖 16 个元素
    }
    comm.queue.submit(Some(encoder.finish()));
    comm.barrier();

    let mut out = TensorBuf::Cpu(vec![0.0; n]);
    comm.copy(&data_buf, &mut out, n);
    let expected: Vec<f32> = (0..n).map(|x| x as f32 + 1.0).collect();
    match out {
        TensorBuf::Cpu(v) => v == expected,
        TensorBuf::Gpu(_) => unreachable!(),
    }
}

/// 跑完整端到端 PoC，返回结果（不打印，便于测试复用）。
pub fn run_poc() -> PocResult {
    // ===== M1：强制 Vulkan backend 枚举设备 =====
    let adapter = pollster::block_on(het_comm::request_vulkan_adapter());
    let info = adapter.get_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "M1 失败：未走 Vulkan backend"
    );

    // ===== M2：统一 buffer + H2D/D2H + smoke compute =====
    let comm = pollster::block_on(VulkanComm::new(&adapter));
    let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut gpu = het_comm::make_gpu_buffer(&comm.device, src.len());
    comm.copy(&TensorBuf::Cpu(src.clone()), &mut gpu, src.len());
    comm.barrier();
    let mut back = TensorBuf::Cpu(vec![0.0; src.len()]);
    comm.copy(&gpu, &mut back, src.len());
    let h2d_roundtrip_ok = match &back {
        TensorBuf::Cpu(v) => *v == src,
        TensorBuf::Gpu(_) => unreachable!(),
    };
    let smoke_compute_ok = smoke_compute(&comm);

    // P3：预热 rayon 线程池，使首次启动开销不计入 matmul 计时。
    warmup_matmul_cpu();

    // ===== M3：双路 matmul + 非均匀列切分（P0 重叠 + P2c 非均匀）=====
    let (m, k, n) = (DEFAULT_M, DEFAULT_K, DEFAULT_N);
    let gpu_cols = (n as f64 * GPU_COL_FRACTION) as usize;
    let cpu_cols = n - gpu_cols;
    assert!(cpu_cols > 0 && gpu_cols > 0, "切分比例必须为正");

    // 确定性伪随机输入，范围 [-0.5, 0.5]。
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 97) as f32) / 97.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 31 % 89) as f32) / 89.0 - 0.5)
        .collect();

    // P2c：B 按非均匀列切成 B0 [K, cpu_cols]（CPU）与 B1 [K, gpu_cols]（GPU）。
    let mut b0 = Vec::with_capacity(k * cpu_cols);
    let mut b1 = Vec::with_capacity(k * gpu_cols);
    for t in 0..k {
        for j in 0..cpu_cols {
            b0.push(b[t * n + j]);
        }
        for j in 0..gpu_cols {
            b1.push(b[t * n + cpu_cols + j]);
        }
    }

    let total_start = Instant::now();

    // 1) 提交 GPU（H2D + compute），立即返回。
    let (pending, h2d_us) = gpu_matmul_submit(&comm, &a, &b1, m, k, gpu_cols);
    let gpu_wall_start = Instant::now();

    // 2) CPU 算左半 C0 —— 与 GPU 执行重叠。
    let t = Instant::now();
    let c0 = matmul_cpu(&a, &b0, m, k, cpu_cols);
    let cpu_us = t.elapsed().as_micros();

    // 3) 直接取回（P2b：compute + D2H copy 已合并为同一次 submit，无需 barrier）。
    let (c1, d2h_us, ts_guard) = gpu_matmul_collect(&comm, pending);
    // 先测 GPU 墙钟，再读 timestamp（读回本身会产生额外 submit/poll）
    let gpu_wall_us = gpu_wall_start.elapsed().as_micros();
    let gpu_compute_us = read_gpu_timestamps(&comm, ts_guard);

    // 4) Gather：按行交错拼成完整 [M, N]（见 HANDOVER.md 坑 3）。
    let parts = [TensorBuf::Cpu(c0), TensorBuf::Cpu(c1)];
    let mut gathered = TensorBuf::Cpu(Vec::with_capacity(m * n));
    comm.gather_to_cpu(&parts, &mut gathered, m, &[cpu_cols, gpu_cols]);
    let total_us = total_start.elapsed().as_micros();
    let got = match gathered {
        TensorBuf::Cpu(v) => v,
        TensorBuf::Gpu(_) => unreachable!(),
    };

    // ===== M4：数值比对 =====
    let dense = dense_matmul(&a, &b, m, k, n);
    let err = max_abs_error(&got, &dense);

    // CPU 计算被 GPU 等待覆盖的部分：
    // GPU 阶段（扣除 D2H）的墙钟 = CPU 时间 + barrier 剩余等待，
    // 重叠节省 = min(CPU 时间, GPU 阶段时间)。
    let gpu_phase_us = gpu_wall_us.saturating_sub(d2h_us);
    let overlap_saving_us = cpu_us.min(gpu_phase_us);

    PocResult {
        adapter_name: info.name,
        backend: info.backend,
        h2d_roundtrip_ok,
        smoke_compute_ok,
        max_abs_error: err,
        h2d_submit_us: h2d_us,
        cpu_matmul_us: cpu_us,
        gpu_wall_us,
        gpu_compute_us,
        d2h_transfer_us: d2h_us,
        total_us,
        overlap_saving_us,
        cpu_cols,
        gpu_cols,
    }
}
