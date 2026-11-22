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

use het_comm::{as_gpu, CommDomain, GpuReducer, TensorBuf, VulkanComm};

use crate::dense::{dense_matmul, max_abs_error};
use crate::matmul_cpu::{matmul_cpu, warmup_matmul_cpu};

/// 默认形状：M=64, K=256, N=128（P4 行并行：K 对半切，各算 [M,N] 部分和）。
pub const DEFAULT_M: usize = 64;
pub const DEFAULT_K: usize = 256;
pub const DEFAULT_N: usize = 128;

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
    /// P4：行并行切分的 K 维度（CPU / GPU），各算 [M,N] 部分和
    pub cpu_k: usize,
    pub gpu_k: usize,
}

/// P5：N 逻辑设备端到端 TP demo 的结果。
#[derive(Debug, Clone)]
pub struct MultiPocResult {
    pub adapter_name: String,
    /// 逻辑设备数（用 N 个独立 GPU storage buffer 模拟）
    pub n_devices: usize,
    /// 每个设备负责的 K 维度长度
    pub k_per_device: usize,
    /// 与 dense 参考的最大绝对误差
    pub max_abs_error: f32,
    /// GPU 端 all-reduce 结果是否与 CPU pairwise 参考一致
    pub all_reduce_ok: bool,
    /// 端到端墙钟（含 H2D / N 路 matmul / all-reduce / D2H）
    pub total_us: u128,
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

    // ===== M3 / P4：真 TP —— 行并行 + all-reduce =====
    // 数学依据：A[M,K] × B[K,N] = [A0|A1] × [B0;B1] = A0×B0 + A1×B1
    //   各设备持有部分权重（A 按列切、B 按行切），
    //   各自算 [M,N] 的部分和，层末 all-reduce 相加 —— 这才是真正的 TP。
    let (m, k, n) = (DEFAULT_M, DEFAULT_K, DEFAULT_N);
    let cpu_k = k / 2;
    let gpu_k = k - cpu_k;

    // 确定性伪随机输入，范围 [-0.5, 0.5]。
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 97) as f32) / 97.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 31 % 89) as f32) / 89.0 - 0.5)
        .collect();

    // A 按列切成 A0 [M, cpu_k]（CPU）与 A1 [M, gpu_k]（GPU）。
    let mut a0 = Vec::with_capacity(m * cpu_k);
    let mut a1 = Vec::with_capacity(m * gpu_k);
    for i in 0..m {
        for t in 0..cpu_k {
            a0.push(a[i * k + t]);
        }
        for t in 0..gpu_k {
            a1.push(a[i * k + cpu_k + t]);
        }
    }
    // B 按行切成 B0 [cpu_k, N]（CPU）与 B1 [gpu_k, N]（GPU）。
    let b0 = b[..cpu_k * n].to_vec();
    let b1 = b[cpu_k * n..].to_vec();

    let total_start = Instant::now();

    // 1) 提交 GPU：算 A1 × B1（部分和 C1 [M,N]），立即返回。
    let (pending, h2d_us) = gpu_matmul_submit(&comm, &a1, &b1, m, gpu_k, n);
    let gpu_wall_start = Instant::now();

    // 2) CPU 算 C0 = A0 × B0（部分和 [M,N]）—— 与 GPU 执行重叠。
    let t = Instant::now();
    let c0 = matmul_cpu(&a0, &b0, m, cpu_k, n);
    let cpu_us = t.elapsed().as_micros();

    // 3) 取回 GPU 的部分和 C1。
    let (c1, d2h_us, ts_guard) = gpu_matmul_collect(&comm, pending);
    let gpu_wall_us = gpu_wall_start.elapsed().as_micros();
    let gpu_compute_us = read_gpu_timestamps(&comm, ts_guard);

    // 4) All-reduce：C = C0 + C1（层末规约），得到完整 [M,N]。
    let parts = [TensorBuf::Cpu(c0), TensorBuf::Cpu(c1)];
    let mut reduced = TensorBuf::Cpu(Vec::with_capacity(m * n));
    comm.all_reduce(&parts, &mut reduced, m, n);
    let total_us = total_start.elapsed().as_micros();
    let got = match reduced {
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
        cpu_k,
        gpu_k,
    }
}

/// P5：N 逻辑设备端到端张量并行（行并行 + GPU 端 all-reduce）。
///
/// 本机只有 1 张物理 GPU，用同一 logical device 上的 N 个独立 storage buffer
/// 模拟 N 个逻辑设备：
///   - K 维度切成 N 份，每个 buffer 持有 (A_i, B_i) 并算部分和 C_i = A_i × B_i
///   - GpuReducer 对 N 个 C_i 做 GPU 端 all-reduce（pairwise add + broadcast）
///   - 结果与 dense 参考比对
///
/// 通信算法（all-reduce）与真实多卡一致，物理传输走本 device 显存。
pub fn run_poc_multi(n_devices: usize) -> MultiPocResult {
    assert!(n_devices >= 2, "多设备 demo 至少需要 2 个逻辑设备");
    assert_eq!(DEFAULT_K % n_devices, 0, "K 必须能被设备数整除");

    let adapter = pollster::block_on(het_comm::request_vulkan_adapter());
    let info = adapter.get_info();
    let comm = pollster::block_on(VulkanComm::new(&adapter));
    let reducer = GpuReducer::new(comm.device.clone(), comm.queue.clone());

    let (m, k, n) = (DEFAULT_M, DEFAULT_K, DEFAULT_N);
    let k_i = k / n_devices;

    // 与单设备 demo 相同的确定性输入。
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 97) as f32) / 97.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 31 % 89) as f32) / 89.0 - 0.5)
        .collect();

    // 行并行切分：A 按列切成 N 份，B 按行切成 N 份。
    let mut a_parts = Vec::with_capacity(n_devices);
    let mut b_parts = Vec::with_capacity(n_devices);
    for d in 0..n_devices {
        let mut a_d = Vec::with_capacity(m * k_i);
        for i in 0..m {
            for t in 0..k_i {
                a_d.push(a[i * k + d * k_i + t]);
            }
        }
        a_parts.push(a_d);
        b_parts.push(b[d * k_i * n..(d + 1) * k_i * n].to_vec());
    }

    let total_start = Instant::now();

    // 每个逻辑设备用自己的 (A_i, B_i) 算部分和 C_i [M,N]，全部放 GPU buffer。
    let module = comm.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("multi matmul shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
    });
    let pipeline = comm.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("multi matmul pipeline"),
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });
    let dims_buf = comm.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multi matmul dims"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    comm.queue.write_buffer(
        &dims_buf,
        0,
        bytemuck::cast_slice(&[m as u32, k_i as u32, n as u32, 0u32]),
    );

    // 为 N 个设备各建 A/B/C buffer 与 bind group，一次 submit 跑完 N 路 matmul。
    let mut a_bufs = Vec::with_capacity(n_devices);
    let mut b_bufs = Vec::with_capacity(n_devices);
    let mut c_bufs = Vec::with_capacity(n_devices);
    let mut bind_groups = Vec::with_capacity(n_devices);
    for d in 0..n_devices {
        let a_buf = het_comm::make_gpu_buffer(&comm.device, a_parts[d].len());
        let b_buf = het_comm::make_gpu_buffer(&comm.device, b_parts[d].len());
        let c_buf = het_comm::make_gpu_buffer(&comm.device, m * n);
        comm.queue.write_buffer(
            as_gpu(&a_buf).unwrap(),
            0,
            bytemuck::cast_slice(&a_parts[d]),
        );
        comm.queue.write_buffer(
            as_gpu(&b_buf).unwrap(),
            0,
            bytemuck::cast_slice(&b_parts[d]),
        );
        let bg = comm.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("multi matmul bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: as_gpu(&a_buf).unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: as_gpu(&b_buf).unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: as_gpu(&c_buf).unwrap(),
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
        a_bufs.push(a_buf);
        b_bufs.push(b_buf);
        c_bufs.push(c_buf);
        bind_groups.push(bg);
    }

    let mut encoder = comm.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi matmul encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multi matmul pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        for bg in &bind_groups {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(((m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1);
        }
    }
    comm.queue.submit(Some(encoder.finish()));

    // GPU 端 all-reduce：N 个部分和相加，结果 broadcast 回每个 buffer。
    let mut out = TensorBuf::Cpu(Vec::new());
    reducer.all_reduce(&c_bufs, &mut out, m * n);
    let total_us = total_start.elapsed().as_micros();
    let got = match out {
        TensorBuf::Cpu(v) => v,
        TensorBuf::Gpu(_) => unreachable!(),
    };

    // 与 dense 参考比对。
    let dense = dense_matmul(&a, &b, m, k, n);
    let err = max_abs_error(&got, &dense);

    // 同时用 CPU pairwise 参考验证 all-reduce 本身（GPU reduce 正确性交叉验证）。
    let cpu_sum: Vec<f32> = (0..m * n)
        .map(|idx| {
            // 重新用 CPU 算 N 个部分和的对应元素并累加（用 dense 拆分验证）
            (0..n_devices)
                .map(|d| {
                    let a_d = &a_parts[d];
                    let b_d = &b_parts[d];
                    let row = idx / n;
                    let col = idx % n;
                    (0..k_i)
                        .map(|t| a_d[row * k_i + t] * b_d[t * n + col])
                        .sum::<f32>()
                })
                .sum::<f32>()
        })
        .collect();
    let all_reduce_ok = max_abs_error(&got, &cpu_sum) < 1e-4;

    MultiPocResult {
        adapter_name: info.name,
        n_devices,
        k_per_device: k_i,
        max_abs_error: err,
        all_reduce_ok,
        total_us,
    }
}
