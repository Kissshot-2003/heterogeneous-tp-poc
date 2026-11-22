//! tp-demo 核心逻辑。
//!
//! 抽成 lib 是为了让 `main.rs` 和集成测试共用同一份端到端流程，
//! 避免"测试通过但 demo 行为不同"的偏差。

pub mod dense;
pub mod matmul_cpu;

use std::time::Instant;

use het_comm::{as_gpu, CommDomain, TensorBuf, VulkanComm};

use crate::dense::{dense_matmul, max_abs_error};
use crate::matmul_cpu::matmul_cpu;

/// 默认形状：M=64, K=256, N=128（列切分后每半 64 列）。
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
    /// 分阶段耗时（微秒）
    pub cpu_matmul_us: u128,
    pub h2d_transfer_us: u128,
    pub gpu_matmul_us: u128,
    pub d2h_transfer_us: u128,
    pub total_us: u128,
}

/// GPU 侧三个阶段耗时（微秒）。
struct GpuTimings {
    h2d_us: u128,
    gpu_us: u128,
    d2h_us: u128,
}

/// 在 GPU 上算 C1 = A × B1，返回 (C1, 分阶段耗时)。
fn gpu_matmul(
    comm: &VulkanComm,
    a: &[f32],
    b1: &[f32],
    m: usize,
    k: usize,
    n2: usize,
) -> (Vec<f32>, GpuTimings) {
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

    // --- H2D：上传 A 与 B1 ---
    let t = Instant::now();
    comm.copy(&TensorBuf::Cpu(a.to_vec()), &mut a_buf, a.len());
    comm.copy(&TensorBuf::Cpu(b1.to_vec()), &mut b_buf, b1.len());
    comm.barrier();
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

    // --- GPU：dispatch compute shader ---
    let t = Instant::now();
    let mut encoder = comm.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 保守取整：@workgroup_size(8,8)
        pass.dispatch_workgroups(((m + 7) / 8) as u32, ((n2 + 7) / 8) as u32, 1);
    }
    comm.queue.submit(Some(encoder.finish()));
    comm.barrier();
    let gpu_us = t.elapsed().as_micros();

    // --- D2H：读回 C1 ---
    let t = Instant::now();
    let mut c1_buf = TensorBuf::Cpu(vec![0.0; m * n2]);
    comm.copy(&c_buf, &mut c1_buf, m * n2);
    let d2h_us = t.elapsed().as_micros();

    let c1 = match c1_buf {
        TensorBuf::Cpu(v) => v,
        TensorBuf::Gpu(_) => unreachable!(),
    };
    (
        c1,
        GpuTimings {
            h2d_us,
            gpu_us,
            d2h_us,
        },
    )
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

    // ===== M3：双路 matmul + 列切分合并 =====
    let (m, k, n) = (DEFAULT_M, DEFAULT_K, DEFAULT_N);
    let n2 = n / 2;
    assert_eq!(n % 2, 0, "N 必须能被 2 整除");

    // 确定性伪随机输入，范围 [-0.5, 0.5]。
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 97) as f32) / 97.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 31 % 89) as f32) / 89.0 - 0.5)
        .collect();

    // B 按列切成 B0 [K, N/2] 与 B1 [K, N/2]。
    let mut b0 = Vec::with_capacity(k * n2);
    let mut b1 = Vec::with_capacity(k * n2);
    for t in 0..k {
        for j in 0..n2 {
            b0.push(b[t * n + j]);
            b1.push(b[t * n + n2 + j]);
        }
    }

    let total_start = Instant::now();
    let t = Instant::now();
    let c0 = matmul_cpu(&a, &b0, m, k, n2);
    let cpu_us = t.elapsed().as_micros();
    let (c1, timings) = gpu_matmul(&comm, &a, &b1, m, k, n2);

    // Gather：按行交错拼成完整 [M, N]（见 HANDOVER.md 坑 3）。
    let parts = [TensorBuf::Cpu(c0), TensorBuf::Cpu(c1)];
    let mut gathered = TensorBuf::Cpu(Vec::with_capacity(m * n));
    comm.gather_to_cpu(&parts, &mut gathered, m, n2);
    let total_us = total_start.elapsed().as_micros();
    let got = match gathered {
        TensorBuf::Cpu(v) => v,
        TensorBuf::Gpu(_) => unreachable!(),
    };

    // ===== M4：数值比对 =====
    let dense = dense_matmul(&a, &b, m, k, n);
    let err = max_abs_error(&got, &dense);

    PocResult {
        adapter_name: info.name,
        backend: info.backend,
        h2d_roundtrip_ok,
        smoke_compute_ok,
        max_abs_error: err,
        cpu_matmul_us: cpu_us,
        h2d_transfer_us: timings.h2d_us,
        gpu_matmul_us: timings.gpu_us,
        d2h_transfer_us: timings.d2h_us,
        total_us,
    }
}
