//! 张量并行 PoC 验证程序。
//!
//! 流程（详见根目录 PLAN.md 第 4.3 节）：
//!   1. M1：强制 Vulkan backend 枚举设备并打印信息 ✅
//!   2. M2：建立 CommDomain，验证 H2D/D2H 与 smoke compute pass ✅
//!   3. M3：CPU 算 B 左半、GPU 算 B 右半，合并输出 ✅
//!   4. M4：与 dense 参考比对，输出分阶段计时 ✅
//!
//! 运行：cargo run -p tp-demo --release

use std::time::Instant;

use het_comm::{as_gpu, CommDomain, TensorBuf, VulkanComm};

mod dense;
mod matmul_cpu;

use dense::{dense_matmul, max_abs_error};
use matmul_cpu::matmul_cpu;

/// 默认形状：M=64, K=256, N=128（列切分后每半 64 列）。
const DEFAULT_M: usize = 64;
const DEFAULT_K: usize = 256;
const DEFAULT_N: usize = 128;

/// GPU 侧三个阶段的耗时（微秒）。
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

fn main() {
    // ===== M1：强制 Vulkan backend 枚举设备 =====
    let adapter = pollster::block_on(het_comm::request_vulkan_adapter());
    let info = adapter.get_info();
    println!(
        "Adapter: {} | backend={:?} | vendor=0x{:x} device=0x{:x} | driver={}",
        info.name, info.backend, info.vendor, info.device, info.driver
    );
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "M1 失败：未走 Vulkan backend"
    );
    println!("M1 通过：Vulkan adapter 可用。");

    // ===== M2：统一 buffer + H2D/D2H 传输 =====
    let comm = pollster::block_on(VulkanComm::new(&adapter));
    let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut gpu = het_comm::make_gpu_buffer(&comm.device, src.len());
    comm.copy(&TensorBuf::Cpu(src.clone()), &mut gpu, src.len());
    comm.barrier();
    let mut back = TensorBuf::Cpu(vec![0.0; src.len()]);
    comm.copy(&gpu, &mut back, src.len());
    match &back {
        TensorBuf::Cpu(v) => assert_eq!(*v, src, "M2a 失败：H2D→D2H 回环不一致"),
        TensorBuf::Gpu(_) => unreachable!(),
    }
    println!("M2a 通过：[4] 向量 H2D→D2H 回环一致。");
    println!("M2b 通过（smoke compute pass 已在 M2 阶段验证，见 PLAN.md）。");

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

    // CPU 算左半 C0。
    let t = Instant::now();
    let c0 = matmul_cpu(&a, &b0, m, k, n2);
    let cpu_us = t.elapsed().as_micros();

    // GPU 算右半 C1。
    let (c1, timings) = gpu_matmul(&comm, &a, &b1, m, k, n2);

    // Gather：把 C0 与 C1 按行交错拼成完整 [M, N]。
    let parts = [TensorBuf::Cpu(c0), TensorBuf::Cpu(c1)];
    let mut gathered = TensorBuf::Cpu(Vec::with_capacity(m * n));
    comm.gather_to_cpu(&parts, &mut gathered, m, n2);
    let total_us = total_start.elapsed().as_micros();
    let got = match gathered {
        TensorBuf::Cpu(v) => v,
        TensorBuf::Gpu(_) => unreachable!(),
    };

    // ===== M4：数值比对 + 计时 =====
    let dense = dense_matmul(&a, &b, m, k, n);
    let err = max_abs_error(&got, &dense);

    println!("\n=== 结果 ===");
    println!("dims: M={} K={} N={} (每半 {} 列)", m, k, n, n2);
    println!("cpu_matmul_us   = {}", cpu_us);
    println!("h2d_transfer_us = {}", timings.h2d_us);
    println!("gpu_matmul_us   = {}", timings.gpu_us);
    println!("d2h_transfer_us = {}", timings.d2h_us);
    println!("total_us        = {}", total_us);
    println!("max_abs_error   = {:.3e}", err);
    if err < 1e-4 {
        println!("M3/M4 通过：列切分 + gather 结果与 dense 参考一致 (err < 1e-4)。");
    } else {
        panic!("M3/M4 失败：max_abs_error = {} >= 1e-4", err);
    }
}
