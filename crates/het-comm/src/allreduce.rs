//! GPU 端 all-reduce（P5）。
//!
//! 用 pairwise add shader 把 N 个"部分和" buffer 累加成完整和，
//! 再把结果拷回每个 buffer，使所有参与方都持有相同结果（all-reduce 语义）。
//!
//! 注意：本机只有 1 张物理 GPU，这里用同一 logical device 上的 N 个独立
//! storage buffer 模拟 N 个逻辑设备。算法（pairwise reduce + broadcast）、
//! 同步与 buffer 抽象均与真实多卡一致，物理传输走本 device 显存。

use crate::{device_to_device, TensorBuf};

/// GPU 端归约器，持有一个逐元素相加的 compute pipeline。
pub struct GpuReducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl GpuReducer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reduce_sum shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/reduce_sum.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce_sum pipeline"),
            layout: None,
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
        }
    }

    /// 把 `parts` 中的所有 buffer 按元素相加，结果写入 `out`，
    /// 随后把结果拷回每个输入 buffer（all-reduce：所有参与方持有完整和）。
    pub fn all_reduce(&self, parts: &[TensorBuf], out: &mut TensorBuf, len: usize) {
        assert!(parts.len() >= 2, "all-reduce 至少需要 2 个分片");
        let TensorBuf::Cpu(out_vec) = out else {
            panic!("all_reduce 输出必须是 TensorBuf::Cpu");
        };

        // 1) pairwise reduce：acc = buf0；acc = acc + bufi（i=1..N）
        let mut acc = self.clone_to_new(&parts[0], len);
        for part in &parts[1..] {
            let next = self.add(&acc, part, len);
            acc = next;
        }

        // 2) broadcast：把 acc 拷回每个参与方 buffer（all-reduce 语义）
        for part in parts {
            if let TensorBuf::Gpu(dst) = part {
                device_to_device(&self.device, &self.queue, as_gpu_ref(&acc).unwrap(), dst, len);
            }
        }

        // 3) 输出到 CPU（供比对）；同时 acc 留在 GPU 供 broadcast 使用
        *out_vec = crate::device_to_host(&self.device, &self.queue, as_gpu_ref(&acc).unwrap(), len);
    }

    /// 把 src buffer 复制到一个新的 GPU buffer。
    fn clone_to_new(&self, src: &TensorBuf, len: usize) -> TensorBuf {
        let dst = crate::make_gpu_buffer(&self.device, len);
        if let TensorBuf::Gpu(s) = src {
            device_to_device(&self.device, &self.queue, s, as_gpu_ref(&dst).unwrap(), len);
        }
        dst
    }

    /// out = a + b（逐元素），返回新的 GPU buffer。
    fn add(&self, a: &TensorBuf, b: &TensorBuf, len: usize) -> TensorBuf {
        let a_buf = as_gpu_ref(a).expect("add 的 a 必须是 GPU buffer");
        let b_buf = as_gpu_ref(b).expect("add 的 b 必须是 GPU buffer");
        let out = crate::make_gpu_buffer(&self.device, len);
        let out_buf = as_gpu_ref(&out).unwrap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_sum bind group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: a_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: out_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce_sum encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce_sum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((len + 63) / 64) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        out
    }
}

/// 从 TensorBuf 取 GPU buffer 引用（本地 helper，避免循环依赖 transport）。
fn as_gpu_ref(buf: &TensorBuf) -> Option<&wgpu::Buffer> {
    match buf {
        TensorBuf::Gpu(b) => Some(b),
        TensorBuf::Cpu(_) => None,
    }
}
