//! 具体通信域实现（M2）。
//!
//! `VulkanComm` 持有 Vulkan 的 `Device` / `Queue`，实现 `CommDomain` trait。
//! 上层 TP 逻辑只依赖 trait，未来换 oneDNN / 多卡时不改调用方。

use crate::{
    collectives::gather_to_cpu,
    device_to_host,
    host_to_device,
    CommDomain,
    TensorBuf,
};

/// 基于 Vulkan(wgpu) 的通信域。
pub struct VulkanComm {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl VulkanComm {
    /// 从已枚举的 Vulkan adapter 创建 device/queue。
    pub async fn new(adapter: &wgpu::Adapter) -> Self {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("het-comm device"),
                ..Default::default()
            })
            .await
            .expect("创建 Vulkan device 失败");
        Self { device, queue }
    }
}

impl CommDomain for VulkanComm {
    fn copy(&self, src: &TensorBuf, dst: &mut TensorBuf, len: usize) {
        match (src, dst) {
            (TensorBuf::Cpu(s), TensorBuf::Gpu(d)) => {
                host_to_device(&self.queue, &s[..len], d);
            }
            (TensorBuf::Gpu(s), TensorBuf::Cpu(d)) => {
                let v = device_to_host(&self.device, &self.queue, s, len);
                d.clear();
                d.extend_from_slice(&v);
            }
            (TensorBuf::Cpu(s), TensorBuf::Cpu(d)) => {
                d.clear();
                d.extend_from_slice(&s[..len]);
            }
            (TensorBuf::Gpu(_), TensorBuf::Gpu(_)) => {
                // 本 PoC 不需要 GPU→GPU；留到多卡阶段用 vkCmdCopyBuffer 实现。
                todo!("M2: GPU→GPU 拷贝不在 1 CPU + 1 GPU 的 PoC 范围")
            }
        }
    }

    fn gather_to_cpu(&self, parts: &[TensorBuf], out: &mut TensorBuf, rows: usize, cols: usize) {
        gather_to_cpu(&self.device, &self.queue, parts, out, rows, cols)
    }

    fn barrier(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}
