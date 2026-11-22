//! 跨设备传输实现（M2）。
//!
//! H2D：`queue.write_buffer`。
//! D2H：`map_async(MapMode::Read)` → `poll` → `get_mapped_range` → `unmap`。
//!
//! 注意：wgpu 的 buffer map 是异步的，同一 buffer 处于 mapped 状态时不能再次写入。
//! 本模块严格遵循 map → read → drop view → unmap 顺序。

use crate::TensorBuf;

/// 创建一个可用于 storage + 双向拷贝的 GPU buffer。
pub fn make_gpu_buffer(device: &wgpu::Device, len: usize) -> TensorBuf {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("het-comm gpu buffer"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    TensorBuf::Gpu(buffer)
}

/// 从 `TensorBuf` 中取出 GPU buffer 引用（非 GPU 变体返回 None）。
pub fn as_gpu(buf: &TensorBuf) -> Option<&wgpu::Buffer> {
    match buf {
        TensorBuf::Gpu(b) => Some(b),
        TensorBuf::Cpu(_) => None,
    }
}

/// H2D：把 CPU 切片写入 GPU storage buffer。
pub fn host_to_device(queue: &wgpu::Queue, src: &[f32], dst: &wgpu::Buffer) {
    queue.write_buffer(dst, 0, bytemuck::cast_slice(src));
}

/// GPU→GPU：同一 logical device 内两个 buffer 之间的拷贝（P5）。
///
/// 真实多物理卡场景下跨 device 拷贝受 wgpu 限制，需回 host staging 中转；
/// 本 PoC 用同一 device 上的两个独立 storage buffer 模拟两个逻辑设备，
/// 其通信算法（ring all-reduce）与真实多卡一致，只是物理路径走本 device 显存。
pub fn device_to_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    len: usize,
) {
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("het-comm gpu2gpu copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, byte_len);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}

/// D2H：把 GPU buffer 读回 CPU，返回前 `len` 个 f32。
///
/// WebGPU 规范不允许 `MAP_READ` 与 `STORAGE` 同时存在，因此走 staging buffer：
/// `copy_buffer_to_buffer(src → staging)` → map staging → 读取 → unmap。
pub fn device_to_host(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let byte_len = (len * std::mem::size_of::<f32>()) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("het-comm d2h staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("het-comm d2h copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_result| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let view = staging
        .get_mapped_range(..)
        .expect("D2H 失败：staging buffer 未成功映射为可读");
    let floats = bytemuck::cast_slice::<u8, f32>(&view[..]);
    let out = floats[..len].to_vec();

    drop(view);
    staging.unmap();
    out
}
