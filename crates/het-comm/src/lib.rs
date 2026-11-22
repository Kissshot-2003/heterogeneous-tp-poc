//! 统一通信域（heterogeneous communication domain）。
//!
//! 本 crate 是 PoC 的核心交付物：向上层屏蔽 CPU 内存与 Vulkan GPU 显存差异，
//! 让张量并行逻辑只依赖 `CommDomain` trait，不直接操作 wgpu。
//!
//! 里程碑状态见项目根目录 `PLAN.md`。

mod collectives;
mod device;
mod domain;
mod transport;

pub use collectives::*;
pub use device::*;
pub use domain::*;
pub use transport::*;

/// 统一张量缓冲区：屏蔽 CPU 内存与 GPU 显存。
///
/// - `Cpu`：宿主内存，`f32` 行主序连续存储。
/// - `Gpu`：wgpu storage buffer，需 `COPY_SRC | COPY_DST | STORAGE` usage。
pub enum TensorBuf {
    Cpu(Vec<f32>),
    Gpu(wgpu::Buffer),
}

/// 通信域 trait：上层 TP 逻辑只依赖此接口。
///
/// 未来接入 oneDNN、多 GPU 或换后端时，只需替换实现，不改调用方。
pub trait CommDomain {
    /// 把 `src` 的前 `len` 个 f32 送到 `dst`。
    /// 实现内部自动判断 H2D / D2H / CPU→CPU。
    fn copy(&self, src: &TensorBuf, dst: &mut TensorBuf, len: usize);

    /// 把各设备分片按行交错收集到 CPU 输出（支持不等宽分片）。
    /// `parts` 为各设备分片（行主序的 `rows × widths[i]`）；
    /// `out` 为拼接后的 CPU 输出，宽度为 `sum(widths)`。
    fn gather_to_cpu(
        &self,
        parts: &[TensorBuf],
        out: &mut TensorBuf,
        rows: usize,
        widths: &[usize],
    );

    /// 对所有分片按元素求和（all-reduce 的 reduce-to-CPU 侧）。
    /// `parts` 为各设备的 `rows × cols` 部分和分片；`out` 为累加结果。
    /// 2 设备场景下 broadcast 回 GPU 可省（最终输出在 CPU），留到多卡阶段。
    fn all_reduce(&self, parts: &[TensorBuf], out: &mut TensorBuf, rows: usize, cols: usize);

    /// 等待所有在途工作完成。PoC 阶段用阻塞等待，正确性优先。
    fn barrier(&self);
}
