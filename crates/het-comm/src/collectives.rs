//! Collectives 实现（M2）。
//!
//! 本 PoC 只需要 2 路 gather：把 GPU 算出的右半结果读回 CPU，
//! 与 CPU 算出的左半结果拼接成完整输出。
//! 真正的 all-reduce 留到 PoC 通过后的下一阶段（见 PLAN.md 第 9 节）。

use std::borrow::Cow;

use crate::{device_to_host, TensorBuf};

/// 把各设备分片按顺序拼接回 CPU 输出。
///
/// - `parts`：每个分片是 `rows × cols` 的行主序 f32；本 PoC 最多 2 片。
/// - `out`：拼接结果，大小应为 `rows × cols × parts.len()`。
pub fn gather_to_cpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    parts: &[TensorBuf],
    out: &mut TensorBuf,
    rows: usize,
    cols: usize,
) {
    let TensorBuf::Cpu(out_vec) = out else {
        panic!("gather_to_cpu 的输出必须是 TensorBuf::Cpu");
    };

    let part_len = rows * cols;
    out_vec.clear();
    out_vec.reserve(part_len * parts.len());

    // 先把所有分片取到 CPU（GPU 分片经 staging buffer 读回）。
    let mut matrices: Vec<Cow<'_, [f32]>> = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            TensorBuf::Cpu(v) => {
                assert_eq!(v.len(), part_len, "CPU 分片尺寸不符");
                matrices.push(Cow::Borrowed(v));
            }
            TensorBuf::Gpu(buf) => {
                matrices.push(Cow::Owned(device_to_host(device, queue, buf, part_len)));
            }
        }
    }

    // 列切分的分片每行只有 cols 个元素，而输出每行有 cols * n_parts 个元素，
    // 因此必须按行交错拼接，不能整块拼接。
    for i in 0..rows {
        for mat in &matrices {
            out_vec.extend_from_slice(&mat[i * cols..(i + 1) * cols]);
        }
    }
}
