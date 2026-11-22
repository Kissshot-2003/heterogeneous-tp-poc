//! Collectives 实现（M2 / P2c）。
//!
//! 本 PoC 只需要 2 路 gather：把 GPU 算出的分片读回 CPU，
//! 与 CPU 算出的分片拼接成完整输出。
//! P2c 起支持**不等宽分片**（非均匀切分）：每个 part 有自己的宽度。
//! 真正的 all-reduce 留到后续阶段（见 HANDOVER.md 第 9 节）。

use std::borrow::Cow;

use crate::{device_to_host, TensorBuf};

/// 把各设备分片按行交错拼接回 CPU 输出（支持不等宽分片）。
///
/// - `parts`：每个分片是 `rows × widths[i]` 的行主序 f32。
/// - `widths`：每个分片的列数，长度必须与 `parts` 相同。
/// - `out`：拼接结果，大小为 `rows × sum(widths)`。
pub fn gather_to_cpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    parts: &[TensorBuf],
    out: &mut TensorBuf,
    rows: usize,
    widths: &[usize],
) {
    assert_eq!(parts.len(), widths.len(), "parts 与 widths 长度不一致");
    let TensorBuf::Cpu(out_vec) = out else {
        panic!("gather_to_cpu 的输出必须是 TensorBuf::Cpu");
    };

    out_vec.clear();
    out_vec.reserve(rows * widths.iter().sum::<usize>());

    // 先把所有分片取到 CPU（GPU 分片经 staging buffer 读回）。
    let mut matrices: Vec<Cow<'_, [f32]>> = Vec::with_capacity(parts.len());
    for (part, &width) in parts.iter().zip(widths.iter()) {
        let part_len = rows * width;
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

    // 列切分的分片每行宽度可能不同，必须按行交错拼接，不能整块拼接（坑 3）。
    for i in 0..rows {
        for (mat, &width) in matrices.iter().zip(widths.iter()) {
            out_vec.extend_from_slice(&mat[i * width..(i + 1) * width]);
        }
    }
}

/// 对所有分片按元素求和（all-reduce 的 reduce-to-CPU 侧，P4）。
///
/// 真正的 TP（行并行）里，各设备持有 `A_i × B_i` 的部分和
/// `[M, N]`，层末 all-reduce 把它们相加得到完整输出。
/// 本实现把 GPU 分片读回 CPU 后逐元素累加；2 设备下等价于 reduce-to-root。
pub fn all_reduce_to_cpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    parts: &[TensorBuf],
    out: &mut TensorBuf,
    rows: usize,
    cols: usize,
) {
    let len = rows * cols;
    let TensorBuf::Cpu(out_vec) = out else {
        panic!("all_reduce_to_cpu 的输出必须是 TensorBuf::Cpu");
    };
    out_vec.clear();
    out_vec.resize(len, 0.0f32);

    for part in parts {
        match part {
            TensorBuf::Cpu(v) => {
                assert_eq!(v.len(), len, "CPU 分片尺寸不符");
                for (o, &x) in out_vec.iter_mut().zip(v.iter()) {
                    *o += x;
                }
            }
            TensorBuf::Gpu(buf) => {
                let v = device_to_host(device, queue, buf, len);
                for (o, &x) in out_vec.iter_mut().zip(v.iter()) {
                    *o += x;
                }
            }
        }
    }
}
