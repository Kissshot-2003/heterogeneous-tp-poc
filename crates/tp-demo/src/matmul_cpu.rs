//! CPU 侧矩阵乘法（M3 / P3）。
//!
//! P3：cache-aware 分块 + rayon 并行。简单按行并行会因多线程争抢 B 矩阵
//! 缓存而变慢；按输出块（BLOCK 行 × 全列）划分，每个线程有独立的 C 块，
//! B 的读取局部性更好。若后续接 oneDNN，只需替换本文件实现。

use rayon::prelude::*;

const BLOCK: usize = 64;

/// C[M,N] = A[M,K] × B[K,N]，全部为 f32 行主序连续数组。
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A 尺寸不符");
    assert_eq!(b.len(), k * n, "B 尺寸不符");

    let mut c = vec![0.0f32; m * n];

    // 按输出行块并行：par_chunks_mut 给出不重叠的可变切片，
    // 每个 rayon 任务负责 BLOCK 行 × 全列，任务数少、B 读取局部性好。
    c.par_chunks_mut(BLOCK * n)
        .enumerate()
        .for_each(|(bi, c_block)| {
            let i0 = bi * BLOCK;
            for (local_i, c_row) in c_block.chunks_mut(n).enumerate() {
                let i = i0 + local_i;
                for t in 0..k {
                    let av = a[i * k + t];
                    let b_row = &b[t * n..(t + 1) * n];
                    for (j, &bv) in b_row.iter().enumerate() {
                        c_row[j] += av * bv;
                    }
                }
            }
        });
    c
}

/// 预热 rayon 线程池，使首次启动开销不计入正式计时。
pub fn warmup_matmul_cpu() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build_global();
    let a = vec![0.0f32; 4];
    let b = vec![0.0f32; 4];
    let _ = matmul_cpu(&a, &b, 2, 2, 2);
}
