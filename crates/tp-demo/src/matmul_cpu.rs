//! CPU 侧矩阵乘法（M3）。
//!
//! 当前为 naive 实现。PoC 通过后，把此函数改为 FFI 调用 oneDNN，
//! `CommDomain` 与上层调度逻辑无需改动（见 PLAN.md 第 9 节）。

/// C[M,N] = A[M,K] × B[K,N]，全部为 f32 行主序连续数组。
///
/// 循环顺序 i → t → j：A 按行复用、B 按行连续访问，对缓存友好。
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A 尺寸不符");
    assert_eq!(b.len(), k * n, "B 尺寸不符");

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for t in 0..k {
            let av = a[i * k + t];
            for j in 0..n {
                c[i * n + j] += av * b[t * n + j];
            }
        }
    }
    c
}
