//! Dense 参考计算与数值比对（M3/M4）。
//!
//! 参考结果在 CPU 上独立算一遍完整 `[M,K] × [K,N]`，
//! 再与 TP 切分+合并的结果逐元素比对。
//! 为避免使用同源 bug 互相掩盖，这里用与 `matmul_cpu` 不同的循环顺序
//! （i → j → t，内层 accum）。

/// 独立的 dense 参考实现。
pub fn dense_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// 返回逐元素最大绝对误差。
pub fn max_abs_error(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "比对数组长度不一致");
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max)
}
