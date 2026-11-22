// GPU compute matmul（WGSL → naga → SPIR-V，无需 Vulkan SDK / glslc）
//
// C[M,N] = A[M,K] × B[K,N]
// 本 PoC 中 GPU 只算右半：B1 为 [K, N/2]，输出 C1 为 [M, N/2]。
// dims.n 传入 N/2（右半列数），因此 C1 的行主序 stride 为 N/2。

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform> dims : Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= dims.m || col >= dims.n) {
        return;
    }

    var sum = 0.0;
    for (var t = 0u; t < dims.k; t = t + 1u) {
        sum = sum + a[row * dims.k + t] * b[t * dims.n + col];
    }
    c[row * dims.n + col] = sum;
}
