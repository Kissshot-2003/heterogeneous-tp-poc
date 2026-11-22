// 逐元素相加：out[i] = a[i] + b[i]
// 用于 GPU 端 all-reduce 的 pairwise 累加（P5）

@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    out[i] = a[i] + b[i];
}
