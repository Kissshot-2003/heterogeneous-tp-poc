// GPU compute matmul —— shared memory tiling 版（P1）
//
// C[M,N] = A[M,K] × B[K,N]
// GPU 只算右半：B1 为 [K, N/2]，输出 C1 为 [M, N/2]。
// dims.n 传入 N/2（右半列数），C1 行主序 stride 为 N/2。
//
// 优化点：每 workgroup 负责一个 16×16 输出 tile，把 A/B 的 16×16 子块
// 加载到 shared memory 后复用，内层循环从 shared memory 读取，
// 全局内存读取量从 O(M·K·N) 降到 O((M/16)·(K/16)·(N/16)·16²·2)。

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

const TILE: u32 = 16u;

var<workgroup> a_tile : array<array<f32, 16>, 16>;
var<workgroup> b_tile : array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>,
) {
    let row = wid.x * TILE + lid.x;   // 输出行
    let col = wid.y * TILE + lid.y;   // 输出列
    var sum = 0.0;

    for (var t = 0u; t < dims.k; t = t + TILE) {
        // 加载 A 子块：A[row, t + lid.y] → a_tile[lid.x][lid.y]
        let ar = wid.x * TILE + lid.x;
        let ac = t + lid.y;
        if (ar < dims.m && ac < dims.k) {
            a_tile[lid.x][lid.y] = a[ar * dims.k + ac];
        } else {
            a_tile[lid.x][lid.y] = 0.0;
        }

        // 加载 B 子块：B[t + lid.x, col] → b_tile[lid.x][lid.y]
        let br = t + lid.x;
        let bc = wid.y * TILE + lid.y;
        if (br < dims.k && bc < dims.n) {
            b_tile[lid.x][lid.y] = b[br * dims.n + bc];
        } else {
            b_tile[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        // 从 shared memory 读取并累加
        for (var i = 0u; i < TILE; i = i + 1u) {
            sum = sum + a_tile[lid.x][i] * b_tile[i][lid.y];
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = sum;
    }
}
