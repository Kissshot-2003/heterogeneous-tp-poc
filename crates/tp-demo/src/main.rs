//! 张量并行 PoC 验证程序入口。
//!
//! 核心逻辑在 `lib.rs::run_poc()`，集成测试共用同一份流程，
//! 避免"测试通过但 demo 行为不同"的偏差。
//!
//! 运行：cargo run --offline -p tp-demo --release

use tp_demo::{run_poc, DEFAULT_K, DEFAULT_M, DEFAULT_N};

fn main() {
    let r = run_poc();

    println!("Adapter: {} | backend={:?}", r.adapter_name, r.backend);
    println!("M1 通过：Vulkan adapter 可用。");
    println!(
        "M2a {}：[4] 向量 H2D→D2H 回环。",
        if r.h2d_roundtrip_ok { "通过" } else { "失败" }
    );
    println!(
        "M2b {}：smoke compute pass (data[i]+=1)。",
        if r.smoke_compute_ok { "通过" } else { "失败" }
    );

    println!("\n=== 结果 ===");
    println!(
        "dims: M={} K={} N={} (每半 {} 列)",
        DEFAULT_M, DEFAULT_K, DEFAULT_N, DEFAULT_N / 2
    );
    println!("cpu_matmul_us   = {}", r.cpu_matmul_us);
    println!("h2d_transfer_us = {}", r.h2d_transfer_us);
    println!("gpu_matmul_us   = {}", r.gpu_matmul_us);
    println!("d2h_transfer_us = {}", r.d2h_transfer_us);
    println!("total_us        = {}", r.total_us);
    println!("max_abs_error   = {:.3e}", r.max_abs_error);

    assert!(r.h2d_roundtrip_ok, "M2a 失败：H2D→D2H 回环不一致");
    assert!(r.smoke_compute_ok, "M2b 失败：smoke compute pass 结果不符");
    assert!(
        r.max_abs_error < 1e-4,
        "M3/M4 失败：max_abs_error = {} >= 1e-4",
        r.max_abs_error
    );
    println!("M3/M4 通过：列切分 + gather 结果与 dense 参考一致 (err < 1e-4)。");
}
