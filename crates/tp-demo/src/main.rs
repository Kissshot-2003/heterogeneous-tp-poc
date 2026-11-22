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

    println!("\n=== 结果（P0：CPU 与 GPU 重叠）===");
    println!(
        "dims: M={} K={} N={} | 非均匀切分: CPU={} 列 / GPU={} 列 ({:.0}%)",
        DEFAULT_M,
        DEFAULT_K,
        DEFAULT_N,
        r.cpu_cols,
        r.gpu_cols,
        r.gpu_cols as f64 / DEFAULT_N as f64 * 100.0
    );
    println!("h2d_submit_us    = {}", r.h2d_submit_us);
    println!(
        "cpu_matmul_us    = {}   (与 GPU 执行重叠)",
        r.cpu_matmul_us
    );
    println!(
        "gpu_wall_us      = {}   (submit→collect，含 CPU 并行)",
        r.gpu_wall_us
    );
    println!(
        "gpu_compute_us   = {}   (P2 timestamp query 纯 GPU 时间)",
        r.gpu_compute_us
            .map(|v| v.to_string())
            .unwrap_or_else(|| "N/A (不支持 TIMESTAMP_QUERY)".to_string())
    );
    println!("d2h_transfer_us  = {}", r.d2h_transfer_us);
    println!("total_us         = {}", r.total_us);
    println!("overlap_saving_us= {}", r.overlap_saving_us);
    println!("max_abs_error    = {:.3e}", r.max_abs_error);

    assert!(r.h2d_roundtrip_ok, "M2a 失败：H2D→D2H 回环不一致");
    assert!(r.smoke_compute_ok, "M2b 失败：smoke compute pass 结果不符");
    assert!(
        r.max_abs_error < 1e-4,
        "M3/M4 失败：max_abs_error = {} >= 1e-4",
        r.max_abs_error
    );
    println!("M3/M4 通过：列切分 + gather 结果与 dense 参考一致 (err < 1e-4)。");
}
