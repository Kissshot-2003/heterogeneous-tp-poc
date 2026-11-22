//! 张量并行 PoC 验证程序入口。
//!
//! 核心逻辑在 `lib.rs::run_poc()`，集成测试共用同一份流程，
//! 避免"测试通过但 demo 行为不同"的偏差。
//!
//! 运行：cargo run --offline -p tp-demo --release

use tp_demo::{run_poc, run_poc_multi, DEFAULT_K, DEFAULT_M, DEFAULT_N};

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

    println!("\n=== 结果（P4：真 TP 行并行 + all-reduce）===");
    println!(
        "dims: M={} K={} N={} | 行并行: CPU K={} / GPU K={} (各算 [M,N] 部分和)",
        DEFAULT_M, DEFAULT_K, DEFAULT_N, r.cpu_k, r.gpu_k
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
        "P4 失败：max_abs_error = {} >= 1e-4",
        r.max_abs_error
    );
    println!("P4 通过：行并行 + all-reduce 结果与 dense 参考一致 (err < 1e-4)。");

    // ===== P5：N 逻辑设备端到端 TP（单卡多 buffer 模拟）=====
    println!("\n=== 结果（P5：4 逻辑设备 TP + GPU all-reduce）===");
    let multi = run_poc_multi(4);
    println!(
        "逻辑设备数={} | K 每设备={} | Adapter={}",
        multi.n_devices, multi.k_per_device, multi.adapter_name
    );
    println!("all_reduce_ok   = {}", multi.all_reduce_ok);
    println!("total_us        = {}", multi.total_us);
    println!("max_abs_error   = {:.3e}", multi.max_abs_error);
    assert!(
        multi.all_reduce_ok,
        "P5 失败：GPU all-reduce 与 CPU pairwise 参考不一致"
    );
    assert!(
        multi.max_abs_error < 1e-4,
        "P5 失败：max_abs_error = {} >= 1e-4",
        multi.max_abs_error
    );
    println!("P5 通过：4 设备行并行 + GPU 端 all-reduce 结果与 dense 参考一致。");
}
