//! 端到端回归测试。
//!
//! 把 HANDOVER.md 第 6 节的验收标准变成一键可跑的 `cargo test`，
//! 防止 P0（异步重叠）等后续改动破坏已验证的正确性。

use std::sync::Mutex;

use het_comm::{CommDomain, TensorBuf, VulkanComm};
use tp_demo::dense::{dense_matmul, max_abs_error};
use tp_demo::matmul_cpu::matmul_cpu;
use tp_demo::run_poc;

/// 多个测试同时请求 Vulkan adapter 时会偶发 NotFound（active_backends=0x0），
/// 用进程内 Mutex 把需要 GPU 的测试串行化。
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// 验收标准 1~5 的端到端验证（需要本机有 Vulkan GPU）。
#[test]
fn poc_end_to_end() {
    let _guard = GPU_LOCK.lock().unwrap();
    let r = run_poc();

    // 1. 必须走 Vulkan backend（不断言具体型号，保证测试可移植）
    assert_eq!(r.backend, wgpu::Backend::Vulkan, "未走 Vulkan backend");
    assert!(!r.adapter_name.is_empty(), "adapter 名称为空");

    // 2. M2a：H2D→D2H 回环
    assert!(r.h2d_roundtrip_ok, "H2D→D2H 回环不一致");

    // 3. M2b：smoke compute pass 真实执行且结果正确
    assert!(r.smoke_compute_ok, "smoke compute pass 结果不符");

    // 4. M4：数值正确
    assert!(
        r.max_abs_error < 1e-4,
        "max_abs_error = {} >= 1e-4",
        r.max_abs_error
    );

    // 5. 计时字段全部产生
    assert!(r.cpu_matmul_us > 0);
    assert!(r.h2d_submit_us > 0);
    assert!(r.gpu_wall_us > 0);
    assert!(r.d2h_transfer_us > 0);
    assert!(r.total_us > 0);
}

/// P0 验证：CPU 计算确实被塞进 GPU 等待窗口里。
///
/// 断言 `overlap_saving_us > 0` 且不超过 CPU 时间本身；
/// 这保证 submit → collect 之间确实有 CPU 工作发生（而非串行执行）。
#[test]
fn p0_cpu_work_overlaps_gpu_wait() {
    let _guard = GPU_LOCK.lock().unwrap();
    let r = run_poc();
    assert!(
        r.overlap_saving_us > 0,
        "P0 失败：没有观察到 CPU/GPU 重叠"
    );
    assert!(
        r.overlap_saving_us <= r.cpu_matmul_us,
        "重叠节省不应超过 CPU 计算时间本身"
    );
    assert!(
        r.overlap_saving_us <= r.gpu_wall_us,
        "重叠节省不应超过 GPU 阶段墙钟"
    );
}

/// 坑 3 回归测试：列切分的 gather 必须按行交错，不能整块拼接。
///
/// 若有人把 gather 改回整块拼接，这个测试会立刻失败（误差达 3.9）。
#[test]
fn gather_is_row_interleaved() {
    let _guard = GPU_LOCK.lock().unwrap();
    let adapter = pollster::block_on(het_comm::request_vulkan_adapter());
    let comm = pollster::block_on(VulkanComm::new(&adapter));

    // 两个 [2,2] 分片：C0 的行是 [1,2] [3,4]，C1 的行是 [5,6] [7,8]。
    let parts = [
        TensorBuf::Cpu(vec![1.0, 2.0, 3.0, 4.0]),
        TensorBuf::Cpu(vec![5.0, 6.0, 7.0, 8.0]),
    ];
    let mut out = TensorBuf::Cpu(Vec::new());
    comm.gather_to_cpu(&parts, &mut out, 2, 2);

    // 期望按行交错：[C0行0, C1行0, C0行1, C1行1]
    let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    match out {
        TensorBuf::Cpu(v) => assert_eq!(v, expected, "gather 必须按行交错拼接"),
        TensorBuf::Gpu(_) => unreachable!(),
    }
}

/// CPU matmul 与 dense 参考在小矩阵上逐元素相等。
#[test]
fn matmul_cpu_matches_dense() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let got = matmul_cpu(&a, &b, 2, 2, 2);
    let dense = dense_matmul(&a, &b, 2, 2, 2);
    assert_eq!(got, dense);
    assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);
}

/// 比对函数本身：能检测出不一致。
#[test]
fn max_abs_error_detects_mismatch() {
    assert_eq!(max_abs_error(&[1.0, 2.0], &[1.5, 2.0]), 0.5);
    assert_eq!(max_abs_error(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
}
