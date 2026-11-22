# 异构张量并行 PoC（1 CPU + 1 Vulkan GPU）

最小工具化验证：建立跨 CPU / Vulkan GPU 的统一通信域，跑通一次真实的张量并行切分。

## 一句话目标

把一次矩阵乘法 `[M,K] × [K,N]` 按列切成两半：**CPU 算左半、GPU 算右半**，再把结果汇总回 CPU，与 dense 参考逐元素比对，并输出分阶段耗时。

## 快速开始

```powershell
cd C:\通用工程\10_heterogeneous-tp-poc
cargo run --offline -p tp-demo --release   # 跑 Demo
cargo test --offline --workspace           # 一键回归测试（4 个测试全绿）
```

预期输出（示意，数字为热启动典型值）：

```
Adapter: AMD Radeon RX 5700 XT | backend=Vulkan
M1 通过：Vulkan adapter 可用。
M2a 通过：[4] 向量 H2D→D2H 回环。
M2b 通过：smoke compute pass (data[i]+=1)。

=== 结果（P0：CPU 与 GPU 重叠）===
dims: M=64 K=256 N=128 (每半 64 列)
h2d_submit_us     =  41
cpu_matmul_us     =  161   (与 GPU 执行重叠)
gpu_wall_us       =  392   (submit→collect，含 CPU 并行)
d2h_transfer_us   =  141
total_us          =  780
overlap_saving_us =  161
max_abs_error     =  0.000e0   # vs dense reference  (< 1e-4 => PASS)
```

## 当前状态

| 里程碑 | 状态 |
|--------|------|
| M1 环境验证（wgpu 枚举 Vulkan 设备） | ✅ 已通过 |
| M2 统一 buffer + H2D/D2H 传输 + smoke compute pass | ✅ 已通过 |
| M3 CPU/GPU 双路 matmul + 列切分合并 | ✅ 已通过 |
| M4 数值比对 + 计时报告 | ✅ 已通过 |
| 自动化回归测试（5 个） | ✅ 全绿，含 gather 按行交错 + P0 重叠的回归测试 |
| P0 CPU/GPU 异步重叠 | ✅ 已完成，典型收益 23%（1011µs → 780µs） |
| P1 shared memory tiling | ✅ 已完成，GPU 执行改善 20% 但端到端不可见 |
| P2 timestamp query | ✅ 已完成，**纯 GPU 计算仅 34µs，88% 墙钟是提交/同步开销** |
| P2b 合并 submit | ✅ 已完成，D2H 140µs→42µs；**新瓶颈：CPU matmul 成关键路径** |
| P2c 非均匀切分 | ✅ 已尝试，失败：D2H 存在 16KB staging 阈值，GPU>50% 反而慢 8 倍，已回退 50/50 |
| P3 加速 CPU | ✅ 已完成（纯 Rust 分块 + rayon，未接 oneDNN）：小形状 160→140µs，大形状 25ms→8ms（3 倍） |

**完整 PoC 已跑通**：`max_abs_error = 0.000e0`（M=64, K=256, N=128）。已 `git init`，基线提交 `836d14f`。

> 注意：首次运行 GPU 阶段会包含 shader 编译开销（可达 1107µs），热启动稳定在 357~510µs；`total_us` 还包含初始化、gather 与 dense 参考计算，不要直接相加各阶段核对。详见 HANDOVER.md 第 8 节。

## 文档

- [HANDOVER.md](HANDOVER.md) — **交接文档**（接手人先读这个：架构、踩坑、性能解读、下一步路线）
- [PLAN.md](PLAN.md) — 完整项目方案（目标、架构、API、里程碑、验收标准、风险）
- [crates/het-comm](crates/het-comm) — 统一通信域 crate
- [crates/tp-demo](crates/tp-demo) — 张量并行验证程序

## 相关 upstream 工作

- [Candle #3828 张量并行跟踪卡](../07_开源共建/upstream-contributions/candle-tensor-parallel/STATUS.md) —— 向 `huggingface/candle` 贡献可复用的 `ColumnParallelLinear` / `RowParallelLinear`。本 PoC 是**自研验证**（CPU + Vulkan GPU），#3828 是**上游组件化**（多设备张量并行层），两者可互相借鉴实现，但互不依赖。

## 为什么用 Rust + wgpu

本机没有 C++ 编译器、没有 Vulkan SDK（`glslc` 缺失），但 Rust 工具链完整，`wgpu`/`naga`/`pollster`/`bytemuck` 已在本地 cargo 缓存。wgpu 用 WGSL 由 naga 编译成 SPIR-V，**完全不需要 Vulkan SDK**，且能锁定 Vulkan backend 直接操作 AMD GPU。详见 PLAN.md 第 3 节。
