# heterogeneous-tp-poc

**一次 CPU + Vulkan 异构张量并行的探索：从通信域到 4 个反直觉的实测发现**

> 用 Rust + wgpu 建立跨 CPU / Vulkan GPU 的统一通信域，跑通真正的张量并行（行并行 + all-reduce）。
> 这是一个**可复现的实验性 PoC**，不是生产级推理引擎——它的价值在于实测数据与踩坑记录。

## 起点

一个简单的问题：**推理引擎的张量并行，是否都必须同型号 GPU？**

主流框架（vLLM / TensorRT-LLM）默认同构——TP 的同步通信、显存对齐、NCCL 拓扑都偏好同型号卡。想打破这个限制、把 CPU 和 Vulkan GPU 纳入进来时，会撞上一个现实问题：

> **没有跨 CPU / GPU 的统一通信域**（NCCL 只覆盖 NVIDIA GPU）。

这个仓库记录了一次从零造最小通信域、并跑通异构 TP 的完整探索。

## 四个反直觉的实测发现

这不是一个"性能突破"的故事。恰恰相反，端到端异构方案**比纯 CPU 还慢**——但过程中挖出了几条纸上推不出来、社区少有实证的结论：

| # | 发现 | 数据 |
|---|------|------|
| 1 | **GPU 本身很快，慢的是提交/同步** | timestamp query 实测：纯 GPU 计算仅 34µs，但 GPU 墙钟 ~300µs——**88% 是 submit/poll/map 的固定开销**。调 shader 是盲调 |
| 2 | **wgpu 有 16KB staging 快路径阈值** | D2H 的 staging buffer 超过 16KB 后，`map` 耗时从 ~40µs 暴涨到 ~300µs（8×）。直接否决了"让 GPU 多干活"的纸上最优切分 |
| 3 | **简单多线程并行反而更慢** | rayon 按行并行 30ms vs naive 单线程 25ms；改成 cache-aware 分块后才 8ms（3× 加速）——多线程争抢 B 缓存抵消了收益 |
| 4 | **map_async 等待在途 compute 有 ~1.7ms 异常延迟** | 把 compute 与 D2H copy 合并进同一次 submit 后规避；这也成了 P2b 优化的关键 |

这些数字强依赖测试硬件（见下），换机器需重新测量。

## 快速开始

```powershell
git clone <your-repo-url> heterogeneous-tp-poc
cd heterogeneous-tp-poc
cargo run --offline -p tp-demo --release   # 跑 Demo（P4 单设备 + P5 4 设备两段）
cargo test --offline --workspace           # 9 个回归测试全绿
```

> 首次构建：`--offline` 要求依赖已在本地缓存。换机器时去掉 `--offline` 让 cargo 联网拉取（wgpu 30 / rayon / bytemuck / pollster 等）。

预期输出（热启动典型值，数字允许波动）：

```
=== 结果（P4：真 TP 行并行 + all-reduce）===
dims: M=64 K=256 N=128 | 行并行: CPU K=128 / GPU K=128
gpu_compute_us = 45        (P2 timestamp 纯 GPU 时间)
max_abs_error  = 1.192e-6  (< 1e-4 => PASS)

=== 结果（P5：4 逻辑设备 TP + GPU all-reduce）===
逻辑设备数=4 | K 每设备=64
all_reduce_ok  = true
max_abs_error  = 1.490e-6
```

## 测试硬件

| 项 | 值 |
|----|----|
| CPU | AMD Ryzen 5 5600（6 核 12 线程） |
| GPU | AMD Radeon RX 5700 XT（Vulkan 1.4，独显） |
| 工具链 | Rust + wgpu 30（Vulkan backend）+ WGSL（naga 编译 SPIR-V） |
| 环境 | 无 C++ 编译器、无 Vulkan SDK——wgpu 让 WGSL 直接编译成 SPIR-V，零 SDK 依赖 |

## 做了什么（P0 → P5）

| 阶段 | 内容 | 结论 |
|------|------|------|
| 底座 | `TensorBuf` / `CommDomain` 统一通信域 + H2D/D2H(staging) | 列切分 + gather 验证可行性 |
| P0 | submit/collect 拆分，CPU/GPU 重叠 | 收益 23% |
| P1 | 16×16 shared memory tiling | GPU 执行快 20%，端到端不可见 |
| P2 | timestamp query 精确测量 | 发现 1、2 |
| P2b | 合并 compute+D2H 到单次 submit | D2H 140→42µs |
| P2c | 非均匀切分尝试 | 失败，但发现阈值（发现 2） |
| P3 | cache-aware 分块 + rayon | 发现 3；大形状 3× 加速 |
| P4 | 行并行 + all-reduce | **真正的 TP**，误差 1.19e-6 |
| P5 | GPU→GPU copy + GPU 端 all-reduce | 4 逻辑设备端到端 TP，误差 1.49e-6 |

## 当前状态

```
cargo test: 9 passed; 0 failed
P4 max_abs_error: 1.192e-6
P5 4-device max_abs_error: 1.490e-6
```

## 重要说明 / 局限

- **这是 PoC，不是生产代码**。端到端没有净性能收益（比纯 CPU 慢），价值在通信域与实测数据。
- **"多设备"是单卡模拟**：本机只有 1 张物理 GPU，用同一 logical device 上的 N 个独立 storage buffer 模拟 N 个逻辑设备。通信算法与真实多卡一致，物理传输走本 device 显存。
- **跨厂商未验证**：没有第二张物理 GPU，无法测真实 P2P / ring all-reduce。
- **形状很小**：默认 64×256×128，只够验证逻辑；大模型下 16KB 阈值会被跨越，非均匀切分可能重新有利。

## 文档

- [HANDOVER.md](HANDOVER.md) — 完整交接文档（架构、全部踩坑、性能解读、路线图）
- [PLAN.md](PLAN.md) — 项目方案（目标、API、里程碑、验收标准）
- [BLOG.md](BLOG.md) — 社区文章底稿（结论摘要，可直接改写发布）
- `crates/het-comm` — 统一通信域 crate
- `crates/tp-demo` — 验证程序

## 许可

MIT — 见 [LICENSE](LICENSE)。
