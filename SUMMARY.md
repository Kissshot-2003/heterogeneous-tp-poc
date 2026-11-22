# 一页纸总结（Executive Summary）

> 整个项目的最终结论、关键数据与边界。只想知道"做出来什么、学到了什么、能用来干嘛"的，看这一页就够。
> 详细过程见 [HANDOVER.md](HANDOVER.md)，社区叙事见 [README.md](README.md)，文章底稿见 [BLOG.md](BLOG.md)。

---

## 一句话结论

**跨 CPU / Vulkan GPU 的统一通信域能支撑真正的张量并行（行并行 + all-reduce），但端到端没有净性能收益——GPU 的算力优势被提交/同步开销吞噬。项目的价值不在"更快"，而在用实测揭示了异构推理的真实成本结构与几个隐藏陷阱。**

---

## 做出来什么（交付物）

一个可复现的 Rust 实验性 PoC，一条命令跑通全部：

| 交付物 | 内容 |
|--------|------|
| 统一通信域 `het-comm` | `TensorBuf` + `CommDomain`（copy / gather / all-reduce / barrier），屏蔽 CPU 内存与 GPU 显存 |
| 真 TP `tp-demo` | 行并行（A 按列切、B 按行切，各设备持有部分权重）+ 层末 all-reduce；4 逻辑设备端到端 |
| 测量能力 | timestamp query 分离纯 GPU 时间；5 项分阶段计时 |
| 9 个回归测试 | 覆盖回环、gather 行交错、重叠、GPU→GPU、all-reduce、多设备端到端 |
| 硬件 | AMD Ryzen 5 5600 + RX 5700 XT；Rust + wgpu 30 + WGSL（无 C++ 编译器 / 无 Vulkan SDK） |

**状态**：`cargo test` 9/9 通过；P4 误差 1.192e-6，P5 4 设备误差 1.490e-6（均远小于 1e-4）。

---

## 学到什么（4 个核心发现）

这是项目最有价值的部分——都是纸上推不出来、必须实测的东西：

| # | 发现 | 关键数据 | 直接推论 |
|---|------|---------|---------|
| 1 | **GPU 快，慢的是提交/同步** | 纯 GPU compute 34µs，GPU 墙钟 ~300µs，**88% 是固定开销** | 别优化 shader，优化 submit/poll/map 路径 |
| 2 | **wgpu 有 16KB staging 快路径阈值** | staging 超 16KB 后 D2H map 从 ~40µs 暴涨到 ~300µs（8×） | "GPU 多干活"的纸上最优切分被实证否决 |
| 3 | **简单多线程并行让 matmul 更慢** | naive 并行 30ms vs 单线程 25ms；cache-aware 分块后 8ms（3×） | 不解决数据 locality 的并行是负优化 |
| 4 | **map_async 等待在途 compute 有 ~1.7ms 异常延迟** | 合并 compute + D2H 到单次 submit 后规避 | wgpu Vulkan 后端的具体坑，合并 submit 是关键 |

**方法论结论**：这类系统里，直觉指向的瓶颈（shader）往往不是真瓶颈；timestamp 这类精确测量比猜测可靠；负面结果和被否决的路线本身就有价值。

---

## 关键数据速查（热启动，本机硬件）

| 指标 | 值 |
|------|-----|
| 纯 GPU compute（timestamp） | 34µs（8 次采样恒定） |
| GPU 阶段墙钟 | 289~474µs |
| 提交/同步开销 | ~266µs（P2）→ 202µs（P2b 合并 submit 后） |
| D2H 读回 | ~140µs → 42µs（P2b） |
| CPU matmul | 160µs（naive）→ 140µs（分块+rayon）；大形状 25ms→8ms |
| 端到端 total | ~900~1300µs |
| P4 / P5 误差 | 1.192e-6 / 1.490e-6 |

> 数字强依赖这套硬件，换机器必须重测。冷启动含 shader 编译会虚高 3~4 倍。

---

## 能做什么 / 不能做什么

**能：**
- 验证跨设备 TP 的可行性（通信域抽象、同步、all-reduce 算法都是真的）
- 作为 wgpu/Vulkan 异构计算的踩坑参考与教学样本
- 在目标形状上复现测量、对比不同通信策略

**不能（明确边界）：**
- ❌ 不是生产级推理引擎，端到端比纯 CPU 慢
- ❌ "多设备"是单卡多 buffer 模拟，无真实多物理卡
- ❌ 只测了 1 张 AMD GPU，无跨厂商验证
- ❌ 形状很小（64×256×128），大模型行为需重新测量
- ❌ 未接 vLLM / llama.cpp / oneDNN

---

## 回答最初的问题

**"推理引擎的张量并行是否都是同型号 GPU？"**

- **工程事实**：是，事实标准——TP 的同步通信、显存对齐、NCCL 拓扑都偏好同构。
- **技术本质**：不是强制。本 PoC 证明跨 CPU/GPU 的 TP 能跑通，限制不在算力而在通信开销。
- **扩大到 oneDNN + Vulkan**：两者是并列后端（oneDNN GPU 走 SYCL，不走 Vulkan），真正的统一通信域需自建——本 PoC 给出了最小可行实现。

---

## 文档索引

| 文档 | 用途 |
|------|------|
| 本文件 SUMMARY.md | 一页纸结论与速查 |
| README.md | 社区叙事：问题引入 + 4 个发现 + 快速开始 |
| HANDOVER.md | 完整交接：架构、全部踩坑、性能解读、路线图、第一天清单 |
| PLAN.md | 原始方案：API、里程碑、验收标准 |
| BLOG.md | 社区文章底稿（可直接改写发布） |
