# 异构张量并行 PoC —— 交接文档

> 交接日期：2026-11-22
> 项目路径：`C:\通用工程\10_heterogeneous-tp-poc`
> 状态：**PoC 四个里程碑全部完成，端到端跑通，数值误差为 0**

---

## 1. 一页速览（TL;DR）

| 项 | 内容 |
|----|------|
| **做了什么** | 用 Rust + wgpu 建立跨 CPU / Vulkan GPU 的统一通信域，跑通一次真实的张量并行切分：矩阵按列切成两半，CPU 算左半、GPU 算右半，按行交错 gather 回 CPU，与 dense 参考逐元素比对 |
| **结果** | `max_abs_error = 0.000e0`（M=64, K=256, N=128），四个里程碑全绿 |
| **硬件** | AMD Ryzen 5 5600 + AMD Radeon RX 5700 XT（Vulkan 1.4） |
| **技术栈** | Rust + wgpu 30（Vulkan backend）+ WGSL（naga 编译 SPIR-V） |
| **怎么跑** | `cd C:\通用工程\10_heterogeneous-tp-poc && cargo run --offline -p tp-demo --release` |
| **下一步** | 优先级最高的是 **P0：event-based 异步重叠**（见第 9 节） |

一句话结论：**可行性已验证，但当前版本没有性能收益**——GPU shader 是朴素实现，端到端比纯 CPU 还慢。这是预期内的，PoC 的目的是验证通信域底座，不是性能。

---

## 2. 背景：为什么做这个

讨论链条：

1. 主流推理引擎（vLLM / TensorRT-LLM / SGLang）的张量并行（TP）默认同型号 GPU——因为 TP 是同步紧耦合的，木桶效应 + 显存对齐 + NCCL 通信拓扑都要求同构。
2. 想打破这个限制，先扩大到 **oneDNN（CPU）+ Vulkan（跨厂商 GPU）**。
3. 调研发现：oneDNN 和 Vulkan 在同一个框架里是**并列后端**（oneDNN 的 GPU 引擎走 SYCL/OpenCL，不走 Vulkan），不是组合关系；异构 TP 主要是学术探索（HeteGen、HexGen、AsymGroup 等）。
4. 真正的瓶颈是**没有统一的跨设备通信域**——没有 NCCL 这种现成工具能覆盖 CPU ↔ Vulkan GPU。
5. 因此本 PoC 的最小目标：先造一个最小的统一通信域，证明它能支撑一次 TP 切分并得到正确结果。

相关工作（供接手人继续调研）：
- **HeteGen**（MLSys 2024）：CPU+GPU 异构 TP，非均匀切分权重 + 通信/计算异步重叠，OPT-30B 最高提升 317%
- **HexGen**（MLR 2024）：跨数据中心异构 GPU，TP+PP 非对称划分
- **AsymGroup**（IEEE Access 2025）：按设备能力比例分配 2D TP 工作负载
- **Dovetail**（EMNLP 2025）：CPU/GPU 异构推测解码
- llama.cpp：SYCL 后端用 oneDNN 做 GEMM，支持 `--split-mode tensor`（仅 2 卡优化）

---

## 3. 环境事实与关键决策

### 3.1 实测环境

| 项 | 值 |
|----|----|
| CPU | AMD Ryzen 5 5600，6 核 12 线程 |
| GPU | AMD Radeon RX 5700 XT（独显），Vulkan 1.4.315，驱动 25.10.30.02 |
| C++ 编译器 | **无**（`cl`/`clang`/`g++` 均不在 PATH，vswhere 无输出） |
| Vulkan SDK | **无**（`VULKAN_SDK` 未设置，`glslc` 缺失） |
| Rust | `cargo`/`rustc` 可用 |
| 本地 crate 缓存 | `wgpu 30.0.2`、`naga 30`、`pollster`、`bytemuck`、`ash 0.38` 已缓存 |

### 3.2 为什么是 Rust + wgpu 而不是 C++ + Vulkan SDK

本机没有 C++ 编译器和 Vulkan SDK，装它们成本不必要。wgpu 用 WGSL 由 naga 编译成 SPIR-V，**完全不需要 Vulkan SDK**，且能强制锁定 Vulkan backend 直接操作 AMD GPU。`--offline` 可离线构建，因为依赖已在本地缓存。

### 3.3 oneDNN 为什么没用

- 本机是 AMD CPU，oneDNN 的 AVX-512/AMX 红利拿不到；
- oneDNN 的 GPU 引擎走 SYCL/OpenCL，**不走 Vulkan**，两者是并列后端而非组合；
- PoC 阶段的核心是通信域，不是算子性能。`matmul_cpu` 已做成独立函数，后续通过 FFI 替换成 oneDNN 即可，通信域代码不用改。

---

## 4. 项目结构与关键文件

```
10_heterogeneous-tp-poc/
├── Cargo.toml              # workspace：het-comm + tp-demo，统一依赖版本
├── Cargo.lock              # 已生成，离线可复现
├── README.md               # 快速开始 + 状态表
├── PLAN.md                 # 完整方案（架构/API/里程碑/验收/风险）
├── HANDOVER.md             # 本文档
├── .gitignore
├── crates/
│   ├── het-comm/           # 【核心交付物】统一通信域 crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # TensorBuf + CommDomain 定义、模块导出
│   │       ├── device.rs       # request_vulkan_adapter()：强制 Vulkan 枚举
│   │       ├── transport.rs    # make_gpu_buffer / as_gpu / H2D / D2H（staging buffer）
│   │       ├── collectives.rs  # gather_to_cpu：按行交错拼接
│   │       └── domain.rs       # VulkanComm：CommDomain 的具体实现
│   └── tp-demo/            # 验证程序
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs         # M1~M4 流程编排 + gpu_matmul() + 计时报告
│           ├── matmul_cpu.rs   # CPU naive matmul（未来 FFI 换 oneDNN 的位置）
│           ├── dense.rs        # dense 参考 + max_abs_error
│           └── shaders/
│               └── matmul.wgsl # GPU compute matmul（WGSL → SPIR-V）
```

**接手人优先读**：`het-comm/src/lib.rs`（接口）→ `domain.rs`（实现）→ `tp-demo/src/main.rs`（用法）。

---

## 5. 核心 API（必须理解）

### 5.1 `TensorBuf` —— 统一缓冲区

```rust
pub enum TensorBuf {
    Cpu(Vec<f32>),        // 宿主内存，f32 行主序
    Gpu(wgpu::Buffer),    // GPU storage buffer
}
```

所有计算和传输都以它为输入输出，避免各模块自己管内存。

### 5.2 `CommDomain` trait —— 上层唯一依赖的边界

```rust
pub trait CommDomain {
    fn copy(&self, src: &TensorBuf, dst: &mut TensorBuf, len: usize);
    fn gather_to_cpu(&self, parts: &[TensorBuf], out: &mut TensorBuf, rows: usize, cols: usize);
    fn barrier(&self);
}
```

设计原则：上层 TP 逻辑**只依赖这个 trait**，不直接碰 wgpu。未来接 oneDNN、多 GPU 或换后端时，只替换实现，不改调用方。

### 5.3 `VulkanComm` —— 当前实现

```rust
pub struct VulkanComm {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}
impl VulkanComm {
    pub async fn new(adapter: &wgpu::Adapter) -> Self { ... }
}
```

`copy` 内部自动分派：CPU→CPU、CPU→GPU（`queue.write_buffer`）、GPU→CPU（staging buffer + map）。GPU→GPU 目前是 `todo!()`（单 GPU PoC 不需要）。

### 5.4 数据流（列切分 TP）

```
A [M,K] (CPU)        B [K,N] (CPU)
                         │
                列切分 → B0 [K,N/2] (CPU)  +  B1 [K,N/2] (CPU)
                         │
   ┌─────────────────────┴─────────────────────┐
   │                                           │
CPU: C0 = A × B0                      H2D: A, B1 → GPU
   │                                  GPU: C1 = A × B1 (WGSL shader)
   │                                  D2H: C1 → CPU
   └─────────────┬─────────────────────────────┘
                 ▼
       gather（按行交错）→ C [M,N] (CPU)
                 ▼
       与 dense A × B 比对 → max_abs_error
```

**注意**：这是"伪 TP"——真正的 TP 会连 `A` 一起切，各设备持有部分权重，通信是 all-reduce。本 PoC 用列切分 + gather 是为了用最小通信量验证通信域底座。all-reduce 留到后续阶段。

---

## 6. 运行与验证

```powershell
cd C:\通用工程\10_heterogeneous-tp-poc
cargo run --offline -p tp-demo --release
```

预期输出：

```
Adapter: AMD Radeon RX 5700 XT | backend=Vulkan | vendor=0x1002 device=0x731f
M1 通过：Vulkan adapter 可用。
M2a 通过：[4] 向量 H2D→D2H 回环一致。
M2b 通过（smoke compute pass 已在 M2 阶段验证，见 PLAN.md）。

=== 结果 ===
dims: M=64 K=256 N=128 (每半 64 列)
cpu_matmul_us   = 272
h2d_transfer_us = 50
gpu_matmul_us   = 1107
d2h_transfer_us = 123
total_us        = 2310
max_abs_error   = 0.000e0
M3/M4 通过：列切分 + gather 结果与 dense 参考一致 (err < 1e-4)。
```

验收标准（5 条，全部满足才算通过）：
1. `adapter.get_info().backend == Backends::VULKAN`，设备名为 AMD Radeon RX 5700 XT
2. `max_abs_error < 1e-4`（默认形状）
3. H2D 与 D2H 各至少一次，且回传数据参与最终 gather
4. 输出 5 项耗时
5. 连续 3 次运行结果一致

> Windows PowerShell 下中文输出末尾可能出现乱码（GBK 解码 UTF-8），功能正常，不影响验收。

---

## 7. 已踩坑与解决方案（必读）

### 坑 1：wgpu 30 API 与常见教程不同

- `InstanceDescriptor` **没有实现 `Default`** → 用 `InstanceDescriptor::new_without_display_handle()` 构造后设置 `backends`。
- `Instance::new` 接受**值**而非引用。
- `RequestAdapterOptions` 多了 `apply_limit_buckets` 字段。
- `MapMode` 是 `MapMode::Read/Write`（不是 `READ/WRITE`）。
- `PollType::Wait` 是带字段的变体 → 用便捷方法 `PollType::wait_indefinitely()`。

### 坑 2：`MAP_READ` 不能与 `STORAGE` 共存

WebGPU 规范禁止两者同时出现在同一 buffer 的 usage 中。所以 D2H **不能直接 map 计算用的 storage buffer**，必须走 staging buffer：

```
copy_buffer_to_buffer(src → staging[MAP_READ|COPY_DST]) → map_async → poll → read → unmap
```

错误表现：`Usage flags BufferUsages(COPY_SRC | COPY_DST | STORAGE) ... do not contain required usage flags BufferUsages(MAP_READ)`。

### 坑 3：列切分的 gather 必须按行交错拼接（最隐蔽）

最初实现是"把 C0 整块 + C1 整块"拼接，结果 `max_abs_error = 3.93`，但 GPU 的 C1 单独比对误差为 0。

原因：每个分片是 `rows × cols`（cols = N/2）行主序，而输出是 `rows × N` 行主序。整块拼接后第 i 行开头对应的是 `C0[i*cols]`，但期望的是 `C0[i*(N/2)]` 从第 i*N 位置开始——必须用**按行交错**：

```rust
for i in 0..rows {
    for mat in &matrices {           // matrices = [C0, C1]
        out_vec.extend_from_slice(&mat[i*cols..(i+1)*cols]);
    }
}
```

这是本次调试花时间最多的地方，接手人改 gather 时务必保留这个逻辑。

### 坑 4（编译期）：闭包 `move` 捕获 `b` 导致后续借用失败

切分 B 时最初用 `flat_map(move |j| b[...])`，把 `b` 移进闭包，后面 dense 参考还要用 `b`。改成显式 `for` 循环解决。

---

## 8. 性能数据与解读

| 阶段 | 耗时（µs） |
|------|-----------|
| CPU matmul（算左半） | 272 |
| H2D 传输 | 50 |
| GPU matmul（算右半） | 1107 |
| D2H 传输 | 123 |
| **端到端 total** | **2310** |

解读：

- **GPU 比 CPU 慢得多**：1107µs vs 272µs。原因是 WGSL shader 是朴素实现——每个输出元素一个线程，串行累加 K=256 次，没有 tiling/shared memory，且 workgroup 只有 8×8。
- **传输不是主要瓶颈**：H2D 50µs + D2H 123µs 远小于 GPU 计算时间。但这是因为形状小（A=64×256，B1=256×64，C1=64×64，总共约 48KB）。
- **`total_us` 大于各阶段之和**：因为 total 包含 shader 编译、pipeline/object 创建等一次性初始化开销（在 `gpu_matmul()` 内部、计时开始前）。PoC 未做预热。
- **结论**：当前版本没有性能收益，纯 CPU 算全量大约 550µs，比异构方案快 4 倍。这**不是 PoC 失败**——PoC 验证的是可行性。性能优化是下一步的事。

---

## 9. 下一步路线（按优先级）

### P0：event-based 异步重叠（最高优先级，建议先做）

**目标**：让 CPU 计算与 H2D/D2H 传输重叠，验证异构 TP 的核心价值主张。

**为什么**：这是 HeteGen 的核心优化（论文中 317% 提升的主要来源）。当前 `barrier()` 是阻塞 poll，所有阶段串行。

**改动范围**：集中在 `CommDomain` 的同步实现，不动上层逻辑——正好体现 trait 边界的价值。

**思路**：
- 把 `barrier()` 改为返回 event/handle，或用 wgpu 的 `on_submitted_work_done` / `PollType::wait_for(submission_index)`
- 让 GPU 提交后立即启动 CPU 计算，等 GPU 完成再取结果
- 关键等式（来自 HeteGen）：`T_CPU = T_GPU + T_COM`（让 CPU 计算时间刚好覆盖 GPU 计算 + 通信）

### P1：GPU shader 优化

- workgroup size 从 8×8 调到 16×16 或 32×8
- 用 shared memory 做 tiling，减少全局内存读取
- 目标：GPU matmul 低于 CPU 的 272µs（当前 1107µs）

### P2：更大形状测试

- 试 M=512, K=1024, N=1024，看趋势是否逆转（GPU 在大规模下才有优势）
- 注意 buffer 大小和 workgroup 数量

### P3：接入 oneDNN

- 把 `matmul_cpu` 改为 FFI 调用 oneDNN
- **前置条件**：本机需要装 C++ 工具链（VS Build Tools）+ oneDNN 库；或用 Rust 的 oneDNN binding
- 注意 AMD CPU 上拿不到 AVX-512/AMX 红利

### P4：真 TP + all-reduce

- 切分 `A` 和 `B`，各设备持有部分权重
- 实现 all-reduce（ring 或 tree）替代当前的 gather
- 可参考 AsymGroup 的非均匀切分

### P5：多 GPU 与跨厂商

- 用 `ash` 直接管理多个 Vulkan 设备
- 探索跨厂商 GPU 之间 P2P 是否可用（大概率不可用，需回 host 中转）
- 实现 `CommDomain::copy` 的 GPU→GPU 分支（当前是 `todo!()`）

---

## 10. 已知限制 / 不要踩的雷

1. **同步阻塞**：`barrier()` 用 `device.poll(wait_indefinitely())`，正确性优先，无异步重叠。P0 要解决。
2. **没有 all-reduce**：当前是列切分 + gather，不是真 TP。别把它当成完整的 TP 实现。
3. **GPU→GPU 是 `todo!()`**：单 GPU PoC 用不到，多卡时必须实现。
4. **一次性初始化未计入分阶段耗时**：`total_us` 包含 shader 编译/pipeline 创建，比较性能时别直接相加各阶段。
5. **数值比对用 f32**：当前误差为 0，但换更大 K 或用不同累加顺序时可能到 1e-5 量级，验收阈值 1e-4 留有空间。
6. **`gather_to_cpu` 的按行交错逻辑不要改回整块拼接**：见坑 3。
7. **依赖离线可复现**：用 `--offline` 时依赖必须已在本地缓存；换机器时去掉 `--offline` 让 cargo 拉取。

---

## 11. 参考

- 项目内文档：`README.md`（快速开始）、`PLAN.md`（完整方案与踩坑记录）
- HeteGen, MLSys 2024 — CPU+GPU 异构 TP 与非均匀切分
- HexGen, MLR 2024 — 异构跨数据中心 TP+PP
- AsymGroup, IEEE Access 2025 — 按设备能力分配 2D TP
- Dovetail, EMNLP 2025 — CPU/GPU 异构推测解码
- wgpu 文档：<https://wgpu.rs/>
- llama.cpp SYCL/Vulkan 后端文档（GitHub `ggml-org/llama.cpp/docs/backend/`）

---

## 12. 接手人第一天清单

- [ ] 跑一遍 `cargo run --offline -p tp-demo --release`，确认 5 条验收标准
- [ ] 读 `het-comm/src/lib.rs` + `domain.rs`，理解 `TensorBuf` / `CommDomain` / `VulkanComm`
- [ ] 读 `tp-demo/src/main.rs` 的 `gpu_matmul()`，理解 H2D→compute→D2H 流程
- [ ] 读第 7 节踩坑记录，特别是坑 3（gather 按行交错）
- [ ] 决定 P0（异步重叠）是否启动；启动则先设计 event/handle 方案再动手
