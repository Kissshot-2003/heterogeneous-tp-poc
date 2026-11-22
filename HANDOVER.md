# 异构张量并行 PoC —— 交接文档

> 交接日期：2026-11-22
> 项目路径：`C:\通用工程\10_heterogeneous-tp-poc`
> 状态：**PoC 四个里程碑全部完成，4 个集成测试全绿，git 已初始化并有基线提交**

---

## 1. 一页速览（TL;DR）

| 项 | 内容 |
|----|------|
| **做了什么** | 用 Rust + wgpu 建立跨 CPU / Vulkan GPU 的统一通信域，跑通一次真实的张量并行切分：矩阵按列切成两半，CPU 算左半、GPU 算右半，按行交错 gather 回 CPU，与 dense 参考逐元素比对 |
| **结果** | `max_abs_error = 0.000e0`（M=64, K=256, N=128），四个里程碑全绿 |
| **硬件** | AMD Ryzen 5 5600 + AMD Radeon RX 5700 XT（Vulkan 1.4） |
| **技术栈** | Rust + wgpu 30（Vulkan backend）+ WGSL（naga 编译 SPIR-V） |
| **怎么跑** | `cargo run --offline -p tp-demo --release` |
| **怎么回归** | `cargo test --offline --workspace`（4 个测试全绿，含坑 3 的回归测试） |
| **版本控制** | 已 `git init`，基线提交 `836d14f`，改坏可回滚 |
| **下一步** | 优先级最高的是 **P0：event-based 异步重叠**（见第 9 节） |

一句话结论：**可行性已验证，稳态下没有净性能收益**——朴素 WGSL shader 略慢于 CPU，端到端还包含传输与初始化。这是预期内的，PoC 的目的是验证通信域底座，不是性能；详见第 8 节（含冷/热启动区分）。

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

### 3.4 考虑过但否决的方案（避免重复调研）

| 方案 | 否决原因 |
|------|---------|
| **C++ + Vulkan SDK 直写** | 本机无 C++ 编译器、无 `glslc`；装 VS Build Tools + SDK 成本远超 PoC 需要。WGSL + naga 能达成同样目的且零安装 |
| **oneDNN 作为 CPU 算子后端** | 本机 AMD CPU 拿不到 AVX-512/AMX 红利；oneDNN 的 GPU 引擎走 SYCL/OpenCL 不走 Vulkan，与本 PoC 路径不兼容。FFI 接入留到 P3 |
| **MPI / UCX 做控制面与传输** | 能解决 host 端传输，但 GPU-direct 跨厂商不可用（主要支持 CUDA/ROCm），且引入额外依赖；PoC 用 `write_buffer` + staging buffer 已够 |
| **`ash` 直接操作 Vulkan** | 控制力最强，但需手写内存管理、shader 编译、同步原语，开发量数倍于 wgpu。PoC 不需要这种控制力 |
| **直接实现 all-reduce** | 当前列切分只需 2 路 gather；all-reduce 需切分 A 与 B 并对等通信，超出"最小 PoC"范围，留到 P4 |
| **一开始接入多 GPU** | 单 CPU + 单 GPU 已能验证统一通信域；多卡引入 P2P/拓扑/同步复杂度，顺序上放在 P5 |

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
├── .git/                   # 已 git init；基线提交 836d14f
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
│       ├── src/
│       │   ├── lib.rs          # 核心：run_poc() / gpu_matmul() / smoke_compute() + PocResult
│       │   ├── main.rs         # 只负责打印 run_poc() 结果并断言
│       │   ├── matmul_cpu.rs   # CPU naive matmul（未来 FFI 换 oneDNN 的位置）
│       │   ├── dense.rs        # dense 参考 + max_abs_error
│       │   └── shaders/
│       │       └── matmul.wgsl # GPU compute matmul（WGSL → SPIR-V）
│       └── tests/
│           └── poc.rs          # 4 个集成测试（含坑 3 的回归测试）
```

**接手人优先读**：`het-comm/src/lib.rs`（接口）→ `domain.rs`（实现）→ `tp-demo/src/lib.rs::run_poc()`（端到端用法）。`main.rs` 只是打印壳，不用读。

---

## 5. 核心 API（最小摘要）

> 详细设计与数据流图见 [PLAN.md 第 4 节](PLAN.md#4-架构设计)，这里只保留接手人必须知道的最小集合。

- **`TensorBuf`**：`Cpu(Vec<f32>)` 或 `Gpu(wgpu::Buffer)`，统一 CPU/GPU 内存。
- **`CommDomain` trait**：`copy` / `gather_to_cpu` / `barrier` 三个方法。上层只依赖它、不直接碰 wgpu——这是 P0 异步重叠能"只改实现、不动调用方"的关键。
- **`VulkanComm`**：当前唯一实现，持有 `device`/`queue`。`copy` 自动分派 CPU→CPU / CPU→GPU（`write_buffer`）/ GPU→CPU（staging buffer）；**GPU→GPU 是 `todo!()`**，单卡 PoC 用不到。
- **`run_poc()`**（`tp-demo/src/lib.rs`）：端到端流程，返回 `PocResult`（含 5 项耗时与误差）。`main.rs` 和集成测试共用它，避免"测试通过但 demo 行为不同"。

**关键边界**：当前是列切分 + gather 的**"伪 TP"，没有 all-reduce**。真正的 TP 要切分 A 和 B、各设备持有部分权重、层末 all-reduce。别把 gather 当成完整 TP 实现。

---

## 6. 运行与验证

### 6.1 跑 Demo

```powershell
cd C:\通用工程\10_heterogeneous-tp-poc
cargo run --offline -p tp-demo --release
```

预期输出（数字为热启动典型值，允许波动，见第 8 节）：

```
Adapter: AMD Radeon RX 5700 XT | backend=Vulkan
M1 通过：Vulkan adapter 可用。
M2a 通过：[4] 向量 H2D→D2H 回环。
M2b 通过：smoke compute pass (data[i]+=1)。

=== 结果 ===
dims: M=64 K=256 N=128 (每半 64 列)
cpu_matmul_us   = 165
h2d_transfer_us = 50
gpu_matmul_us   = 316
d2h_transfer_us = 132
total_us        = 1011
max_abs_error   = 0.000e0
M3/M4 通过：列切分 + gather 结果与 dense 参考一致 (err < 1e-4)。
```

### 6.2 一键回归测试（推荐）

5 条验收标准已固化为自动化测试：

```powershell
cargo test --offline --workspace
```

预期：`running 4 tests ... test result: ok. 4 passed; 0 failed`，覆盖：
- `poc_end_to_end` — 端到端验收
- `gather_is_row_interleaved` — **坑 3 的回归测试**，改回整块拼接会立刻失败
- `matmul_cpu_matches_dense` — CPU 算子正确性
- `max_abs_error_detects_mismatch` — 比对函数本身

### 6.3 验收标准

1. `adapter.get_info().backend == Backends::VULKAN`（测试覆盖）；具体型号 AMD Radeon RX 5700 XT 由人工确认
2. `max_abs_error < 1e-4`（测试覆盖）
3. H2D 与 D2H 各至少一次，且回传数据参与最终 gather（测试覆盖）
4. 输出 5 项耗时（测试覆盖）
5. 连续 3 次运行结果一致 —— 由端到端测试 + 第 8 节 3 次采样覆盖

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

### 坑 5（P0）：`map_async` + poll 等待在途 compute 有 ~1.7ms 异常延迟

P0 初版省掉了 collect 前的显式 `barrier()`，想让 `device_to_host` 内部的 `map_async + poll` 顺带等待 compute 完成。实测 D2H 从 ~150µs 暴涨到 1700~2164µs（wgpu Vulkan 后端行为），端到端反而慢 2~3 倍。

**修复**：CPU 计算后先 `comm.barrier()` 显式等待 compute，再 `collect`。D2H 回落到 109~175µs。这**仍是异步重叠**——CPU 工作发生在 submit 与 barrier 之间，barrier 只等待 `max(0, T_GPU − T_CPU)` 的剩余时间。

---

## 8. 性能数据与解读

**P0 重叠版，5 次热启动采样（排除首次编译开销）：**

| 阶段 | 范围（µs） | 典型值 |
|------|-----------|--------|
| H2D 提交 | 39 ~ 62 | 41 |
| CPU matmul（与 GPU 重叠） | 161 ~ 314 | 161 |
| GPU 阶段（submit→collect） | 357 ~ 510 | 392 |
| D2H 读回 | 109 ~ 175 | 141 |
| **端到端 total** | **725 ~ 1070** | 780 |
| `overlap_saving_us` | 161 ~ 314 | 161 |
| `max_abs_error` | 0 | 0 |

对比 P0 之前的串行版：total 899~1415µs（典型 1011），**典型收益约 23%**。

⚠️ **冷启动陷阱**：首次运行时 GPU 阶段曾达到 1107~1845µs，因为 shader 编译、pipeline/buffer 创建被计入计时。热启动稳定在 357~510µs。对比性能前务必预热。

### P2 决定性发现：GPU 计算只占墙钟的 12%

用 wgpu timestamp query 分离纯 GPU 时间后，8 次采样得到：

| 指标 | 值 |
|------|-----|
| **纯 GPU compute（timestamp）** | **34µs（8 次完全一致）** |
| GPU 阶段墙钟（submit→collect） | 289~474µs（偶发抖动到 8.7ms） |
| CPU matmul | 161~582µs |
| 提交/同步开销（P2 时 wall − compute） | ~266µs，占墙钟 88%（**P2b 已降至 ~202µs**） |

**这解释了此前所有现象**：

- P1 tiling 收益不可见：真实 GPU 计算只有 34µs，优化它省下的几 µs 对 ~300µs 墙钟毫无影响。
- GPU 本身比 CPU 快 **5~10 倍**（34µs vs 161~582µs）——异构计算的算力优势是真实存在的。
- **真正的瓶颈不是 shader，而是 wgpu/Vulkan 的提交与同步开销**：`queue.write_buffer` × 3、`queue.submit`、`device.poll(wait_indefinitely)` 用于 barrier、staging buffer 的 map/unmap——这些固定成本约 266µs，完全吞噬了 GPU 的速度优势。

**优化方向随之改变**：继续调 tiling 是盲调；应减少 submit / poll / buffer 创建次数（批处理、复用 staging buffer、避免显式 barrier），或用更轻量的同步原语。见下面 P2b。

### P2b：合并 submit，开销下降但 CPU 成为新瓶颈 ✅ 已完成

把 compute + timestamp resolve + **D2H staging copy** 塞进同一次 submit，collect 只做 map+poll+read，省掉一次 submit 和显式 barrier（坑 5 的 1.7ms 延迟也随之规避）。8 次采样：

| 指标 | P2（分开 submit） | P2b（合并 submit） |
|------|------------------|-------------------|
| D2H 读回 | ~140µs | **~42µs** |
| GPU 墙钟 | ~300µs | **~236µs** |
| 纯 GPU compute | 34µs | 34µs |
| 提交/同步开销 | ~266µs | ~202µs |

**新瓶颈**：GPU 墙钟 ≈ CPU 时间（160µs）+ D2H（42µs），即 **CPU matmul 已成为关键路径**——GPU 34µs 的计算被 CPU 160µs 完全盖住。再优化 GPU 同步收益有限，下一步要么加速 CPU（P3 oneDNN），要么按 HeteGen 思路做**非均匀切分**（GPU 能力强就多分给它，让 CPU 少算）。

解读：

- **CPU 时间被 GPU 完全隐藏**：P0 后 CPU 的 161~314µs 发生在 GPU 执行窗口内，`overlap_saving_us` 直接量化了这部分收益。barrier 只等待 `max(0, T_GPU − T_CPU)` 的剩余时间。
- **GPU 仍慢于 CPU**：GPU 阶段 357~510µs vs CPU 161~314µs。WGSL shader 是朴素实现——每输出元素一线程、串行累加 K=256、无 tiling/shared memory、workgroup 仅 8×8。这是 P1 的优化目标。
- **传输占比不低但非主导**：H2D 39~62µs + D2H 109~175µs。当前形状小（A=64×256、B1=256×64、C1=64×64，约 48KB），传输还未放大成瓶颈。
- **`total_us` 大于各阶段之和**：total 从 GPU submit 前开始计时，包含一次性初始化、gather 与 dense 参考计算。**不要直接相加各阶段来核对 total**。
- **结论**：P0 让异构方案从典型 1011µs 降到 780µs，但仍慢于纯 CPU 算全量（约 330~410µs）。这**不是 PoC 失败**：PoC 验证的是通信域可行性。下一步 P1（shader 优化）才有望逆转。

---

## 9. 下一步路线（按优先级）

### P0：event-based 异步重叠 ✅ 已完成

**目标**：让 CPU 计算与 GPU 执行重叠，验证异构 TP 的核心价值主张（HeteGen 的核心优化）。

**实现**：把 `gpu_matmul` 拆成 `gpu_matmul_submit`（H2D + compute submit，**立即返回不等待**）和 `gpu_matmul_collect`（barrier + D2H），中间插入 CPU matmul。`CommDomain` 的 trait **完全不用改**——只改编排层，兑现了 trait 边界的价值。

**实测（5 次热启动采样）**：

| 指标 | 串行版（P0 之前） | P0 重叠版 |
|------|------------------|-----------|
| 端到端 total | 899~1415µs（典型 1011） | 725~1070µs（典型 780） |
| CPU matmul | 164~377µs | 161~314µs（被 GPU 执行覆盖） |
| `overlap_saving_us` | — | 161~314µs |

典型收益约 **23%**，CPU 计算时间被 GPU 完全隐藏。`max_abs_error` 仍全为 0，测试增至 5 个全绿（新增 `p0_cpu_work_overlaps_gpu_wait`）。

**关键等式**（HeteGen）：`T_CPU = T_GPU + T_COM`，让 CPU 计算时间刚好覆盖 GPU 计算 + 通信。

**踩坑**：collect 前的显式 `barrier()` 不能省——见坑 5。

### P1：GPU shader 优化 ✅ 已完成（收益有限）

- 已实现 16×16 shared memory tiling（`matmul.wgsl`），dispatch 改为 `(M+15)/16`
- **GPU 执行**：392µs → ~312µs（约 20% 改善）
- **端到端无收益**：tiling shader 的 naga 编译 / pipeline 创建更慢，抵消了执行收益（GPU 阶段不含编译，total 包含）
- **TILE 对比**：实测 TILE=16 优于 TILE=8（barrier 次数减半），已固定 16
- **大形状验证（M=256,K=512,N=512）**：GPU 仍慢于 CPU（~3900µs vs ~3100µs），RX 5700 XT + 朴素 shader 难超越现代 CPU 的 naive matmul；但异构端到端 4800µs < 纯 CPU 全量 6200µs（P0 重叠收益在大规模下依然成立）

### P2：timestamp query 精确测量 ✅ 已完成（见第 8 节决定性发现）

- device 按需请求 `Features::TIMESTAMP_QUERY`（不支持则指标显示 N/A）
- compute pass 写入起止 timestamp，resolve 后读回，`gpu_compute_us` 进入 `PocResult`
- **结论**：纯 GPU 计算仅 34µs，提交/同步开销 ~266µs 占墙钟 88%——瓶颈转移，后续优化应针对提交/同步而非 shader

### P2b：减少提交与同步开销 ✅ 已完成（见上文）

- 已合并 compute + D2H copy 到同一次 submit；开销 266µs → 202µs，D2H 140µs → 42µs
- 坑 5 的 1.7ms 延迟在合并 submit 后不再复现
- 剩余可选项：复用 H2D 的 staging / 进一步探索 callback 同步（收益已递减）

### P2c：非均匀切分（新的最高优先级之一）

- GPU 34µs vs CPU 160µs，当前 50/50 列切分让 CPU 成为关键路径
- 按 HeteGen 思路把更大比例的工作分给 GPU（如 GPU 70% / CPU 30%），让两侧完成时间趋近 `T_CPU ≈ T_GPU + T_COM`
- 与 P3（oneDNN 加速 CPU）二选一或组合推进

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
6. **`gather_to_cpu` 的按行交错逻辑不要改回整块拼接**：见坑 3；已有回归测试守护。
7. **依赖离线可复现**：用 `--offline` 时依赖必须已在本地缓存；换机器时去掉 `--offline` 让 cargo 拉取。
8. **性能对比必须先预热**：冷启动把 shader 编译计入 GPU 计时，数字会虚高 3~4 倍，不可与稳态值直接比较。
9. **验收标准已自动化**：改完代码先跑 `cargo test --offline --workspace`，别再靠肉眼读 demo 输出判断是否回归。

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

- [ ] `git log --oneline` 确认基线提交 `836d14f`，工作区干净
- [ ] 跑 `cargo test --offline --workspace`，确认 4 个测试全绿
- [ ] 跑 `cargo run --offline -p tp-demo --release`，对照第 6.1 节预期输出
- [ ] 读 `het-comm/src/lib.rs` + `domain.rs`，理解 `TensorBuf` / `CommDomain` / `VulkanComm`
- [ ] 读 `tp-demo/src/lib.rs` 的 `run_poc()` / `gpu_matmul()`，理解 H2D→compute→D2H 流程（`main.rs` 只是打印壳）
- [ ] 读第 7 节踩坑记录，特别是坑 3（gather 按行交错）
- [ ] 决定 P0（异步重叠）是否启动；启动则先设计 event/handle 方案再动手，改完跑 `cargo test` 回归
