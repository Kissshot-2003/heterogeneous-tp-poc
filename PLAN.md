# 异构张量并行 PoC —— 项目方案

> 范围：`1 CPU + 1 Vulkan GPU` 的最小工具化验证
> 日期：2026-11-22
> 前置讨论：打破"TP 必须同型号 GPU"的假设，先验证跨 CPU/GPU 通信域是否可行

---

## 1. 目标与范围

### 1.1 一句话目标

构建一套**最小的跨设备通信域**（unified communication domain），跑通一次真实的张量并行切分：把矩阵乘法按列切成两半，CPU 与 GPU 各算一半，汇总回 CPU 后与 dense 参考逐元素比对，并测量各阶段耗时。

### 1.2 In Scope（本 PoC 做）

| # | 内容 |
|---|------|
| 1 | 统一 buffer 抽象：`CpuBuffer(Vec<f32>)` 与 `GpuBuffer(wgpu::Buffer)` |
| 2 | 跨设备传输：H2D（`queue.write_buffer`）与 D2H（`map_async` + `get_mapped_range`） |
| 3 | 一次 2 路 gather：把 GPU 结果切片收集回 CPU 并 cat |
| 4 | CPU 侧 naive matmul + GPU 侧 WGSL compute matmul |
| 5 | 数值校验：与 dense 参考比对，`max_abs_error < 1e-4` |
| 6 | 分阶段计时：CPU 计算 / H2D / GPU 计算 / D2H / 总计 |
| 7 | 确认 Vulkan backend 实际落在 AMD RX 5700 XT 上 |

### 1.3 Out of Scope（本 PoC 不做，但留接口）

- **oneDNN 接入**：CPU 侧先用 Rust naive matmul；`matmul_cpu` 做成独立函数，后续通过 FFI 替换为 oneDNN，通信域不变。
- 多 GPU / 多节点 / 跨机网络。
- ring-allreduce、tree-allreduce 等真正 collectives（本 PoC 只需要 gather）。
- 生产级错误处理、超时、容错、动态拓扑发现。
- 接入真实 LLM / 与 vLLM、llama.cpp 集成。

---

## 2. 环境事实（已在本机确认）

| 项 | 值 |
|----|----|
| CPU | AMD Ryzen 5 5600，6 核 12 线程 |
| GPU | AMD Radeon RX 5700 XT（独显），Vulkan 1.4.315，驱动 25.10.30.02 |
| Vulkan runtime | `vulkaninfo` 可用，GPU 被正确枚举 |
| C++ 编译器 | **无**（`cl`/`clang`/`g++` 均不在 PATH，vswhere 无输出） |
| Vulkan SDK | **无**（`VULKAN_SDK` 未设置，`glslc` 缺失） |
| Rust | `cargo` / `rustc` 可用 |
| 本地 crate 缓存 | `wgpu 30.0.1/30.0.2`、`naga 30`、`pollster 0.4/1.0`、`bytemuck 1.25.2`、`ash 0.38` 已缓存 |

**结论**：C++ + Vulkan SDK 路线在本机不可行（装 SDK 成本不必要）；Rust + wgpu 路线可离线构建。

---

## 3. 技术路线决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 实现语言 | Rust | 工具链现成；本机有 Rust 项目经验（candle 相关）；无需 C++ 编译器 |
| GPU API | wgpu 30，强制 Vulkan backend | 本地缓存有 wgpu 30；可直接枚举 AMD 设备；抽象层薄，够用 |
| Shader 语言 | WGSL（naga → SPIR-V） | **无需 Vulkan SDK / glslc**；wgpu 原生支持 |
| 同步方式 | submit 立即返回 + 显式 barrier + `map_async` | P0 已实现 CPU/GPU 重叠，barrier 只等待剩余时间 |
| CPU matmul | Rust naive（三层循环 + 分块预留） | 先验证通信域，性能不是 PoC 目标；后续 FFI 换 oneDNN |
| 数据布局 | `f32` 行主序，连续内存 | 简化传输与比对；避免 stride 坑 |

### 3.1 为什么不直接上 oneDNN

- 本机是 **AMD CPU**，oneDNN 的 x86 优化红利（AVX-512 / AMX）拿不到，PoC 阶段用它收益不明显。
- oneDNN 的 GPU 引擎走 SYCL/OpenCL，**不走 Vulkan**——它和 Vulkan 是并列后端，不是组合关系。
- PoC 的核心是验证通信域，不是验证算子性能。先把 `matmul_cpu` 做成独立函数，后续 FFI 接入 oneDNN 即可，通信域代码不用改。

### 3.2 wgpu 在这里的定位

wgpu 是安全抽象层，但本 PoC 需要的底层能力它都暴露了：

- `wgpu::Buffer` 的 `COPY_SRC | COPY_DST | STORAGE` usage → 足以做传输和 compute 读写。
- `queue.write_buffer` → H2D。
- `buffer.slice(..).map_async(MAP_READ)` → D2H。
- `adapter.get_info()` → 确认 vendor/device/backend。
- `request_adapter` 时指定 `backends: Backends::VULKAN` → 强制 Vulkan。

**不隐藏的点**：fence/semaphore 的具体行为由 wgpu 管理，PoC 阶段用 `device.poll(PollType::Wait)` 同步即可，不追求精细控制。

---

## 4. 架构设计

### 4.1 分层图

```
┌─────────────────────────────────────────────────┐
│  tp-demo（验证应用）                              │
│   - dense 参考计算                                │
│   - 列切分调度：CPU 左半 / GPU 右半                │
│   - 数值比对 + 计时报告                           │
├─────────────────────────────────────────────────┤
│  het-comm（统一通信域，本 PoC 的核心交付物）       │
│   - TensorBuf: Cpu(Vec<f32>) | Gpu(wgpu::Buffer) │
│   - CommDomain: copy / gather_to_cpu / barrier   │
│   - H2D / D2H 传输封装                            │
├─────────────────────────────────────────────────┤
│  wgpu (Vulkan backend) + naga/WGSL                │
│   - GPU compute matmul shader                    │
└─────────────────────────────────────────────────┘
```

### 4.2 核心 API 草案（`het-comm`）

```rust
/// 统一张量缓冲区：屏蔽 CPU 内存与 GPU 显存
pub enum TensorBuf {
    /// CPU 侧宿主内存，f32 行主序
    Cpu(Vec<f32>),
    /// GPU 侧 storage buffer
    Gpu(wgpu::Buffer),
}

/// 通信域 Trait：未来多设备/多节点时替换实现即可
pub trait CommDomain {
    /// 把 src 的 len 个 f32 送到 dst（自动判断 H2D / D2H / CPU→CPU）
    fn copy(&self, src: &TensorBuf, dst: &mut TensorBuf, len: usize);
    /// 把 GPU 分片收集到 CPU 输出（本 PoC 只有 1 个 GPU 分片）
    fn gather_to_cpu(&self, parts: &[TensorBuf], out: &mut TensorBuf, rows: usize, cols: usize);
    /// 等待所有在途工作完成
    fn barrier(&self);
}
```

设计原则：

1. **trait 是边界**：上层 TP 逻辑只依赖 `CommDomain`，不直接碰 wgpu。后续接 oneDNN 或换后端，只改实现。
2. **`TensorBuf` 是统一货币**：计算和传输都以它为输入输出，避免各模块自己管内存。
3. **同步点显式化**：submit 后立即返回让 CPU 计算与 GPU 重叠，最后显式 barrier 等待剩余时间（P0 已实现）。注意 collect 前不能省 barrier，否则 `map_async + poll` 有 ~1.7ms 异常延迟。

### 4.3 数据流（TP 列切分）

```
A [M,K] (CPU)          B [K,N] (CPU)
                           │
                 列切分 B → B0 [K,N/2] (CPU)  +  B1 [K,N/2] (CPU)
                           │
   ┌───────────────────────┴───────────────────────┐
   │                                               │
CPU: C0 = A × B0                          H2D: A, B1 → GPU
   │                                      GPU: C1 = A × B1
   │                                      D2H: C1 → CPU
   └───────────────┬───────────────────────────────┘
                   ▼
            cat(C0, C1) → C [M,N] (CPU)
                   ▼
       与 dense: A × B 比对 → max_abs_error
```

**注意**：这是"伪 TP"——真正的 TP 会把 `A` 也切分并让每卡持有部分 `B`，通信模式是 all-reduce。本 PoC 用列切分 + gather 是为了用最小通信量验证"跨设备统一通信域"这个底座。all-reduce 留到下一阶段。

---

## 5. 目录结构

```
10_heterogeneous-tp-poc/
├── Cargo.toml              # workspace 定义
├── README.md               # 快速开始
├── PLAN.md                 # 本文档
├── .gitignore
├── crates/
│   ├── het-comm/           # 统一通信域 crate（核心交付物）
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # TensorBuf + CommDomain 定义、模块导出
│   │       ├── device.rs       # GPU 设备初始化 / adapter 查询（待 M1）
│   │       ├── transport.rs    # H2D / D2H 实现（待 M2）
│   │       └── collectives.rs  # gather_to_cpu 实现（待 M2）
│   └── tp-demo/            # 验证程序
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs         # 流程编排 + 计时（待 M3/M4）
│           ├── matmul_cpu.rs   # CPU naive matmul（待 M3）
│           ├── dense.rs        # dense 参考 + 数值比对（待 M3/M4）
│           └── shaders/
│               └── matmul.wgsl # GPU compute matmul（待 M1/M3）
```

---

## 6. 里程碑

| 里程碑 | 内容 | 验收点 | 预估 | 状态 |
|--------|------|--------|------|------|
| **M1** | wgpu 强制 Vulkan 枚举设备 | 打印 `AMD Radeon RX 5700 XT` / `backend=Vulkan` | 0.5 天 | ✅ 已完成 |
| **M2** | `TensorBuf` + `CommDomain` + H2D/D2H + smoke compute pass | `[4]` 向量回环相等；`data[i]+=1` 结果正确 | 1 天 | ✅ 已完成 |
| **M3** | CPU/GPU 双路 matmul + 列切分合并 | `max_abs_error < 1e-4`（M=64 K=256 N=128） | 1–2 天 | ✅ 已完成 |
| **M4** | 计时报告 + 文档收尾 | 输出 5 项耗时；数值比对通过 | 0.5 天 | ✅ 已完成 |

**实际耗时：约 1 个工作日**（含 2 个 bug 修复：wgpu 30 API 变更、`MAP_READ` 需 staging buffer、gather 按行交错拼接）。

> 踩坑记录：
> 1. wgpu 30 的 `InstanceDescriptor` 无 `Default`，需用 `new_without_display_handle()`；`RequestAdapterOptions` 多了 `apply_limit_buckets` 字段。
> 2. `MAP_READ` 不能与 `STORAGE` 共存 → D2H 必须走 staging buffer + `copy_buffer_to_buffer`。
> 3. 列切分的 gather 必须**按行交错**拼接，不能整块拼接（否则误差达 3.9）。

---

## 7. 验收标准

PoC 通过必须同时满足（**已固化为 `cargo test --offline --workspace` 的 5 个自动化测试**，见 `crates/tp-demo/tests/poc.rs`）：

1. **后端正确**：`adapter.get_info().backend == Backends::VULKAN`；具体型号 AMD Radeon RX 5700 XT 由人工确认（测试不断言型号以保证可移植）。
2. **数值正确**：`max(|C[i,j] - C_dense[i,j]|) < 1e-4`，默认形状 `M=64, K=256, N=128`。
3. **传输真实发生**：H2D 与 D2H 各至少一次，且回传数据参与最终 cat（不是 CPU 自己算完假装的）。
4. **计时完整**：输出 `cpu_matmul_us / h2d_transfer_us / gpu_matmul_us / d2h_transfer_us / total_us`。
5. **可重复**：3 次采样误差均为 0、耗时在稳定范围（见 HANDOVER.md 第 8 节）；由端到端测试 + 采样覆盖。

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| wgpu 30 的 `map_async` 同步坑（未 `unmap` 导致下次写失败） | D2H 卡死或数据错 | 严格 `map → read → drop view → unmap`；`barrier()` 里 `device.poll(Wait)` |
| Vulkan compute shader workgroup size 不匹配导致性能极差或编译失败 | GPU matmul 跑不起来 | 先用 `@workgroup_size(8,8)` 保守值；M1 先跑空 shader 验证管线 |
| D2H 走 PCIe，传输时间可能吞掉 GPU 计算收益 | PoC 没有性能意义 | 这是**要测量的数据**而非失败；PoC 目标是验证可行性，性能优化留到 M4 之后 |
| 本机无 C++ 编译器，后续接 oneDNN 受阻 | 第二阶段卡住 | 提前验证 `oneDNN` 的 Rust binding 或准备装 VS Build Tools；PoC 阶段不受影响 |
| f32 累加顺序导致误差 > 1e-4 | 数值验收失败 | 放宽到 `1e-3` 或使用 Kahan 求和；默认 K=256 误差应在 1e-5 量级 |

---

## 9. 后续演进（PoC 通过之后）

1. **接 oneDNN**：`matmul_cpu` 改为 FFI 调用 oneDNN；`CommDomain` 不变。需要先解决 C++ 工具链。
2. **真 TP + all-reduce**：切分 `A` 与 `B`，两台设备各持有部分权重，每层末做一次 all-reduce（参考 HeteGen 的非均匀切分思路）。
3. **异步重叠** ✅ **已完成（P0）**：`gpu_matmul` 拆分为 submit / collect，CPU matmul 塞进 GPU 执行窗口；典型收益 23%。详见 HANDOVER.md 第 9 节 P0。
4. **多 GPU 扩展**：用 `ash` 直接管理多 Vulkan 设备，探索跨厂商 P2P 是否可用。
5. **并行范式对比**：用同一套 `CommDomain` 再实现 PP/层拆分，对比 TP 与 PP 在异构场景下的收益。

---

## 10. 参考

- HeteGen (MLSys 2024)：CPU+GPU 异构 TP，非均匀切分 + 通信/计算重叠
- HexGen (MLR 2024)：异构跨数据中心 TP+PP 非对称划分
- AsymGroup (IEEE Access 2025)：按设备能力比例分配 2D TP 工作负载
- llama.cpp SYCL 后端：`--split-mode tensor`（2 GPU 优化）与 `--split-mode layer`（PP）
- wgpu 文档：<https://wgpu.rs/>
