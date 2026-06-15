# CS336 Assignment 2: Systems & Parallelism — 初学者教程报告

> 本教程面向 AI 初学者，将这份 48 页的 Stanford CS336 作业拆解为 **概念解读 → 要做什么实验 → 要写什么代码 → 交付物清单**，帮你建立全局视野。

---

## 目录

1. [作业全貌与知识地图](#1-作业全貌与知识地图)
2. [Part 1: Profiling & Benchmarking（性能分析与基准测试）](#2-part-1-profiling--benchmarking)
3. [Part 2: Activation Checkpointing（激活值检查点）](#3-part-2-activation-checkpointing)
4. [Part 3: Flash Attention 2（闪存注意力）](#4-part-3-flash-attention-2)
5. [Part 4: Distributed Data Parallel (DDP)（分布式数据并行）](#5-part-4-ddp)
6. [Part 5: Optimizer State Sharding（优化器状态分片）](#6-part-5-optimizer-state-sharding)
7. [Part 6: Fully Sharded Data Parallel (FSDP)（全分片数据并行）](#7-part-6-fsdp)
8. [Part 7: Leaderboard（排行榜）](#8-part-7-leaderboard)
9. [代码结构与开发路线图](#9-代码结构与开发路线图)
10. [核心概念速查表](#10-核心概念速查表)

---

## 1. 作业全貌与知识地图

### 一句话总结

> **这份作业的目标是：让你从"能跑的Transformer"进化到"跑得快、占内存少、还能多卡并行"的Transformer。**

### 知识依赖关系

```
Assignment 1 (基础Transformer模型)
       │
       ▼
┌──────────────────────────────────────────────────┐
│           Assignment 2: Systems                   │
│                                                    │
│  ① Profiling/Benchmarking ──▶ 发现瓶颈              │
│       │                                            │
│       ▼                                            │
│  ② Activation Checkpointing ──▶ 省显存              │
│       │                                            │
│       ▼                                            │
│  ③ Flash Attention 2 (Triton) ──▶ 加速注意力        │
│       │                                            │
│       ▼                                            │
│  ④ DDP (分布式数据并行) ──▶ 多卡训练                 │
│       │                                            │
│       ▼                                            │
│  ⑤ Optimizer State Sharding ──▶ 省优化器显存         │
│       │                                            │
│       ▼                                            │
│  ⑥ FSDP (全分片数据并行) ──▶ 终极多卡方案            │
│       │                                            │
│       ▼                                            │
│  ⑦ Leaderboard ──▶ 综合优化竞赛                     │
└──────────────────────────────────────────────────┘
```

### 模型规格参考

作业中会反复使用这几个模型尺寸：

| 名称 | d_model | d_ff | num_layers | num_heads |
|------|---------|------|------------|-----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| **xl** | **2560** | **10240** | **32** | **32** |
| 10B | 4608 | 12288 | 50 | 36 |

> 默认 context_length = 512

---

## 2. Part 1: Profiling & Benchmarking

### 🧠 概念解读

**Profiling（性能剖析）** 就是给程序做"体检"，搞清楚：
- 时间花在哪了？（是矩阵乘法？还是注意力？）
- 内存被什么占了？（是模型参数？还是中间激活值？）

#### 关键概念 1: CUDA 异步执行

```
CPU:  调用matmul() ──▶ 立即返回 ──▶ 继续做其他事
GPU:  .............. ──▶ 开始计算 ──▶ 计算完成
```

- GPU 操作是**异步**的：CPU 发出指令后不等 GPU 做完就继续了
- 所以用 `time.time()` 直接测量是**不准的**
- 必须调用 `torch.cuda.synchronize()` 让 CPU 等 GPU 做完，才能准确计时

#### 关键概念 2: Warm-up（预热）

- 第一次运行 GPU 代码会有额外开销（CUDA 上下文初始化、JIT 编译等）
- 所以基准测试要先跑几步"预热"，再开始计时

#### 关键概念 3: torch.compile

- PyTorch 2.x 引入的编译器，会将你的 Python 模型编译为优化过的 GPU 代码
- 可以自动进行**算子融合**（把多个小操作合并成一个大操作，减少内存读写）
- 首次调用慢（需要编译），之后会快很多

### 📋 要做的实验 & 要写的代码

#### 实验 2.1: 基准测试脚本 (4分)

**要写的代码：** 一个 benchmarking 脚本，支持：
- 输入超参数，创建模型
- 生成随机数据
- 先跑 w 步预热，再计时 n 步
- 支持只跑 forward / forward+backward / forward+backward+optimizer
- 用 `torch.cuda.synchronize()` 正确计时
- 支持多种模型变体（不同精度、torch.compile 等）

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (b) 端到端计时 | 对 small/medium/large/xl 模型，batch_size=16，分别测 forward / forward+backward / 完整训练步 | 包含 12 个测量值的表格 |
| (c) torch.compile | 对比 compile vs 不 compile 在 xl 模型上的性能差异 | 测量值 + 1-2 句分析 |
| (d) 混合精度 | 用 `torch.autocast` 测试 bf16 精度在 xl 模型上的效果 | 测量值 + 1-2 句分析 |

#### 实验 2.2: Nsight Systems 性能分析 (3分)

**概念：** Nsight Systems 是 NVIDIA 的可视化性能分析工具，可以看到 CPU 和 GPU 上每个操作的时间线。

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (a) Nsight trace | 对 xl 模型生成 Nsight trace，找出前向传播中最耗时的 3 个 CUDA kernel | 截图 + kernel 名称 + 时间占比 |
| (b) compile 对比 | 对比 compile vs 不 compile 的 Nsight trace | 截图 + 区别分析 |
| (c) mixed precision 对比 | 对比 bf16 vs fp32 的 Nsight trace | 截图 + 区别分析 |

**如何运行 Nsight：**
```bash
nsys profile -o trace_name uv run python your_benchmark_script.py
```

#### 实验 2.3: 内存分析 (3分)

**概念：** GPU 内存主要被三部分占据：
1. **模型参数** — 权重矩阵
2. **优化器状态** — AdamW 需要为每个参数存储 2 个额外 float（一阶/二阶动量）
3. **激活值/中间结果** — 前向传播中保存的张量，用于反向传播

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (a) 参数/优化器内存 | 计算 xl 模型参数量 + 优化器状态占用内存 | fp32 和 bf16 下的计算结果 |
| (b) Peak 内存 | 用 `torch.cuda.max_memory_allocated()` 测量 xl 模型训练时的峰值内存 | 测量值 + 与 (a) 对比分析 |
| (c) 内存快照 | 用 `torch.cuda.memory._record_memory_history` 生成内存时间线 | 可视化截图 + 分析 |

---

## 3. Part 2: Activation Checkpointing

### 🧠 概念解读

#### 问题：为什么训练比推理吃更多显存？

训练时，前向传播中的**所有中间结果**都要保存下来，因为反向传播需要用到它们来计算梯度。

举例说明：
```
前向传播: x → [Layer1] → a1 → [Layer2] → a2 → [Layer3] → a3 → loss
                        ↑保存           ↑保存           ↑保存
```

对于 xl 模型（32 层），每一层大约需要 ~3.6 GiB 的激活值存储，32 层就是 **~114 GiB**！这比模型本身大得多。

#### 解决方案：Activation Checkpointing（梯度检查点/激活值重计算）

**核心思想：** 不保存所有中间结果，只保存几个"检查点"。反向传播到某一层时，从最近的检查点重新计算需要的中间结果。

```
不用 checkpointing: 保存所有 32 层的激活值 → 114 GiB
用 checkpointing:   每 2 层保存一个检查点   → 只需 ~几百 MiB 的检查点 + 2 层的激活值
```

**代价：** 需要额外的计算时间（重新跑一次前向传播），是一种**时间换空间**的策略。

### 📋 要做的实验 & 要写的代码

#### 实验 3.1: Activation Checkpointing (5分)

**要写的代码：**
- 修改 Transformer 模型，支持可配置的 activation checkpointing
- 使用 `torch.utils.checkpoint.checkpoint` API
- 可以选择每隔多少层做一个 checkpoint

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (a) 峰值内存 | 对比 xl 模型有/无 checkpointing 的峰值内存 | 测量值 + 内存节省量 |
| (b) 训练速度 | 对比有/无 checkpointing 的训练速度 | 测量值 + 速度开销分析 |
| (c) 最大 batch | 在 xl 模型上，对比有/无 checkpointing 能用的最大 batch_size | batch_size 数值 |

---

## 4. Part 3: Flash Attention 2

### 🧠 概念解读

#### 标准 Attention 的问题

标准的 self-attention 公式：
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

问题在于：
1. `Q × K^T` 生成一个 `[seq_len × seq_len]` 的矩阵 — **O(n²) 内存**
2. 这个矩阵要写到 GPU 高带宽内存 (HBM)，再读回来做 softmax — **大量内存读写**

当 seq_len=2048 时，这个矩阵有 400 万个元素，非常浪费。

#### Flash Attention 的核心思想

**不要一次性算出整个 attention 矩阵！** 而是：
1. 将 Q、K、V 切成小块（tiles）
2. 每次只在 GPU 的**片上缓存 (SRAM)** 中处理一小块
3. 用 **online softmax** 算法边算边更新结果，不需要看到完整矩阵

```
传统方式:
  Q×K^T (写入HBM) → softmax (读HBM,写HBM) → ×V (读HBM)
  
Flash Attention:
  对每个 Q 的小块:
    对每个 K,V 的小块:
      在 SRAM 中计算 Q_tile × K_tile^T
      在 SRAM 中更新 softmax（online）
      在 SRAM 中累加 × V_tile
    写最终结果到 HBM（只写一次！）
```

#### Online Softmax 算法

普通 softmax 需要两遍扫描：
1. 第一遍找最大值 `max(x)`
2. 第二遍算 `exp(x - max) / sum(exp(x - max))`

**Online softmax** 只需一遍扫描，边看数据边更新：
- 维护一个"当前最大值" `m` 和"当前 softmax 分母" `l`
- 每看到新数据，更新 `m` 和 `l`，并对之前的结果做修正

#### Triton 是什么？

Triton 是一个 GPU 编程框架，比 CUDA 简单得多：
- CUDA：你需要手动管理线程、共享内存、同步...
- Triton：你只需要按"块/tile"思考，Triton 帮你处理底层细节

核心概念：
- **Program instance（程序实例）**：类似 CUDA 的 thread block，每个实例处理一个数据块
- **Block pointer**：指向内存中一块数据的指针
- **tl.load / tl.store**：从内存加载/存储一整块数据
- **boundary_check**：处理边界情况（数据不能整除 tile 大小时）

### 📋 要做的实验 & 要写的代码

#### 实验 4.1: PyTorch 版 Flash Attention 2 (15分)

**要写的代码：**
- 用纯 PyTorch 实现 Flash Attention 2 算法
- 实现为 `torch.autograd.Function` 子类（需要写 forward 和 backward）
- 要支持 **causal masking**（因果掩码，让模型看不到未来的 token）
- 需要实现 **online softmax**

**关键实现要点：**
- Forward: 按 tile 遍历 K/V，在线更新 softmax 统计量，累加输出
- Backward: 给定 `dO`（输出梯度），计算 `dQ`、`dK`、`dV`
- Backward 的技巧：先算 `D = rowsum(O ⊙ dO)`，然后用它来高效计算注意力矩阵的梯度

**适配器连接：** 实现 `tests/adapters.py` 中的 `get_flashattention_autograd_function_pytorch()`

**测试命令：**
```bash
uv run pytest tests/test_attention.py -k pytorch
```

#### 实验 4.2: Triton 版 Flash Attention 2 (15分)

**要写的代码：**
- 用 Triton 重写 Flash Attention 2 的 forward kernel
- Backward 可以用 `torch.compile` 自动导出（或自己写 Triton kernel）

**适配器连接：** 实现 `get_flashattention_autograd_function_triton()`

**测试命令：**
```bash
uv run pytest tests/test_attention.py -k triton
```

#### 实验 4.3: Flash Attention Benchmarking (3分)

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (a) 速度对比 | 对比标准 attention vs Flash Attention（PyTorch版 & Triton版）在不同 seq_len 下的速度 | 图表 |
| (b) 内存对比 | 对比标准 attention vs Flash Attention 的峰值内存 | 图表 |
| (c) 端到端 | Flash Attention 集成到完整模型后的训练速度和峰值内存 | 测量值 + 分析 |

---

## 5. Part 4: DDP（分布式数据并行）

### 🧠 概念解读

#### 为什么需要多 GPU？

单 GPU 的内存和计算力有限。想训练大模型或加速训练，需要用多块 GPU。

#### DDP 的基本原理

```
GPU 0: 模型副本A ──处理数据批次0──▶ 梯度A ─┐
                                            ├── All-Reduce（求平均）──▶ 统一梯度 ──▶ 各自更新参数
GPU 1: 模型副本B ──处理数据批次1──▶ 梯度B ─┘
```

1. 每张 GPU 都有完整模型的**一份拷贝**
2. 每张 GPU 处理**不同的数据**
3. 反向传播后，所有 GPU 的梯度通过 **All-Reduce** 求平均
4. 每张 GPU 用相同的平均梯度更新参数（参数保持同步）

#### All-Reduce 是什么？

一种分布式通信操作：所有 GPU 把各自的张量求和（或求平均），结果同步到所有 GPU 上。

```
GPU0: [1, 2, 3]  ─┐
                   ├── All-Reduce(sum) ──▶ 所有 GPU 都得到 [4, 6, 8]
GPU1: [3, 4, 5]  ─┘
```

#### 进程组与分布式初始化

- 使用 `torch.distributed` 模块
- 每个 GPU 对应一个"进程"（rank）
- 需要初始化进程组：`dist.init_process_group(backend="nccl")`
- NCCL 是 NVIDIA 专门为 GPU 间通信优化的库

### 📋 要做的实验 & 要写的代码

#### 实验 5.1: Minimal DDP (3分)

**要写的代码：** 最简单的 DDP 实现——训练结束后对每个参数的梯度做 all-reduce

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| (a) 实现 | 写一个简单的 DDP 训练脚本 | 代码 |
| (b) 基准测试 | 在 2 GPU 上测量 xl 模型的训练速度，并计算通信开销 | 时间测量 + 分析 |

#### 实验 5.2: DDP with Flat Gradients (2分)

**优化思路：** 把所有参数的梯度拼接成一个大张量，做一次 all-reduce，减少通信次数。

**要做的实验：** 对比逐参数 all-reduce vs 拼接后一次 all-reduce 的性能差异

#### 实验 5.3: DDP with Overlapping (5分)

**核心优化：** 反向传播是逐层计算梯度的。某一层的梯度算好了，就可以**立即开始通信**，同时 GPU 继续算下一层的梯度。

```
时间线（不重叠）:
  [── backward 计算 ──][── all-reduce 通信 ──]

时间线（重叠）:
  [── backward Layer N ──][── backward Layer N-1 ──]...
       └── all-reduce N ──┘     └── all-reduce N-1 ──┘
```

**要写的代码：**
- 实现一个 DDP wrapper 类，支持计算与通信重叠
- 使用 `register_post_accumulate_grad_hook` 在梯度就绪时触发 all-reduce
- 使用 `async_op=True` 进行异步通信
- 实现 `finish_gradient_synchronization()` 方法

**适配器连接：** 实现 `get_ddp(module)` 和 `ddp_on_after_backward()`

**测试命令：**
```bash
uv run pytest tests/test_ddp.py
```

**要做的实验：**
- 对比三种 DDP 实现的性能
- 用 Nsight 分析通信是否与计算重叠

---

## 6. Part 5: Optimizer State Sharding

### 🧠 概念解读

#### 问题：优化器状态太占内存

以 AdamW 为例，每个参数需要存储：
- 参数本身: 1x
- 一阶动量 (m): 1x
- 二阶动量 (v): 1x
- **总计: 参数内存的 3 倍！**

在 DDP 中，**每张 GPU 都存了完整的优化器状态** — 这是浪费！

#### 解决方案：分片（Sharding）

把优化器状态**分配到不同 GPU** 上，每张 GPU 只管一部分参数的优化：

```
DDP（不分片）:                    优化器分片:
GPU 0: 参数[全部] + Adam[全部]    GPU 0: 参数[全部] + Adam[前半]
GPU 1: 参数[全部] + Adam[全部]    GPU 1: 参数[全部] + Adam[后半]
```

每个 GPU 优化自己那部分参数后，通过 **broadcast** 把更新后的参数同步给其他 GPU。

这就是 ZeRO 论文的核心思想（Stage 1: 优化器状态分片）。

### 📋 要做的实验 & 要写的代码

#### 实验 6.1: Optimizer State Sharding (15分)

**要写的代码：**
- 实现 `ShardedOptimizer` 类，继承 `torch.optim.Optimizer`
- 将参数均匀分配给各 rank
- 每个 rank 只对自己负责的参数调用优化器
- `step()` 后通过 broadcast 同步参数

**适配器连接：** 实现 `get_sharded_optimizer()`

**测试命令：**
```bash
uv run pytest tests/test_sharded_optimizer.py
```

**要做的实验：**
- 验证分片后训练结果与不分片一致
- 测量内存节省量

---

## 7. Part 6: FSDP（全分片数据并行）

### 🧠 概念解读

#### 从 DDP → FSDP：更激进的分片

DDP + 优化器分片后，每张 GPU 仍然存着**完整的模型参数**。FSDP 更进一步：**连参数也分片！**

```
FSDP 训练一步的流程:

1. 前向传播某一层前:
   All-Gather —— 从所有 GPU 收集该层的完整参数
   
2. 用完整参数做前向计算

3. 前向计算完:
   释放完整参数，只保留自己的分片

4. 反向传播某一层前:
   All-Gather —— 再次收集完整参数

5. 反向计算完:
   Reduce-Scatter —— 每个 GPU 只拿回自己分片对应的梯度（已经求和）
   释放完整参数

6. 优化器更新:
   每个 GPU 只更新自己分片的参数
```

#### 关键通信原语

| 操作 | 说明 |
|------|------|
| **All-Gather** | 每个 GPU 有一小块数据，收集后所有 GPU 都拿到完整数据 |
| **Reduce-Scatter** | 每个 GPU 有一份完整梯度，求和后每个 GPU 只拿回属于自己的那一块 |

```
All-Gather 示例:
GPU0 有 [A]，GPU1 有 [B]  →  两者都得到 [A, B]

Reduce-Scatter 示例:
GPU0 有 [1,2,3,4]，GPU1 有 [5,6,7,8]
→ GPU0 得到 [1+5, 2+6] = [6, 8]
→ GPU1 得到 [3+7, 4+8] = [10, 12]
```

### 📋 要做的实验 & 要写的代码

#### 实验 7.1: FSDP (20分)

**要写的代码：**
- 实现一个 FSDP wrapper 类
- 支持模型参数的分片存储
- 前向/反向时自动 All-Gather 收集完整参数
- 反向完成时用 Reduce-Scatter 分发梯度
- 支持低精度计算 + 全精度参数存储（mixed precision）
- 实现 `gather_full_params()` 方法用于评估/保存模型

**适配器连接：** 实现 `get_fsdp()`、`fsdp_on_after_backward()`、`fsdp_gather_full_params()`

**测试命令：**
```bash
uv run pytest tests/test_fsdp.py
```

**要做的实验：**

| 子实验 | 内容 | 交付物 |
|--------|------|--------|
| Benchmarking | 2 GPU 上对比 FSDP vs DDP 的训练速度 | 时间测量 |
| 内存对比 | 对比 FSDP vs DDP 的峰值内存 | 内存测量 |
| 扩展性 | 测试不同 GPU 数量下的性能变化 | 图表 |

---

## 8. Part 7: Leaderboard

### 🧠 概念解读

这是一个**综合优化竞赛**——把前面学到的所有技巧组合起来，在 2 张 B200 GPU 上实现最快的训练步。

### 📋 可用的优化手段

| 技术 | 效果 |
|------|------|
| `torch.compile` | 算子融合，减少内存读写 |
| 混合精度 (bf16) | 计算量减半 |
| Flash Attention (Triton) | 注意力层加速 + 省内存 |
| Activation Checkpointing | 省内存（可选，换取更大 batch） |
| FSDP | 多卡训练，参数分片 |
| Fused Cross-Entropy | 合并 log-softmax 和交叉熵 |
| 通信/计算重叠 | 减少通信等待时间 |

**Baseline：10 秒。** 你的目标是打败它。

---

## 9. 代码结构与开发路线图

### 项目文件结构

```
assignment2-systems/
├── cs336_systems/           # 🎯 你写代码的地方！
│   └── __init__.py          # 目前为空
├── cs336-basics/            # 作业1的参考实现（模型代码在这里）
│   └── cs336_basics/
│       └── model.py         # Transformer 模型
├── tests/
│   ├── adapters.py          # 🎯 你要实现的适配器接口
│   ├── test_attention.py    # Flash Attention 测试
│   ├── test_ddp.py          # DDP 测试
│   ├── test_fsdp.py         # FSDP 测试
│   └── test_sharded_optimizer.py  # 分片优化器测试
├── pyproject.toml           # 项目配置
└── cs336_assignment2_systems.pdf  # 作业说明
```

### 建议开发顺序

```
第一周: Profiling & Benchmarking
  ├─ 写 benchmarking 脚本
  ├─ 跑 Nsight traces
  ├─ 做内存分析
  └─ 实现 Activation Checkpointing

第二周: Flash Attention
  ├─ 先实现 PyTorch 版（理解算法）
  ├─ 学习 Triton 教程代码
  └─ 实现 Triton 版 + benchmarking

第三周: 分布式训练
  ├─ 实现 Minimal DDP
  ├─ 实现 Overlapping DDP
  ├─ 实现 Optimizer State Sharding
  └─ 实现 FSDP

第四周: 综合优化 + Leaderboard
  ├─ 整合所有优化
  ├─ 写 writeup
  └─ 提交 Leaderboard
```

### 要实现的适配器函数（tests/adapters.py）

| 适配器函数 | 对应的实现 |
|-----------|-----------|
| `get_flashattention_autograd_function_pytorch()` | 返回你的 PyTorch Flash Attention 类 |
| `get_flashattention_autograd_function_triton()` | 返回你的 Triton Flash Attention 类 |
| `get_ddp(module)` | 返回你的 DDP wrapper 实例 |
| `ddp_on_after_backward(ddp_model, optimizer)` | DDP backward 后的同步调用 |
| `get_fsdp(module, compute_dtype)` | 返回你的 FSDP wrapper 实例 |
| `fsdp_on_after_backward(fsdp_model, optimizer)` | FSDP backward 后的同步调用 |
| `fsdp_gather_full_params(fsdp_model)` | FSDP 收集完整参数 |
| `get_sharded_optimizer(params, optimizer_cls, **kwargs)` | 返回你的 Sharded Optimizer 实例 |

### 运行测试

```bash
# 测试 Flash Attention
uv run pytest tests/test_attention.py

# 测试 DDP（建议多跑几次）
uv run pytest tests/test_ddp.py

# 测试 FSDP
uv run pytest tests/test_fsdp.py

# 测试分片优化器
uv run pytest tests/test_sharded_optimizer.py

# 运行全部测试并打包提交
bash test_and_make_submission.sh
```

---

## 10. 核心概念速查表

### GPU 内存层次

| 层级 | 名称 | 大小 | 速度 | 说明 |
|------|------|------|------|------|
| L1 | 寄存器/SRAM | ~20 MB | 最快 | GPU 核心内部缓存 |
| L2 | HBM（高带宽内存） | 40-80 GB | 快 | GPU 主内存 |
| L3 | CPU 内存 | 128+ GB | 慢 | 需要通过 PCIe 传输 |

Flash Attention 的本质就是尽量在 SRAM 中完成计算，减少 HBM 读写。

### 精度类型

| 类型 | 位数 | 范围 | 用途 |
|------|------|------|------|
| fp32 | 32 | 高精度 | 参数存储、优化器状态 |
| fp16 | 16 | 容易溢出 | 计算（需要 loss scaling） |
| bf16 | 16 | 范围同fp32 | 计算（推荐，不需要 loss scaling） |

### 分布式通信原语

| 操作 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **Broadcast** | 一个 rank 有数据 | 所有 rank 都有 | 初始化参数同步 |
| **All-Reduce** | 每个 rank 有一份 | 所有 rank 得到总和 | DDP 梯度同步 |
| **All-Gather** | 每个 rank 有一小块 | 所有 rank 得到完整数据 | FSDP 前向收集参数 |
| **Reduce-Scatter** | 每个 rank 有一份 | 每个 rank 得到对应分块的总和 | FSDP 反向分发梯度 |

### 内存优化技术对比

| 技术 | 节省什么 | 代价 |
|------|---------|------|
| 混合精度 | 激活值/梯度内存减半 | 精度可能略降 |
| Activation Checkpointing | 大幅减少激活值存储 | 需要重计算（~33% 额外时间） |
| Flash Attention | 注意力层 O(n²)→O(n) 内存 | 需要写 Triton kernel |
| 优化器分片 | 优化器状态减少到 1/N | 需要额外通信 |
| FSDP | 参数+优化器+梯度都分片 | 更多通信开销 |

---

## 📝 提交清单

最终你需要提交：

1. **writeup.pdf** — 包含所有实验结果、图表、截图和分析
2. **code.zip** — 运行 `bash test_and_make_submission.sh` 生成

writeup 中需要回答的所有问题汇总：

| # | 问题 | 分值 |
|---|------|------|
| 2.1 | Benchmarking Script (端到端计时表格 + compile/bf16 对比) | 4 |
| 2.2 | Nsight Profiling (截图 + 分析) | 3 |
| 2.3 | Memory Profiling (计算 + 测量 + 内存快照) | 3 |
| 3.1 | Activation Checkpointing (内存/速度/最大batch对比) | 5 |
| 4.1 | Flash Attention PyTorch (实现 + 测试通过) | 15 |
| 4.2 | Flash Attention Triton (实现 + 测试通过) | 15 |
| 4.3 | Flash Attention Benchmarking (图表) | 3 |
| 5.1 | Minimal DDP + Benchmarking | 5 |
| 5.2 | DDP Flat Gradients Benchmarking | 2 |
| 5.3 | DDP Overlapping + Benchmarking + Nsight | 6 |
| 6 | Optimizer State Sharding | 15 |
| 7 | FSDP | 20 |
| 8 | Leaderboard | 10 |
| | **总计** | **~106** |

---

> 💡 **建议：** 按照上述"开发顺序"逐步推进，每完成一个模块就运行对应的测试确保正确性，再进入下一个模块。遇到卡壳时，PDF 中的代码示例（特别是 Triton 的 weighted sum 教程）是非常好的参考。

祝学习顺利！🚀
