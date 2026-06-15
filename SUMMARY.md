# CS336 Assignment 2: Systems — 实现总结文档

## 1. 环境配置

| 项目 | 配置 |
|------|------|
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu124（原版要求 2.11，因驱动限制降级） |
| Triton | 3.1.0 |
| GPU | 2 × NVIDIA H20 |
| Driver | 535.247.01（支持 CUDA 12.2） |
| uv | `/root/aigame/.tools/uv` |

**运行测试需要设置环境变量：**
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 /root/aigame/.tools/uv run pytest tests/ -v
```

---

## 2. 代码文件清单

```
cs336_systems/
├── __init__.py                  # 包初始化（原有）
├── flash_attention.py           # ① PyTorch 版 Flash Attention 2
├── flash_attention_triton.py    # ② Triton 版 Flash Attention 2
├── ddp.py                       # ③ DDP（分布式数据并行）
├── sharded_optimizer.py         # ④ 优化器状态分片
├── fsdp.py                      # ⑤ FSDP（全分片数据并行）
└── benchmark.py                 # ⑥ 基准测试脚本

tests/
└── adapters.py                  # 适配器（已实现，连接代码到测试框架）
```

---

## 3. 各模块实现详解

### 3.1 Flash Attention 2 — PyTorch 版 (`flash_attention.py`)

**类：** `FlashAttentionPyTorch(torch.autograd.Function)`

**Forward 实现思路：**
- 将 Q 按行分成 tile（Br=64），K/V 按列分成 tile（Bc=64）
- 对每个 Q tile，遍历所有 K/V tile，使用 **online softmax** 逐步累加结果
- 维护三个累加量：`m`（当前最大值）、`l`（softmax 分母）、`o`（加权输出）
- 每看到新的 K/V tile，用修正因子 `alpha = exp(m_old - m_new)` 修正之前的累加结果
- 最终保存 `q, k, v, o, L`（L = logsumexp）供 backward 使用

**Backward 实现思路：**
- 先算 `D = rowsum(O ⊙ dO)`
- 外层循环遍历 K/V tile，内层遍历 Q tile
- 重算 attention scores `S`，用保存的 `L` 恢复 attention weights `P = exp(S - L)`
- `dV += P^T @ dO`，`dP = dO @ V^T`，`dS = P * (dP - D)`
- `dQ += dS @ K * scale`，`dK += dS^T @ Q * scale`

**Causal masking：** 根据 Q/K 的位置索引，跳过 `q_pos < k_pos` 的 tile，掩码填 `-inf`。

### 3.2 Flash Attention 2 — Triton 版 (`flash_attention_triton.py`)

**类：** `FlashAttentionTriton(torch.autograd.Function)`

**Forward：** 使用 Triton kernel `_flash_attn_fwd_kernel`
- 2D launch grid：`(ceil(n_queries / BLOCK_Q), batch_size)`
- 每个 program instance 处理一个 Q tile，遍历所有 K/V tile
- 使用 `tl.dot` 计算矩阵乘法（要求维度是 2 的幂次）
- Online softmax 逻辑与 PyTorch 版相同
- Causal masking 通过 `tl.where` 实现

**Backward：** 复用 PyTorch 实现的 `_flash_attn_backward_pytorch`（可以用 `torch.compile` 加速）

### 3.3 DDP (`ddp.py`)

**类：** `DDP(nn.Module)`

**核心设计：**
1. **初始化时**：`dist.broadcast(param.data, src=0)` 将 rank 0 的参数广播到所有 rank
2. **前向传播**：直接调用 `self.module(*inputs, **kwargs)`
3. **梯度同步（与 backward 重叠）**：
   - 用 `register_post_accumulate_grad_hook` 注册钩子
   - 每个参数的梯度计算完毕后，立即发起异步 `all_reduce`（`async_op=True`）
   - 这样 backward 计算和梯度通信可以重叠执行
4. **`finish_gradient_synchronization()`**：等待所有异步操作完成，对梯度除以 `world_size` 求平均

**Tied weights 处理：** 用 `seen_params` set 跟踪已注册的参数 id，避免对同一参数重复注册 hook。

### 3.4 Optimizer State Sharding (`sharded_optimizer.py`)

**类：** `ShardedOptimizer(torch.optim.Optimizer)`

**核心设计：**
1. **参数分配**：将去重后的参数按 `i % world_size` 分配给各 rank
2. **内部优化器**：每个 rank 只为自己分到的参数创建 `optimizer_cls` 实例
3. **`step()`**：
   - 只调用自己那部分参数的 `inner_optimizer.step()`
   - 然后遍历所有 rank，用 `dist.broadcast(p.data, src=owner_rank)` 同步更新后的参数
4. **内存节省**：每个 rank 的优化器状态（AdamW 的 m 和 v）从完整变为 ~1/world_size

### 3.5 FSDP (`fsdp.py`)

**类：** `FSDP(nn.Module)`

**参数分类：**
- **Sharded**：`Linear` 和 `Embedding` 的 weight → 按 rank 分片存储
- **Replicated**：其他参数（如 `RMSNorm` 的 weight） → 每个 rank 保留完整副本

**Forward 流程：**
1. `_all_gather_params()`：all-gather 收集所有 shard 拼成完整参数
2. 运行 `self.module(*inputs, **kwargs)`
3. 在 output 上注册 backward hook，用于 backward 开始时再次 all-gather
4. `_reshard_params()`：恢复 shard 存储

**Backward 流程：**
1. `_on_backward_start()`：通过 output hook 触发，再次 all-gather 参数
2. 梯度计算完毕后，通过 `post_accumulate_grad_hook` 触发通信：
   - Sharded 参数：`all_reduce` 梯度（因 gloo 不支持 `reduce_scatter`），然后取本 rank 的 shard
   - Replicated 参数：`all_reduce` 梯度

**`finish_gradient_synchronization()`：**
1. 先 `_reshard_params()` 恢复参数为 shard 形状
2. 等待所有异步通信完成
3. 为 sharded 参数：从 all-reduce 结果中取出对应 shard 作为梯度
4. 对所有梯度除以 `world_size` 求平均

**Mixed precision 支持：**
- Master weights 始终以 fp32 存储
- `compute_dtype` 指定后，all-gather 出的完整参数会 cast 到该 dtype 再参与计算
- 梯度最终转回 `param.data.dtype` 以匹配参数

**gloo 兼容性：** PyTorch 2.5 的 gloo 后端不支持 `reduce_scatter`，改用 `all_reduce` + 手动取 shard 的方式实现等价功能。

### 3.6 Benchmarking 脚本 (`benchmark.py`)

支持命令行参数：
```bash
uv run python -m cs336_systems.benchmark \
    --size xl --batch-size 16 --mode full \
    --compile --use-amp --amp-dtype bfloat16
```

功能：
- 支持 small/medium/large/xl/10B 五种模型尺寸
- 支持 forward / forward_backward / full 三种模式
- 支持 `torch.compile` 和混合精度 (`torch.autocast`)
- 正确使用 `torch.cuda.synchronize()` 计时
- 输出平均耗时和峰值 GPU 内存

---

## 4. 测试结果

```
tests/test_attention.py
  ✅ test_flash_forward_pass_pytorch
  ✅ test_flash_forward_pass_triton[False]
  ✅ test_flash_forward_pass_triton[True]
  ✅ test_flash_backward_pytorch
  ✅ test_flash_backward_triton[False]
  ✅ test_flash_backward_triton[True]

tests/test_ddp.py
  ✅ test_DistributedDataParallel[ToyModel]
  ✅ test_DistributedDataParallel[ToyModelWithTiedWeights]

tests/test_fsdp.py
  ✅ test_fsdp_correctness[fp32]
  ✅ test_fsdp_correctness[fp16]
  ✅ test_fsdp_gradient_sync[fp32]
  ✅ test_fsdp_gradient_sync[fp16]

tests/test_sharded_optimizer.py
  ✅ test_sharded_optimizer[ToyModel]
  ✅ test_sharded_optimizer[ToyModelWithTiedWeights]

总计: 14/14 passed ✅    用时 ~4 分 17 秒
```

---

## 5. 已知限制与注意事项

| 问题 | 说明 |
|------|------|
| PyTorch 版本 | 从 2.11 降到 2.5.1 以兼容驱动。如恢复原版需升级 GPU 驱动到 560+ |
| FSDP reduce_scatter | gloo 后端不支持，用 all_reduce + 取 shard 替代。nccl 后端可直接用 reduce_scatter |
| CUBLAS_WORKSPACE_CONFIG | FSDP 测试要求 deterministic 模式，GPU 上运行必须设置此环境变量 |
| Flash Attention Triton backward | 当前使用 PyTorch 实现，可进一步优化为 Triton kernel |
| Benchmarking 实验 | 脚本已就绪，具体实验数据需在 GPU 上运行收集 |

---

## 6. 运行指南

```bash
# 设置环境变量
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 运行全部测试
/root/aigame/.tools/uv run pytest tests/ -v

# 单独运行各模块测试
/root/aigame/.tools/uv run pytest tests/test_attention.py -v      # Flash Attention
/root/aigame/.tools/uv run pytest tests/test_ddp.py -v            # DDP
/root/aigame/.tools/uv run pytest tests/test_sharded_optimizer.py -v  # Sharded Optimizer
/root/aigame/.tools/uv run pytest tests/test_fsdp.py -v           # FSDP

# 运行 benchmarking
/root/aigame/.tools/uv run python -m cs336_systems.benchmark --size xl --batch-size 16

# 打包提交
bash test_and_make_submission.sh
```
