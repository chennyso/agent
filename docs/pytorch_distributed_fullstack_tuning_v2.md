# PyTorch 分布式训练：全栈精细化控制与源码级调优手册 (终极版 v2.0)

本手册面向工程与研究人员，覆盖 PyTorch 原生分布式训练的底层控制与源码级调优策略。内容以 “可推理、可学习、可复用” 的结构组织，覆盖 Mesh、PP、TP、CP、EP、FSDP2 六大模块，并对 FSDP2 做深度解析。

版本约定：
- 适用 PyTorch >= 2.4（FSDP2/fully_shard）
- 所有源码定位为 PyTorch 源码/测试目录中的参考锚点

---

## 0. 统一的“可推理策略结构”

将工程经验抽象为统一的策略卡（Strategy Card），方便复用与自动化决策。

```yaml
StrategyCard:
  module: device_mesh | fsdp2 | pp | tp | cp | ep
  intent: "优化目标（如显存/通信/吞吐/稳定性）"
  preconditions:
    - "硬件/拓扑/模型规模的前置条件"
  control_points:
    - name: "参数/接口/策略点"
      effect: "影响哪些路径或成本"
  observables:
    - "可观测指标（通信占比、显存峰值、collective 次数等）"
  policies:
    - "具体策略与触发条件"
  risks:
    - "可能的风险/回退条件"
  source_refs:
    - "源码定位或测试用例"
  example:
    - "最小可执行示例"
```

推荐的推理流程（单一问题场景）：
1) 确定硬件与拓扑约束 → 2) 设定 Device Mesh → 3) 设定 FSDP2 基础布局与分片策略 → 4) 叠加 PP/TP/CP/EP → 5) 观测证据并迭代。

---

## 1. 核心架构：设备网格 (Device Mesh) 的“手术刀级”控制

### 1.1 网格维度显式命名与物理映射
控制点：`mesh_dim_names` 顺序决定通信层级  
源码定位：`test_pp_composability.py:234`

精细化策略：
- 命名规则：遵循 “Bandwidth-Frequency Law”，从低带宽到高带宽排列维度。
- 物理映射：TP（Tensor Parallel）必须位于 `mesh_dim_names` 的最后一个索引，优先使用机内最高带宽链路。

代码示例：
```python
# 8 卡无 NVLink，构建 3D 网格：PP=4, DP=1, TP=2
device_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(4, 1, 2),
    mesh_dim_names=("pp", "dp", "tp")  # rank 0-1 做 TP (PCIe 邻居)
)
tp_mesh = device_mesh["tp"]
pp_mesh = device_mesh["pp"]
```

### 1.2 专家并行 (EP) 的“网格折叠与展开”
控制点：`_unflatten` 方法  
源码定位：`test_device_mesh.py:1017 (test_unflatten_mesh_3d)`

精细化策略：
- EP 本质上是特殊 DP：在 EP 维度上不同 Rank 只拥有部分 Experts，而在非 Expert 层保持 Replicated。
- 从全局 world 网格出发，通过 `_unflatten` 将 DP 维度切开为 `(dp_replicate, ep)`。

代码示例：
```python
global_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("world",))
# 将 8 卡切分为 DP=2, EP=2, TP=2
ep_mesh_3d = global_mesh._unflatten(0, (2, 2, 2), ("dp", "ep", "tp"))
```

---

## 2. FSDP2 (Fully Sharded Data Parallel) 的深度定制

### 2.1 显存回收与通信权衡 (Resharding Strategy)
控制点：`reshard_after_forward`  
源码定位：`_fully_shard.py:144-161`, `test_fully_shard_comm.py:376`

精细化策略：
- `True`（默认）：前向结束后立即释放 unsharded 参数，后向时重新 all-gather。最省显存。
- `False`：参数常驻显存，速度最快，显存开销大（32B 模型在 24GB 卡上严禁）。
- `int`（例如 `2`）：将参数 reshard 到大小为 2 的子组（如 Intra-node），在 PCIe 瓶颈时减少全局 all-gather。

最佳实践：
- Root Module 设置 `False`，减少后向初始 all-gather。
- 非根模块默认 `True`，保守保显存。

### 2.2 分片布局自定义 (Shard Placement)
控制点：`shard_placement_fn` 回调  
源码定位：`_fully_shard.py:168-175`, `test_fully_shard_init.py:1183`

精细化策略：
- 默认行为：切分第 0 维。
- Embedding/LM Head 常见维度 `(Vocab, Hidden)`，切分 dim0 合理。
- 特殊线性层若 dim1 更大，可强制切分 dim1。

代码示例：
```python
def custom_placement(param: nn.Parameter) -> Optional[Shard]:
    if "embed" in param.name and param.shape[0] > 32000:
        return Shard(0)
    if "out_proj" in param.name:
        return Shard(1)
    return Shard(0)

fully_shard(model, shard_placement_fn=custom_placement)
```

### 2.3 HSDP (Hybrid Sharding) 2D 网格支持
控制点：`mesh` 参数传入 2D 网格  
源码定位：`_fully_shard.py:136-141`

精细化策略：
- 第一维 Replicate，第二维 Shard：组内 FSDP，组间 DDP。
- 相比纯 FSDP 减少通信量，相比纯 DDP 节省显存。

代码示例：
```python
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))
fully_shard(model, mesh=mesh_2d)
```

### 2.4 混合精度与卸载策略 (Policy)
控制点：`MixedPrecisionPolicy`, `CPUOffloadPolicy`  
源码定位：`_fsdp_api.py`

精细化策略：
- `param_dtype`: 推荐 `torch.bfloat16`。
- `reduce_dtype`: 关键权衡点。
  - `float32`：默认，数值稳定但通信量大。
  - `bfloat16`：PCIe 环境推荐，通信量减半，速度提升明显。
- `CPUOffloadPolicy(pin_memory=True)`：配合 PP 对非活跃 stage 卸载，需预取控制。

### 2.5 梯度分桶与动态分片 (Internal Optimizations)
控制点：内部逻辑（理解有助于调优）  
源码定位：`_fully_shard.py:121-130`

机制要点：
- 梯度分桶：合并小参数梯度，减少 collective 启动开销。
- 动态分片：未指定 `shard_placement_fn` 时，系统根据参数形状自动选择分片维度。

---

## 3. 流水线并行 (Pipeline Parallel - PP) 的调度控制

### 3.1 调度器 (Schedules) 选择
控制点：`ScheduleClass`  
源码定位：`schedules.py`

精细化策略：
- `Schedule1F1B`：标准模式，显存/速度平衡。
- `ScheduleInterleaved1F1B`：虚拟流水线；`n_virtual=2` 等会增加 activation 显存。
- `ScheduleInterleavedZeroBubble`：无气泡模式，适合算力富余但显存紧张。
- `ScheduleDualPipeV`：双向流水线（DeepSeek 风格），`schedules.py:2983`。

### 3.2 微批次 (Microbatches) 与阶段划分
控制点：`n_microbatches`, `split_points`

精细化策略：
- 经验公式：`n_microbatches >= 4 * pp_degree`。
- 微批次越多，气泡占比越小，但梯度累积显存开销上升。
- 32B/24GB 场景建议：`pp_degree=4`，`n_microbatches=8/16`。

---

## 4. 张量并行 (Tensor Parallel - TP) 的算子级定制

### 4.1 并行样式 (Parallel Styles)
控制点：`ColwiseParallel`, `RowwiseParallel`  
源码定位：`style.py`, `api.py`

精细化策略：
- `use_local_output=False`：在 Colwise 层保持 DTensor 输出，可直接衔接 Rowwise，省一次 all-gather。
- Input Layouts：输入可指定 Replicated 或 Sharded。

### 4.2 序列并行 (Sequence Parallel - SP)
控制点：`SequenceParallel` 样式

精细化策略：
- 将 LayerNorm/Dropout 等非矩阵操作按序列维度切分。
- 在 `TP > 1` 时建议始终开启，显著减少 activation 显存。

---

## 5. 上下文并行 (Context Parallel - CP) 的深度定制

### 5.1 序列维度与 Attention 类型
控制点：`_ContextParallel` 初始化参数  
源码定位：`_attention.py:1309`

精细化策略：
- `seq_dim`：指定序列维度位置（通常为 1 或 2）。
- `attention_type`：
  - `AttentionType.FLEX`：支持自定义 block mask，适合变长序列/因果掩码（`test_attention.py:535`）。
  - `AttentionType.SDPA`：原生 FlashAttention，速度最快。

### 5.2 负载均衡 (Load Balancing)
控制点：`load_balancer`  
源码定位：`_attention.py:18`

可选策略：
- `_HeadTailLoadBalancer`：首尾平衡。
- `_PerDocumentHeadTailLoadBalancer`：文档级平衡。
- Ring Attention：`test_attention.py:131` 展示 `_test_ring_attention_sdpa`，P2P 轮转实现。

### 5.3 缓冲区管理 (Buffer Management)
控制点：`context_parallel` 上下文管理器参数  
源码定位：`_attention.py:1496`

精细化策略：
- `buffers`：显式指定随序列切分的 Tensor（如 Positional Embedding, Labels）。
- `no_restore_buffers`：CP 结束后不恢复的 buffer，节省通信。

### 5.4 KV 旋转方法 (Rotate Method)
控制点：`set_rotate_method`  
源码定位：`_attention.py:1669`

精细化策略：
- `allgather`：简单易用，但显存开销大。
- `alltoall`：推荐，P2P 轮转 KV block，显存恒定，适合超长序列。

---

## 6. 专家并行 (Expert Parallel - EP) 与 MoE

### 6.1 混合并行拓扑
控制点：网格组合 `(dp, ep, tp)`  
源码定位：`common_fsdp.py:773 (MixtureOfExperts)`

精细化策略：
- Expert 参数：使用 `ep_mesh` 做 FSDP 或 TP。
- Shared 参数：使用 `dp_mesh` 做 Replicate 或 FSDP。

代码逻辑：
```python
if wrap_fsdp:
    expert_group = torch.distributed.new_group([group.rank()])
    expert = FSDP(expert, expert_group)
```

---

## 7. 并行策略的“工程经验 → 可复用规则”

### 7.1 单一问题场景的决策模板
```yaml
DecisionFlow:
  - step: "设定 mesh_dim_names"
    rule: "TP 放在最后；跨节点维度在前"
  - step: "FSDP2 reshard"
    rule: "Root 设 False；其他 True；PCIe 瓶颈可考虑 int 子组"
  - step: "TP/PP/CP/EP 叠加"
    rule: "先 TP（算子瓶颈），再 PP（显存墙），最后 CP/EP（长序列或 MoE）"
  - step: "调度与微批次"
    rule: "n_microbatches >= 4 * pp_degree"
```

### 7.2 可复用策略卡示例（FSDP2 Reshard）
```yaml
StrategyCard:
  module: fsdp2
  intent: "显存节省 vs 通信开销"
  preconditions:
    - "模型参数量大"
    - "通信带宽受限"
  control_points:
    - name: reshard_after_forward
      effect: "param lifetime + all-gather 压力"
  observables:
    - "comm_ratio"
    - "peak_memory"
  policies:
    - "root: False, others: True"
    - "PCIe 瓶颈时使用 int 子组"
  risks:
    - "False 可能造成显存爆炸"
  source_refs:
    - "_fully_shard.py:144-161"
  example:
    - "fully_shard(model, reshard_after_forward=False)"
```

---

## 8. 组合策略小结（可落地的单一结构）

1) 定义 mesh 维度顺序（TP 最后）  
2) FSDP2 设定 reshard/shard placement/mixed precision  
3) 选择 PP 调度与 microbatch  
4) 添加 TP/SP，必要时加入 CP/EP  
5) 使用统一策略卡记录“控制点-证据-动作-风险-源码”

这套结构可直接用于人工调优，也可作为自动化策略搜索或 LLM 决策的输入协议。
