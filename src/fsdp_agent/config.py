from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

# Mesh / topology ----------------------------------------------------------------


@dataclass
class MeshConfig:
    """Device mesh topology. Only supports 1D DP or 2D HSDP for 4 GPUs."""

    layout: Literal["dp_1d", "hsdp_2d"] = "dp_1d"
    dp_world: int = 4
    hs_replicate_dim: Optional[int] = None  # optional replication dim for HSDP
    mesh_shape: Optional[Sequence[int]] = None  # 支持多节点自定义 shape
    mesh_ranks: Optional[Sequence[int]] = None  # 扁平 rank 列表


# Grouping / wrapping -------------------------------------------------------------


@dataclass
class GroupingConfig:
    """Module grouping and wrap policy for FSDP2."""

    module_grouping: Literal["block", "mem_eff", "all", "manual"] = "block"
    use_size_based_auto_wrap: bool = False
    size_based_min_num_params: int = int(1e8)
    treat_embeddings_special: bool = True
    manual_groups: List[List[str]] = field(default_factory=list)  # 层名/正则列表


# Shard and reshard ---------------------------------------------------------------


@dataclass
class ShardReshardConfig:
    """Parameter sharding dims and reshard policy."""

    use_custom_shard_placement: bool = False
    custom_shard_rules: Dict[str, int] = field(default_factory=dict)
    # memory_saving=True, no_reshard=False, intra_node=int, auto=None
    reshard_behavior: Literal["memory_saving", "no_reshard", "intra_node", "auto"] = (
        "auto"
    )
    per_group_reshard: Dict[str, Literal["memory_saving", "no_reshard", "intra_node", "auto"]] = field(
        default_factory=dict
    )


# Communication / overlap / prefetch ---------------------------------------------


@dataclass
class CommOverlapConfig:
    """Communication overlap and prefetch knobs."""

    unshard_async_op: bool = False
    unshard_in_backward_embeddings: bool = False
    backward_prefetch_num: int = 0
    forward_prefetch_num: int = 0
    force_sum_reduction_for_comms: bool = False
    disable_hsdp_all_reduce: bool = False


# Precision / offload -------------------------------------------------------------


@dataclass
class PrecisionOffloadConfig:
    mp_policy: Literal["bf16", "fp16", "none"] = "bf16"
    offload_param: bool = False
    offload_grad: bool = False
    offload_optim: bool = False


@dataclass
class DatasetStats:
    """数据集/队列的形状统计，供 Agent/Judge 推理和控制显存风险。"""

    seq_len_p50: int = 2048
    seq_len_p90: int = 2048
    seq_len_p99: int = 2048
    seq_len_max: int = 2048
    pad_ratio: float = 0.0
    entropy_mean: float = 0.0
    entropy_var: float = 0.0
    squeue: str = "default"  # 任务/队列标签


# Full strategy -------------------------------------------------------------------


@dataclass
class Fsdp2Strategy:
    mesh: MeshConfig = field(default_factory=MeshConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    shard_reshard: ShardReshardConfig = field(default_factory=ShardReshardConfig)
    comm_overlap: CommOverlapConfig = field(default_factory=CommOverlapConfig)
    precision_offload: PrecisionOffloadConfig = field(
        default_factory=PrecisionOffloadConfig
    )
    train_hyper: Dict = field(default_factory=dict)  # micro_batch, grad_accum, seq_len override
    dataset_stats: DatasetStats = field(default_factory=DatasetStats)
    schema_version: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)


# Strategy seeds for reproducible baselines --------------------------------------


def default_strategy() -> Fsdp2Strategy:
    """Official-ish FSDP2 default style baseline."""
    return Fsdp2Strategy()


def heuristic_mem_eff_strategy() -> Fsdp2Strategy:
    """Human heuristic focusing on memory efficiency + some prefetch."""
    return Fsdp2Strategy(
        grouping=GroupingConfig(module_grouping="mem_eff", treat_embeddings_special=True),
        shard_reshard=ShardReshardConfig(reshard_behavior="memory_saving"),
        comm_overlap=CommOverlapConfig(
            backward_prefetch_num=2, forward_prefetch_num=1, unshard_async_op=True
        ),
    )


def random_strategy(space_overrides: Optional[Dict[str, List]] = None) -> Fsdp2Strategy:
    """
    Simple random sampler for ablations.

    space_overrides lets callers constrain options. This keeps the space discrete and
    small so the LLM loop can be compared against a cheap baseline.
    """
    import random

    space_overrides = space_overrides or {}

    layout = random.choice(space_overrides.get("layout", ["dp_1d", "hsdp_2d"]))
    grouping = random.choice(
        space_overrides.get("module_grouping", ["block", "mem_eff", "all"])
    )
    reshard = random.choice(
        space_overrides.get(
            "reshard_behavior", ["memory_saving", "no_reshard", "intra_node", "auto"]
        )
    )
    backward_prefetch_num = random.choice(
        space_overrides.get("backward_prefetch_num", [0, 1, 2])
    )
    forward_prefetch_num = random.choice(
        space_overrides.get("forward_prefetch_num", [0, 1])
    )
    mp_policy = random.choice(space_overrides.get("mp_policy", ["bf16", "fp16"]))

    return Fsdp2Strategy(
        mesh=MeshConfig(layout=layout),
        grouping=GroupingConfig(module_grouping=grouping),
        shard_reshard=ShardReshardConfig(reshard_behavior=reshard),
        comm_overlap=CommOverlapConfig(
            backward_prefetch_num=backward_prefetch_num,
            forward_prefetch_num=forward_prefetch_num,
            unshard_async_op=random.choice([False, True]),
        ),
        precision_offload=PrecisionOffloadConfig(mp_policy=mp_policy),
    )


def strategy_from_dict(payload: Dict) -> Fsdp2Strategy:
    """Hydrate Fsdp2Strategy from a plain dict (e.g., parsed LLM JSON)."""
    return Fsdp2Strategy(
        mesh=MeshConfig(**payload.get("mesh", {})),
        grouping=GroupingConfig(**payload.get("grouping", {})),
        shard_reshard=ShardReshardConfig(**payload.get("shard_reshard", {})),
        comm_overlap=CommOverlapConfig(**payload.get("comm_overlap", {})),
        precision_offload=PrecisionOffloadConfig(
            **payload.get("precision_offload", {})
        ),
        train_hyper=payload.get("train_hyper", {}),
        dataset_stats=DatasetStats(**payload.get("dataset_stats", {})),
        schema_version=payload.get("schema_version", 1),
    )


def validate_strategy(strategy: Fsdp2Strategy, mem_limit_gb: float) -> Fsdp2Strategy:
    """简单合法性和保守兜底。"""
    if strategy.mesh.layout not in ("dp_1d", "hsdp_2d"):
        strategy.mesh.layout = "dp_1d"
    if strategy.shard_reshard.reshard_behavior not in (
        "memory_saving",
        "no_reshard",
        "intra_node",
        "auto",
    ):
        strategy.shard_reshard.reshard_behavior = "auto"
    if mem_limit_gb < 70 and strategy.shard_reshard.reshard_behavior == "no_reshard":
        strategy.shard_reshard.reshard_behavior = "memory_saving"
    return strategy
