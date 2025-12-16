from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Union

# ---------------------------------------------------------------
# FSDP2 策略协议：分层覆盖 + 特殊模块覆盖
# ---------------------------------------------------------------


@dataclass
class Fsdp2Layout:
    """单个模块/层的布局与精度配置。"""

    mesh_topology: Literal["1D", "2D"] = "1D"  # 1D DP 或 2D HSDP
    sharding_strategy: Literal["FULL", "HYBRID", "NO"] = "FULL"
    # True/False 或 int（用于 intra-node reshard）
    reshard_after_forward: Union[bool, int] = True
    shard_plan: Literal["DIM0", "DIM1", "LARGEST"] = "DIM0"
    offload_params: bool = False
    mp_policy: Literal["bf16", "fp32"] = "bf16"


@dataclass
class LayerOverride:
    """按层范围覆盖策略。"""

    start_layer: int  # 包含
    end_layer: int  # 不包含
    layout: Fsdp2Layout


@dataclass
class Fsdp2Strategy:
    """Agent 输出的完整策略对象。"""

    global_layout: Fsdp2Layout
    layer_overrides: List[LayerOverride] = field(default_factory=list)
    named_overrides: Dict[str, Fsdp2Layout] = field(default_factory=dict)
    dataset_stats: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    schema_version: int = 2

    def to_dict(self) -> Dict:
        return {
            "global_layout": asdict(self.global_layout),
            "layer_overrides": [
                {
                    "start_layer": o.start_layer,
                    "end_layer": o.end_layer,
                    "layout": asdict(o.layout),
                }
                for o in self.layer_overrides
            ],
            "named_overrides": {k: asdict(v) for k, v in self.named_overrides.items()},
            "dataset_stats": self.dataset_stats,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: Dict) -> "Fsdp2Strategy":
        gl = Fsdp2Layout(**data.get("global_layout", {}))
        layer_overrides = [
            LayerOverride(
                start_layer=o["start_layer"],
                end_layer=o["end_layer"],
                layout=Fsdp2Layout(**o["layout"]),
            )
            for o in data.get("layer_overrides", [])
        ]
        named_overrides = {
            k: Fsdp2Layout(**v) for k, v in data.get("named_overrides", {}).items()
        }
        return Fsdp2Strategy(
            global_layout=gl,
            layer_overrides=layer_overrides,
            named_overrides=named_overrides,
            dataset_stats=data.get("dataset_stats", {}),
            schema_version=data.get("schema_version", 2),
        )


# ---------------------------------------------------------------
# 默认策略与示例种子
# ---------------------------------------------------------------


def default_strategy() -> Fsdp2Strategy:
    """安全基线：全局 FULL + reshard=True（memory saving）。"""
    return Fsdp2Strategy(global_layout=Fsdp2Layout())


def sandwich_sample_strategy() -> Fsdp2Strategy:
    """
    示例：前 4 层 HYBRID+不 reshard（减少通信），中间默认，Embedding 不切（NO）。
    """
    return Fsdp2Strategy(
        global_layout=Fsdp2Layout(),
        layer_overrides=[
            LayerOverride(
                start_layer=0,
                end_layer=4,
                layout=Fsdp2Layout(
                    mesh_topology="2D",
                    sharding_strategy="HYBRID",
                    reshard_after_forward=False,
                ),
            )
        ],
        named_overrides={
            "embed_tokens": Fsdp2Layout(
                sharding_strategy="NO", reshard_after_forward=False
            )
        },
    )


def validate_strategy(strategy: Fsdp2Strategy, mem_limit_gb: float = 70.0) -> Fsdp2Strategy:
    """基础合法性检查，防止 LLM 输出非法字段。"""

    def _clip_layout(layout: Fsdp2Layout) -> Fsdp2Layout:
        if layout.mesh_topology not in ("1D", "2D"):
            layout.mesh_topology = "1D"
        if layout.sharding_strategy not in ("FULL", "HYBRID", "NO"):
            layout.sharding_strategy = "FULL"
        if isinstance(layout.reshard_after_forward, bool):
            pass
        elif isinstance(layout.reshard_after_forward, int):
            layout.reshard_after_forward = max(1, layout.reshard_after_forward)
        else:
            layout.reshard_after_forward = True
        if layout.shard_plan not in ("DIM0", "DIM1", "LARGEST"):
            layout.shard_plan = "DIM0"
        if layout.mp_policy not in ("bf16", "fp32"):
            layout.mp_policy = "bf16"
        layout.offload_params = bool(layout.offload_params)
        return layout

    strategy.global_layout = _clip_layout(strategy.global_layout)
    strategy.layer_overrides = [
        LayerOverride(
            start_layer=o.start_layer,
            end_layer=o.end_layer,
            layout=_clip_layout(o.layout),
        )
        for o in strategy.layer_overrides
        if o.end_layer > o.start_layer
    ]
    strategy.named_overrides = {
        k: _clip_layout(v) for k, v in strategy.named_overrides.items()
    }
    return strategy


# ---------------------------------------------------------------
# 性能导向提示（供 Prompt 使用，不直接作为种子）
# ---------------------------------------------------------------


def perf_tier_candidates() -> Dict[str, Dict]:
    """预置的性能导向提示，帮助提示词强调方向。"""
    return {
        "overlap_first": {
            "global_layout": {"mesh_topology": "1D", "reshard_after_forward": True},
            "hints": "先开启 overlap，再观察显存，再逐步放宽分片策略。",
        },
        "no_reshard": {
            "global_layout": {"reshard_after_forward": False},
            "hints": "显存充足时减少一次 all-gather 通信。",
        },
        "mem_eff": {
            "layer_overrides": [
                {"start_layer": 0, "end_layer": 4, "layout": {"reshard_after_forward": False}}
            ],
            "hints": "前几层不 reshard，减少通信次数。",
        },
    }


# 兼容调用入口
def strategy_from_dict(payload: Dict) -> Fsdp2Strategy:
    return Fsdp2Strategy.from_dict(payload)
