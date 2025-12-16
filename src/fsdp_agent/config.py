from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Union


# -----------------------------
# Data statistics (for prompt)
# -----------------------------
@dataclass
class DatasetStats:
    seq_len_p50: int = 2048
    seq_len_p90: int = 2048
    seq_len_p99: int = 2048
    seq_len_max: int = 2048
    pad_ratio: float = 0.0
    entropy_mean: float = 0.0
    entropy_var: float = 0.0
    squeue: str = "default"


# -----------------------------
# FSDP2 strategy protocol
# -----------------------------
@dataclass
class Fsdp2Layout:
    mesh_topology: Literal["1D", "2D"] = "1D"
    sharding_strategy: Literal["FULL", "HYBRID", "NO"] = "FULL"
    # bool / int: int 可表示“在更小 mesh 上 reshard”
    reshard_after_forward: Union[bool, int] = True
    shard_plan: Literal["DIM0", "DIM1", "LARGEST"] = "DIM0"
    offload_params: bool = False
    mp_policy: Literal["bf16", "fp32"] = "bf16"


@dataclass
class LayerOverride:
    start_layer: int
    end_layer: int
    layout: Fsdp2Layout


@dataclass
class Fsdp2Strategy:
    global_layout: Fsdp2Layout
    layer_overrides: List[LayerOverride] = field(default_factory=list)
    named_overrides: Dict[str, Fsdp2Layout] = field(default_factory=dict)
    schema_version: int = 2
    dataset_stats: Optional[DatasetStats] = None

    # -------- serialization helpers --------
    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
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
            "dataset_stats": asdict(self.dataset_stats) if self.dataset_stats else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "Fsdp2Strategy":
        schema_version = payload.get("schema_version", 2)
        gl = Fsdp2Layout(**payload["global_layout"])
        ovrs = []
        for o in payload.get("layer_overrides", []):
            ovrs.append(
                LayerOverride(
                    start_layer=int(o["start_layer"]),
                    end_layer=int(o["end_layer"]),
                    layout=Fsdp2Layout(**o["layout"]),
                )
            )
        named = {k: Fsdp2Layout(**v) for k, v in payload.get("named_overrides", {}).items()}
        ds = payload.get("dataset_stats")
        ds_obj = DatasetStats(**ds) if ds else None
        return cls(global_layout=gl, layer_overrides=ovrs, named_overrides=named, schema_version=schema_version, dataset_stats=ds_obj)

    # -------- semantic normalization & hash --------
    def normalized(self) -> "Fsdp2Strategy":
        """Merge equivalent overrides, drop redundant ones, sort deterministically."""
        norm = copy.deepcopy(self)

        # 1) drop overrides equal to global layout
        filtered: List[LayerOverride] = []
        for o in norm.layer_overrides:
            if o.start_layer >= o.end_layer:
                continue
            if asdict(o.layout) == asdict(norm.global_layout):
                continue
            filtered.append(o)

        # 2) sort by range
        filtered.sort(key=lambda x: (x.start_layer, x.end_layer))

        # 3) merge adjacent ranges with identical layout
        merged: List[LayerOverride] = []
        for o in filtered:
            if merged and asdict(merged[-1].layout) == asdict(o.layout) and merged[-1].end_layer == o.start_layer:
                merged[-1].end_layer = o.end_layer
            else:
                merged.append(copy.deepcopy(o))
        norm.layer_overrides = merged

        # 4) sort named overrides
        norm.named_overrides = dict(sorted(norm.named_overrides.items(), key=lambda kv: kv[0]))
        return norm

    def semantic_hash(self) -> str:
        norm = self.normalized()
        blob = json.dumps(norm.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Validators & helpers
# -----------------------------
_ALLOWED_MESH = {"1D", "2D"}
_ALLOWED_SHARD = {"FULL", "HYBRID", "NO"}
_ALLOWED_PLAN = {"DIM0", "DIM1", "LARGEST"}
_ALLOWED_MP = {"bf16", "fp32"}


def _validate_layout(layout: Fsdp2Layout) -> Fsdp2Layout:
    l = copy.deepcopy(layout)
    if l.mesh_topology not in _ALLOWED_MESH:
        l.mesh_topology = "1D"
    if l.sharding_strategy not in _ALLOWED_SHARD:
        l.sharding_strategy = "FULL"
    if isinstance(l.reshard_after_forward, int) and l.reshard_after_forward < 1:
        l.reshard_after_forward = 1
    if not isinstance(l.reshard_after_forward, (bool, int)):
        l.reshard_after_forward = True
    if l.shard_plan not in _ALLOWED_PLAN:
        l.shard_plan = "DIM0"
    if l.mp_policy not in _ALLOWED_MP:
        l.mp_policy = "bf16"
    l.offload_params = bool(l.offload_params)
    return l


def validate_strategy(strategy: Fsdp2Strategy, mem_limit_gb: float = 999.0) -> Fsdp2Strategy:
    """Lightweight sanity filter; does not estimate exact memory."""
    gl = _validate_layout(strategy.global_layout)

    ovrs: List[LayerOverride] = []
    for o in strategy.layer_overrides:
        if o.start_layer >= o.end_layer:
            continue
        ovrs.append(LayerOverride(start_layer=int(o.start_layer), end_layer=int(o.end_layer), layout=_validate_layout(o.layout)))

    named = {k: _validate_layout(v) for k, v in strategy.named_overrides.items()}
    ds = strategy.dataset_stats

    validated = Fsdp2Strategy(global_layout=gl, layer_overrides=ovrs, named_overrides=named, schema_version=strategy.schema_version, dataset_stats=ds)
    return validated.normalized()


# -----------------------------
# Seed strategies
# -----------------------------
def default_strategy() -> Fsdp2Strategy:
    gl = Fsdp2Layout()
    return Fsdp2Strategy(global_layout=gl)


def sandwich_sample_strategy() -> Fsdp2Strategy:
    """A simple heterogeneous seed: keep ends fast, middle memory-saving."""
    gl = Fsdp2Layout(reshard_after_forward=True)
    fast = Fsdp2Layout(reshard_after_forward=False)
    ovrs = [
        LayerOverride(start_layer=0, end_layer=4, layout=fast),
        LayerOverride(start_layer=20, end_layer=24, layout=fast),
    ]
    return Fsdp2Strategy(global_layout=gl, layer_overrides=ovrs)


# 可选：性能优先候选，用于提示或枚举
def perf_tier_candidates() -> List[Fsdp2Strategy]:
    fast = Fsdp2Layout(reshard_after_forward=False)
    overlap = Fsdp2Layout(reshard_after_forward=False, shard_plan="DIM0")
    return [
        default_strategy(),
        sandwich_sample_strategy(),
        Fsdp2Strategy(global_layout=overlap),
    ]
