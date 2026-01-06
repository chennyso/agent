from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
import re
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
    # bool / int / None: None means FSDP2 auto (root=False, others=True).
    reshard_after_forward: Optional[Union[bool, int]] = None
    shard_plan: Literal["DIM0", "DIM1", "LARGEST"] = "DIM0"
    offload_params: bool = False
    offload_pin_memory: bool = True
    mp_policy: Literal["bf16", "fp32"] = "bf16"


@dataclass
class GroupingConfig:
    """FSDP grouping mode (controls fully_shard call granularity)."""

    mode: Literal["block", "merged"] = "block"
    merge_factor: int = 1


@dataclass
class LayerOverride:
    start_layer: Optional[int]
    end_layer: Optional[int]
    layout: Fsdp2Layout
    layers: Optional[List[int]] = None


@dataclass
class ParallelSpec:
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    cp_degree: int = 1
    sp_enabled: bool = False
    tp_plan: str = "auto"
    pp_microbatches: int = 1
    pp_schedule: str = "1f1b"
    pp_stages: Optional[List[List[int]]] = None


@dataclass
class Fsdp2Strategy:
    global_layout: Fsdp2Layout
    layer_overrides: List[LayerOverride] = field(default_factory=list)
    named_overrides: Dict[str, Fsdp2Layout] = field(default_factory=dict)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
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
                    "layers": o.layers,
                    "layout": asdict(o.layout),
                }
                for o in self.layer_overrides
            ],
            "named_overrides": {k: asdict(v) for k, v in self.named_overrides.items()},
            "grouping": asdict(self.grouping),
            "parallel": asdict(self.parallel),
            "dataset_stats": asdict(self.dataset_stats) if self.dataset_stats else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "Fsdp2Strategy":
        schema_version = payload.get("schema_version", 2)
        gl = Fsdp2Layout(**payload["global_layout"])
        grouping = GroupingConfig(**payload.get("grouping", {})) if payload.get("grouping") else GroupingConfig()
        ovrs = []
        for o in payload.get("layer_overrides", []):
            ovrs.append(
                LayerOverride(
                    start_layer=int(o["start_layer"]) if o.get("start_layer") is not None else None,
                    end_layer=int(o["end_layer"]) if o.get("end_layer") is not None else None,
                    layout=Fsdp2Layout(**o["layout"]),
                    layers=[int(x) for x in o.get("layers", [])] if o.get("layers") else None,
                )
            )
        named = {k: Fsdp2Layout(**v) for k, v in payload.get("named_overrides", {}).items()}
        ds = payload.get("dataset_stats")
        ds_obj = DatasetStats(**ds) if ds else None
        parallel_payload = payload.get("parallel") or {}
        parallel = ParallelSpec(
            tp_degree=parallel_payload.get("tp_degree", 1),
            pp_degree=parallel_payload.get("pp_degree", 1),
            ep_degree=parallel_payload.get("ep_degree", 1),
            cp_degree=parallel_payload.get("cp_degree", 1),
            sp_enabled=parallel_payload.get("sp_enabled", False),
            tp_plan=parallel_payload.get("tp_plan", "auto"),
            pp_microbatches=parallel_payload.get("pp_microbatches", 1),
            pp_schedule=parallel_payload.get("pp_schedule", "1f1b"),
            pp_stages=parallel_payload.get("pp_stages"),
        )
        return cls(
            global_layout=gl,
            layer_overrides=ovrs,
            named_overrides=named,
            grouping=grouping,
            parallel=parallel,
            schema_version=schema_version,
            dataset_stats=ds_obj,
        )

    # -------- semantic normalization & hash --------
    def normalized(self) -> "Fsdp2Strategy":
        """Merge equivalent overrides, drop redundant ones, sort deterministically."""
        norm = copy.deepcopy(self)

        # 1) drop overrides equal to global layout
        gl_layout_dict = asdict(norm.global_layout)
        range_overrides: List[LayerOverride] = []
        list_overrides: List[LayerOverride] = []
        for o in norm.layer_overrides:
            if asdict(o.layout) == gl_layout_dict:
                continue
            has_layers = False
            if o.layers:
                uniq_layers = sorted({int(x) for x in o.layers if int(x) >= 0})
                if uniq_layers:
                    clone = copy.deepcopy(o)
                    clone.layers = uniq_layers
                    list_overrides.append(clone)
                    has_layers = True
            has_range = o.start_layer is not None and o.end_layer is not None
            if has_range:
                if o.start_layer >= o.end_layer:
                    continue
                range_overrides.append(copy.deepcopy(o))
            elif not has_layers:
                continue

        # 2) sort by range
        range_overrides.sort(key=lambda x: (x.start_layer, x.end_layer))

        # 3) merge adjacent ranges with identical layout
        merged: List[LayerOverride] = []
        for o in range_overrides:
            if merged and asdict(merged[-1].layout) == asdict(o.layout) and merged[-1].end_layer == o.start_layer:
                merged[-1].end_layer = o.end_layer
            else:
                merged.append(o)

        list_overrides.sort(key=lambda x: x.layers[0] if x.layers else -1)
        norm.layer_overrides = merged + list_overrides

        # 4) sort named overrides
        norm.named_overrides = dict(sorted(norm.named_overrides.items(), key=lambda kv: kv[0]))
        return norm

    def semantic_hash(self) -> str:
        norm = self.normalized()
        payload = norm.to_dict()
        # dataset_stats does not affect strategy semantics; exclude to avoid noisy hashes.
        payload["dataset_stats"] = None
        payload["_hash_salt"] = _SEMANTIC_HASH_SALT
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Validators & helpers
# -----------------------------
_ALLOWED_MESH = {"1D", "2D"}
_ALLOWED_SHARD = {"FULL", "HYBRID", "NO"}
_ALLOWED_PLAN = {"DIM0", "DIM1", "LARGEST"}
_ALLOWED_MP = {"bf16", "fp32"}
_SEMANTIC_HASH_SALT = "fsdp2_strategy_v3"


def _validate_layout(layout: Fsdp2Layout, context: str = "layout") -> Fsdp2Layout:
    l = copy.deepcopy(layout)
    if l.mesh_topology not in _ALLOWED_MESH:
        raise ValueError(f"{context}.mesh_topology must be in {_ALLOWED_MESH}; got {l.mesh_topology}")
    if l.sharding_strategy not in _ALLOWED_SHARD:
        raise ValueError(f"{context}.sharding_strategy must be in {_ALLOWED_SHARD}; got {l.sharding_strategy}")
    # Normalize redundant fields: mesh_topology is the actual control in FSDP2; sharding_strategy is derived.
    if l.sharding_strategy != "NO":
        l.sharding_strategy = "HYBRID" if l.mesh_topology == "2D" else "FULL"
    # Note: bool is a subclass of int in Python, so check bool before int.
    raf = l.reshard_after_forward
    if raf is None:
        l.reshard_after_forward = None
    elif isinstance(raf, str):
        low = raf.strip().lower()
        if low in {"none", "null"}:
            raf = None
        elif low in {"true", "false"}:
            raf = (low == "true")
        else:
            try:
                raf = int(low)
            except Exception as exc:
                raise ValueError(f"{context}.reshard_after_forward must be bool or int; got {raf!r}") from exc
        l.reshard_after_forward = raf

    if l.reshard_after_forward is None:
        pass
    elif isinstance(l.reshard_after_forward, bool):
        pass
    elif isinstance(l.reshard_after_forward, int):
        # Allow 0 as a permissive encoding of False (common from LLMs / config UIs).
        if l.reshard_after_forward == 0:
            l.reshard_after_forward = False
        elif l.reshard_after_forward < 2:
            raise ValueError(f"{context}.reshard_after_forward int value must be >=2; got {l.reshard_after_forward}")
    else:
        raise ValueError(
            f"{context}.reshard_after_forward must be bool or int; got {type(l.reshard_after_forward).__name__}"
        )
    if l.shard_plan not in _ALLOWED_PLAN:
        raise ValueError(f"{context}.shard_plan must be in {_ALLOWED_PLAN}; got {l.shard_plan}")
    if l.mp_policy not in _ALLOWED_MP:
        raise ValueError(f"{context}.mp_policy must be in {_ALLOWED_MP}; got {l.mp_policy}")
    l.offload_params = bool(l.offload_params)
    l.offload_pin_memory = bool(l.offload_pin_memory)
    return l


def _validate_parallel(parallel: ParallelSpec) -> ParallelSpec:
    p = copy.deepcopy(parallel) if parallel is not None else ParallelSpec()
    try:
        p.tp_degree = int(p.tp_degree)
        p.pp_degree = int(p.pp_degree)
        p.ep_degree = int(p.ep_degree)
        p.cp_degree = int(p.cp_degree)
        p.pp_microbatches = int(p.pp_microbatches)
    except Exception as exc:
        raise ValueError("parallel degrees and pp_microbatches must be integers") from exc
    if p.tp_degree < 1 or p.pp_degree < 1 or p.ep_degree < 1 or p.cp_degree < 1:
        raise ValueError("parallel degrees must be >= 1")
    if p.pp_microbatches < 1:
        raise ValueError("parallel.pp_microbatches must be >= 1")
    p.sp_enabled = bool(p.sp_enabled)
    p.tp_plan = str(p.tp_plan or "auto")
    p.pp_schedule = str(p.pp_schedule or "1f1b")
    if p.pp_stages is not None:
        stages = []
        for idx, item in enumerate(p.pp_stages):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"parallel.pp_stages[{idx}] must be a [start, end] list")
            start, end = int(item[0]), int(item[1])
            if start < 0 or end < start:
                raise ValueError(f"parallel.pp_stages[{idx}] must satisfy 0 <= start <= end")
            stages.append([start, end])
        p.pp_stages = stages
    return p


def _parse_layer_index(value: object, context: str) -> int:
    if isinstance(value, int):
        idx = value
    else:
        s = str(value).strip()
        if s.isdigit():
            idx = int(s)
        else:
            # Accept common strings like "layers.22" from LLM output.
            matches = re.findall(r"\d+", s)
            if not matches:
                raise ValueError(f"{context}.layers must contain integers; got {value!r}")
            idx = int(matches[-1])
    if idx < 0:
        raise ValueError(f"{context}.layers must contain non-negative integers; got {idx}")
    return idx


def _coerce_layers(layers: Optional[List[int]], context: str) -> Optional[List[int]]:
    if layers is None:
        return None
    try:
        normalized = sorted({_parse_layer_index(x, context) for x in layers})
    except Exception as exc:
        raise ValueError(f"{context}.layers must be a list of non-negative integers") from exc
    if not normalized:
        raise ValueError(f"{context}.layers must contain at least one non-negative integer")
    return normalized


def validate_strategy(strategy: Fsdp2Strategy, mem_limit_gb: float = 999.0) -> Fsdp2Strategy:
    """Lightweight sanity filter; does not estimate exact memory."""
    gl = _validate_layout(strategy.global_layout, "global_layout")

    grouping = copy.deepcopy(strategy.grouping) if getattr(strategy, "grouping", None) else GroupingConfig()
    if grouping.mode not in {"block", "merged"}:
        raise ValueError("grouping.mode must be 'block' or 'merged'")
    try:
        grouping.merge_factor = int(grouping.merge_factor)
    except Exception as exc:
        raise ValueError("grouping.merge_factor must be an integer") from exc
    if grouping.merge_factor < 1:
        raise ValueError("grouping.merge_factor must be >= 1")

    ovrs: List[LayerOverride] = []
    for idx, o in enumerate(strategy.layer_overrides):
        ctx = f"layer_overrides[{idx}]"
        layout = _validate_layout(o.layout, f"{ctx}.layout")
        layers_list = _coerce_layers(o.layers, ctx)
        start = end = None
        if o.start_layer is not None or o.end_layer is not None:
            if o.start_layer is None or o.end_layer is None:
                raise ValueError(f"{ctx}: start_layer and end_layer must both be provided")
            start = int(o.start_layer)
            end = int(o.end_layer)
            if start >= end:
                raise ValueError(f"{ctx}: start_layer must be < end_layer")
        if start is None and layers_list is None:
            raise ValueError(f"{ctx}: specify either (start_layer,end_layer) or a non-empty layers list")
        ovrs.append(LayerOverride(start_layer=start, end_layer=end, layout=layout, layers=layers_list))

    named = {k: _validate_layout(v, f"named_overrides['{k}']") for k, v in strategy.named_overrides.items()}
    parallel = _validate_parallel(strategy.parallel)
    ds = strategy.dataset_stats

    validated = Fsdp2Strategy(
        global_layout=gl,
        layer_overrides=ovrs,
        named_overrides=named,
        grouping=grouping,
        parallel=parallel,
        schema_version=strategy.schema_version,
        dataset_stats=ds,
    )
    # Reject "explicit default mimic" via contradictory reshard config.
    if _looks_like_default_mimic(validated):
        raise ValueError(
            "Invalid reshard configuration: do not simulate auto-reshard via overrides; use reshard_after_forward=None."
        )
    return validated.normalized()


def _looks_like_default_mimic(strategy: Fsdp2Strategy) -> bool:
    gl = strategy.global_layout
    if gl.reshard_after_forward is not False:
        return False
    if not strategy.layer_overrides:
        return False
    # Detect a full-range override that flips reshard back to True without other changes.
    for o in strategy.layer_overrides:
        if o.layout.reshard_after_forward is not True:
            continue
        same_other = (
            o.layout.mesh_topology == gl.mesh_topology
            and o.layout.shard_plan == gl.shard_plan
            and o.layout.offload_params == gl.offload_params
            and o.layout.mp_policy == gl.mp_policy
        )
        if not same_other:
            continue
        if o.start_layer == 0 and o.end_layer is not None and o.end_layer >= 10**8:
            return True
    return False


# -----------------------------
# Seed strategies
# -----------------------------
def default_strategy() -> Fsdp2Strategy:
    gl = Fsdp2Layout()
    return Fsdp2Strategy(global_layout=gl)


def sandwich_sample_strategy(num_layers: Optional[int] = None, span: int = 4) -> Fsdp2Strategy:
    """Heterogeneous seed: edges favor speed (unsharded), middle favors memory (reshard).
    - num_layers=None uses a conservative default (24) to avoid loading the model in controller.
    - span is the edge layer count (default 4).
    """
    span = int(span)
    if span < 1:
        span = 1

    n = int(num_layers) if num_layers is not None else 24
    if n < 1:
        n = 24

    gl = Fsdp2Layout(reshard_after_forward=True)
    fast = Fsdp2Layout(reshard_after_forward=False)

    head_end = min(span, n)
    tail_start = max(n - span, 0)
    ovrs = []
    if head_end > 0:
        ovrs.append(LayerOverride(start_layer=0, end_layer=head_end, layout=fast))
    # Avoid duplicate ranges when n is small.
    if tail_start < n and tail_start >= head_end:
        ovrs.append(LayerOverride(start_layer=tail_start, end_layer=n, layout=fast))
    return Fsdp2Strategy(global_layout=gl, layer_overrides=ovrs)


# Optional performance-first candidates for prompting or search.
def perf_tier_candidates() -> List[Fsdp2Strategy]:
    fast = Fsdp2Layout(reshard_after_forward=False)
    overlap = Fsdp2Layout(reshard_after_forward=False, shard_plan="DIM0")
    return [
        default_strategy(),
        sandwich_sample_strategy(),
        Fsdp2Strategy(global_layout=overlap),
    ]
