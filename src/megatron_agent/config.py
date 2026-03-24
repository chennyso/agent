from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_SEMANTIC_HASH_SALT = "megatron_strategy_v2"
_PROGRAM_HASH_SALT = "megatron_program_v1"
_DEFAULT_DENSE_MODULE_FAMILIES = ["embedding", "decoder", "loss"]
_DEFAULT_MOE_MODULE_FAMILIES = ["embedding", "decoder", "experts", "loss"]


def _stable_hash(payload: Dict[str, Any], salt: str) -> str:
    blob = json.dumps({**payload, "_hash_salt": salt}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class MegatronParallelSpec:
    tp_degree: int = 1
    pp_degree: int = 1
    vpp_degree: int = 1
    ep_degree: int = 1
    cp_degree: int = 1
    expert_tp_degree: int = 1
    sp_enabled: bool = False

    def normalized(self) -> "MegatronParallelSpec":
        norm = copy.deepcopy(self)
        norm.tp_degree = max(int(norm.tp_degree), 1)
        norm.pp_degree = max(int(norm.pp_degree), 1)
        norm.vpp_degree = max(int(norm.vpp_degree), 1)
        norm.ep_degree = max(int(norm.ep_degree), 1)
        norm.cp_degree = max(int(norm.cp_degree), 1)
        norm.expert_tp_degree = max(int(norm.expert_tp_degree), 1)
        norm.sp_enabled = bool(norm.sp_enabled)
        return norm


@dataclass
class MegatronStrategy:
    parallel: MegatronParallelSpec = field(default_factory=MegatronParallelSpec)
    micro_batch_size: int = 1
    global_batch_size: int = 8
    seq_len: int = 2048
    use_bf16: bool = True
    use_fp16: bool = False
    recompute_granularity: Optional[str] = None
    extra_args: Optional[List[str]] = None
    schema_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parallel": asdict(self.parallel),
            "micro_batch_size": int(self.micro_batch_size),
            "global_batch_size": int(self.global_batch_size),
            "seq_len": int(self.seq_len),
            "use_bf16": bool(self.use_bf16),
            "use_fp16": bool(self.use_fp16),
            "recompute_granularity": self.recompute_granularity,
            "extra_args": list(self.extra_args or []),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MegatronStrategy":
        parallel_raw = payload.get("parallel") or {}
        parallel = MegatronParallelSpec(
            tp_degree=int(parallel_raw.get("tp_degree", 1) or 1),
            pp_degree=int(parallel_raw.get("pp_degree", 1) or 1),
            vpp_degree=int(parallel_raw.get("vpp_degree", 1) or 1),
            ep_degree=int(parallel_raw.get("ep_degree", 1) or 1),
            cp_degree=int(parallel_raw.get("cp_degree", 1) or 1),
            expert_tp_degree=int(parallel_raw.get("expert_tp_degree", 1) or 1),
            sp_enabled=bool(parallel_raw.get("sp_enabled", False)),
        )
        return cls(
            parallel=parallel,
            micro_batch_size=int(payload.get("micro_batch_size", 1) or 1),
            global_batch_size=int(payload.get("global_batch_size", 8) or 1),
            seq_len=int(payload.get("seq_len", 2048) or 1),
            use_bf16=bool(payload.get("use_bf16", True)),
            use_fp16=bool(payload.get("use_fp16", False)),
            recompute_granularity=payload.get("recompute_granularity"),
            extra_args=list(payload.get("extra_args") or []),
            schema_version=int(payload.get("schema_version", 2) or 2),
        )

    def normalized(self) -> "MegatronStrategy":
        norm = copy.deepcopy(self)
        if norm.use_bf16:
            norm.use_fp16 = False
        norm.parallel = norm.parallel.normalized()
        norm.micro_batch_size = max(int(norm.micro_batch_size), 1)
        norm.global_batch_size = max(int(norm.global_batch_size), 1)
        norm.seq_len = max(int(norm.seq_len), 1)
        if norm.extra_args is not None:
            norm.extra_args = [str(x) for x in norm.extra_args if str(x).strip()]
        return norm

    def semantic_hash(self) -> str:
        return _stable_hash(self.normalized().to_dict(), _SEMANTIC_HASH_SALT)


def validate_strategy(strategy: MegatronStrategy) -> MegatronStrategy:
    s = strategy.normalized()
    if s.parallel.sp_enabled and s.parallel.tp_degree <= 1:
        raise ValueError("sequence parallel requires tp_degree > 1")
    if s.parallel.vpp_degree > 1 and s.parallel.pp_degree <= 1:
        raise ValueError("virtual pipeline requires pp_degree > 1")
    if (
        s.parallel.tp_degree
        * s.parallel.pp_degree
        * s.parallel.ep_degree
        * s.parallel.cp_degree
        * s.parallel.expert_tp_degree
        <= 0
    ):
        raise ValueError("parallel degrees must be positive")
    return s


@dataclass
class TopologyDomainSpec:
    name: str
    bandwidth_gbps: Optional[float] = None
    latency_us: Optional[float] = None
    node_local: bool = False


@dataclass
class ClusterSpec:
    target: str = "single_g5"
    nodes: List[str] = field(default_factory=lambda: ["g5"])
    gpus_per_node: int = 8
    topology_domains: List[TopologyDomainSpec] = field(
        default_factory=lambda: [
            TopologyDomainSpec(name="intra_node", bandwidth_gbps=900.0, node_local=True),
            TopologyDomainSpec(name="cross_node_ib", bandwidth_gbps=200.0, node_local=False),
        ]
    )
    device_memory_gb: Optional[int] = None
    notes: Optional[str] = None

    def normalized(self) -> "ClusterSpec":
        norm = copy.deepcopy(self)
        norm.target = str(norm.target or "single_g5")
        norm.nodes = [str(node) for node in (norm.nodes or ["g5"])]
        norm.gpus_per_node = max(int(norm.gpus_per_node), 1)
        norm.topology_domains = [copy.deepcopy(domain) for domain in (norm.topology_domains or [])]
        return norm

    @property
    def world_size(self) -> int:
        return max(len(self.nodes), 1) * max(int(self.gpus_per_node), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "nodes": list(self.nodes),
            "gpus_per_node": int(self.gpus_per_node),
            "device_memory_gb": self.device_memory_gb,
            "notes": self.notes,
            "topology_domains": [asdict(domain) for domain in self.topology_domains],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ClusterSpec":
        return cls(
            target=str(payload.get("target", "single_g5")),
            nodes=[str(node) for node in (payload.get("nodes") or ["g5"])],
            gpus_per_node=int(payload.get("gpus_per_node", 8) or 8),
            device_memory_gb=payload.get("device_memory_gb"),
            notes=payload.get("notes"),
            topology_domains=[
                TopologyDomainSpec(
                    name=str(raw.get("name")),
                    bandwidth_gbps=raw.get("bandwidth_gbps"),
                    latency_us=raw.get("latency_us"),
                    node_local=bool(raw.get("node_local", False)),
                )
                for raw in (payload.get("topology_domains") or [])
            ]
            or [
                TopologyDomainSpec(name="intra_node", bandwidth_gbps=900.0, node_local=True),
                TopologyDomainSpec(name="cross_node_ib", bandwidth_gbps=200.0, node_local=False),
            ],
        )


@dataclass
class MachineProfile:
    name: str = "consumer_single_node_g5"
    device_class: str = "consumer_gpu"
    device_memory_gb: Optional[int] = None
    interconnect_class: str = "single_node_pcie"
    communication_sensitivity: str = "medium"
    prefer_small_tp: bool = True
    prefer_pp_for_scaling: bool = True
    supports_te_path: bool = False
    notes: Optional[str] = None

    def normalized(self) -> "MachineProfile":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "consumer_single_node_g5")
        norm.device_class = str(norm.device_class or "consumer_gpu")
        if norm.device_memory_gb is not None:
            norm.device_memory_gb = max(int(norm.device_memory_gb), 1)
        norm.interconnect_class = str(norm.interconnect_class or "single_node_pcie")
        norm.communication_sensitivity = str(norm.communication_sensitivity or "medium").lower()
        norm.prefer_small_tp = bool(norm.prefer_small_tp)
        norm.prefer_pp_for_scaling = bool(norm.prefer_pp_for_scaling)
        norm.supports_te_path = bool(norm.supports_te_path)
        if norm.notes is not None:
            norm.notes = str(norm.notes).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "device_class": norm.device_class,
            "device_memory_gb": norm.device_memory_gb,
            "interconnect_class": norm.interconnect_class,
            "communication_sensitivity": norm.communication_sensitivity,
            "prefer_small_tp": bool(norm.prefer_small_tp),
            "prefer_pp_for_scaling": bool(norm.prefer_pp_for_scaling),
            "supports_te_path": bool(norm.supports_te_path),
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MachineProfile":
        return cls(
            name=str(payload.get("name", "consumer_single_node_g5")),
            device_class=str(payload.get("device_class", "consumer_gpu")),
            device_memory_gb=payload.get("device_memory_gb"),
            interconnect_class=str(payload.get("interconnect_class", "single_node_pcie")),
            communication_sensitivity=str(payload.get("communication_sensitivity", "medium")),
            prefer_small_tp=bool(payload.get("prefer_small_tp", True)),
            prefer_pp_for_scaling=bool(payload.get("prefer_pp_for_scaling", True)),
            supports_te_path=bool(payload.get("supports_te_path", False)),
            notes=payload.get("notes"),
        )


@dataclass
class BackendCaps:
    transformer_impl: str = "local"
    supports_sequence_parallel: bool = False
    supports_tp_comm_overlap: bool = False
    supports_rope_fusion: bool = False
    supports_persist_layer_norm: bool = False
    supports_dual_plane: bool = False
    supports_moe: bool = True
    supports_observability_deep: bool = True
    notes: Optional[str] = None

    def normalized(self) -> "BackendCaps":
        norm = copy.deepcopy(self)
        norm.transformer_impl = str(norm.transformer_impl or "local").strip().lower() or "local"
        norm.supports_sequence_parallel = bool(norm.supports_sequence_parallel)
        norm.supports_tp_comm_overlap = bool(norm.supports_tp_comm_overlap)
        norm.supports_rope_fusion = bool(norm.supports_rope_fusion)
        norm.supports_persist_layer_norm = bool(norm.supports_persist_layer_norm)
        norm.supports_dual_plane = bool(norm.supports_dual_plane)
        norm.supports_moe = bool(norm.supports_moe)
        norm.supports_observability_deep = bool(norm.supports_observability_deep)
        if norm.notes is not None:
            norm.notes = str(norm.notes).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "transformer_impl": norm.transformer_impl,
            "supports_sequence_parallel": bool(norm.supports_sequence_parallel),
            "supports_tp_comm_overlap": bool(norm.supports_tp_comm_overlap),
            "supports_rope_fusion": bool(norm.supports_rope_fusion),
            "supports_persist_layer_norm": bool(norm.supports_persist_layer_norm),
            "supports_dual_plane": bool(norm.supports_dual_plane),
            "supports_moe": bool(norm.supports_moe),
            "supports_observability_deep": bool(norm.supports_observability_deep),
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackendCaps":
        return cls(
            transformer_impl=str(payload.get("transformer_impl", "local")),
            supports_sequence_parallel=bool(payload.get("supports_sequence_parallel", False)),
            supports_tp_comm_overlap=bool(payload.get("supports_tp_comm_overlap", False)),
            supports_rope_fusion=bool(payload.get("supports_rope_fusion", False)),
            supports_persist_layer_norm=bool(payload.get("supports_persist_layer_norm", False)),
            supports_dual_plane=bool(payload.get("supports_dual_plane", False)),
            supports_moe=bool(payload.get("supports_moe", True)),
            supports_observability_deep=bool(payload.get("supports_observability_deep", True)),
            notes=payload.get("notes"),
        )


@dataclass
class ModelSpec:
    track: str = "dense"
    model_name: str = "qwen3_14b"
    num_layers: int = 40
    module_families: List[str] = field(default_factory=lambda: list(_DEFAULT_DENSE_MODULE_FAMILIES))
    num_experts: Optional[int] = None
    moe_layer_freq: Optional[int] = None
    mtp_layers: int = 0

    def normalized(self) -> "ModelSpec":
        norm = copy.deepcopy(self)
        norm.track = str(norm.track or "dense").lower()
        norm.model_name = str(norm.model_name or "unknown_model")
        norm.num_layers = max(int(norm.num_layers), 1)
        if not norm.module_families:
            norm.module_families = (
                list(_DEFAULT_MOE_MODULE_FAMILIES) if norm.track == "moe" else list(_DEFAULT_DENSE_MODULE_FAMILIES)
            )
        norm.module_families = [str(item) for item in norm.module_families]
        if norm.num_experts is not None:
            norm.num_experts = max(int(norm.num_experts), 1)
        if norm.moe_layer_freq is not None:
            norm.moe_layer_freq = max(int(norm.moe_layer_freq), 1)
        norm.mtp_layers = max(int(norm.mtp_layers), 0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track": self.track,
            "model_name": self.model_name,
            "num_layers": int(self.num_layers),
            "module_families": list(self.module_families),
            "num_experts": self.num_experts,
            "moe_layer_freq": self.moe_layer_freq,
            "mtp_layers": int(self.mtp_layers),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelSpec":
        return cls(
            track=str(payload.get("track", "dense")),
            model_name=str(payload.get("model_name", "qwen3_14b")),
            num_layers=int(payload.get("num_layers", 40) or 1),
            module_families=[str(item) for item in (payload.get("module_families") or [])],
            num_experts=payload.get("num_experts"),
            moe_layer_freq=payload.get("moe_layer_freq"),
            mtp_layers=int(payload.get("mtp_layers", 0) or 0),
        )


@dataclass
class PartitionStageSpec:
    decoder_layers: int = 0
    special_tokens: List[str] = field(default_factory=list)

    def normalized(self) -> "PartitionStageSpec":
        norm = copy.deepcopy(self)
        norm.decoder_layers = max(int(norm.decoder_layers), 0)
        norm.special_tokens = [str(token) for token in norm.special_tokens if str(token).strip()]
        return norm


@dataclass
class PartitionSpec:
    stages: List[PartitionStageSpec] = field(default_factory=lambda: [PartitionStageSpec(decoder_layers=40, special_tokens=["E", "L"])])

    def normalized(self) -> "PartitionSpec":
        norm = copy.deepcopy(self)
        norm.stages = [stage.normalized() for stage in (norm.stages or [])]
        if not norm.stages:
            norm.stages = [PartitionStageSpec(decoder_layers=0, special_tokens=["E", "L"])]
        return norm

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def total_decoder_layers(self) -> int:
        return sum(int(stage.decoder_layers) for stage in self.stages)

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": [asdict(stage) for stage in self.stages]}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PartitionSpec":
        return cls(
            stages=[
                PartitionStageSpec(
                    decoder_layers=int(raw.get("decoder_layers", 0) or 0),
                    special_tokens=[str(token) for token in (raw.get("special_tokens") or [])],
                )
                for raw in (payload.get("stages") or [])
            ]
        )


@dataclass
class LayoutSpec:
    stage_to_node: List[str] = field(default_factory=lambda: ["g5"])
    vpp_degree: int = 1
    pipeline_layout: Optional[str] = None
    stage_device_counts: Optional[List[int]] = None
    submesh_hints: List[Dict[str, Any]] = field(default_factory=list)

    def normalized(self) -> "LayoutSpec":
        norm = copy.deepcopy(self)
        norm.stage_to_node = [str(node) for node in (norm.stage_to_node or ["g5"])]
        norm.vpp_degree = max(int(norm.vpp_degree), 1)
        if norm.stage_device_counts is not None:
            norm.stage_device_counts = [max(int(item), 1) for item in norm.stage_device_counts]
        if norm.pipeline_layout is not None:
            norm.pipeline_layout = str(norm.pipeline_layout).strip() or None
        norm.submesh_hints = [copy.deepcopy(item) for item in (norm.submesh_hints or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_to_node": list(self.stage_to_node),
            "vpp_degree": int(self.vpp_degree),
            "pipeline_layout": self.pipeline_layout,
            "stage_device_counts": list(self.stage_device_counts) if self.stage_device_counts is not None else None,
            "submesh_hints": copy.deepcopy(self.submesh_hints),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LayoutSpec":
        return cls(
            stage_to_node=[str(node) for node in (payload.get("stage_to_node") or ["g5"])],
            vpp_degree=int(payload.get("vpp_degree", 1) or 1),
            pipeline_layout=payload.get("pipeline_layout"),
            stage_device_counts=[int(item) for item in payload.get("stage_device_counts", [])]
            if payload.get("stage_device_counts") is not None
            else None,
            submesh_hints=copy.deepcopy(payload.get("submesh_hints") or []),
        )


@dataclass
class PlaneParallelSpec:
    tp_degree: int = 1
    cp_degree: int = 1
    ep_degree: int = 1
    expert_tp_degree: int = 1

    def normalized(self) -> "PlaneParallelSpec":
        norm = copy.deepcopy(self)
        norm.tp_degree = max(int(norm.tp_degree), 1)
        norm.cp_degree = max(int(norm.cp_degree), 1)
        norm.ep_degree = max(int(norm.ep_degree), 1)
        norm.expert_tp_degree = max(int(norm.expert_tp_degree), 1)
        return norm


@dataclass
class PlaneMapSpec:
    attention: PlaneParallelSpec = field(default_factory=PlaneParallelSpec)
    moe: Optional[PlaneParallelSpec] = None
    enabled: bool = False

    def normalized(self) -> "PlaneMapSpec":
        norm = copy.deepcopy(self)
        norm.attention = norm.attention.normalized()
        norm.moe = norm.moe.normalized() if norm.moe is not None else None
        norm.enabled = bool(norm.enabled and norm.moe is not None)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention": asdict(self.attention),
            "moe": asdict(self.moe) if self.moe is not None else None,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlaneMapSpec":
        attention_raw = payload.get("attention") or {}
        moe_raw = payload.get("moe")
        return cls(
            attention=PlaneParallelSpec(
                tp_degree=int(attention_raw.get("tp_degree", 1) or 1),
                cp_degree=int(attention_raw.get("cp_degree", 1) or 1),
                ep_degree=int(attention_raw.get("ep_degree", 1) or 1),
                expert_tp_degree=int(attention_raw.get("expert_tp_degree", 1) or 1),
            ),
            moe=PlaneParallelSpec(
                tp_degree=int(moe_raw.get("tp_degree", 1) or 1),
                cp_degree=int(moe_raw.get("cp_degree", 1) or 1),
                ep_degree=int(moe_raw.get("ep_degree", 1) or 1),
                expert_tp_degree=int(moe_raw.get("expert_tp_degree", 1) or 1),
            )
            if moe_raw
            else None,
            enabled=bool(payload.get("enabled", False)),
        )


@dataclass
class ScheduleSpec:
    microbatch_group_size_per_vp_stage: Optional[int] = None
    skeleton: str = "fixed_1f1b"
    dispatch_order: str = "default"

    def normalized(self) -> "ScheduleSpec":
        norm = copy.deepcopy(self)
        if norm.microbatch_group_size_per_vp_stage is not None:
            norm.microbatch_group_size_per_vp_stage = max(int(norm.microbatch_group_size_per_vp_stage), 1)
        norm.skeleton = str(norm.skeleton or "fixed_1f1b")
        norm.dispatch_order = str(norm.dispatch_order or "default")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "microbatch_group_size_per_vp_stage": self.microbatch_group_size_per_vp_stage,
            "skeleton": self.skeleton,
            "dispatch_order": self.dispatch_order,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleSpec":
        return cls(
            microbatch_group_size_per_vp_stage=payload.get("microbatch_group_size_per_vp_stage"),
            skeleton=str(payload.get("skeleton", "fixed_1f1b")),
            dispatch_order=str(payload.get("dispatch_order", "default")),
        )


@dataclass
class ConstraintSpec:
    required_node_local_axes: List[str] = field(default_factory=list)
    requires_runtime_pg_rebuild: bool = False
    requested_heterogeneous_apipe: bool = False
    notes: Optional[str] = None

    def normalized(self) -> "ConstraintSpec":
        norm = copy.deepcopy(self)
        norm.required_node_local_axes = [str(axis) for axis in (norm.required_node_local_axes or [])]
        norm.requires_runtime_pg_rebuild = bool(norm.requires_runtime_pg_rebuild)
        norm.requested_heterogeneous_apipe = bool(norm.requested_heterogeneous_apipe)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_node_local_axes": list(self.required_node_local_axes),
            "requires_runtime_pg_rebuild": bool(self.requires_runtime_pg_rebuild),
            "requested_heterogeneous_apipe": bool(self.requested_heterogeneous_apipe),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConstraintSpec":
        return cls(
            required_node_local_axes=[str(axis) for axis in (payload.get("required_node_local_axes") or [])],
            requires_runtime_pg_rebuild=bool(payload.get("requires_runtime_pg_rebuild", False)),
            requested_heterogeneous_apipe=bool(payload.get("requested_heterogeneous_apipe", False)),
            notes=payload.get("notes"),
        )


@dataclass
class ConstraintRuleSpec:
    name: str
    enabled: bool = True
    rationale: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ConstraintRuleSpec":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "unnamed_rule")
        norm.enabled = bool(norm.enabled)
        norm.params = copy.deepcopy(norm.params or {})
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "enabled": bool(norm.enabled),
            "rationale": norm.rationale,
            "params": copy.deepcopy(norm.params),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConstraintRuleSpec":
        return cls(
            name=str(payload.get("name", "unnamed_rule")),
            enabled=bool(payload.get("enabled", True)),
            rationale=payload.get("rationale"),
            params=copy.deepcopy(payload.get("params") or {}),
        )


@dataclass
class SearchSpaceSpec:
    allow_nonuniform_partition: bool = False
    allow_single_node_pp_split: bool = False
    allow_sequence_parallel_toggle: bool = False
    allow_asymmetric_vpp: bool = False
    allow_dual_plane: bool = False
    allow_stage_aware_schedule: bool = False
    allow_subgraph_submeshes: bool = False
    allow_heterogeneous_apipe: bool = False
    max_tp_size: Optional[int] = None
    max_pp_size: Optional[int] = None
    max_ep_size: Optional[int] = None
    max_cp_size: Optional[int] = None
    max_vpp_size: Optional[int] = None
    required_node_local_axes: List[str] = field(default_factory=list)
    preferred_node_for_module: Dict[str, str] = field(default_factory=dict)
    forbidden_axes_by_node: Dict[str, List[str]] = field(default_factory=dict)
    allowed_schedule_skeletons: List[str] = field(default_factory=lambda: ["fixed_1f1b"])
    rewrite_rules: List[ConstraintRuleSpec] = field(default_factory=list)
    notes: Optional[str] = None

    def normalized(self) -> "SearchSpaceSpec":
        norm = copy.deepcopy(self)
        norm.allow_nonuniform_partition = bool(norm.allow_nonuniform_partition)
        norm.allow_single_node_pp_split = bool(norm.allow_single_node_pp_split)
        norm.allow_sequence_parallel_toggle = bool(norm.allow_sequence_parallel_toggle)
        norm.allow_asymmetric_vpp = bool(norm.allow_asymmetric_vpp)
        norm.allow_dual_plane = bool(norm.allow_dual_plane)
        norm.allow_stage_aware_schedule = bool(norm.allow_stage_aware_schedule)
        norm.allow_subgraph_submeshes = bool(norm.allow_subgraph_submeshes)
        norm.allow_heterogeneous_apipe = bool(norm.allow_heterogeneous_apipe)
        if norm.max_tp_size is not None:
            norm.max_tp_size = max(int(norm.max_tp_size), 1)
        if norm.max_pp_size is not None:
            norm.max_pp_size = max(int(norm.max_pp_size), 1)
        if norm.max_ep_size is not None:
            norm.max_ep_size = max(int(norm.max_ep_size), 1)
        if norm.max_cp_size is not None:
            norm.max_cp_size = max(int(norm.max_cp_size), 1)
        if norm.max_vpp_size is not None:
            norm.max_vpp_size = max(int(norm.max_vpp_size), 1)
        norm.required_node_local_axes = [str(axis) for axis in (norm.required_node_local_axes or [])]
        norm.preferred_node_for_module = {str(key): str(value) for key, value in (norm.preferred_node_for_module or {}).items()}
        norm.forbidden_axes_by_node = {
            str(node): [str(axis) for axis in axes]
            for node, axes in (norm.forbidden_axes_by_node or {}).items()
        }
        norm.allowed_schedule_skeletons = [str(item) for item in (norm.allowed_schedule_skeletons or ["fixed_1f1b"])]
        norm.rewrite_rules = [rule.normalized() for rule in (norm.rewrite_rules or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "allow_nonuniform_partition": bool(norm.allow_nonuniform_partition),
            "allow_single_node_pp_split": bool(norm.allow_single_node_pp_split),
            "allow_sequence_parallel_toggle": bool(norm.allow_sequence_parallel_toggle),
            "allow_asymmetric_vpp": bool(norm.allow_asymmetric_vpp),
            "allow_dual_plane": bool(norm.allow_dual_plane),
            "allow_stage_aware_schedule": bool(norm.allow_stage_aware_schedule),
            "allow_subgraph_submeshes": bool(norm.allow_subgraph_submeshes),
            "allow_heterogeneous_apipe": bool(norm.allow_heterogeneous_apipe),
            "max_tp_size": norm.max_tp_size,
            "max_pp_size": norm.max_pp_size,
            "max_ep_size": norm.max_ep_size,
            "max_cp_size": norm.max_cp_size,
            "max_vpp_size": norm.max_vpp_size,
            "required_node_local_axes": list(norm.required_node_local_axes),
            "preferred_node_for_module": copy.deepcopy(norm.preferred_node_for_module),
            "forbidden_axes_by_node": copy.deepcopy(norm.forbidden_axes_by_node),
            "allowed_schedule_skeletons": list(norm.allowed_schedule_skeletons),
            "rewrite_rules": [rule.to_dict() for rule in norm.rewrite_rules],
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SearchSpaceSpec":
        return cls(
            allow_nonuniform_partition=bool(payload.get("allow_nonuniform_partition", False)),
            allow_single_node_pp_split=bool(payload.get("allow_single_node_pp_split", False)),
            allow_sequence_parallel_toggle=bool(payload.get("allow_sequence_parallel_toggle", False)),
            allow_asymmetric_vpp=bool(payload.get("allow_asymmetric_vpp", False)),
            allow_dual_plane=bool(payload.get("allow_dual_plane", False)),
            allow_stage_aware_schedule=bool(payload.get("allow_stage_aware_schedule", False)),
            allow_subgraph_submeshes=bool(payload.get("allow_subgraph_submeshes", False)),
            allow_heterogeneous_apipe=bool(payload.get("allow_heterogeneous_apipe", False)),
            max_tp_size=payload.get("max_tp_size"),
            max_pp_size=payload.get("max_pp_size"),
            max_ep_size=payload.get("max_ep_size"),
            max_cp_size=payload.get("max_cp_size"),
            max_vpp_size=payload.get("max_vpp_size"),
            required_node_local_axes=[str(axis) for axis in (payload.get("required_node_local_axes") or [])],
            preferred_node_for_module={
                str(key): str(value) for key, value in (payload.get("preferred_node_for_module") or {}).items()
            },
            forbidden_axes_by_node={
                str(node): [str(axis) for axis in axes]
                for node, axes in (payload.get("forbidden_axes_by_node") or {}).items()
            },
            allowed_schedule_skeletons=[str(item) for item in (payload.get("allowed_schedule_skeletons") or ["fixed_1f1b"])],
            rewrite_rules=[ConstraintRuleSpec.from_dict(rule) for rule in (payload.get("rewrite_rules") or [])],
            notes=payload.get("notes"),
        )


@dataclass
class MegatronProgram:
    cluster: ClusterSpec = field(default_factory=ClusterSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    parallel: MegatronParallelSpec = field(default_factory=MegatronParallelSpec)
    partition: PartitionSpec = field(default_factory=PartitionSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    plane_map: PlaneMapSpec = field(default_factory=PlaneMapSpec)
    schedule: ScheduleSpec = field(default_factory=ScheduleSpec)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    search_space: SearchSpaceSpec = field(default_factory=SearchSpaceSpec)
    machine_profile: Optional[MachineProfile] = None
    backend_caps: Optional[BackendCaps] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def normalized(self) -> "MegatronProgram":
        norm = copy.deepcopy(self)
        norm.cluster = norm.cluster.normalized()
        norm.model = norm.model.normalized()
        norm.parallel = norm.parallel.normalized()
        norm.partition = norm.partition.normalized()
        norm.layout = norm.layout.normalized()
        norm.plane_map = norm.plane_map.normalized()
        norm.schedule = norm.schedule.normalized()
        norm.constraints = norm.constraints.normalized()
        norm.search_space = norm.search_space.normalized()
        norm.machine_profile = norm.machine_profile.normalized() if norm.machine_profile is not None else None
        norm.backend_caps = norm.backend_caps.normalized() if norm.backend_caps is not None else None
        norm.metadata = copy.deepcopy(norm.metadata or {})
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "schema_version": norm.schema_version,
            "cluster": norm.cluster.to_dict(),
            "model": norm.model.to_dict(),
            "parallel": asdict(norm.parallel),
            "partition": norm.partition.to_dict(),
            "layout": norm.layout.to_dict(),
            "plane_map": norm.plane_map.to_dict(),
            "schedule": norm.schedule.to_dict(),
            "constraints": norm.constraints.to_dict(),
            "search_space": norm.search_space.to_dict(),
            "machine_profile": norm.machine_profile.to_dict() if norm.machine_profile is not None else None,
            "backend_caps": norm.backend_caps.to_dict() if norm.backend_caps is not None else None,
            "metadata": copy.deepcopy(norm.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MegatronProgram":
        parallel_raw = payload.get("parallel") or {}
        return cls(
            cluster=ClusterSpec.from_dict(payload.get("cluster") or {}),
            model=ModelSpec.from_dict(payload.get("model") or {}),
            parallel=MegatronParallelSpec(
                tp_degree=int(parallel_raw.get("tp_degree", 1) or 1),
                pp_degree=int(parallel_raw.get("pp_degree", 1) or 1),
                vpp_degree=int(parallel_raw.get("vpp_degree", 1) or 1),
                ep_degree=int(parallel_raw.get("ep_degree", 1) or 1),
                cp_degree=int(parallel_raw.get("cp_degree", 1) or 1),
                expert_tp_degree=int(parallel_raw.get("expert_tp_degree", 1) or 1),
                sp_enabled=bool(parallel_raw.get("sp_enabled", False)),
            ),
            partition=PartitionSpec.from_dict(payload.get("partition") or {}),
            layout=LayoutSpec.from_dict(payload.get("layout") or {}),
            plane_map=PlaneMapSpec.from_dict(payload.get("plane_map") or {}),
            schedule=ScheduleSpec.from_dict(payload.get("schedule") or {}),
            constraints=ConstraintSpec.from_dict(payload.get("constraints") or {}),
            search_space=SearchSpaceSpec.from_dict(payload.get("search_space") or {}),
            machine_profile=MachineProfile.from_dict(payload.get("machine_profile") or {})
            if payload.get("machine_profile") is not None
            else None,
            backend_caps=BackendCaps.from_dict(payload.get("backend_caps") or {})
            if payload.get("backend_caps") is not None
            else None,
            metadata=copy.deepcopy(payload.get("metadata") or {}),
            schema_version=int(payload.get("schema_version", 1) or 1),
        )

    def semantic_hash(self) -> str:
        return _stable_hash(self.to_dict(), _PROGRAM_HASH_SALT)


def default_cluster_spec(target: str) -> ClusterSpec:
    if str(target) == "dual_g4_g5":
        return ClusterSpec(target="dual_g4_g5", nodes=["g4", "g5"], gpus_per_node=8)
    if str(target) == "single_g4":
        return ClusterSpec(target="single_g4", nodes=["g4"], gpus_per_node=8)
    return ClusterSpec(target="single_g5", nodes=["g5"], gpus_per_node=8)


def default_machine_profile(target: str) -> MachineProfile:
    if str(target) == "single_g4":
        return MachineProfile(
            name="consumer_single_node_4090d",
            device_class="consumer_gpu",
            device_memory_gb=24,
            interconnect_class="single_node_pcie",
            communication_sensitivity="high",
            prefer_small_tp=True,
            prefer_pp_for_scaling=True,
            supports_te_path=False,
            notes="4090D-like single-node profile with stronger communication and memory caution",
        )
    if str(target) == "dual_g4_g5":
        return MachineProfile(
            name="heterogeneous_dual_4090d_5090d",
            device_class="heterogeneous_consumer_gpu",
            device_memory_gb=24,
            interconnect_class="cross_node_heterogeneous",
            communication_sensitivity="very_high",
            prefer_small_tp=True,
            prefer_pp_for_scaling=True,
            supports_te_path=False,
            notes="heterogeneous dual-node profile with asymmetric devices and cross-node communication penalty",
        )
    return MachineProfile(
        name="consumer_single_node_5090d",
        device_class="consumer_gpu",
        device_memory_gb=32,
        interconnect_class="single_node_pcie",
        communication_sensitivity="medium",
        prefer_small_tp=True,
        prefer_pp_for_scaling=True,
        supports_te_path=True,
        notes="5090D-like single-node profile with conservative TP bias and PP-preferred scaling",
    )


def default_backend_caps(transformer_impl: str = "local") -> BackendCaps:
    impl = str(transformer_impl or "local").strip().lower() or "local"
    if impl == "transformer_engine":
        return BackendCaps(
            transformer_impl="transformer_engine",
            supports_sequence_parallel=True,
            supports_tp_comm_overlap=True,
            supports_rope_fusion=True,
            supports_persist_layer_norm=True,
            supports_dual_plane=True,
            supports_moe=True,
            supports_observability_deep=True,
            notes="Transformer Engine backend caps for the fast path",
        )
    return BackendCaps(
        transformer_impl="local",
        supports_sequence_parallel=False,
        supports_tp_comm_overlap=False,
        supports_rope_fusion=False,
        supports_persist_layer_norm=False,
        supports_dual_plane=False,
        supports_moe=True,
        supports_observability_deep=True,
        notes="Local backend caps keep compatibility guards enabled",
    )


def default_dense_program(target: str = "single_g5") -> MegatronProgram:
    cluster = default_cluster_spec(target)
    parallel = MegatronParallelSpec(tp_degree=4, pp_degree=1, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=True)
    partition = PartitionSpec(stages=[PartitionStageSpec(decoder_layers=40, special_tokens=["E", "L"])])
    layout = LayoutSpec(stage_to_node=[cluster.nodes[-1]], vpp_degree=1)
    if target == "single_g4":
        parallel = MegatronParallelSpec(tp_degree=2, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=True)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=20, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=20, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g4"], vpp_degree=1)
    if target == "dual_g4_g5":
        parallel = MegatronParallelSpec(tp_degree=2, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=True)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=20, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=20, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g5"], vpp_degree=1)
    return MegatronProgram(
        cluster=cluster,
        model=ModelSpec(track="dense", model_name="qwen3_14b", num_layers=40),
        parallel=parallel,
        partition=partition,
        layout=layout,
        plane_map=PlaneMapSpec(
            attention=PlaneParallelSpec(tp_degree=parallel.tp_degree, cp_degree=parallel.cp_degree),
            moe=None,
            enabled=False,
        ),
        schedule=ScheduleSpec(),
        constraints=ConstraintSpec(required_node_local_axes=["tp"]),
        search_space=SearchSpaceSpec(),
        machine_profile=default_machine_profile(target),
        backend_caps=default_backend_caps("local"),
        metadata={"program_kind": "baseline_dense"},
    )


def default_moe_smoke_program(target: str = "single_g5") -> MegatronProgram:
    cluster = default_cluster_spec(target)
    parallel = MegatronParallelSpec(tp_degree=2, pp_degree=1, vpp_degree=1, cp_degree=1, ep_degree=2, expert_tp_degree=1, sp_enabled=True)
    partition = PartitionSpec(stages=[PartitionStageSpec(decoder_layers=8, special_tokens=["E", "L"])])
    layout = LayoutSpec(stage_to_node=[cluster.nodes[-1]], vpp_degree=1)
    if target == "single_g4":
        parallel = MegatronParallelSpec(tp_degree=1, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=2, expert_tp_degree=1, sp_enabled=False)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=4, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=4, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g4"], vpp_degree=1)
    if target == "dual_g4_g5":
        parallel = MegatronParallelSpec(tp_degree=1, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=2, expert_tp_degree=1, sp_enabled=False)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=4, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=4, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g5"], vpp_degree=1)
    return MegatronProgram(
        cluster=cluster,
        model=ModelSpec(
            track="moe",
            model_name="moe_smoke",
            num_layers=8,
            module_families=list(_DEFAULT_MOE_MODULE_FAMILIES),
            num_experts=4,
            moe_layer_freq=2,
        ),
        parallel=parallel,
        partition=partition,
        layout=layout,
        plane_map=PlaneMapSpec(
            attention=PlaneParallelSpec(tp_degree=max(parallel.tp_degree, 1), cp_degree=parallel.cp_degree),
            moe=PlaneParallelSpec(tp_degree=1, cp_degree=1, ep_degree=parallel.ep_degree, expert_tp_degree=parallel.expert_tp_degree),
            enabled=True,
        ),
        schedule=ScheduleSpec(),
        constraints=ConstraintSpec(required_node_local_axes=["tp", "ep"]),
        search_space=SearchSpaceSpec(),
        machine_profile=default_machine_profile(target),
        backend_caps=default_backend_caps("local"),
        metadata={"program_kind": "baseline_moe_smoke"},
    )
