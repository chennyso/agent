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
_DEFAULT_VARIABLE_TIERS = {
    "apipe": "global_low_freq",
    "placement": "global_low_freq",
    "local_parallel": "local_mid_freq",
    "pipe": "runtime_high_freq",
    "morphable_pipe": "runtime_high_freq",
}


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
    template: str = "fixed_1f1b"
    dispatch_order: str = "default"

    def normalized(self) -> "ScheduleSpec":
        norm = copy.deepcopy(self)
        if norm.microbatch_group_size_per_vp_stage is not None:
            norm.microbatch_group_size_per_vp_stage = max(int(norm.microbatch_group_size_per_vp_stage), 1)
        norm.skeleton = str(norm.skeleton or "fixed_1f1b")
        norm.template = str(norm.template or norm.skeleton or "fixed_1f1b")
        norm.dispatch_order = str(norm.dispatch_order or "default")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "microbatch_group_size_per_vp_stage": self.microbatch_group_size_per_vp_stage,
            "skeleton": self.skeleton,
            "template": self.template,
            "dispatch_order": self.dispatch_order,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleSpec":
        return cls(
            microbatch_group_size_per_vp_stage=payload.get("microbatch_group_size_per_vp_stage"),
            skeleton=str(payload.get("skeleton", "fixed_1f1b")),
            template=str(payload.get("template", payload.get("skeleton", "fixed_1f1b"))),
            dispatch_order=str(payload.get("dispatch_order", "default")),
        )


@dataclass
class StageSemanticSpec:
    stage_id: int = 0
    family: str = "unspecified"
    local_dispatch_hint: Optional[str] = None
    prefer_delayed_wgrad: bool = False
    prefer_early_reload: bool = False
    prefer_checkpoint: bool = False
    prefer_offload: bool = False
    overlap_aggressiveness: str = "balanced"
    notes: Optional[str] = None

    def normalized(self) -> "StageSemanticSpec":
        norm = copy.deepcopy(self)
        norm.stage_id = max(int(norm.stage_id), 0)
        norm.family = str(norm.family or "unspecified")
        if norm.local_dispatch_hint is not None:
            norm.local_dispatch_hint = str(norm.local_dispatch_hint).strip() or None
        norm.prefer_delayed_wgrad = bool(norm.prefer_delayed_wgrad)
        norm.prefer_early_reload = bool(norm.prefer_early_reload)
        norm.prefer_checkpoint = bool(norm.prefer_checkpoint)
        norm.prefer_offload = bool(norm.prefer_offload)
        norm.overlap_aggressiveness = str(norm.overlap_aggressiveness or "balanced")
        if norm.notes is not None:
            norm.notes = str(norm.notes).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "stage_id": int(norm.stage_id),
            "family": norm.family,
            "local_dispatch_hint": norm.local_dispatch_hint,
            "prefer_delayed_wgrad": bool(norm.prefer_delayed_wgrad),
            "prefer_early_reload": bool(norm.prefer_early_reload),
            "prefer_checkpoint": bool(norm.prefer_checkpoint),
            "prefer_offload": bool(norm.prefer_offload),
            "overlap_aggressiveness": norm.overlap_aggressiveness,
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StageSemanticSpec":
        return cls(
            stage_id=int(payload.get("stage_id", 0) or 0),
            family=str(payload.get("family", "unspecified")),
            local_dispatch_hint=payload.get("local_dispatch_hint"),
            prefer_delayed_wgrad=bool(payload.get("prefer_delayed_wgrad", False)),
            prefer_early_reload=bool(payload.get("prefer_early_reload", False)),
            prefer_checkpoint=bool(payload.get("prefer_checkpoint", False)),
            prefer_offload=bool(payload.get("prefer_offload", False)),
            overlap_aggressiveness=str(payload.get("overlap_aggressiveness", "balanced")),
            notes=payload.get("notes"),
        )


@dataclass
class OverlapIntentSpec:
    enable_p2p_overlap: bool = False
    enable_grad_reduce_overlap: bool = False
    enable_param_gather_overlap: bool = False
    enable_tp_comm_overlap: bool = False
    enable_optimizer_tail_overlap: bool = False
    enable_reload_overlap: bool = False
    priority_frontier_pairs: List[str] = field(default_factory=list)
    status: str = "direct_now"
    disabled_reasons: List[str] = field(default_factory=list)

    def normalized(self) -> "OverlapIntentSpec":
        norm = copy.deepcopy(self)
        norm.enable_p2p_overlap = bool(norm.enable_p2p_overlap)
        norm.enable_grad_reduce_overlap = bool(norm.enable_grad_reduce_overlap)
        norm.enable_param_gather_overlap = bool(norm.enable_param_gather_overlap)
        norm.enable_tp_comm_overlap = bool(norm.enable_tp_comm_overlap)
        norm.enable_optimizer_tail_overlap = bool(norm.enable_optimizer_tail_overlap)
        norm.enable_reload_overlap = bool(norm.enable_reload_overlap)
        norm.priority_frontier_pairs = [str(item) for item in (norm.priority_frontier_pairs or []) if str(item).strip()]
        norm.status = str(norm.status or "direct_now")
        norm.disabled_reasons = [str(item) for item in (norm.disabled_reasons or []) if str(item).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "enable_p2p_overlap": bool(norm.enable_p2p_overlap),
            "enable_grad_reduce_overlap": bool(norm.enable_grad_reduce_overlap),
            "enable_param_gather_overlap": bool(norm.enable_param_gather_overlap),
            "enable_tp_comm_overlap": bool(norm.enable_tp_comm_overlap),
            "enable_optimizer_tail_overlap": bool(norm.enable_optimizer_tail_overlap),
            "enable_reload_overlap": bool(norm.enable_reload_overlap),
            "priority_frontier_pairs": list(norm.priority_frontier_pairs),
            "status": norm.status,
            "disabled_reasons": list(norm.disabled_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OverlapIntentSpec":
        return cls(
            enable_p2p_overlap=bool(payload.get("enable_p2p_overlap", False)),
            enable_grad_reduce_overlap=bool(payload.get("enable_grad_reduce_overlap", False)),
            enable_param_gather_overlap=bool(payload.get("enable_param_gather_overlap", False)),
            enable_tp_comm_overlap=bool(payload.get("enable_tp_comm_overlap", False)),
            enable_optimizer_tail_overlap=bool(payload.get("enable_optimizer_tail_overlap", False)),
            enable_reload_overlap=bool(payload.get("enable_reload_overlap", False)),
            priority_frontier_pairs=[str(item) for item in (payload.get("priority_frontier_pairs") or [])],
            status=str(payload.get("status", "direct_now")),
            disabled_reasons=[str(item) for item in (payload.get("disabled_reasons") or [])],
        )


@dataclass
class MemoryIntentSpec:
    checkpoint_policy: str = "default"
    recompute_policy: str = "default"
    offload_policy: str = "none"
    reload_policy: str = "default"
    prefetch_policy: str = "default"
    per_stage_policies: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "direct_now"
    notes: Optional[str] = None

    def normalized(self) -> "MemoryIntentSpec":
        norm = copy.deepcopy(self)
        norm.checkpoint_policy = str(norm.checkpoint_policy or "default")
        norm.recompute_policy = str(norm.recompute_policy or "default")
        norm.offload_policy = str(norm.offload_policy or "none")
        norm.reload_policy = str(norm.reload_policy or "default")
        norm.prefetch_policy = str(norm.prefetch_policy or "default")
        norm.per_stage_policies = [copy.deepcopy(item) for item in (norm.per_stage_policies or []) if isinstance(item, dict)]
        norm.status = str(norm.status or "direct_now")
        if norm.notes is not None:
            norm.notes = str(norm.notes).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "checkpoint_policy": norm.checkpoint_policy,
            "recompute_policy": norm.recompute_policy,
            "offload_policy": norm.offload_policy,
            "reload_policy": norm.reload_policy,
            "prefetch_policy": norm.prefetch_policy,
            "per_stage_policies": copy.deepcopy(norm.per_stage_policies),
            "status": norm.status,
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryIntentSpec":
        return cls(
            checkpoint_policy=str(payload.get("checkpoint_policy", "default")),
            recompute_policy=str(payload.get("recompute_policy", "default")),
            offload_policy=str(payload.get("offload_policy", "none")),
            reload_policy=str(payload.get("reload_policy", "default")),
            prefetch_policy=str(payload.get("prefetch_policy", "default")),
            per_stage_policies=[copy.deepcopy(item) for item in (payload.get("per_stage_policies") or [])],
            status=str(payload.get("status", "direct_now")),
            notes=payload.get("notes"),
        )


@dataclass
class PartitionOptimizationSpec:
    partition_mode: str = "uniform"
    allow_nonuniform_partition: bool = False
    stage_layer_counts: List[int] = field(default_factory=list)
    stage_local_vpp_vector: List[int] = field(default_factory=list)
    stage_local_vpp_forward_vector: List[int] = field(default_factory=list)
    stage_local_vpp_backward_vector: List[int] = field(default_factory=list)
    exposed_cost_weights: Dict[str, float] = field(default_factory=dict)
    preferred_boundary_modules: List[str] = field(default_factory=list)
    anti_boundary_modules: List[str] = field(default_factory=list)
    asymmetry_notes: List[str] = field(default_factory=list)

    def normalized(self) -> "PartitionOptimizationSpec":
        norm = copy.deepcopy(self)
        norm.partition_mode = str(norm.partition_mode or "uniform")
        norm.allow_nonuniform_partition = bool(norm.allow_nonuniform_partition)
        norm.stage_layer_counts = [max(int(item), 0) for item in (norm.stage_layer_counts or [])]
        norm.stage_local_vpp_vector = [max(int(item), 1) for item in (norm.stage_local_vpp_vector or [])]
        norm.stage_local_vpp_forward_vector = [max(int(item), 1) for item in (norm.stage_local_vpp_forward_vector or [])]
        norm.stage_local_vpp_backward_vector = [max(int(item), 1) for item in (norm.stage_local_vpp_backward_vector or [])]
        if not norm.stage_local_vpp_forward_vector and norm.stage_local_vpp_vector:
            norm.stage_local_vpp_forward_vector = list(norm.stage_local_vpp_vector)
        if not norm.stage_local_vpp_backward_vector and norm.stage_local_vpp_vector:
            norm.stage_local_vpp_backward_vector = list(norm.stage_local_vpp_vector)
        raw_weights = dict(norm.exposed_cost_weights or {})
        alpha = max(float(raw_weights.get("alpha", 0.45) or 0.45), 0.0)
        beta = max(float(raw_weights.get("beta", 0.25) or 0.25), 0.0)
        gamma = max(float(raw_weights.get("gamma", 0.20) or 0.20), 0.0)
        eta = max(float(raw_weights.get("eta", 0.10) or 0.10), 0.0)
        norm.exposed_cost_weights = {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "eta": float(eta),
        }
        norm.preferred_boundary_modules = [str(item) for item in (norm.preferred_boundary_modules or []) if str(item).strip()]
        norm.anti_boundary_modules = [str(item) for item in (norm.anti_boundary_modules or []) if str(item).strip()]
        norm.asymmetry_notes = [str(item) for item in (norm.asymmetry_notes or []) if str(item).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "partition_mode": norm.partition_mode,
            "allow_nonuniform_partition": bool(norm.allow_nonuniform_partition),
            "stage_layer_counts": list(norm.stage_layer_counts),
            "stage_local_vpp_vector": list(norm.stage_local_vpp_vector),
            "stage_local_vpp_forward_vector": list(norm.stage_local_vpp_forward_vector),
            "stage_local_vpp_backward_vector": list(norm.stage_local_vpp_backward_vector),
            "exposed_cost_weights": copy.deepcopy(norm.exposed_cost_weights),
            "preferred_boundary_modules": list(norm.preferred_boundary_modules),
            "anti_boundary_modules": list(norm.anti_boundary_modules),
            "asymmetry_notes": list(norm.asymmetry_notes),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PartitionOptimizationSpec":
        return cls(
            partition_mode=str(payload.get("partition_mode", "uniform")),
            allow_nonuniform_partition=bool(payload.get("allow_nonuniform_partition", False)),
            stage_layer_counts=[int(item) for item in (payload.get("stage_layer_counts") or [])],
            stage_local_vpp_vector=[int(item) for item in (payload.get("stage_local_vpp_vector") or [])],
            stage_local_vpp_forward_vector=[int(item) for item in (payload.get("stage_local_vpp_forward_vector") or [])],
            stage_local_vpp_backward_vector=[int(item) for item in (payload.get("stage_local_vpp_backward_vector") or [])],
            exposed_cost_weights={
                str(key): float(value)
                for key, value in dict(payload.get("exposed_cost_weights") or {}).items()
            },
            preferred_boundary_modules=[str(item) for item in (payload.get("preferred_boundary_modules") or [])],
            anti_boundary_modules=[str(item) for item in (payload.get("anti_boundary_modules") or [])],
            asymmetry_notes=[str(item) for item in (payload.get("asymmetry_notes") or [])],
        )


@dataclass
class ProgramPatchSpec:
    patch_id: str = ""
    base_program_hash: str = ""
    patch_family: str = "baseline"
    target_scope: str = "program"
    changes: Dict[str, Any] = field(default_factory=dict)
    expected_effects: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    derived_program_hash: Optional[str] = None

    def normalized(self) -> "ProgramPatchSpec":
        norm = copy.deepcopy(self)
        norm.patch_id = str(norm.patch_id or "")
        norm.base_program_hash = str(norm.base_program_hash or "")
        norm.patch_family = str(norm.patch_family or "baseline")
        norm.target_scope = str(norm.target_scope or "program")
        norm.changes = copy.deepcopy(norm.changes or {})
        norm.expected_effects = copy.deepcopy(norm.expected_effects or {})
        norm.risk_flags = [str(item) for item in (norm.risk_flags or []) if str(item).strip()]
        if norm.derived_program_hash is not None:
            norm.derived_program_hash = str(norm.derived_program_hash).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "patch_id": norm.patch_id,
            "base_program_hash": norm.base_program_hash,
            "patch_family": norm.patch_family,
            "target_scope": norm.target_scope,
            "changes": copy.deepcopy(norm.changes),
            "expected_effects": copy.deepcopy(norm.expected_effects),
            "risk_flags": list(norm.risk_flags),
            "derived_program_hash": norm.derived_program_hash,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProgramPatchSpec":
        return cls(
            patch_id=str(payload.get("patch_id") or ""),
            base_program_hash=str(payload.get("base_program_hash") or ""),
            patch_family=str(payload.get("patch_family", "baseline")),
            target_scope=str(payload.get("target_scope", "program")),
            changes=copy.deepcopy(payload.get("changes") or {}),
            expected_effects=copy.deepcopy(payload.get("expected_effects") or {}),
            risk_flags=[str(item) for item in (payload.get("risk_flags") or [])],
            derived_program_hash=payload.get("derived_program_hash"),
        )


@dataclass
class PatchProposal:
    target_bottleneck: str = ""
    patch_family: str = "baseline"
    target_scope: str = "program"
    expected_effects: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    search_priority: int = 0
    rationale: Optional[str] = None

    def normalized(self) -> "PatchProposal":
        norm = copy.deepcopy(self)
        norm.target_bottleneck = str(norm.target_bottleneck or "").strip()
        norm.patch_family = str(norm.patch_family or "baseline").strip() or "baseline"
        norm.target_scope = str(norm.target_scope or "program").strip() or "program"
        norm.expected_effects = copy.deepcopy(norm.expected_effects or {})
        norm.risk_flags = [str(item).strip() for item in (norm.risk_flags or []) if str(item).strip()]
        norm.search_priority = int(norm.search_priority or 0)
        norm.rationale = str(norm.rationale).strip() if norm.rationale is not None else None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "target_bottleneck": norm.target_bottleneck,
            "patch_family": norm.patch_family,
            "target_scope": norm.target_scope,
            "expected_effects": copy.deepcopy(norm.expected_effects),
            "risk_flags": list(norm.risk_flags),
            "search_priority": int(norm.search_priority),
            "rationale": norm.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PatchProposal":
        return cls(
            target_bottleneck=str(payload.get("target_bottleneck") or ""),
            patch_family=str(payload.get("patch_family") or "baseline"),
            target_scope=str(payload.get("target_scope") or "program"),
            expected_effects=copy.deepcopy(payload.get("expected_effects") or {}),
            risk_flags=[str(item) for item in (payload.get("risk_flags") or [])],
            search_priority=int(payload.get("search_priority", 0) or 0),
            rationale=payload.get("rationale"),
        )


@dataclass
class ScheduleActionSpec:
    action_type: str = "WAIT"
    stage_id: int = 0
    lane_id: int = 0
    microbatch_id: int = 0
    vchunk_id: int = 0
    time_slot: int = 0
    duration_hint: float = 0.0
    dependency_ids: List[str] = field(default_factory=list)
    memory_delta: float = 0.0
    stream_or_channel: Optional[str] = None
    weight_version_tag: Optional[str] = None

    def normalized(self) -> "ScheduleActionSpec":
        norm = copy.deepcopy(self)
        norm.action_type = str(norm.action_type or "WAIT").upper()
        norm.stage_id = max(int(norm.stage_id), 0)
        norm.lane_id = max(int(norm.lane_id), 0)
        norm.microbatch_id = max(int(norm.microbatch_id), 0)
        norm.vchunk_id = max(int(norm.vchunk_id), 0)
        norm.time_slot = max(int(norm.time_slot), 0)
        norm.duration_hint = max(float(norm.duration_hint or 0.0), 0.0)
        norm.dependency_ids = [str(item) for item in (norm.dependency_ids or []) if str(item).strip()]
        norm.memory_delta = float(norm.memory_delta or 0.0)
        if norm.stream_or_channel is not None:
            norm.stream_or_channel = str(norm.stream_or_channel).strip() or None
        if norm.weight_version_tag is not None:
            norm.weight_version_tag = str(norm.weight_version_tag).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "action_type": norm.action_type,
            "stage_id": int(norm.stage_id),
            "lane_id": int(norm.lane_id),
            "microbatch_id": int(norm.microbatch_id),
            "vchunk_id": int(norm.vchunk_id),
            "time_slot": int(norm.time_slot),
            "duration_hint": float(norm.duration_hint),
            "dependency_ids": list(norm.dependency_ids),
            "memory_delta": float(norm.memory_delta),
            "stream_or_channel": norm.stream_or_channel,
            "weight_version_tag": norm.weight_version_tag,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleActionSpec":
        return cls(
            action_type=str(payload.get("action_type", "WAIT")),
            stage_id=int(payload.get("stage_id", 0) or 0),
            lane_id=int(payload.get("lane_id", 0) or 0),
            microbatch_id=int(payload.get("microbatch_id", 0) or 0),
            vchunk_id=int(payload.get("vchunk_id", 0) or 0),
            time_slot=int(payload.get("time_slot", 0) or 0),
            duration_hint=float(payload.get("duration_hint", 0.0) or 0.0),
            dependency_ids=[str(item) for item in (payload.get("dependency_ids") or [])],
            memory_delta=float(payload.get("memory_delta", 0.0) or 0.0),
            stream_or_channel=payload.get("stream_or_channel"),
            weight_version_tag=payload.get("weight_version_tag"),
        )


@dataclass
class ScheduleGridSpec:
    lanes: int = 1
    time_slots: int = 0
    cells: List[Dict[str, Any]] = field(default_factory=list)
    family: str = "fixed_1f1b"
    stage_count: int = 1
    vstage_count: int = 1
    microbatch_count: int = 1
    weight_version_policy: str = "default"
    constraints: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def normalized(self) -> "ScheduleGridSpec":
        norm = copy.deepcopy(self)
        norm.lanes = max(int(norm.lanes), 1)
        norm.time_slots = max(int(norm.time_slots), 0)
        norm.cells = [copy.deepcopy(item) for item in (norm.cells or []) if isinstance(item, dict)]
        norm.family = str(norm.family or "fixed_1f1b")
        norm.stage_count = max(int(norm.stage_count), 1)
        norm.vstage_count = max(int(norm.vstage_count), 1)
        norm.microbatch_count = max(int(norm.microbatch_count), 1)
        norm.weight_version_policy = str(norm.weight_version_policy or "default")
        norm.constraints = copy.deepcopy(norm.constraints or {})
        norm.notes = [str(item) for item in (norm.notes or []) if str(item).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "lanes": int(norm.lanes),
            "time_slots": int(norm.time_slots),
            "cells": copy.deepcopy(norm.cells),
            "family": norm.family,
            "stage_count": int(norm.stage_count),
            "vstage_count": int(norm.vstage_count),
            "microbatch_count": int(norm.microbatch_count),
            "weight_version_policy": norm.weight_version_policy,
            "constraints": copy.deepcopy(norm.constraints),
            "notes": list(norm.notes),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleGridSpec":
        return cls(
            lanes=int(payload.get("lanes", 1) or 1),
            time_slots=int(payload.get("time_slots", 0) or 0),
            cells=[copy.deepcopy(item) for item in (payload.get("cells") or [])],
            family=str(payload.get("family", "fixed_1f1b")),
            stage_count=int(payload.get("stage_count", 1) or 1),
            vstage_count=int(payload.get("vstage_count", 1) or 1),
            microbatch_count=int(payload.get("microbatch_count", 1) or 1),
            weight_version_policy=str(payload.get("weight_version_policy", "default")),
            constraints=copy.deepcopy(payload.get("constraints") or {}),
            notes=[str(item) for item in (payload.get("notes") or [])],
        )


@dataclass
class ScheduleIRSpec:
    family: str = "fixed_1f1b"
    skeleton: str = "fixed_1f1b"
    microbatch_lanes: int = 1
    microbatch_group_size_per_vp_stage: Optional[int] = None
    dispatch_order: str = "default"
    warmup_policy: str = "default"
    steady_state_policy: str = "default"
    cooldown_policy: str = "default"
    weight_version_policy: str = "default"
    virtual_stage_grouping: List[int] = field(default_factory=list)
    runtime_requirements: Dict[str, Any] = field(default_factory=dict)
    stage_semantics: List[StageSemanticSpec] = field(default_factory=list)
    overlap_intents: OverlapIntentSpec = field(default_factory=OverlapIntentSpec)
    memory_intents: MemoryIntentSpec = field(default_factory=MemoryIntentSpec)
    execution_hints: Dict[str, Any] = field(default_factory=dict)
    schedule_grid: Optional[ScheduleGridSpec] = None
    derived_actions: List[ScheduleActionSpec] = field(default_factory=list)

    def normalized(self) -> "ScheduleIRSpec":
        norm = copy.deepcopy(self)
        norm.family = str(norm.family or "fixed_1f1b")
        norm.skeleton = str(norm.skeleton or norm.family or "fixed_1f1b")
        norm.microbatch_lanes = max(int(norm.microbatch_lanes), 1)
        if norm.microbatch_group_size_per_vp_stage is not None:
            norm.microbatch_group_size_per_vp_stage = max(int(norm.microbatch_group_size_per_vp_stage), 1)
        norm.dispatch_order = str(norm.dispatch_order or "default")
        norm.warmup_policy = str(norm.warmup_policy or "default")
        norm.steady_state_policy = str(norm.steady_state_policy or "default")
        norm.cooldown_policy = str(norm.cooldown_policy or "default")
        norm.weight_version_policy = str(norm.weight_version_policy or "default")
        norm.virtual_stage_grouping = [max(int(item), 1) for item in (norm.virtual_stage_grouping or [])]
        norm.runtime_requirements = copy.deepcopy(norm.runtime_requirements or {})
        norm.stage_semantics = [item.normalized() for item in (norm.stage_semantics or [])]
        norm.overlap_intents = norm.overlap_intents.normalized()
        norm.memory_intents = norm.memory_intents.normalized()
        norm.execution_hints = copy.deepcopy(norm.execution_hints or {})
        norm.schedule_grid = norm.schedule_grid.normalized() if norm.schedule_grid is not None else None
        norm.derived_actions = [item.normalized() for item in (norm.derived_actions or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "family": norm.family,
            "skeleton": norm.skeleton,
            "microbatch_lanes": int(norm.microbatch_lanes),
            "microbatch_group_size_per_vp_stage": norm.microbatch_group_size_per_vp_stage,
            "dispatch_order": norm.dispatch_order,
            "warmup_policy": norm.warmup_policy,
            "steady_state_policy": norm.steady_state_policy,
            "cooldown_policy": norm.cooldown_policy,
            "weight_version_policy": norm.weight_version_policy,
            "virtual_stage_grouping": list(norm.virtual_stage_grouping),
            "runtime_requirements": copy.deepcopy(norm.runtime_requirements),
            "stage_semantics": [item.to_dict() for item in norm.stage_semantics],
            "overlap_intents": norm.overlap_intents.to_dict(),
            "memory_intents": norm.memory_intents.to_dict(),
            "execution_hints": copy.deepcopy(norm.execution_hints),
            "schedule_grid": norm.schedule_grid.to_dict() if norm.schedule_grid is not None else None,
            "derived_actions": [item.to_dict() for item in norm.derived_actions],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleIRSpec":
        return cls(
            family=str(payload.get("family", "fixed_1f1b")),
            skeleton=str(payload.get("skeleton", "fixed_1f1b")),
            microbatch_lanes=int(payload.get("microbatch_lanes", 1) or 1),
            microbatch_group_size_per_vp_stage=payload.get("microbatch_group_size_per_vp_stage"),
            dispatch_order=str(payload.get("dispatch_order", "default")),
            warmup_policy=str(payload.get("warmup_policy", "default")),
            steady_state_policy=str(payload.get("steady_state_policy", "default")),
            cooldown_policy=str(payload.get("cooldown_policy", "default")),
            weight_version_policy=str(payload.get("weight_version_policy", "default")),
            virtual_stage_grouping=[int(item) for item in (payload.get("virtual_stage_grouping") or [])],
            runtime_requirements=copy.deepcopy(payload.get("runtime_requirements") or {}),
            stage_semantics=[StageSemanticSpec.from_dict(item) for item in (payload.get("stage_semantics") or [])],
            overlap_intents=OverlapIntentSpec.from_dict(payload.get("overlap_intents") or {}),
            memory_intents=MemoryIntentSpec.from_dict(payload.get("memory_intents") or {}),
            execution_hints=copy.deepcopy(payload.get("execution_hints") or {}),
            schedule_grid=ScheduleGridSpec.from_dict(payload.get("schedule_grid") or {})
            if payload.get("schedule_grid") is not None
            else None,
            derived_actions=[ScheduleActionSpec.from_dict(item) for item in (payload.get("derived_actions") or [])],
        )


@dataclass
class LayerGroupSpec:
    group_id: str = ""
    stage_id: int = 0
    layer_range: List[int] = field(default_factory=list)
    module_family: str = "decoder"
    fwd_time_ms: float = 0.0
    bwd_input_time_ms: float = 0.0
    bwd_weight_time_ms: float = 0.0
    activation_size_mb: float = 0.0
    parameter_size_mb: float = 0.0
    optimizer_state_size_mb: float = 0.0
    offload_cost_ms: float = 0.0
    reload_cost_ms: float = 0.0
    comm_boundary_cost_ms: float = 0.0

    def normalized(self) -> "LayerGroupSpec":
        norm = copy.deepcopy(self)
        norm.group_id = str(norm.group_id or "")
        norm.stage_id = max(int(norm.stage_id), 0)
        raw_range = list(norm.layer_range or [])
        if len(raw_range) >= 2:
            start = max(int(raw_range[0]), 0)
            end = max(int(raw_range[1]), start)
            norm.layer_range = [start, end]
        elif len(raw_range) == 1:
            start = max(int(raw_range[0]), 0)
            norm.layer_range = [start, start]
        else:
            norm.layer_range = [0, 0]
        norm.module_family = str(norm.module_family or "decoder")
        norm.fwd_time_ms = max(float(norm.fwd_time_ms or 0.0), 0.0)
        norm.bwd_input_time_ms = max(float(norm.bwd_input_time_ms or 0.0), 0.0)
        norm.bwd_weight_time_ms = max(float(norm.bwd_weight_time_ms or 0.0), 0.0)
        norm.activation_size_mb = max(float(norm.activation_size_mb or 0.0), 0.0)
        norm.parameter_size_mb = max(float(norm.parameter_size_mb or 0.0), 0.0)
        norm.optimizer_state_size_mb = max(float(norm.optimizer_state_size_mb or 0.0), 0.0)
        norm.offload_cost_ms = max(float(norm.offload_cost_ms or 0.0), 0.0)
        norm.reload_cost_ms = max(float(norm.reload_cost_ms or 0.0), 0.0)
        norm.comm_boundary_cost_ms = max(float(norm.comm_boundary_cost_ms or 0.0), 0.0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "group_id": norm.group_id,
            "stage_id": int(norm.stage_id),
            "layer_range": list(norm.layer_range),
            "module_family": norm.module_family,
            "fwd_time_ms": float(norm.fwd_time_ms),
            "bwd_input_time_ms": float(norm.bwd_input_time_ms),
            "bwd_weight_time_ms": float(norm.bwd_weight_time_ms),
            "activation_size_mb": float(norm.activation_size_mb),
            "parameter_size_mb": float(norm.parameter_size_mb),
            "optimizer_state_size_mb": float(norm.optimizer_state_size_mb),
            "offload_cost_ms": float(norm.offload_cost_ms),
            "reload_cost_ms": float(norm.reload_cost_ms),
            "comm_boundary_cost_ms": float(norm.comm_boundary_cost_ms),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LayerGroupSpec":
        return cls(
            group_id=str(payload.get("group_id") or ""),
            stage_id=int(payload.get("stage_id", 0) or 0),
            layer_range=[int(item) for item in (payload.get("layer_range") or [])],
            module_family=str(payload.get("module_family", "decoder")),
            fwd_time_ms=float(payload.get("fwd_time_ms", 0.0) or 0.0),
            bwd_input_time_ms=float(payload.get("bwd_input_time_ms", 0.0) or 0.0),
            bwd_weight_time_ms=float(payload.get("bwd_weight_time_ms", 0.0) or 0.0),
            activation_size_mb=float(payload.get("activation_size_mb", 0.0) or 0.0),
            parameter_size_mb=float(payload.get("parameter_size_mb", 0.0) or 0.0),
            optimizer_state_size_mb=float(payload.get("optimizer_state_size_mb", 0.0) or 0.0),
            offload_cost_ms=float(payload.get("offload_cost_ms", 0.0) or 0.0),
            reload_cost_ms=float(payload.get("reload_cost_ms", 0.0) or 0.0),
            comm_boundary_cost_ms=float(payload.get("comm_boundary_cost_ms", 0.0) or 0.0),
        )


@dataclass
class StateObjectSpec:
    state_id: str = ""
    state_type: str = "activation"
    owner_stage: int = 0
    owner_layer_group: str = ""
    size_mb: float = 0.0
    offloadable: bool = False
    prefetchable: bool = False

    def normalized(self) -> "StateObjectSpec":
        norm = copy.deepcopy(self)
        norm.state_id = str(norm.state_id or "")
        norm.state_type = str(norm.state_type or "activation")
        norm.owner_stage = max(int(norm.owner_stage), 0)
        norm.owner_layer_group = str(norm.owner_layer_group or "")
        norm.size_mb = max(float(norm.size_mb or 0.0), 0.0)
        norm.offloadable = bool(norm.offloadable)
        norm.prefetchable = bool(norm.prefetchable)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "state_id": norm.state_id,
            "state_type": norm.state_type,
            "owner_stage": int(norm.owner_stage),
            "owner_layer_group": norm.owner_layer_group,
            "size_mb": float(norm.size_mb),
            "offloadable": bool(norm.offloadable),
            "prefetchable": bool(norm.prefetchable),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StateObjectSpec":
        return cls(
            state_id=str(payload.get("state_id") or ""),
            state_type=str(payload.get("state_type", "activation")),
            owner_stage=int(payload.get("owner_stage", 0) or 0),
            owner_layer_group=str(payload.get("owner_layer_group") or ""),
            size_mb=float(payload.get("size_mb", 0.0) or 0.0),
            offloadable=bool(payload.get("offloadable", False)),
            prefetchable=bool(payload.get("prefetchable", False)),
        )


@dataclass
class StatePlacementSpec:
    state_id: str = ""
    placement: str = "hbm"
    ready_for_use: bool = True
    valid_from_slot: int = 0
    valid_until_slot: Optional[int] = None

    def normalized(self) -> "StatePlacementSpec":
        norm = copy.deepcopy(self)
        norm.state_id = str(norm.state_id or "")
        norm.placement = str(norm.placement or "hbm")
        norm.ready_for_use = bool(norm.ready_for_use)
        norm.valid_from_slot = max(int(norm.valid_from_slot or 0), 0)
        if norm.valid_until_slot is not None:
            norm.valid_until_slot = max(int(norm.valid_until_slot), norm.valid_from_slot)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "state_id": norm.state_id,
            "placement": norm.placement,
            "ready_for_use": bool(norm.ready_for_use),
            "valid_from_slot": int(norm.valid_from_slot),
            "valid_until_slot": norm.valid_until_slot,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StatePlacementSpec":
        return cls(
            state_id=str(payload.get("state_id") or ""),
            placement=str(payload.get("placement", "hbm")),
            ready_for_use=bool(payload.get("ready_for_use", True)),
            valid_from_slot=int(payload.get("valid_from_slot", 0) or 0),
            valid_until_slot=payload.get("valid_until_slot"),
        )


@dataclass
class ScheduleNodeSpec:
    node_id: str = ""
    node_type: str = "wait"
    stage_id: int = 0
    microbatch_id: int = 0
    layer_group_id: str = ""
    lane_id: int = 0
    chunk_id: int = 0
    duration_hint_ms: float = 0.0
    state_refs: List[str] = field(default_factory=list)
    resource_class: str = "compute"

    def normalized(self) -> "ScheduleNodeSpec":
        norm = copy.deepcopy(self)
        norm.node_id = str(norm.node_id or "")
        norm.node_type = str(norm.node_type or "wait")
        norm.stage_id = max(int(norm.stage_id), 0)
        norm.microbatch_id = max(int(norm.microbatch_id), 0)
        norm.layer_group_id = str(norm.layer_group_id or "")
        norm.lane_id = max(int(norm.lane_id), 0)
        norm.chunk_id = max(int(norm.chunk_id), 0)
        norm.duration_hint_ms = max(float(norm.duration_hint_ms or 0.0), 0.0)
        norm.state_refs = [str(item) for item in (norm.state_refs or []) if str(item).strip()]
        norm.resource_class = str(norm.resource_class or "compute")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "node_id": norm.node_id,
            "node_type": norm.node_type,
            "stage_id": int(norm.stage_id),
            "microbatch_id": int(norm.microbatch_id),
            "layer_group_id": norm.layer_group_id,
            "lane_id": int(norm.lane_id),
            "chunk_id": int(norm.chunk_id),
            "duration_hint_ms": float(norm.duration_hint_ms),
            "state_refs": list(norm.state_refs),
            "resource_class": norm.resource_class,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleNodeSpec":
        return cls(
            node_id=str(payload.get("node_id") or ""),
            node_type=str(payload.get("node_type", "wait")),
            stage_id=int(payload.get("stage_id", 0) or 0),
            microbatch_id=int(payload.get("microbatch_id", 0) or 0),
            layer_group_id=str(payload.get("layer_group_id") or ""),
            lane_id=int(payload.get("lane_id", 0) or 0),
            chunk_id=int(payload.get("chunk_id", 0) or 0),
            duration_hint_ms=float(payload.get("duration_hint_ms", 0.0) or 0.0),
            state_refs=[str(item) for item in (payload.get("state_refs") or [])],
            resource_class=str(payload.get("resource_class", "compute")),
        )


@dataclass
class ScheduleEdgeSpec:
    src: str = ""
    dst: str = ""
    edge_type: str = "data_dep"
    required: bool = True
    slack_hint_ms: float = 0.0

    def normalized(self) -> "ScheduleEdgeSpec":
        norm = copy.deepcopy(self)
        norm.src = str(norm.src or "")
        norm.dst = str(norm.dst or "")
        norm.edge_type = str(norm.edge_type or "data_dep")
        norm.required = bool(norm.required)
        norm.slack_hint_ms = max(float(norm.slack_hint_ms or 0.0), 0.0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "src": norm.src,
            "dst": norm.dst,
            "edge_type": norm.edge_type,
            "required": bool(norm.required),
            "slack_hint_ms": float(norm.slack_hint_ms),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScheduleEdgeSpec":
        return cls(
            src=str(payload.get("src") or ""),
            dst=str(payload.get("dst") or ""),
            edge_type=str(payload.get("edge_type", "data_dep")),
            required=bool(payload.get("required", True)),
            slack_hint_ms=float(payload.get("slack_hint_ms", 0.0) or 0.0),
        )


@dataclass
class StatePlanSpec:
    objects: List[StateObjectSpec] = field(default_factory=list)
    placements: List[StatePlacementSpec] = field(default_factory=list)
    offload_budget_mb: float = 0.0
    reload_prefetch_window: int = 1

    def normalized(self) -> "StatePlanSpec":
        norm = copy.deepcopy(self)
        norm.objects = [item.normalized() for item in (norm.objects or [])]
        norm.placements = [item.normalized() for item in (norm.placements or [])]
        norm.offload_budget_mb = max(float(norm.offload_budget_mb or 0.0), 0.0)
        norm.reload_prefetch_window = max(int(norm.reload_prefetch_window or 1), 0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "objects": [item.to_dict() for item in norm.objects],
            "placements": [item.to_dict() for item in norm.placements],
            "offload_budget_mb": float(norm.offload_budget_mb),
            "reload_prefetch_window": int(norm.reload_prefetch_window),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StatePlanSpec":
        return cls(
            objects=[StateObjectSpec.from_dict(item) for item in (payload.get("objects") or [])],
            placements=[StatePlacementSpec.from_dict(item) for item in (payload.get("placements") or [])],
            offload_budget_mb=float(payload.get("offload_budget_mb", 0.0) or 0.0),
            reload_prefetch_window=int(payload.get("reload_prefetch_window", 1) or 1),
        )


@dataclass
class VPPFlowVirtualChunkSpec:
    chunk_id: str
    stage_id: int
    local_vchunk_id: int
    layer_group_ids: List[str] = field(default_factory=list)
    device_group: str = ""
    lifecycle_unit: str = "activation"
    compute_ms: float = 0.0
    activation_mb: float = 0.0
    boundary_comm_ms: float = 0.0
    reload_cost_ms: float = 0.0
    offload_cost_ms: float = 0.0
    memory_pressure: float = 0.0
    priority: int = 0

    def normalized(self) -> "VPPFlowVirtualChunkSpec":
        norm = copy.deepcopy(self)
        norm.chunk_id = str(norm.chunk_id or f"s{int(norm.stage_id)}v{int(norm.local_vchunk_id)}")
        norm.stage_id = max(int(norm.stage_id or 0), 0)
        norm.local_vchunk_id = max(int(norm.local_vchunk_id or 0), 0)
        norm.layer_group_ids = [str(item) for item in (norm.layer_group_ids or []) if str(item).strip()]
        norm.device_group = str(norm.device_group or f"pp{norm.stage_id}")
        norm.lifecycle_unit = str(norm.lifecycle_unit or "activation")
        norm.compute_ms = max(float(norm.compute_ms or 0.0), 0.0)
        norm.activation_mb = max(float(norm.activation_mb or 0.0), 0.0)
        norm.boundary_comm_ms = max(float(norm.boundary_comm_ms or 0.0), 0.0)
        norm.reload_cost_ms = max(float(norm.reload_cost_ms or 0.0), 0.0)
        norm.offload_cost_ms = max(float(norm.offload_cost_ms or 0.0), 0.0)
        norm.memory_pressure = max(float(norm.memory_pressure or 0.0), 0.0)
        norm.priority = max(int(norm.priority or 0), 0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "chunk_id": norm.chunk_id,
            "stage_id": int(norm.stage_id),
            "local_vchunk_id": int(norm.local_vchunk_id),
            "layer_group_ids": list(norm.layer_group_ids),
            "device_group": norm.device_group,
            "lifecycle_unit": norm.lifecycle_unit,
            "compute_ms": float(norm.compute_ms),
            "activation_mb": float(norm.activation_mb),
            "boundary_comm_ms": float(norm.boundary_comm_ms),
            "reload_cost_ms": float(norm.reload_cost_ms),
            "offload_cost_ms": float(norm.offload_cost_ms),
            "memory_pressure": float(norm.memory_pressure),
            "priority": int(norm.priority),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VPPFlowVirtualChunkSpec":
        return cls(
            chunk_id=str(payload.get("chunk_id") or ""),
            stage_id=int(payload.get("stage_id", 0) or 0),
            local_vchunk_id=int(payload.get("local_vchunk_id", 0) or 0),
            layer_group_ids=[str(item) for item in (payload.get("layer_group_ids") or [])],
            device_group=str(payload.get("device_group") or ""),
            lifecycle_unit=str(payload.get("lifecycle_unit") or "activation"),
            compute_ms=float(payload.get("compute_ms", 0.0) or 0.0),
            activation_mb=float(payload.get("activation_mb", 0.0) or 0.0),
            boundary_comm_ms=float(payload.get("boundary_comm_ms", 0.0) or 0.0),
            reload_cost_ms=float(payload.get("reload_cost_ms", 0.0) or 0.0),
            offload_cost_ms=float(payload.get("offload_cost_ms", 0.0) or 0.0),
            memory_pressure=float(payload.get("memory_pressure", 0.0) or 0.0),
            priority=int(payload.get("priority", 0) or 0),
        )


@dataclass
class VPPFlowActivationPolicySpec:
    stage_id: int
    chunk_id: str
    microbatch_class: str = "stable"
    phase: str = "steady"
    family: str = "activation"
    policy: str = "resident"
    prefetch_distance: int = 1
    reload_deadline_slot: int = 0
    forbid_tail_reload: bool = False
    rationale: str = ""

    def normalized(self) -> "VPPFlowActivationPolicySpec":
        norm = copy.deepcopy(self)
        norm.stage_id = max(int(norm.stage_id or 0), 0)
        norm.chunk_id = str(norm.chunk_id or f"s{norm.stage_id}v0")
        norm.microbatch_class = str(norm.microbatch_class or "stable").strip().lower() or "stable"
        if norm.microbatch_class not in {"stable", "edge"}:
            norm.microbatch_class = "stable"
        norm.phase = str(norm.phase or "steady").strip().lower() or "steady"
        if norm.phase not in {"warmup", "steady", "cooldown"}:
            norm.phase = "steady"
        norm.family = str(norm.family or "activation").strip().lower() or "activation"
        norm.policy = str(norm.policy or "resident").strip().lower() or "resident"
        if norm.policy not in {"resident", "offload", "reload", "recompute"}:
            norm.policy = "resident"
        norm.prefetch_distance = max(int(norm.prefetch_distance or 0), 0)
        norm.reload_deadline_slot = max(int(norm.reload_deadline_slot or 0), 0)
        norm.forbid_tail_reload = bool(norm.forbid_tail_reload)
        norm.rationale = str(norm.rationale or "")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "stage_id": int(norm.stage_id),
            "chunk_id": norm.chunk_id,
            "microbatch_class": norm.microbatch_class,
            "phase": norm.phase,
            "family": norm.family,
            "policy": norm.policy,
            "prefetch_distance": int(norm.prefetch_distance),
            "reload_deadline_slot": int(norm.reload_deadline_slot),
            "forbid_tail_reload": bool(norm.forbid_tail_reload),
            "rationale": norm.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VPPFlowActivationPolicySpec":
        return cls(
            stage_id=int(payload.get("stage_id", 0) or 0),
            chunk_id=str(payload.get("chunk_id") or ""),
            microbatch_class=str(payload.get("microbatch_class") or "stable"),
            phase=str(payload.get("phase") or "steady"),
            family=str(payload.get("family") or "activation"),
            policy=str(payload.get("policy") or "resident"),
            prefetch_distance=int(payload.get("prefetch_distance", 1) or 1),
            reload_deadline_slot=int(payload.get("reload_deadline_slot", 0) or 0),
            forbid_tail_reload=bool(payload.get("forbid_tail_reload", False)),
            rationale=str(payload.get("rationale") or ""),
        )


@dataclass
class VPPFlowCreditPolicySpec:
    resource: str = "h2d"
    capacity: int = 1
    priority_order: List[str] = field(default_factory=list)
    watermark_policy: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def normalized(self) -> "VPPFlowCreditPolicySpec":
        norm = copy.deepcopy(self)
        norm.resource = str(norm.resource or "h2d").strip().lower() or "h2d"
        norm.capacity = max(int(norm.capacity or 1), 1)
        default_order = [
            "near_deadline_reload",
            "cross_node_boundary_send",
            "normal_send",
            "normal_reload",
            "d2h_offload",
        ]
        norm.priority_order = [
            str(item).strip()
            for item in (norm.priority_order or default_order)
            if str(item).strip()
        ] or default_order
        norm.watermark_policy = copy.deepcopy(norm.watermark_policy or {})
        norm.notes = str(norm.notes or "")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "resource": norm.resource,
            "capacity": int(norm.capacity),
            "priority_order": list(norm.priority_order),
            "watermark_policy": copy.deepcopy(norm.watermark_policy),
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VPPFlowCreditPolicySpec":
        return cls(
            resource=str(payload.get("resource") or "h2d"),
            capacity=int(payload.get("capacity", 1) or 1),
            priority_order=[str(item) for item in (payload.get("priority_order") or [])],
            watermark_policy=copy.deepcopy(payload.get("watermark_policy") or {}),
            notes=str(payload.get("notes") or ""),
        )


@dataclass
class VPPFlowExposedCostSpec:
    terms: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    total_exposed_ms: float = 0.0
    dominant_term: str = ""

    def normalized(self) -> "VPPFlowExposedCostSpec":
        norm = copy.deepcopy(self)
        norm.terms = {str(key): max(float(value or 0.0), 0.0) for key, value in (norm.terms or {}).items()}
        default_weights = {
            "stage_exposed_ms": 1.0,
            "comm_exposed_ms": 1.0,
            "reload_stall_ms": 1.0,
            "copy_stall_ms": 0.6,
            "uncovered_bubble_ms": 0.8,
            "optimizer_tail_ms": 0.7,
            "straggler_penalty_ms": 0.8,
        }
        norm.weights = {
            str(key): max(float(value or 0.0), 0.0)
            for key, value in ({**default_weights, **(norm.weights or {})}).items()
        }
        if norm.total_exposed_ms <= 0.0:
            norm.total_exposed_ms = sum(
                float(norm.terms.get(key, 0.0)) * float(norm.weights.get(key, 1.0))
                for key in norm.terms
            )
        norm.total_exposed_ms = max(float(norm.total_exposed_ms or 0.0), 0.0)
        if not norm.dominant_term and norm.terms:
            norm.dominant_term = max(norm.terms, key=lambda key: float(norm.terms.get(key, 0.0)))
        norm.dominant_term = str(norm.dominant_term or "")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "terms": {str(key): float(value) for key, value in norm.terms.items()},
            "weights": {str(key): float(value) for key, value in norm.weights.items()},
            "total_exposed_ms": float(norm.total_exposed_ms),
            "dominant_term": norm.dominant_term,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VPPFlowExposedCostSpec":
        return cls(
            terms={str(key): float(value or 0.0) for key, value in (payload.get("terms") or {}).items()},
            weights={str(key): float(value or 0.0) for key, value in (payload.get("weights") or {}).items()},
            total_exposed_ms=float(payload.get("total_exposed_ms", 0.0) or 0.0),
            dominant_term=str(payload.get("dominant_term") or ""),
        )


@dataclass
class VPPFlowPolicySpec:
    enabled: bool = True
    policy_version: str = "vpp_flow_v1"
    objective: str = "minimize_exposed_critical_path"
    virtual_chunks: List[VPPFlowVirtualChunkSpec] = field(default_factory=list)
    activation_lifecycle: List[VPPFlowActivationPolicySpec] = field(default_factory=list)
    flow_credit_policy: List[VPPFlowCreditPolicySpec] = field(default_factory=list)
    exposed_cost: VPPFlowExposedCostSpec = field(default_factory=VPPFlowExposedCostSpec)
    constraints: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def normalized(self) -> "VPPFlowPolicySpec":
        norm = copy.deepcopy(self)
        norm.enabled = bool(norm.enabled)
        norm.policy_version = str(norm.policy_version or "vpp_flow_v1")
        norm.objective = str(norm.objective or "minimize_exposed_critical_path")
        norm.virtual_chunks = [item.normalized() for item in (norm.virtual_chunks or [])]
        norm.activation_lifecycle = [item.normalized() for item in (norm.activation_lifecycle or [])]
        norm.flow_credit_policy = [item.normalized() for item in (norm.flow_credit_policy or [])]
        norm.exposed_cost = norm.exposed_cost.normalized()
        norm.constraints = copy.deepcopy(norm.constraints or {})
        norm.notes = [str(item) for item in (norm.notes or []) if str(item).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "enabled": bool(norm.enabled),
            "policy_version": norm.policy_version,
            "objective": norm.objective,
            "virtual_chunks": [item.to_dict() for item in norm.virtual_chunks],
            "activation_lifecycle": [item.to_dict() for item in norm.activation_lifecycle],
            "flow_credit_policy": [item.to_dict() for item in norm.flow_credit_policy],
            "exposed_cost": norm.exposed_cost.to_dict(),
            "constraints": copy.deepcopy(norm.constraints),
            "notes": list(norm.notes),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VPPFlowPolicySpec":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            policy_version=str(payload.get("policy_version") or "vpp_flow_v1"),
            objective=str(payload.get("objective") or "minimize_exposed_critical_path"),
            virtual_chunks=[VPPFlowVirtualChunkSpec.from_dict(item) for item in (payload.get("virtual_chunks") or [])],
            activation_lifecycle=[
                VPPFlowActivationPolicySpec.from_dict(item) for item in (payload.get("activation_lifecycle") or [])
            ],
            flow_credit_policy=[
                VPPFlowCreditPolicySpec.from_dict(item) for item in (payload.get("flow_credit_policy") or [])
            ],
            exposed_cost=VPPFlowExposedCostSpec.from_dict(payload.get("exposed_cost") or {}),
            constraints=copy.deepcopy(payload.get("constraints") or {}),
            notes=[str(item) for item in (payload.get("notes") or [])],
        )


@dataclass
class TelemetryBudgetSpec:
    level: str = "summary"
    max_trace_mb: int = 128
    max_events_per_rank: int = 20000
    sampled_windows: int = 2
    emit_compare_svg: bool = False

    def normalized(self) -> "TelemetryBudgetSpec":
        norm = copy.deepcopy(self)
        norm.level = str(norm.level or "summary").strip().lower() or "summary"
        if norm.level not in {"summary", "aggregated_grid", "full_debug"}:
            norm.level = "summary"
        norm.max_trace_mb = max(int(norm.max_trace_mb or 128), 1)
        norm.max_events_per_rank = max(int(norm.max_events_per_rank or 20000), 0)
        norm.sampled_windows = max(int(norm.sampled_windows or 2), 0)
        norm.emit_compare_svg = bool(norm.emit_compare_svg)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "level": norm.level,
            "max_trace_mb": int(norm.max_trace_mb),
            "max_events_per_rank": int(norm.max_events_per_rank),
            "sampled_windows": int(norm.sampled_windows),
            "emit_compare_svg": bool(norm.emit_compare_svg),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TelemetryBudgetSpec":
        return cls(
            level=str(payload.get("level", "summary")),
            max_trace_mb=int(payload.get("max_trace_mb", 128) or 128),
            max_events_per_rank=int(payload.get("max_events_per_rank", 20000) or 20000),
            sampled_windows=int(payload.get("sampled_windows", 2) or 2),
            emit_compare_svg=bool(payload.get("emit_compare_svg", False)),
        )


@dataclass
class WindowReconfigSpec:
    window_steps: int = 4
    allowed_patch_categories: List[str] = field(default_factory=lambda: ["schedule", "memory", "overlap", "partition"])
    rollback_guard_steps: int = 1
    promotion_threshold: float = 0.03
    demotion_threshold: float = 0.05

    def normalized(self) -> "WindowReconfigSpec":
        norm = copy.deepcopy(self)
        norm.window_steps = max(int(norm.window_steps or 4), 1)
        normalized_categories: List[str] = []
        for item in (norm.allowed_patch_categories or []):
            token = str(item).strip().lower()
            if token in {"partition", "schedule", "memory", "overlap"} and token not in normalized_categories:
                normalized_categories.append(token)
        norm.allowed_patch_categories = normalized_categories or ["schedule", "memory", "overlap", "partition"]
        norm.rollback_guard_steps = max(int(norm.rollback_guard_steps or 1), 0)
        norm.promotion_threshold = max(float(norm.promotion_threshold or 0.03), 0.0)
        norm.demotion_threshold = max(float(norm.demotion_threshold or 0.05), 0.0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "window_steps": int(norm.window_steps),
            "allowed_patch_categories": list(norm.allowed_patch_categories),
            "rollback_guard_steps": int(norm.rollback_guard_steps),
            "promotion_threshold": float(norm.promotion_threshold),
            "demotion_threshold": float(norm.demotion_threshold),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WindowReconfigSpec":
        return cls(
            window_steps=int(payload.get("window_steps", 4) or 4),
            allowed_patch_categories=[str(item) for item in (payload.get("allowed_patch_categories") or [])],
            rollback_guard_steps=int(payload.get("rollback_guard_steps", 1) or 1),
            promotion_threshold=float(payload.get("promotion_threshold", 0.03) or 0.03),
            demotion_threshold=float(payload.get("demotion_threshold", 0.05) or 0.05),
        )


@dataclass
class GlobalStrategyPlanSpec:
    primary_parallel_mode: str = "pp_vpp"
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    vpp_degree: int = 1
    stage_count: int = 1
    stage_boundaries: List[List[int]] = field(default_factory=list)
    layer_group_to_stage: Dict[str, int] = field(default_factory=dict)
    activation_offload_enabled_groups: List[str] = field(default_factory=list)
    overlap_enabled_channels: List[str] = field(default_factory=list)
    selection_rationale: List[str] = field(default_factory=list)

    def normalized(self) -> "GlobalStrategyPlanSpec":
        norm = copy.deepcopy(self)
        mode = str(norm.primary_parallel_mode or "pp_vpp").strip().lower()
        norm.primary_parallel_mode = mode if mode in {"pp_vpp", "fsdp_zero"} else "pp_vpp"
        norm.dp_degree = max(int(norm.dp_degree or 1), 1)
        norm.tp_degree = max(int(norm.tp_degree or 1), 1)
        norm.pp_degree = max(int(norm.pp_degree or 1), 1)
        norm.vpp_degree = max(int(norm.vpp_degree or 1), 1)
        norm.stage_count = max(int(norm.stage_count or norm.pp_degree or 1), 1)
        normalized_boundaries: List[List[int]] = []
        for item in (norm.stage_boundaries or []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            start = int(item[0])
            end = int(item[1])
            if end < start:
                start, end = end, start
            normalized_boundaries.append([start, end])
        norm.stage_boundaries = normalized_boundaries
        norm.layer_group_to_stage = {
            str(key): int(value)
            for key, value in dict(norm.layer_group_to_stage or {}).items()
            if str(key).strip()
        }
        norm.activation_offload_enabled_groups = [
            str(item) for item in (norm.activation_offload_enabled_groups or []) if str(item).strip()
        ]
        channels: List[str] = []
        for item in (norm.overlap_enabled_channels or []):
            token = str(item or "").strip().lower()
            if token and token not in channels:
                channels.append(token)
        norm.overlap_enabled_channels = channels
        norm.selection_rationale = [str(item) for item in (norm.selection_rationale or []) if str(item).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "primary_parallel_mode": str(norm.primary_parallel_mode),
            "dp_degree": int(norm.dp_degree),
            "tp_degree": int(norm.tp_degree),
            "pp_degree": int(norm.pp_degree),
            "vpp_degree": int(norm.vpp_degree),
            "stage_count": int(norm.stage_count),
            "stage_boundaries": [list(item) for item in norm.stage_boundaries],
            "layer_group_to_stage": dict(norm.layer_group_to_stage),
            "activation_offload_enabled_groups": list(norm.activation_offload_enabled_groups),
            "overlap_enabled_channels": list(norm.overlap_enabled_channels),
            "selection_rationale": list(norm.selection_rationale),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GlobalStrategyPlanSpec":
        return cls(
            primary_parallel_mode=str(payload.get("primary_parallel_mode", "pp_vpp") or "pp_vpp"),
            dp_degree=int(payload.get("dp_degree", 1) or 1),
            tp_degree=int(payload.get("tp_degree", 1) or 1),
            pp_degree=int(payload.get("pp_degree", 1) or 1),
            vpp_degree=int(payload.get("vpp_degree", 1) or 1),
            stage_count=int(payload.get("stage_count", 1) or 1),
            stage_boundaries=[
                [int(item[0]), int(item[1])]
                for item in (payload.get("stage_boundaries") or [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ],
            layer_group_to_stage={
                str(key): int(value)
                for key, value in dict(payload.get("layer_group_to_stage") or {}).items()
            },
            activation_offload_enabled_groups=[
                str(item) for item in (payload.get("activation_offload_enabled_groups") or [])
            ],
            overlap_enabled_channels=[str(item) for item in (payload.get("overlap_enabled_channels") or [])],
            selection_rationale=[str(item) for item in (payload.get("selection_rationale") or [])],
        )


@dataclass
class RewriteActionSpec:
    rewrite_type: str = "reload_shift"
    target_stage_ids: List[int] = field(default_factory=list)
    target_layer_group_ids: List[str] = field(default_factory=list)
    target_state_ids: List[str] = field(default_factory=list)
    direction: str = "hold"
    magnitude: float = 0.0
    expected_gain: float = 0.0
    bottleneck_match_score: float = 0.0
    target_compatibility_score: float = 0.0
    rollback_risk: float = 0.0
    expected_mfu_gain: float = 0.0
    memory_safety_margin: float = 0.0
    counterfactual_score: float = 0.0
    diagnostic_labels: List[str] = field(default_factory=list)
    strategy_source: Optional[str] = None
    llm_rationale: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)

    def normalized(self) -> "RewriteActionSpec":
        norm = copy.deepcopy(self)
        rewrite_type = str(norm.rewrite_type or "reload_shift").strip().lower()
        if rewrite_type not in {
            "reload_shift",
            "adaptive_chunking",
            "local_verticalization",
            "offload_timing_shift",
            "selective_reload_prefetch",
            "overlap_window_switch",
            "tail_optimizer_relief",
            "optimizer_state_partition_rewrite",
            "tp_sp_recomposition",
            "schedule_family_switch",
            "chunk_priority_rewrite",
            "cpu_offload_scope_switch",
            "pp_family_exploration",
            "optimizer_offload_policy_rewrite",
            "recompute_policy_rewrite",
            "pp_vpp_partition_rewrite",
        }:
            rewrite_type = "reload_shift"
        norm.rewrite_type = rewrite_type
        norm.target_stage_ids = sorted({int(item) for item in (norm.target_stage_ids or [])})
        norm.target_layer_group_ids = [str(item) for item in (norm.target_layer_group_ids or []) if str(item).strip()]
        norm.target_state_ids = [str(item) for item in (norm.target_state_ids or []) if str(item).strip()]
        norm.direction = str(norm.direction or "hold").strip().lower() or "hold"
        norm.magnitude = float(norm.magnitude or 0.0)
        norm.expected_gain = float(norm.expected_gain or 0.0)
        norm.bottleneck_match_score = min(max(float(norm.bottleneck_match_score or 0.0), 0.0), 1.0)
        norm.target_compatibility_score = min(max(float(norm.target_compatibility_score or 0.0), 0.0), 1.0)
        norm.rollback_risk = min(max(float(norm.rollback_risk or 0.0), 0.0), 1.0)
        norm.expected_mfu_gain = max(float(norm.expected_mfu_gain or 0.0), 0.0)
        norm.memory_safety_margin = min(max(float(norm.memory_safety_margin or 0.0), 0.0), 1.0)
        norm.counterfactual_score = min(max(float(norm.counterfactual_score or 0.0), -1.0), 1.0)
        norm.diagnostic_labels = [
            str(item).strip().lower()
            for item in (norm.diagnostic_labels or [])
            if str(item).strip()
        ]
        if norm.strategy_source is not None:
            norm.strategy_source = str(norm.strategy_source).strip() or None
        if norm.llm_rationale is not None:
            norm.llm_rationale = str(norm.llm_rationale).strip() or None
        norm.risk_flags = [str(item) for item in (norm.risk_flags or []) if str(item).strip()]
        if norm.expected_mfu_gain <= 0.0 and norm.expected_gain > 0.0:
            norm.expected_mfu_gain = float(norm.expected_gain)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "rewrite_type": str(norm.rewrite_type),
            "target_stage_ids": list(norm.target_stage_ids),
            "target_layer_group_ids": list(norm.target_layer_group_ids),
            "target_state_ids": list(norm.target_state_ids),
            "direction": str(norm.direction),
            "magnitude": float(norm.magnitude),
            "expected_gain": float(norm.expected_gain),
            "bottleneck_match_score": float(norm.bottleneck_match_score),
            "target_compatibility_score": float(norm.target_compatibility_score),
            "rollback_risk": float(norm.rollback_risk),
            "expected_mfu_gain": float(norm.expected_mfu_gain),
            "memory_safety_margin": float(norm.memory_safety_margin),
            "counterfactual_score": float(norm.counterfactual_score),
            "diagnostic_labels": list(norm.diagnostic_labels),
            "strategy_source": norm.strategy_source,
            "llm_rationale": norm.llm_rationale,
            "risk_flags": list(norm.risk_flags),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RewriteActionSpec":
        return cls(
            rewrite_type=str(payload.get("rewrite_type", "reload_shift") or "reload_shift"),
            target_stage_ids=[int(item) for item in (payload.get("target_stage_ids") or [])],
            target_layer_group_ids=[str(item) for item in (payload.get("target_layer_group_ids") or [])],
            target_state_ids=[str(item) for item in (payload.get("target_state_ids") or [])],
            direction=str(payload.get("direction", "hold") or "hold"),
            magnitude=float(payload.get("magnitude", 0.0) or 0.0),
            expected_gain=float(payload.get("expected_gain", 0.0) or 0.0),
            bottleneck_match_score=float(payload.get("bottleneck_match_score", 0.0) or 0.0),
            target_compatibility_score=float(payload.get("target_compatibility_score", 0.0) or 0.0),
            rollback_risk=float(payload.get("rollback_risk", 0.0) or 0.0),
            expected_mfu_gain=float(
                payload.get("expected_mfu_gain", payload.get("expected_gain", 0.0)) or 0.0
            ),
            memory_safety_margin=float(payload.get("memory_safety_margin", 0.0) or 0.0),
            counterfactual_score=float(payload.get("counterfactual_score", 0.0) or 0.0),
            diagnostic_labels=[str(item) for item in (payload.get("diagnostic_labels") or [])],
            strategy_source=payload.get("strategy_source"),
            llm_rationale=payload.get("llm_rationale"),
            risk_flags=[str(item) for item in (payload.get("risk_flags") or [])],
        )


@dataclass
class WindowFeedbackSpec:
    window_index: int = 0
    policy_signature: str = ""
    critical_stage_id: int = -1
    critical_layer_group_id: str = ""
    critical_component_type: str = "forward"
    step_time_ms_p50: float = 0.0
    throughput_tokens_per_s: float = 0.0
    bubble_ratio: float = 0.0
    reload_stall_ms: float = 0.0
    comm_exposure_ratio: float = 0.0
    offload_overlap_success_ratio: float = 0.0
    critical_path_breakdown: Dict[str, Any] = field(default_factory=dict)
    recommended_rewrites: List[RewriteActionSpec] = field(default_factory=list)
    rollback_triggered: bool = False

    def normalized(self) -> "WindowFeedbackSpec":
        norm = copy.deepcopy(self)
        norm.window_index = max(int(norm.window_index or 0), 0)
        norm.policy_signature = str(norm.policy_signature or "").strip()
        norm.critical_stage_id = int(norm.critical_stage_id if norm.critical_stage_id is not None else -1)
        norm.critical_layer_group_id = str(norm.critical_layer_group_id or "").strip()
        norm.critical_component_type = str(norm.critical_component_type or "forward").strip().lower() or "forward"
        norm.step_time_ms_p50 = max(float(norm.step_time_ms_p50 or 0.0), 0.0)
        norm.throughput_tokens_per_s = max(float(norm.throughput_tokens_per_s or 0.0), 0.0)
        norm.bubble_ratio = max(float(norm.bubble_ratio or 0.0), 0.0)
        norm.reload_stall_ms = max(float(norm.reload_stall_ms or 0.0), 0.0)
        norm.comm_exposure_ratio = max(float(norm.comm_exposure_ratio or 0.0), 0.0)
        norm.offload_overlap_success_ratio = max(float(norm.offload_overlap_success_ratio or 0.0), 0.0)
        norm.critical_path_breakdown = copy.deepcopy(norm.critical_path_breakdown or {})
        norm.recommended_rewrites = [item.normalized() for item in (norm.recommended_rewrites or [])]
        norm.rollback_triggered = bool(norm.rollback_triggered)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "window_index": int(norm.window_index),
            "policy_signature": str(norm.policy_signature),
            "critical_stage_id": int(norm.critical_stage_id),
            "critical_layer_group_id": str(norm.critical_layer_group_id),
            "critical_component_type": str(norm.critical_component_type),
            "step_time_ms_p50": float(norm.step_time_ms_p50),
            "throughput_tokens_per_s": float(norm.throughput_tokens_per_s),
            "bubble_ratio": float(norm.bubble_ratio),
            "reload_stall_ms": float(norm.reload_stall_ms),
            "comm_exposure_ratio": float(norm.comm_exposure_ratio),
            "offload_overlap_success_ratio": float(norm.offload_overlap_success_ratio),
            "critical_path_breakdown": copy.deepcopy(norm.critical_path_breakdown),
            "recommended_rewrites": [item.to_dict() for item in norm.recommended_rewrites],
            "rollback_triggered": bool(norm.rollback_triggered),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WindowFeedbackSpec":
        return cls(
            window_index=int(payload.get("window_index", 0) or 0),
            policy_signature=str(payload.get("policy_signature", "") or ""),
            critical_stage_id=int(payload.get("critical_stage_id", -1) or -1),
            critical_layer_group_id=str(payload.get("critical_layer_group_id", "") or ""),
            critical_component_type=str(payload.get("critical_component_type", "forward") or "forward"),
            step_time_ms_p50=float(payload.get("step_time_ms_p50", 0.0) or 0.0),
            throughput_tokens_per_s=float(payload.get("throughput_tokens_per_s", 0.0) or 0.0),
            bubble_ratio=float(payload.get("bubble_ratio", 0.0) or 0.0),
            reload_stall_ms=float(payload.get("reload_stall_ms", 0.0) or 0.0),
            comm_exposure_ratio=float(payload.get("comm_exposure_ratio", 0.0) or 0.0),
            offload_overlap_success_ratio=float(payload.get("offload_overlap_success_ratio", 0.0) or 0.0),
            critical_path_breakdown=copy.deepcopy(payload.get("critical_path_breakdown") or {}),
            recommended_rewrites=[
                RewriteActionSpec.from_dict(item) for item in (payload.get("recommended_rewrites") or [])
            ],
            rollback_triggered=bool(payload.get("rollback_triggered", False)),
        )


@dataclass
class RewriteExecutionPlanSpec:
    global_strategy: Optional[GlobalStrategyPlanSpec] = None
    rewrite_actions: List[RewriteActionSpec] = field(default_factory=list)
    telemetry_budget: Optional[TelemetryBudgetSpec] = None
    window_reconfig: Optional[WindowReconfigSpec] = None
    version_tag: str = "v1"

    def normalized(self) -> "RewriteExecutionPlanSpec":
        norm = copy.deepcopy(self)
        norm.global_strategy = (
            norm.global_strategy.normalized() if isinstance(norm.global_strategy, GlobalStrategyPlanSpec) else None
        )
        norm.rewrite_actions = [item.normalized() for item in (norm.rewrite_actions or [])]
        norm.telemetry_budget = (
            norm.telemetry_budget.normalized() if isinstance(norm.telemetry_budget, TelemetryBudgetSpec) else None
        )
        norm.window_reconfig = (
            norm.window_reconfig.normalized() if isinstance(norm.window_reconfig, WindowReconfigSpec) else None
        )
        norm.version_tag = str(norm.version_tag or "v1").strip() or "v1"
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "global_strategy": norm.global_strategy.to_dict() if norm.global_strategy is not None else None,
            "rewrite_actions": [item.to_dict() for item in norm.rewrite_actions],
            "telemetry_budget": norm.telemetry_budget.to_dict() if norm.telemetry_budget is not None else None,
            "window_reconfig": norm.window_reconfig.to_dict() if norm.window_reconfig is not None else None,
            "version_tag": str(norm.version_tag),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RewriteExecutionPlanSpec":
        return cls(
            global_strategy=GlobalStrategyPlanSpec.from_dict(payload.get("global_strategy") or {})
            if payload.get("global_strategy") is not None
            else None,
            rewrite_actions=[RewriteActionSpec.from_dict(item) for item in (payload.get("rewrite_actions") or [])],
            telemetry_budget=TelemetryBudgetSpec.from_dict(payload.get("telemetry_budget") or {})
            if payload.get("telemetry_budget") is not None
            else None,
            window_reconfig=WindowReconfigSpec.from_dict(payload.get("window_reconfig") or {})
            if payload.get("window_reconfig") is not None
            else None,
            version_tag=str(payload.get("version_tag", "v1") or "v1"),
        )


@dataclass
class BatchPlanSpec:
    micro_batch_size: int = 1
    global_batch_size: int = 16
    grad_accum_steps: Optional[int] = None
    target_tokens_per_step: Optional[int] = None

    def normalized(self) -> "BatchPlanSpec":
        norm = copy.deepcopy(self)
        norm.micro_batch_size = max(int(norm.micro_batch_size), 1)
        norm.global_batch_size = max(int(norm.global_batch_size), 1)
        if norm.grad_accum_steps is not None:
            norm.grad_accum_steps = max(int(norm.grad_accum_steps), 1)
        if norm.target_tokens_per_step is not None:
            norm.target_tokens_per_step = max(int(norm.target_tokens_per_step), 1)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "micro_batch_size": int(norm.micro_batch_size),
            "global_batch_size": int(norm.global_batch_size),
            "grad_accum_steps": norm.grad_accum_steps,
            "target_tokens_per_step": norm.target_tokens_per_step,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BatchPlanSpec":
        return cls(
            micro_batch_size=int(payload.get("micro_batch_size", 1) or 1),
            global_batch_size=int(payload.get("global_batch_size", 16) or 1),
            grad_accum_steps=payload.get("grad_accum_steps"),
            target_tokens_per_step=payload.get("target_tokens_per_step"),
        )


@dataclass
class LengthBucketPolicy:
    name: str
    min_seq_len: int = 1
    max_seq_len: Optional[int] = None
    preferred_program_kind: Optional[str] = None
    cp_cap: Optional[int] = None
    micro_batch_cap: Optional[int] = None
    schedule_templates: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def normalized(self) -> "LengthBucketPolicy":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "default")
        norm.min_seq_len = max(int(norm.min_seq_len), 1)
        if norm.max_seq_len is not None:
            norm.max_seq_len = max(int(norm.max_seq_len), norm.min_seq_len)
        if norm.cp_cap is not None:
            norm.cp_cap = max(int(norm.cp_cap), 1)
        if norm.micro_batch_cap is not None:
            norm.micro_batch_cap = max(int(norm.micro_batch_cap), 1)
        norm.schedule_templates = [str(item) for item in (norm.schedule_templates or [])]
        if norm.preferred_program_kind is not None:
            norm.preferred_program_kind = str(norm.preferred_program_kind).strip() or None
        if norm.notes is not None:
            norm.notes = str(norm.notes).strip() or None
        return norm

    def matches(self, seq_len: int) -> bool:
        norm = self.normalized()
        current = max(int(seq_len), 1)
        if current < norm.min_seq_len:
            return False
        if norm.max_seq_len is not None and current > norm.max_seq_len:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "min_seq_len": int(norm.min_seq_len),
            "max_seq_len": norm.max_seq_len,
            "preferred_program_kind": norm.preferred_program_kind,
            "cp_cap": norm.cp_cap,
            "micro_batch_cap": norm.micro_batch_cap,
            "schedule_templates": list(norm.schedule_templates),
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LengthBucketPolicy":
        return cls(
            name=str(payload.get("name", "default")),
            min_seq_len=int(payload.get("min_seq_len", 1) or 1),
            max_seq_len=payload.get("max_seq_len"),
            preferred_program_kind=payload.get("preferred_program_kind"),
            cp_cap=payload.get("cp_cap"),
            micro_batch_cap=payload.get("micro_batch_cap"),
            schedule_templates=[str(item) for item in (payload.get("schedule_templates") or [])],
            notes=payload.get("notes"),
        )


@dataclass
class SubgraphSpec:
    name: str
    stage_index: int
    decoder_start: int = 0
    decoder_end: int = 0
    module_family: str = "decoder"
    special_tokens: List[str] = field(default_factory=list)
    attention_heavy: bool = False
    loss_heavy: bool = False

    def normalized(self) -> "SubgraphSpec":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "subgraph")
        norm.stage_index = max(int(norm.stage_index), 0)
        norm.decoder_start = max(int(norm.decoder_start), 0)
        norm.decoder_end = max(int(norm.decoder_end), norm.decoder_start)
        norm.module_family = str(norm.module_family or "decoder")
        norm.special_tokens = [str(token) for token in (norm.special_tokens or []) if str(token).strip()]
        norm.attention_heavy = bool(norm.attention_heavy)
        norm.loss_heavy = bool(norm.loss_heavy)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "stage_index": int(norm.stage_index),
            "decoder_start": int(norm.decoder_start),
            "decoder_end": int(norm.decoder_end),
            "module_family": norm.module_family,
            "special_tokens": list(norm.special_tokens),
            "attention_heavy": bool(norm.attention_heavy),
            "loss_heavy": bool(norm.loss_heavy),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SubgraphSpec":
        return cls(
            name=str(payload.get("name", "subgraph")),
            stage_index=int(payload.get("stage_index", 0) or 0),
            decoder_start=int(payload.get("decoder_start", 0) or 0),
            decoder_end=int(payload.get("decoder_end", 0) or 0),
            module_family=str(payload.get("module_family", "decoder")),
            special_tokens=[str(token) for token in (payload.get("special_tokens") or [])],
            attention_heavy=bool(payload.get("attention_heavy", False)),
            loss_heavy=bool(payload.get("loss_heavy", False)),
        )


@dataclass
class PlacementEntrySpec:
    subgraph: str
    nodes: List[str] = field(default_factory=list)
    device_group_size: int = 1
    device_type: Optional[str] = None
    topology_domain: Optional[str] = None

    def normalized(self) -> "PlacementEntrySpec":
        norm = copy.deepcopy(self)
        norm.subgraph = str(norm.subgraph or "subgraph")
        norm.nodes = [str(node) for node in (norm.nodes or [])]
        norm.device_group_size = max(int(norm.device_group_size), 1)
        if norm.device_type is not None:
            norm.device_type = str(norm.device_type).strip() or None
        if norm.topology_domain is not None:
            norm.topology_domain = str(norm.topology_domain).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "subgraph": norm.subgraph,
            "nodes": list(norm.nodes),
            "device_group_size": int(norm.device_group_size),
            "device_type": norm.device_type,
            "topology_domain": norm.topology_domain,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlacementEntrySpec":
        return cls(
            subgraph=str(payload.get("subgraph", "subgraph")),
            nodes=[str(node) for node in (payload.get("nodes") or [])],
            device_group_size=int(payload.get("device_group_size", 1) or 1),
            device_type=payload.get("device_type"),
            topology_domain=payload.get("topology_domain"),
        )


@dataclass
class LocalParallelSpec:
    subgraph: str
    vpp_degree: int = 1
    cp_degree: int = 1
    fsdp_scope: str = "none"
    shard_strategy: str = "none"
    reshard_policy: str = "default"
    shard_group_size: Optional[int] = None
    replicate_group_size: Optional[int] = None
    offload_policy: str = "none"
    reduce_dtype: Optional[str] = None
    device_group_size: Optional[int] = None
    device_group_type: Optional[str] = None

    def normalized(self) -> "LocalParallelSpec":
        norm = copy.deepcopy(self)
        norm.subgraph = str(norm.subgraph or "subgraph")
        norm.vpp_degree = max(int(norm.vpp_degree), 1)
        norm.cp_degree = max(int(norm.cp_degree), 1)
        norm.fsdp_scope = str(norm.fsdp_scope or "none").strip().lower() or "none"
        norm.shard_strategy = str(norm.shard_strategy or "none").strip().lower() or "none"
        norm.reshard_policy = str(norm.reshard_policy or "default").strip().lower() or "default"
        if norm.shard_group_size is not None:
            norm.shard_group_size = max(int(norm.shard_group_size), 1)
        if norm.replicate_group_size is not None:
            norm.replicate_group_size = max(int(norm.replicate_group_size), 1)
        norm.offload_policy = str(norm.offload_policy or "none").strip().lower() or "none"
        if norm.reduce_dtype is not None:
            norm.reduce_dtype = str(norm.reduce_dtype).strip().lower() or None
        if norm.device_group_size is not None:
            norm.device_group_size = max(int(norm.device_group_size), 1)
        if norm.device_group_type is not None:
            norm.device_group_type = str(norm.device_group_type).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "subgraph": norm.subgraph,
            "vpp_degree": int(norm.vpp_degree),
            "cp_degree": int(norm.cp_degree),
            "fsdp_scope": norm.fsdp_scope,
            "shard_strategy": norm.shard_strategy,
            "reshard_policy": norm.reshard_policy,
            "shard_group_size": norm.shard_group_size,
            "replicate_group_size": norm.replicate_group_size,
            "offload_policy": norm.offload_policy,
            "reduce_dtype": norm.reduce_dtype,
            "device_group_size": norm.device_group_size,
            "device_group_type": norm.device_group_type,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LocalParallelSpec":
        return cls(
            subgraph=str(payload.get("subgraph", "subgraph")),
            vpp_degree=int(payload.get("vpp_degree", 1) or 1),
            cp_degree=int(payload.get("cp_degree", 1) or 1),
            fsdp_scope=str(payload.get("fsdp_scope", "none")),
            shard_strategy=str(payload.get("shard_strategy", "none")),
            reshard_policy=str(payload.get("reshard_policy", "default")),
            shard_group_size=payload.get("shard_group_size"),
            replicate_group_size=payload.get("replicate_group_size"),
            offload_policy=str(payload.get("offload_policy", "none")),
            reduce_dtype=payload.get("reduce_dtype"),
            device_group_size=payload.get("device_group_size"),
            device_group_type=payload.get("device_group_type"),
        )


@dataclass
class MorphableUnitSpec:
    name: str
    semantic_role: str = "decoder"
    atom_kind: str = "decoder_block"
    parent_subgraph: Optional[str] = None
    stage_index: int = 0
    decoder_start: int = 0
    decoder_end: int = 0
    compute_weight: float = 0.0
    memory_weight: float = 0.0
    communication_weight: float = 0.0
    boundary_cost: float = 0.0
    liveness_weight: float = 0.0
    special_tokens: List[str] = field(default_factory=list)

    def normalized(self) -> "MorphableUnitSpec":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "unit")
        norm.semantic_role = str(norm.semantic_role or "decoder").strip().lower() or "decoder"
        norm.atom_kind = str(norm.atom_kind or "decoder_block").strip().lower() or "decoder_block"
        if norm.parent_subgraph is not None:
            norm.parent_subgraph = str(norm.parent_subgraph).strip() or None
        norm.stage_index = max(int(norm.stage_index), 0)
        norm.decoder_start = max(int(norm.decoder_start), 0)
        norm.decoder_end = max(int(norm.decoder_end), norm.decoder_start)
        norm.compute_weight = max(float(norm.compute_weight), 0.0)
        norm.memory_weight = max(float(norm.memory_weight), 0.0)
        norm.communication_weight = max(float(norm.communication_weight), 0.0)
        norm.boundary_cost = max(float(norm.boundary_cost), 0.0)
        norm.liveness_weight = max(float(norm.liveness_weight), 0.0)
        norm.special_tokens = [str(token) for token in (norm.special_tokens or []) if str(token).strip()]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "semantic_role": norm.semantic_role,
            "atom_kind": norm.atom_kind,
            "parent_subgraph": norm.parent_subgraph,
            "stage_index": int(norm.stage_index),
            "decoder_start": int(norm.decoder_start),
            "decoder_end": int(norm.decoder_end),
            "compute_weight": float(norm.compute_weight),
            "memory_weight": float(norm.memory_weight),
            "communication_weight": float(norm.communication_weight),
            "boundary_cost": float(norm.boundary_cost),
            "liveness_weight": float(norm.liveness_weight),
            "special_tokens": list(norm.special_tokens),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MorphableUnitSpec":
        return cls(
            name=str(payload.get("name", "unit")),
            semantic_role=str(payload.get("semantic_role", "decoder")),
            atom_kind=str(payload.get("atom_kind", "decoder_block")),
            parent_subgraph=payload.get("parent_subgraph"),
            stage_index=int(payload.get("stage_index", 0) or 0),
            decoder_start=int(payload.get("decoder_start", 0) or 0),
            decoder_end=int(payload.get("decoder_end", payload.get("decoder_start", 0)) or 0),
            compute_weight=float(payload.get("compute_weight", 0.0) or 0.0),
            memory_weight=float(payload.get("memory_weight", 0.0) or 0.0),
            communication_weight=float(payload.get("communication_weight", 0.0) or 0.0),
            boundary_cost=float(payload.get("boundary_cost", 0.0) or 0.0),
            liveness_weight=float(payload.get("liveness_weight", 0.0) or 0.0),
            special_tokens=[str(token) for token in (payload.get("special_tokens") or [])],
        )


@dataclass
class MorphableEdgeSpec:
    src: str
    dst: str
    semantic: str = "structure"
    criticality: float = 0.0
    cost: float = 0.0

    def normalized(self) -> "MorphableEdgeSpec":
        norm = copy.deepcopy(self)
        norm.src = str(norm.src or "src")
        norm.dst = str(norm.dst or "dst")
        norm.semantic = str(norm.semantic or "structure").strip().lower() or "structure"
        norm.criticality = max(float(norm.criticality), 0.0)
        norm.cost = max(float(norm.cost), 0.0)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "src": norm.src,
            "dst": norm.dst,
            "semantic": norm.semantic,
            "criticality": float(norm.criticality),
            "cost": float(norm.cost),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MorphableEdgeSpec":
        return cls(
            src=str(payload.get("src", "src")),
            dst=str(payload.get("dst", "dst")),
            semantic=str(payload.get("semantic", "structure")),
            criticality=float(payload.get("criticality", 0.0) or 0.0),
            cost=float(payload.get("cost", 0.0) or 0.0),
        )


@dataclass
class MorphableStageFamilySpec:
    stage_index: int
    family: str = "balanced_interleave"
    semantic_role: str = "decoder"
    preferred_template: Optional[str] = None
    dispatch_order: Optional[str] = None
    warmup_policy: Optional[str] = None
    cooldown_policy: Optional[str] = None
    checkpoint_policy: Optional[str] = None
    p2p_policy: Optional[str] = None
    combined_policy: Optional[str] = None
    recompute_modules: List[str] = field(default_factory=list)
    offload_modules: List[str] = field(default_factory=list)
    chunk_priority_hints: List[int] = field(default_factory=list)

    def normalized(self) -> "MorphableStageFamilySpec":
        norm = copy.deepcopy(self)
        norm.stage_index = max(int(norm.stage_index), 0)
        norm.family = str(norm.family or "balanced_interleave").strip().lower() or "balanced_interleave"
        norm.semantic_role = str(norm.semantic_role or "decoder").strip().lower() or "decoder"
        if norm.preferred_template is not None:
            norm.preferred_template = str(norm.preferred_template).strip() or None
        if norm.dispatch_order is not None:
            norm.dispatch_order = str(norm.dispatch_order).strip() or None
        if norm.warmup_policy is not None:
            norm.warmup_policy = str(norm.warmup_policy).strip() or None
        if norm.cooldown_policy is not None:
            norm.cooldown_policy = str(norm.cooldown_policy).strip() or None
        if norm.checkpoint_policy is not None:
            norm.checkpoint_policy = str(norm.checkpoint_policy).strip().lower() or None
        if norm.p2p_policy is not None:
            norm.p2p_policy = str(norm.p2p_policy).strip().lower() or None
        if norm.combined_policy is not None:
            norm.combined_policy = str(norm.combined_policy).strip().lower() or None
        norm.recompute_modules = [str(item) for item in (norm.recompute_modules or []) if str(item).strip()]
        norm.offload_modules = [str(item) for item in (norm.offload_modules or []) if str(item).strip()]
        norm.chunk_priority_hints = [max(int(item), 0) for item in (norm.chunk_priority_hints or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "stage_index": int(norm.stage_index),
            "family": norm.family,
            "semantic_role": norm.semantic_role,
            "preferred_template": norm.preferred_template,
            "dispatch_order": norm.dispatch_order,
            "warmup_policy": norm.warmup_policy,
            "cooldown_policy": norm.cooldown_policy,
            "checkpoint_policy": norm.checkpoint_policy,
            "p2p_policy": norm.p2p_policy,
            "combined_policy": norm.combined_policy,
            "recompute_modules": list(norm.recompute_modules),
            "offload_modules": list(norm.offload_modules),
            "chunk_priority_hints": list(norm.chunk_priority_hints),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MorphableStageFamilySpec":
        return cls(
            stage_index=int(payload.get("stage_index", 0) or 0),
            family=str(payload.get("family", "balanced_interleave")),
            semantic_role=str(payload.get("semantic_role", "decoder")),
            preferred_template=payload.get("preferred_template"),
            dispatch_order=payload.get("dispatch_order"),
            warmup_policy=payload.get("warmup_policy"),
            cooldown_policy=payload.get("cooldown_policy"),
            checkpoint_policy=payload.get("checkpoint_policy"),
            p2p_policy=payload.get("p2p_policy"),
            combined_policy=payload.get("combined_policy"),
            recompute_modules=[str(item) for item in (payload.get("recompute_modules") or [])],
            offload_modules=[str(item) for item in (payload.get("offload_modules") or [])],
            chunk_priority_hints=[int(item) for item in (payload.get("chunk_priority_hints") or [])],
        )


@dataclass
class MorphablePipelineSpec:
    shape_objective: str = "memory_constrained_throughput_maximization"
    units: List[MorphableUnitSpec] = field(default_factory=list)
    structure_edges: List[MorphableEdgeSpec] = field(default_factory=list)
    memory_edges: List[MorphableEdgeSpec] = field(default_factory=list)
    communication_edges: List[MorphableEdgeSpec] = field(default_factory=list)
    stage_families: List[MorphableStageFamilySpec] = field(default_factory=list)
    chunk_shape_vector: List[int] = field(default_factory=list)
    search_levels: List[str] = field(
        default_factory=lambda: ["structural_regroup", "stage_chunk_form", "family_policy_select"]
    )
    legality_guards: Dict[str, Any] = field(default_factory=dict)
    shape_signature: Optional[str] = None

    def normalized(self) -> "MorphablePipelineSpec":
        norm = copy.deepcopy(self)
        norm.shape_objective = (
            str(norm.shape_objective or "memory_constrained_throughput_maximization").strip().lower()
            or "memory_constrained_throughput_maximization"
        )
        norm.units = [item.normalized() for item in (norm.units or [])]
        norm.structure_edges = [item.normalized() for item in (norm.structure_edges or [])]
        norm.memory_edges = [item.normalized() for item in (norm.memory_edges or [])]
        norm.communication_edges = [item.normalized() for item in (norm.communication_edges or [])]
        norm.stage_families = [item.normalized() for item in (norm.stage_families or [])]
        norm.chunk_shape_vector = [max(int(item), 1) for item in (norm.chunk_shape_vector or [])]
        norm.search_levels = [str(item) for item in (norm.search_levels or []) if str(item).strip()]
        if not norm.search_levels:
            norm.search_levels = ["structural_regroup", "stage_chunk_form", "family_policy_select"]
        norm.legality_guards = copy.deepcopy(norm.legality_guards or {})
        if norm.shape_signature is not None:
            norm.shape_signature = str(norm.shape_signature).strip() or None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "shape_objective": norm.shape_objective,
            "units": [item.to_dict() for item in norm.units],
            "structure_edges": [item.to_dict() for item in norm.structure_edges],
            "memory_edges": [item.to_dict() for item in norm.memory_edges],
            "communication_edges": [item.to_dict() for item in norm.communication_edges],
            "stage_families": [item.to_dict() for item in norm.stage_families],
            "chunk_shape_vector": list(norm.chunk_shape_vector),
            "search_levels": list(norm.search_levels),
            "legality_guards": copy.deepcopy(norm.legality_guards),
            "shape_signature": norm.shape_signature,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MorphablePipelineSpec":
        return cls(
            shape_objective=str(payload.get("shape_objective", "memory_constrained_throughput_maximization")),
            units=[MorphableUnitSpec.from_dict(item) for item in (payload.get("units") or [])],
            structure_edges=[MorphableEdgeSpec.from_dict(item) for item in (payload.get("structure_edges") or [])],
            memory_edges=[MorphableEdgeSpec.from_dict(item) for item in (payload.get("memory_edges") or [])],
            communication_edges=[MorphableEdgeSpec.from_dict(item) for item in (payload.get("communication_edges") or [])],
            stage_families=[
                MorphableStageFamilySpec.from_dict(item) for item in (payload.get("stage_families") or [])
            ],
            chunk_shape_vector=[int(item) for item in (payload.get("chunk_shape_vector") or [])],
            search_levels=[str(item) for item in (payload.get("search_levels") or [])],
            legality_guards=copy.deepcopy(payload.get("legality_guards") or {}),
            shape_signature=payload.get("shape_signature"),
        )


@dataclass
class PipeRuntimeSpec:
    template: str = "fixed_1f1b"
    microbatch_order: str = "default"
    steady_state_group_size: Optional[int] = None
    warmup_policy: str = "default"
    cooldown_policy: str = "default"

    def normalized(self) -> "PipeRuntimeSpec":
        norm = copy.deepcopy(self)
        norm.template = str(norm.template or "fixed_1f1b")
        norm.microbatch_order = str(norm.microbatch_order or "default")
        if norm.steady_state_group_size is not None:
            norm.steady_state_group_size = max(int(norm.steady_state_group_size), 1)
        norm.warmup_policy = str(norm.warmup_policy or "default")
        norm.cooldown_policy = str(norm.cooldown_policy or "default")
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "template": norm.template,
            "microbatch_order": norm.microbatch_order,
            "steady_state_group_size": norm.steady_state_group_size,
            "warmup_policy": norm.warmup_policy,
            "cooldown_policy": norm.cooldown_policy,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipeRuntimeSpec":
        return cls(
            template=str(payload.get("template", "fixed_1f1b")),
            microbatch_order=str(payload.get("microbatch_order", "default")),
            steady_state_group_size=payload.get("steady_state_group_size"),
            warmup_policy=str(payload.get("warmup_policy", "default")),
            cooldown_policy=str(payload.get("cooldown_policy", "default")),
        )


@dataclass
class StrategyIRSpec:
    apipe: List[SubgraphSpec] = field(default_factory=list)
    placement: List[PlacementEntrySpec] = field(default_factory=list)
    local_parallel: List[LocalParallelSpec] = field(default_factory=list)
    pipe: PipeRuntimeSpec = field(default_factory=PipeRuntimeSpec)
    morphable_pipe: MorphablePipelineSpec = field(default_factory=MorphablePipelineSpec)
    variable_tiers: Dict[str, str] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_VARIABLE_TIERS))

    def normalized(self) -> "StrategyIRSpec":
        norm = copy.deepcopy(self)
        norm.apipe = [item.normalized() for item in (norm.apipe or [])]
        norm.placement = [item.normalized() for item in (norm.placement or [])]
        norm.local_parallel = [item.normalized() for item in (norm.local_parallel or [])]
        norm.pipe = norm.pipe.normalized()
        norm.morphable_pipe = norm.morphable_pipe.normalized()
        tiers = copy.deepcopy(_DEFAULT_VARIABLE_TIERS)
        for key, value in (norm.variable_tiers or {}).items():
            tiers[str(key)] = str(value or tiers.get(str(key), "global_low_freq"))
        norm.variable_tiers = tiers
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "apipe": [item.to_dict() for item in norm.apipe],
            "placement": [item.to_dict() for item in norm.placement],
            "local_parallel": [item.to_dict() for item in norm.local_parallel],
            "pipe": norm.pipe.to_dict(),
            "morphable_pipe": norm.morphable_pipe.to_dict(),
            "variable_tiers": copy.deepcopy(norm.variable_tiers),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StrategyIRSpec":
        return cls(
            apipe=[SubgraphSpec.from_dict(item) for item in (payload.get("apipe") or [])],
            placement=[PlacementEntrySpec.from_dict(item) for item in (payload.get("placement") or [])],
            local_parallel=[LocalParallelSpec.from_dict(item) for item in (payload.get("local_parallel") or [])],
            pipe=PipeRuntimeSpec.from_dict(payload.get("pipe") or {}),
            morphable_pipe=MorphablePipelineSpec.from_dict(payload.get("morphable_pipe") or {}),
            variable_tiers={
                str(key): str(value)
                for key, value in (payload.get("variable_tiers") or {}).items()
            },
        )


@dataclass
class ConstraintSpec:
    required_node_local_axes: List[str] = field(default_factory=list)
    memory_budget_gb: Optional[float] = None
    requires_runtime_pg_rebuild: bool = False
    requested_heterogeneous_apipe: bool = False
    notes: Optional[str] = None

    def normalized(self) -> "ConstraintSpec":
        norm = copy.deepcopy(self)
        norm.required_node_local_axes = [str(axis) for axis in (norm.required_node_local_axes or [])]
        if norm.memory_budget_gb is not None:
            norm.memory_budget_gb = max(float(norm.memory_budget_gb), 1.0)
        norm.requires_runtime_pg_rebuild = bool(norm.requires_runtime_pg_rebuild)
        norm.requested_heterogeneous_apipe = bool(norm.requested_heterogeneous_apipe)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_node_local_axes": list(self.required_node_local_axes),
            "memory_budget_gb": self.memory_budget_gb,
            "requires_runtime_pg_rebuild": bool(self.requires_runtime_pg_rebuild),
            "requested_heterogeneous_apipe": bool(self.requested_heterogeneous_apipe),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConstraintSpec":
        return cls(
            required_node_local_axes=[str(axis) for axis in (payload.get("required_node_local_axes") or [])],
            memory_budget_gb=payload.get("memory_budget_gb"),
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
    allow_hybrid_shard: bool = False
    allow_torchtitan_schedule_sandbox: bool = False
    allow_subgraph_submeshes: bool = False
    allow_heterogeneous_apipe: bool = False
    allow_morphable_pipeline: bool = False
    max_tp_size: Optional[int] = None
    max_pp_size: Optional[int] = None
    max_ep_size: Optional[int] = None
    max_cp_size: Optional[int] = None
    max_vpp_size: Optional[int] = None
    max_shard_group_size: Optional[int] = None
    max_replicate_group_size: Optional[int] = None
    max_micro_batch_size: Optional[int] = None
    max_estimated_memory_pressure: Optional[float] = None
    prefer_memory_relief: bool = False
    required_node_local_axes: List[str] = field(default_factory=list)
    preferred_node_for_module: Dict[str, str] = field(default_factory=dict)
    forbidden_axes_by_node: Dict[str, List[str]] = field(default_factory=dict)
    allowed_schedule_skeletons: List[str] = field(default_factory=lambda: ["fixed_1f1b"])
    allowed_schedule_templates: List[str] = field(default_factory=lambda: ["fixed_1f1b"])
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
        norm.allow_hybrid_shard = bool(norm.allow_hybrid_shard)
        norm.allow_torchtitan_schedule_sandbox = bool(norm.allow_torchtitan_schedule_sandbox)
        norm.allow_subgraph_submeshes = bool(norm.allow_subgraph_submeshes)
        norm.allow_heterogeneous_apipe = bool(norm.allow_heterogeneous_apipe)
        norm.allow_morphable_pipeline = bool(norm.allow_morphable_pipeline)
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
        if norm.max_shard_group_size is not None:
            norm.max_shard_group_size = max(int(norm.max_shard_group_size), 1)
        if norm.max_replicate_group_size is not None:
            norm.max_replicate_group_size = max(int(norm.max_replicate_group_size), 1)
        if norm.max_micro_batch_size is not None:
            norm.max_micro_batch_size = max(int(norm.max_micro_batch_size), 1)
        if norm.max_estimated_memory_pressure is not None:
            norm.max_estimated_memory_pressure = max(float(norm.max_estimated_memory_pressure), 0.1)
        norm.prefer_memory_relief = bool(norm.prefer_memory_relief)
        norm.required_node_local_axes = [str(axis) for axis in (norm.required_node_local_axes or [])]
        norm.preferred_node_for_module = {str(key): str(value) for key, value in (norm.preferred_node_for_module or {}).items()}
        norm.forbidden_axes_by_node = {
            str(node): [str(axis) for axis in axes]
            for node, axes in (norm.forbidden_axes_by_node or {}).items()
        }
        norm.allowed_schedule_skeletons = [str(item) for item in (norm.allowed_schedule_skeletons or ["fixed_1f1b"])]
        norm.allowed_schedule_templates = [str(item) for item in (norm.allowed_schedule_templates or ["fixed_1f1b"])]
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
            "allow_hybrid_shard": bool(norm.allow_hybrid_shard),
            "allow_torchtitan_schedule_sandbox": bool(norm.allow_torchtitan_schedule_sandbox),
            "allow_subgraph_submeshes": bool(norm.allow_subgraph_submeshes),
            "allow_heterogeneous_apipe": bool(norm.allow_heterogeneous_apipe),
            "allow_morphable_pipeline": bool(norm.allow_morphable_pipeline),
            "max_tp_size": norm.max_tp_size,
            "max_pp_size": norm.max_pp_size,
            "max_ep_size": norm.max_ep_size,
            "max_cp_size": norm.max_cp_size,
            "max_vpp_size": norm.max_vpp_size,
            "max_shard_group_size": norm.max_shard_group_size,
            "max_replicate_group_size": norm.max_replicate_group_size,
            "max_micro_batch_size": norm.max_micro_batch_size,
            "max_estimated_memory_pressure": norm.max_estimated_memory_pressure,
            "prefer_memory_relief": bool(norm.prefer_memory_relief),
            "required_node_local_axes": list(norm.required_node_local_axes),
            "preferred_node_for_module": copy.deepcopy(norm.preferred_node_for_module),
            "forbidden_axes_by_node": copy.deepcopy(norm.forbidden_axes_by_node),
            "allowed_schedule_skeletons": list(norm.allowed_schedule_skeletons),
            "allowed_schedule_templates": list(norm.allowed_schedule_templates),
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
            allow_hybrid_shard=bool(payload.get("allow_hybrid_shard", False)),
            allow_torchtitan_schedule_sandbox=bool(payload.get("allow_torchtitan_schedule_sandbox", False)),
            allow_subgraph_submeshes=bool(payload.get("allow_subgraph_submeshes", False)),
            allow_heterogeneous_apipe=bool(payload.get("allow_heterogeneous_apipe", False)),
            allow_morphable_pipeline=bool(payload.get("allow_morphable_pipeline", False)),
            max_tp_size=payload.get("max_tp_size"),
            max_pp_size=payload.get("max_pp_size"),
            max_ep_size=payload.get("max_ep_size"),
            max_cp_size=payload.get("max_cp_size"),
            max_vpp_size=payload.get("max_vpp_size"),
            max_shard_group_size=payload.get("max_shard_group_size"),
            max_replicate_group_size=payload.get("max_replicate_group_size"),
            max_micro_batch_size=payload.get("max_micro_batch_size"),
            max_estimated_memory_pressure=payload.get("max_estimated_memory_pressure"),
            prefer_memory_relief=bool(payload.get("prefer_memory_relief", False)),
            required_node_local_axes=[str(axis) for axis in (payload.get("required_node_local_axes") or [])],
            preferred_node_for_module={
                str(key): str(value) for key, value in (payload.get("preferred_node_for_module") or {}).items()
            },
            forbidden_axes_by_node={
                str(node): [str(axis) for axis in axes]
                for node, axes in (payload.get("forbidden_axes_by_node") or {}).items()
            },
            allowed_schedule_skeletons=[str(item) for item in (payload.get("allowed_schedule_skeletons") or ["fixed_1f1b"])],
            allowed_schedule_templates=[str(item) for item in (payload.get("allowed_schedule_templates") or ["fixed_1f1b"])],
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
    batch_plan: BatchPlanSpec = field(default_factory=BatchPlanSpec)
    strategy_ir: StrategyIRSpec = field(default_factory=StrategyIRSpec)
    schedule_ir: Optional[ScheduleIRSpec] = None
    partition_optimization: Optional[PartitionOptimizationSpec] = None
    applied_patch: Optional[ProgramPatchSpec] = None
    baseline_family: Optional[str] = None
    policy_objective: Optional[str] = None
    layer_groups: List[LayerGroupSpec] = field(default_factory=list)
    schedule_graph_nodes: List[ScheduleNodeSpec] = field(default_factory=list)
    schedule_graph_edges: List[ScheduleEdgeSpec] = field(default_factory=list)
    state_plan: Optional[StatePlanSpec] = None
    vpp_flow: Optional[VPPFlowPolicySpec] = None
    global_strategy_plan: Optional[GlobalStrategyPlanSpec] = None
    rewrite_plan: Optional[RewriteExecutionPlanSpec] = None
    telemetry_budget: Optional[TelemetryBudgetSpec] = None
    window_reconfig: Optional[WindowReconfigSpec] = None
    stage_local_vpp: List[int] = field(default_factory=list)
    overlap_policy: Optional[OverlapIntentSpec] = None
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    search_space: SearchSpaceSpec = field(default_factory=SearchSpaceSpec)
    length_bucket_policies: List[LengthBucketPolicy] = field(default_factory=list)
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
        norm.batch_plan = norm.batch_plan.normalized()
        norm.strategy_ir = norm.strategy_ir.normalized()
        if norm.schedule_ir is not None:
            norm.schedule_ir = norm.schedule_ir.normalized()
        if norm.partition_optimization is not None:
            norm.partition_optimization = norm.partition_optimization.normalized()
        if norm.applied_patch is not None:
            norm.applied_patch = norm.applied_patch.normalized()
        if norm.baseline_family is not None:
            norm.baseline_family = str(norm.baseline_family).strip() or None
        if norm.policy_objective is not None:
            norm.policy_objective = str(norm.policy_objective).strip() or None
        norm.layer_groups = [item.normalized() for item in (norm.layer_groups or [])]
        norm.schedule_graph_nodes = [item.normalized() for item in (norm.schedule_graph_nodes or [])]
        norm.schedule_graph_edges = [item.normalized() for item in (norm.schedule_graph_edges or [])]
        if norm.state_plan is not None:
            norm.state_plan = norm.state_plan.normalized()
        if norm.vpp_flow is not None:
            norm.vpp_flow = norm.vpp_flow.normalized()
        if norm.global_strategy_plan is not None:
            norm.global_strategy_plan = norm.global_strategy_plan.normalized()
        if norm.rewrite_plan is not None:
            norm.rewrite_plan = norm.rewrite_plan.normalized()
        if norm.telemetry_budget is not None:
            norm.telemetry_budget = norm.telemetry_budget.normalized()
        if norm.window_reconfig is not None:
            norm.window_reconfig = norm.window_reconfig.normalized()
        norm.stage_local_vpp = [max(int(item), 1) for item in (norm.stage_local_vpp or [])]
        if norm.overlap_policy is not None:
            norm.overlap_policy = norm.overlap_policy.normalized()
        norm.constraints = norm.constraints.normalized()
        norm.search_space = norm.search_space.normalized()
        norm.length_bucket_policies = [item.normalized() for item in (norm.length_bucket_policies or [])]
        norm.machine_profile = norm.machine_profile.normalized() if norm.machine_profile is not None else None
        norm.backend_caps = norm.backend_caps.normalized() if norm.backend_caps is not None else None
        norm.metadata = copy.deepcopy(norm.metadata or {})
        norm.metadata["micro_batch_size"] = int(norm.batch_plan.micro_batch_size)
        norm.metadata["global_batch_size"] = int(norm.batch_plan.global_batch_size)
        if norm.batch_plan.grad_accum_steps is not None:
            norm.metadata["grad_accum_steps"] = int(norm.batch_plan.grad_accum_steps)
        if norm.batch_plan.target_tokens_per_step is not None:
            norm.metadata["target_tokens_per_step"] = int(norm.batch_plan.target_tokens_per_step)
        if not norm.strategy_ir.apipe or not norm.strategy_ir.placement or not norm.strategy_ir.local_parallel:
            norm.strategy_ir = _derive_strategy_ir(norm)
        derived_schedule_ir = _derive_schedule_ir(norm)
        if norm.schedule_ir is None:
            norm.schedule_ir = derived_schedule_ir
        elif _legacy_schedule_fields_override_schedule_ir(norm, norm.schedule_ir):
            norm.schedule_ir = derived_schedule_ir
        else:
            norm.schedule_ir = norm.schedule_ir.normalized()
        derived_partition_optimization = _derive_partition_optimization(norm)
        if norm.partition_optimization is None:
            norm.partition_optimization = derived_partition_optimization
        elif _legacy_partition_fields_override_partition_optimization(
            norm,
            norm.partition_optimization,
            derived_partition_optimization,
        ):
            norm.partition_optimization = derived_partition_optimization
        else:
            norm.partition_optimization = norm.partition_optimization.normalized()
        if norm.applied_patch is None:
            norm.applied_patch = _derive_program_patch(norm)
        else:
            norm.applied_patch = norm.applied_patch.normalized()
        if norm.baseline_family is None:
            norm.baseline_family = str((norm.metadata or {}).get("baseline_family") or norm.schedule_ir.family or norm.schedule.template)
        if norm.policy_objective is None:
            norm.policy_objective = str(
                (norm.metadata or {}).get("policy_objective") or "maximize_throughput_under_memory_and_legality_constraints"
            )
        if not norm.layer_groups:
            norm.layer_groups = _derive_layer_groups(norm)
        if norm.state_plan is None:
            norm.state_plan = _derive_state_plan(norm)
        if norm.vpp_flow is None:
            norm.vpp_flow = _derive_vpp_flow_policy(norm)
        else:
            requested_vpp = [
                int(item)
                for item in list((norm.metadata or {}).get("stage_local_vpp_vector") or [])
                if int(item) > 0
            ]
            current_vpp = [
                int(item)
                for item in list((norm.vpp_flow.constraints or {}).get("stage_local_vpp_vector") or [])
                if int(item) > 0
            ]
            if requested_vpp and requested_vpp != current_vpp:
                norm.vpp_flow = _derive_vpp_flow_policy(norm)
        if not norm.schedule_graph_nodes or not norm.schedule_graph_edges:
            derived_nodes, derived_edges = _derive_schedule_graph(norm)
            if not norm.schedule_graph_nodes:
                norm.schedule_graph_nodes = derived_nodes
            if not norm.schedule_graph_edges:
                norm.schedule_graph_edges = derived_edges
        if norm.global_strategy_plan is None:
            norm.global_strategy_plan = _derive_global_strategy_plan(norm)
        if norm.telemetry_budget is None:
            norm.telemetry_budget = _derive_telemetry_budget(norm)
        if norm.window_reconfig is None:
            norm.window_reconfig = _derive_window_reconfig(norm)
        if norm.rewrite_plan is None:
            norm.rewrite_plan = _derive_rewrite_plan(norm)
        if not norm.stage_local_vpp:
            norm.stage_local_vpp = list(norm.partition_optimization.stage_local_vpp_vector or [])
        if norm.overlap_policy is None:
            norm.overlap_policy = norm.schedule_ir.overlap_intents.normalized()
        _backfill_legacy_policy_fields(norm)
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
            "batch_plan": norm.batch_plan.to_dict(),
            "strategy_ir": norm.strategy_ir.to_dict(),
            "schedule_ir": norm.schedule_ir.to_dict() if norm.schedule_ir is not None else None,
            "partition_optimization": norm.partition_optimization.to_dict()
            if norm.partition_optimization is not None
            else None,
            "applied_patch": norm.applied_patch.to_dict() if norm.applied_patch is not None else None,
            "baseline_family": norm.baseline_family,
            "policy_objective": norm.policy_objective,
            "layer_groups": [item.to_dict() for item in norm.layer_groups],
            "schedule_graph_nodes": [item.to_dict() for item in norm.schedule_graph_nodes],
            "schedule_graph_edges": [item.to_dict() for item in norm.schedule_graph_edges],
            "state_plan": norm.state_plan.to_dict() if norm.state_plan is not None else None,
            "vpp_flow": norm.vpp_flow.to_dict() if norm.vpp_flow is not None else None,
            "global_strategy_plan": norm.global_strategy_plan.to_dict() if norm.global_strategy_plan is not None else None,
            "rewrite_plan": norm.rewrite_plan.to_dict() if norm.rewrite_plan is not None else None,
            "telemetry_budget": norm.telemetry_budget.to_dict() if norm.telemetry_budget is not None else None,
            "window_reconfig": norm.window_reconfig.to_dict() if norm.window_reconfig is not None else None,
            "stage_local_vpp": list(norm.stage_local_vpp),
            "overlap_policy": norm.overlap_policy.to_dict() if norm.overlap_policy is not None else None,
            "constraints": norm.constraints.to_dict(),
            "search_space": norm.search_space.to_dict(),
            "length_bucket_policies": [item.to_dict() for item in norm.length_bucket_policies],
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
            batch_plan=BatchPlanSpec.from_dict(
                payload.get("batch_plan")
                or {
                    "micro_batch_size": (payload.get("metadata") or {}).get("micro_batch_size", 1),
                    "global_batch_size": (payload.get("metadata") or {}).get("global_batch_size", 16),
                    "grad_accum_steps": (payload.get("metadata") or {}).get("grad_accum_steps"),
                    "target_tokens_per_step": (payload.get("metadata") or {}).get("target_tokens_per_step"),
                }
            ),
            strategy_ir=StrategyIRSpec.from_dict(payload.get("strategy_ir") or {}),
            schedule_ir=ScheduleIRSpec.from_dict(payload.get("schedule_ir") or {})
            if payload.get("schedule_ir") is not None
            else None,
            partition_optimization=PartitionOptimizationSpec.from_dict(payload.get("partition_optimization") or {})
            if payload.get("partition_optimization") is not None
            else None,
            applied_patch=ProgramPatchSpec.from_dict(payload.get("applied_patch") or {})
            if payload.get("applied_patch") is not None
            else None,
            baseline_family=payload.get("baseline_family"),
            policy_objective=payload.get("policy_objective"),
            layer_groups=[LayerGroupSpec.from_dict(item) for item in (payload.get("layer_groups") or [])],
            schedule_graph_nodes=[ScheduleNodeSpec.from_dict(item) for item in (payload.get("schedule_graph_nodes") or [])],
            schedule_graph_edges=[ScheduleEdgeSpec.from_dict(item) for item in (payload.get("schedule_graph_edges") or [])],
            state_plan=StatePlanSpec.from_dict(payload.get("state_plan") or {})
            if payload.get("state_plan") is not None
            else None,
            vpp_flow=VPPFlowPolicySpec.from_dict(payload.get("vpp_flow") or {})
            if payload.get("vpp_flow") is not None
            else None,
            global_strategy_plan=GlobalStrategyPlanSpec.from_dict(payload.get("global_strategy_plan") or {})
            if payload.get("global_strategy_plan") is not None
            else None,
            rewrite_plan=RewriteExecutionPlanSpec.from_dict(payload.get("rewrite_plan") or {})
            if payload.get("rewrite_plan") is not None
            else None,
            telemetry_budget=TelemetryBudgetSpec.from_dict(payload.get("telemetry_budget") or {})
            if payload.get("telemetry_budget") is not None
            else None,
            window_reconfig=WindowReconfigSpec.from_dict(payload.get("window_reconfig") or {})
            if payload.get("window_reconfig") is not None
            else None,
            stage_local_vpp=[int(item) for item in (payload.get("stage_local_vpp") or [])],
            overlap_policy=OverlapIntentSpec.from_dict(payload.get("overlap_policy") or {})
            if payload.get("overlap_policy") is not None
            else None,
            constraints=ConstraintSpec.from_dict(payload.get("constraints") or {}),
            search_space=SearchSpaceSpec.from_dict(payload.get("search_space") or {}),
            length_bucket_policies=[
                LengthBucketPolicy.from_dict(item) for item in (payload.get("length_bucket_policies") or [])
            ],
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


@dataclass
class ProgramTemplate:
    name: str
    run_target: str
    model_track: str
    length_bucket: str = "default"
    bottleneck_tags: List[str] = field(default_factory=list)
    selection_score: Optional[float] = None
    program: MegatronProgram = field(default_factory=MegatronProgram)

    def normalized(self) -> "ProgramTemplate":
        norm = copy.deepcopy(self)
        norm.name = str(norm.name or "template")
        norm.run_target = str(norm.run_target or norm.program.cluster.target or "single_g5")
        norm.model_track = str(norm.model_track or norm.program.model.track or "dense")
        norm.length_bucket = str(norm.length_bucket or "default")
        norm.bottleneck_tags = [str(item) for item in (norm.bottleneck_tags or [])]
        norm.program = norm.program.normalized()
        if norm.selection_score is not None:
            norm.selection_score = float(norm.selection_score)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "run_target": norm.run_target,
            "model_track": norm.model_track,
            "length_bucket": norm.length_bucket,
            "bottleneck_tags": list(norm.bottleneck_tags),
            "selection_score": norm.selection_score,
            "program": norm.program.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProgramTemplate":
        return cls(
            name=str(payload.get("name", "template")),
            run_target=str(payload.get("run_target", "single_g5")),
            model_track=str(payload.get("model_track", "dense")),
            length_bucket=str(payload.get("length_bucket", "default")),
            bottleneck_tags=[str(item) for item in (payload.get("bottleneck_tags") or [])],
            selection_score=payload.get("selection_score"),
            program=MegatronProgram.from_dict(payload.get("program") or {}),
        )


@dataclass
class ProgramBank:
    templates: List[ProgramTemplate] = field(default_factory=list)

    def normalized(self) -> "ProgramBank":
        norm = copy.deepcopy(self)
        norm.templates = [item.normalized() for item in (norm.templates or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {"templates": [item.to_dict() for item in norm.templates]}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProgramBank":
        return cls(templates=[ProgramTemplate.from_dict(item) for item in (payload.get("templates") or [])])


PipePlan = PipeRuntimeSpec
LocalParallelPolicy = LocalParallelSpec


@dataclass
class AgentObservation:
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    backend_context: Dict[str, Any] = field(default_factory=dict)
    model_context: Dict[str, Any] = field(default_factory=dict)
    workload_context: Dict[str, Any] = field(default_factory=dict)
    runtime_evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_record: Dict[str, Any] = field(default_factory=dict)
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    derived_bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    optimization_hints: List[Dict[str, Any]] = field(default_factory=list)
    motivation_evidence_manifest: List[Dict[str, Any]] = field(default_factory=list)

    def normalized(self) -> "AgentObservation":
        norm = copy.deepcopy(self)
        norm.hardware_context = copy.deepcopy(norm.hardware_context or {})
        norm.backend_context = copy.deepcopy(norm.backend_context or {})
        norm.model_context = copy.deepcopy(norm.model_context or {})
        norm.workload_context = copy.deepcopy(norm.workload_context or {})
        norm.runtime_evidence = copy.deepcopy(norm.runtime_evidence or {})
        norm.evidence_record = copy.deepcopy(norm.evidence_record or {})
        norm.failure_modes = [copy.deepcopy(item) for item in (norm.failure_modes or [])]
        norm.derived_bottlenecks = [copy.deepcopy(item) for item in (norm.derived_bottlenecks or [])]
        norm.optimization_hints = [copy.deepcopy(item) for item in (norm.optimization_hints or [])]
        norm.motivation_evidence_manifest = [
            copy.deepcopy(item) for item in (norm.motivation_evidence_manifest or [])
        ]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "hardware_context": norm.hardware_context,
            "backend_context": norm.backend_context,
            "model_context": norm.model_context,
            "workload_context": norm.workload_context,
            "runtime_evidence": norm.runtime_evidence,
            "evidence_record": norm.evidence_record,
            "failure_modes": norm.failure_modes,
            "derived_bottlenecks": norm.derived_bottlenecks,
            "optimization_hints": norm.optimization_hints,
            "motivation_evidence_manifest": norm.motivation_evidence_manifest,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentObservation":
        return cls(
            hardware_context=copy.deepcopy(payload.get("hardware_context") or {}),
            backend_context=copy.deepcopy(payload.get("backend_context") or {}),
            model_context=copy.deepcopy(payload.get("model_context") or {}),
            workload_context=copy.deepcopy(payload.get("workload_context") or {}),
            runtime_evidence=copy.deepcopy(payload.get("runtime_evidence") or {}),
            evidence_record=copy.deepcopy(payload.get("evidence_record") or {}),
            failure_modes=[copy.deepcopy(item) for item in (payload.get("failure_modes") or [])],
            derived_bottlenecks=[copy.deepcopy(item) for item in (payload.get("derived_bottlenecks") or [])],
            optimization_hints=[copy.deepcopy(item) for item in (payload.get("optimization_hints") or [])],
            motivation_evidence_manifest=[
                copy.deepcopy(item) for item in (payload.get("motivation_evidence_manifest") or [])
            ],
        )


@dataclass
class VerifierReport:
    is_legal: bool = True
    decision: str = "allow"
    legality: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    diagnosis: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    switch_cost: float = 0.0
    next_scope_hint: Optional[str] = None
    runtime_risk: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_legal": bool(self.is_legal),
            "decision": str(self.decision or "allow"),
            "legality": copy.deepcopy(self.legality),
            "cost": copy.deepcopy(self.cost),
            "diagnosis": list(self.diagnosis),
            "rejection_reason": self.rejection_reason,
            "switch_cost": float(self.switch_cost),
            "next_scope_hint": self.next_scope_hint,
            "runtime_risk": copy.deepcopy(self.runtime_risk),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerifierReport":
        return cls(
            is_legal=bool(payload.get("is_legal", True)),
            decision=str(payload.get("decision", "allow") or "allow"),
            legality=copy.deepcopy(payload.get("legality") or {}),
            cost=copy.deepcopy(payload.get("cost") or {}),
            diagnosis=[str(item) for item in (payload.get("diagnosis") or [])],
            rejection_reason=payload.get("rejection_reason"),
            switch_cost=float(payload.get("switch_cost", 0.0) or 0.0),
            next_scope_hint=payload.get("next_scope_hint"),
            runtime_risk=copy.deepcopy(payload.get("runtime_risk") or {}),
        )


@dataclass
class AgentProposal:
    proposal_id: str
    scope: str
    program: MegatronProgram = field(default_factory=MegatronProgram)
    rationale: Optional[str] = None
    priority_rank: int = 0
    source: str = "heuristic"
    verifier_report: Optional[Dict[str, Any]] = None

    def normalized(self) -> "AgentProposal":
        norm = copy.deepcopy(self)
        norm.proposal_id = str(norm.proposal_id or "proposal")
        norm.scope = str(norm.scope or "local")
        norm.program = norm.program.normalized()
        norm.rationale = str(norm.rationale).strip() if norm.rationale is not None else None
        norm.priority_rank = int(norm.priority_rank or 0)
        norm.source = str(norm.source or "heuristic")
        norm.verifier_report = copy.deepcopy(norm.verifier_report) if norm.verifier_report is not None else None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "proposal_id": norm.proposal_id,
            "scope": norm.scope,
            "program": norm.program.to_dict(),
            "rationale": norm.rationale,
            "priority_rank": int(norm.priority_rank),
            "source": norm.source,
            "verifier_report": copy.deepcopy(norm.verifier_report),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentProposal":
        return cls(
            proposal_id=str(payload.get("proposal_id", "proposal")),
            scope=str(payload.get("scope", "local")),
            program=MegatronProgram.from_dict(payload.get("program") or {}),
            rationale=payload.get("rationale"),
            priority_rank=int(payload.get("priority_rank", 0) or 0),
            source=str(payload.get("source", "heuristic")),
            verifier_report=copy.deepcopy(payload.get("verifier_report")),
        )


@dataclass
class ReplanDecision:
    trigger: str = "steady"
    scope: str = "none"
    rationale: str = "current context does not require replanning"
    expected_switch_cost: float = 0.0
    fallback_if_rejected: str = "none"
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)

    def normalized(self) -> "ReplanDecision":
        norm = copy.deepcopy(self)
        norm.trigger = str(norm.trigger or "steady")
        norm.scope = str(norm.scope or "none")
        norm.rationale = str(norm.rationale or "current context does not require replanning")
        norm.expected_switch_cost = max(float(norm.expected_switch_cost or 0.0), 0.0)
        norm.fallback_if_rejected = str(norm.fallback_if_rejected or "none")
        norm.failure_modes = [copy.deepcopy(item) for item in (norm.failure_modes or [])]
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "trigger": norm.trigger,
            "scope": norm.scope,
            "rationale": norm.rationale,
            "expected_switch_cost": float(norm.expected_switch_cost),
            "fallback_if_rejected": norm.fallback_if_rejected,
            "failure_modes": norm.failure_modes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReplanDecision":
        return cls(
            trigger=str(payload.get("trigger", "steady")),
            scope=str(payload.get("scope", "none")),
            rationale=str(payload.get("rationale", "current context does not require replanning")),
            expected_switch_cost=float(payload.get("expected_switch_cost", 0.0) or 0.0),
            fallback_if_rejected=str(payload.get("fallback_if_rejected", "none")),
            failure_modes=[copy.deepcopy(item) for item in (payload.get("failure_modes") or [])],
        )


@dataclass
class ExperimentSpec:
    experiment_id: str
    category: str
    label: str
    objective: str
    program_kinds: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def normalized(self) -> "ExperimentSpec":
        norm = copy.deepcopy(self)
        norm.experiment_id = str(norm.experiment_id or "experiment")
        norm.category = str(norm.category or "A")
        norm.label = str(norm.label or norm.experiment_id)
        norm.objective = str(norm.objective or "study")
        norm.program_kinds = [str(item) for item in (norm.program_kinds or [])]
        norm.notes = str(norm.notes).strip() if norm.notes is not None else None
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "experiment_id": norm.experiment_id,
            "category": norm.category,
            "label": norm.label,
            "objective": norm.objective,
            "program_kinds": list(norm.program_kinds),
            "notes": norm.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentSpec":
        return cls(
            experiment_id=str(payload.get("experiment_id", "experiment")),
            category=str(payload.get("category", "A")),
            label=str(payload.get("label", payload.get("experiment_id", "experiment"))),
            objective=str(payload.get("objective", "study")),
            program_kinds=[str(item) for item in (payload.get("program_kinds") or [])],
            notes=payload.get("notes"),
        )


def _derive_stage_semantics(program: MegatronProgram) -> List[StageSemanticSpec]:
    metadata = copy.deepcopy(program.metadata or {})
    stage_tags = list(metadata.get("runtime_stage_tags") or [])
    if stage_tags:
        out: List[StageSemanticSpec] = []
        for item in stage_tags:
            if not isinstance(item, dict):
                continue
            out.append(
                StageSemanticSpec(
                    stage_id=int(item.get("stage_id", item.get("stage_index", 0)) or 0),
                    family=str(item.get("family", item.get("stage_family", "unspecified"))),
                    local_dispatch_hint=item.get("local_dispatch_hint") or item.get("dispatch_hint"),
                    prefer_delayed_wgrad=bool(item.get("prefer_delayed_wgrad", False)),
                    prefer_early_reload=bool(item.get("prefer_early_reload", False)),
                    prefer_checkpoint=bool(item.get("prefer_checkpoint", False)),
                    prefer_offload=bool(item.get("prefer_offload", False)),
                    overlap_aggressiveness=str(item.get("overlap_aggressiveness", "balanced")),
                    notes=item.get("notes"),
                ).normalized()
            )
        if out:
            return out
    out = []
    local_policies = list(metadata.get("stage_local_memory_policy") or [])
    for stage_index, stage in enumerate(program.partition.stages):
        local_policy = local_policies[stage_index] if stage_index < len(local_policies) and isinstance(local_policies[stage_index], dict) else {}
        special_tokens = set(stage.special_tokens or [])
        family = "memory_hot" if bool(local_policy) else "tail_heavy" if stage_index == len(program.partition.stages) - 1 else "unspecified"
        if "E" in special_tokens or "L" in special_tokens:
            family = "comm_hot"
        out.append(
            StageSemanticSpec(
                stage_id=stage_index,
                family=family,
                local_dispatch_hint=local_policy.get("dispatch_hint"),
                prefer_delayed_wgrad=bool(local_policy.get("prefer_delayed_wgrad", False)),
                prefer_early_reload=bool(local_policy.get("prefer_early_reload", False)),
                prefer_checkpoint=bool(local_policy.get("checkpoint", False)),
                prefer_offload=bool(local_policy.get("offload", False)),
                overlap_aggressiveness=str(local_policy.get("overlap_aggressiveness", "balanced")),
                notes=str(local_policy.get("notes") or "") or None,
            ).normalized()
        )
    return out


def _derive_overlap_intents(program: MegatronProgram) -> OverlapIntentSpec:
    metadata = copy.deepcopy(program.metadata or {})
    priority_pairs: List[str] = []
    optimizer_mode = str(metadata.get("runtime_optimizer_policy_mode") or "").strip()
    if optimizer_mode:
        priority_pairs.append(f"optimizer_tail:{optimizer_mode}")
    if metadata.get("runtime_enable_overlap_grad_reduce"):
        priority_pairs.append("backward->grad_reduce")
    if metadata.get("runtime_enable_overlap_param_gather"):
        priority_pairs.append("optimizer_tail->param_gather")
    return OverlapIntentSpec(
        enable_p2p_overlap=bool(metadata.get("runtime_enable_p2p_overlap") or metadata.get("runtime_enable_dualpipe_overlap")),
        enable_grad_reduce_overlap=bool(metadata.get("runtime_enable_overlap_grad_reduce")),
        enable_param_gather_overlap=bool(metadata.get("runtime_enable_overlap_param_gather")),
        enable_tp_comm_overlap=bool(metadata.get("runtime_enable_tp_comm_overlap")),
        enable_optimizer_tail_overlap=bool(optimizer_mode),
        enable_reload_overlap=bool(metadata.get("runtime_enable_reload_overlap")),
        priority_frontier_pairs=priority_pairs,
        status=str(metadata.get("overlap_intent_status") or "direct_now"),
        disabled_reasons=[str(item) for item in (metadata.get("overlap_disabled_reasons") or [])],
    ).normalized()


def _derive_memory_intents(program: MegatronProgram) -> MemoryIntentSpec:
    metadata = copy.deepcopy(program.metadata or {})
    per_stage = list(metadata.get("stage_local_memory_policy") or [])
    return MemoryIntentSpec(
        checkpoint_policy=str(
            metadata.get("schedule_warmup_checkpoint_policy")
            or metadata.get("schedule_steady_checkpoint_policy")
            or metadata.get("checkpoint_policy")
            or "default"
        ),
        recompute_policy=str(
            metadata.get("runtime_recompute_policy")
            or ("selective" if metadata.get("runtime_enable_recompute_activations") else "default")
        ),
        offload_policy=str(
            metadata.get("runtime_offload_policy")
            or ("fine_grained" if metadata.get("runtime_enable_fine_grained_activation_offloading") else "none")
        ),
        reload_policy=str(metadata.get("runtime_reload_policy") or "default"),
        prefetch_policy=str(metadata.get("runtime_prefetch_policy") or "default"),
        per_stage_policies=[copy.deepcopy(item) for item in per_stage if isinstance(item, dict)],
        status=str(metadata.get("memory_intent_status") or "direct_now"),
        notes=str(metadata.get("memory_intent_notes") or "") or None,
    ).normalized()


def _derive_schedule_ir(program: MegatronProgram) -> ScheduleIRSpec:
    metadata = copy.deepcopy(program.metadata or {})
    family = str(metadata.get("runtime_schedule_family") or program.schedule.template or program.schedule.skeleton or "fixed_1f1b")
    stage_local_vpp = list(metadata.get("stage_local_vpp_vector") or [])
    stage_local_vpp_fwd = list(metadata.get("stage_local_vpp_forward_vector") or [])
    stage_local_vpp_bwd = list(metadata.get("stage_local_vpp_backward_vector") or [])
    if not stage_local_vpp and stage_local_vpp_fwd:
        stage_local_vpp = list(stage_local_vpp_fwd)
    runtime_requirements = {
        "execution_backend": str(metadata.get("execution_backend") or metadata.get("planner_backend") or ""),
        "sandbox_only": bool(metadata.get("sandbox_only", False)),
        "metadata_only": bool(metadata.get("metadata_only", False)),
        "supports_window_overrides": bool(metadata.get("runtime_window_overrides")),
        "supports_operator_cluster_overrides": bool(metadata.get("runtime_operator_cluster_overrides")),
    }
    execution_hints = {
        "window_overrides": copy.deepcopy(metadata.get("runtime_window_overrides") or []),
        "operator_cluster_overrides": copy.deepcopy(metadata.get("runtime_operator_cluster_overrides") or []),
        "stage_local_vpp_vector": [int(item) for item in stage_local_vpp],
        "stage_local_vpp_forward_vector": [int(item) for item in stage_local_vpp_fwd],
        "stage_local_vpp_backward_vector": [int(item) for item in stage_local_vpp_bwd],
        "vpp_asymmetry_enabled": bool(
            stage_local_vpp_fwd
            and stage_local_vpp_bwd
            and list(stage_local_vpp_fwd) != list(stage_local_vpp_bwd)
        ),
        "program_kind": str(metadata.get("program_kind") or "program"),
    }
    return ScheduleIRSpec(
        family=family,
        skeleton=str(program.schedule.skeleton or family),
        microbatch_lanes=max(int(program.parallel.vpp_degree), 1),
        microbatch_group_size_per_vp_stage=program.schedule.microbatch_group_size_per_vp_stage,
        dispatch_order=str(program.schedule.dispatch_order or "default"),
        warmup_policy=str(program.strategy_ir.pipe.warmup_policy or metadata.get("schedule_warmup_policy") or "default"),
        steady_state_policy=str(program.strategy_ir.pipe.microbatch_order or "default"),
        cooldown_policy=str(program.strategy_ir.pipe.cooldown_policy or metadata.get("schedule_cooldown_policy") or "default"),
        weight_version_policy=str(metadata.get("weight_version_policy") or "default"),
        virtual_stage_grouping=[int(item) for item in stage_local_vpp if int(item) > 0],
        runtime_requirements=runtime_requirements,
        stage_semantics=_derive_stage_semantics(program),
        overlap_intents=_derive_overlap_intents(program),
        memory_intents=_derive_memory_intents(program),
        execution_hints=execution_hints,
    ).normalized()


def _derive_partition_optimization(program: MegatronProgram) -> PartitionOptimizationSpec:
    metadata = copy.deepcopy(program.metadata or {})
    stage_layer_counts = [int(stage.decoder_layers) for stage in (program.partition.stages or [])]
    stage_local_vpp = list(metadata.get("stage_local_vpp_vector") or [])
    stage_local_vpp_fwd = list(metadata.get("stage_local_vpp_forward_vector") or [])
    stage_local_vpp_bwd = list(metadata.get("stage_local_vpp_backward_vector") or [])
    if not stage_local_vpp and stage_local_vpp_fwd:
        stage_local_vpp = list(stage_local_vpp_fwd)
    if not stage_local_vpp_fwd and stage_local_vpp:
        stage_local_vpp_fwd = list(stage_local_vpp)
    if not stage_local_vpp_bwd and stage_local_vpp:
        stage_local_vpp_bwd = list(stage_local_vpp)
    allow_nonuniform = bool(program.search_space.allow_nonuniform_partition or len(set(stage_layer_counts or [0])) > 1)
    boundary_modules: List[str] = []
    anti_boundary_modules: List[str] = []
    boundary_focus = str(metadata.get("boundary_semantic_focus") or "").strip()
    if boundary_focus:
        boundary_modules.append(boundary_focus)
    if str(program.model.track or "") == "moe":
        boundary_modules.append("experts")
    anti_boundary = str(metadata.get("anti_boundary_module") or "").strip()
    if anti_boundary:
        anti_boundary_modules.append(anti_boundary)
    return PartitionOptimizationSpec(
        partition_mode="nonuniform" if allow_nonuniform else "uniform",
        allow_nonuniform_partition=allow_nonuniform,
        stage_layer_counts=stage_layer_counts,
        stage_local_vpp_vector=[int(item) for item in stage_local_vpp if int(item) > 0],
        stage_local_vpp_forward_vector=[int(item) for item in stage_local_vpp_fwd if int(item) > 0],
        stage_local_vpp_backward_vector=[int(item) for item in stage_local_vpp_bwd if int(item) > 0],
        exposed_cost_weights={
            str(key): float(value)
            for key, value in dict(
                metadata.get("partition_exposed_cost_weights")
                or metadata.get("runtime_partition_exposed_cost_weights")
                or {}
            ).items()
            if str(key).strip()
        },
        preferred_boundary_modules=boundary_modules,
        anti_boundary_modules=anti_boundary_modules,
        asymmetry_notes=[str(item) for item in (metadata.get("partition_asymmetry_notes") or [])],
    ).normalized()


def _estimated_microbatch_count(program: MegatronProgram) -> int:
    batch_plan = program.batch_plan.normalized()
    if batch_plan.grad_accum_steps is not None:
        return max(int(batch_plan.grad_accum_steps), 1)
    micro_batch_size = max(int(batch_plan.micro_batch_size), 1)
    global_batch_size = max(int(batch_plan.global_batch_size), micro_batch_size)
    return max(int(global_batch_size // micro_batch_size), 1)


def _stage_module_family(stage_index: int, stage: StageSpec, program: MegatronProgram) -> str:
    special_tokens = set(stage.special_tokens or [])
    if "E" in special_tokens and stage_index == 0:
        return "embedding"
    if "L" in special_tokens and stage_index == (program.partition.num_stages - 1):
        return "loss"
    return "experts" if str(program.model.track or "") == "moe" else "decoder"


def _derive_layer_groups(program: MegatronProgram) -> List[LayerGroupSpec]:
    norm = copy.deepcopy(program)
    seq_len = int((norm.metadata or {}).get("seq_len", 1024) or 1024)
    groups: List[LayerGroupSpec] = []
    global_layer_start = 0
    for stage_index, stage in enumerate(norm.partition.stages):
        layer_count = max(int(stage.decoder_layers), 0)
        if layer_count <= 0:
            continue
        group_count = 2 if layer_count <= 8 else 3 if layer_count <= 24 else 4
        family = _stage_module_family(stage_index, stage, norm)
        cursor = global_layer_start
        for group_index in range(group_count):
            remaining_layers = layer_count - (cursor - global_layer_start)
            remaining_groups = group_count - group_index
            span = max(int(round(float(remaining_layers) / float(remaining_groups))), 1)
            start = cursor
            end = min(cursor + span - 1, global_layer_start + layer_count - 1)
            actual_span = max(end - start + 1, 1)
            base_time = float(actual_span) * (1.3 if family == "decoder" else 1.0)
            activation_size = float(actual_span) * max(float(seq_len) / 1024.0, 1.0) * 96.0
            parameter_size = float(actual_span) * 128.0
            optimizer_size = parameter_size * 2.0
            groups.append(
                LayerGroupSpec(
                    group_id=f"stage{stage_index:02d}_lg{group_index:02d}",
                    stage_id=stage_index,
                    layer_range=[start, end],
                    module_family=family,
                    fwd_time_ms=round(base_time, 4),
                    bwd_input_time_ms=round(base_time * 1.25, 4),
                    bwd_weight_time_ms=round(base_time * 1.15, 4),
                    activation_size_mb=round(activation_size, 4),
                    parameter_size_mb=round(parameter_size, 4),
                    optimizer_state_size_mb=round(optimizer_size, 4),
                    offload_cost_ms=round(max(activation_size / 6400.0, 0.1), 4),
                    reload_cost_ms=round(max(activation_size / 6000.0, 0.1), 4),
                    comm_boundary_cost_ms=round(max(float(actual_span) * 0.15, 0.05), 4),
                ).normalized()
            )
            cursor = end + 1
        global_layer_start += layer_count
    return groups


def _derive_state_plan(program: MegatronProgram) -> StatePlanSpec:
    norm = copy.deepcopy(program)
    groups = list(norm.layer_groups or [])
    allow_activation_offload = str(norm.schedule_ir.memory_intents.offload_policy or "none").strip().lower() not in {"", "none", "off"}
    enable_parameter_state_schedule = bool((norm.metadata or {}).get("enable_parameter_state_schedule", False))
    enable_optimizer_state_schedule = bool((norm.metadata or {}).get("enable_optimizer_state_schedule", False))
    optimizer_cpu_offload = bool((norm.metadata or {}).get("runtime_enable_optimizer_cpu_offload", False)) or bool(
        (norm.metadata or {}).get("optimizer_cpu_offload", False)
    )
    objects: List[StateObjectSpec] = []
    placements: List[StatePlacementSpec] = []
    offload_budget_mb = 0.0
    for group in groups:
        activation_id = f"{group.group_id}:activation"
        parameter_id = f"{group.group_id}:parameter"
        optimizer_id = f"{group.group_id}:optimizer"
        objects.extend(
            [
                StateObjectSpec(
                    state_id=activation_id,
                    state_type="activation",
                    owner_stage=int(group.stage_id),
                    owner_layer_group=str(group.group_id),
                    size_mb=float(group.activation_size_mb),
                    offloadable=allow_activation_offload,
                    prefetchable=allow_activation_offload,
                ).normalized(),
            ]
        )
        if enable_parameter_state_schedule:
            objects.append(
                StateObjectSpec(
                    state_id=parameter_id,
                    state_type="parameter",
                    owner_stage=int(group.stage_id),
                    owner_layer_group=str(group.group_id),
                    size_mb=float(group.parameter_size_mb),
                    offloadable=False,
                    prefetchable=False,
                ).normalized()
            )
        if enable_optimizer_state_schedule or optimizer_cpu_offload:
            objects.append(
                StateObjectSpec(
                    state_id=optimizer_id,
                    state_type="optimizer",
                    owner_stage=int(group.stage_id),
                    owner_layer_group=str(group.group_id),
                    size_mb=float(group.optimizer_state_size_mb),
                    offloadable=optimizer_cpu_offload,
                    prefetchable=optimizer_cpu_offload,
                ).normalized()
            )
        placements.extend(
            [
                StatePlacementSpec(state_id=activation_id, placement="hbm", ready_for_use=True, valid_from_slot=0).normalized(),
            ]
        )
        if enable_parameter_state_schedule:
            placements.append(
                StatePlacementSpec(state_id=parameter_id, placement="hbm", ready_for_use=True, valid_from_slot=0).normalized()
            )
        if enable_optimizer_state_schedule or optimizer_cpu_offload:
            placements.append(
                StatePlacementSpec(
                    state_id=optimizer_id,
                    placement="host" if optimizer_cpu_offload else "hbm",
                    ready_for_use=not optimizer_cpu_offload,
                    valid_from_slot=0,
                ).normalized()
            )
        if allow_activation_offload:
            offload_budget_mb += float(group.activation_size_mb)
    return StatePlanSpec(
        objects=objects,
        placements=placements,
        offload_budget_mb=round(offload_budget_mb, 4),
        reload_prefetch_window=max(int(norm.schedule_ir.memory_intents.prefetch_policy not in {"none", "off"}), 1),
    ).normalized()


def _derive_vpp_flow_policy(program: MegatronProgram) -> VPPFlowPolicySpec:
    norm = copy.deepcopy(program)
    pp_degree = max(int(norm.parallel.pp_degree or 1), 1)
    stage_local_vpp: List[int] = []
    if norm.partition_optimization is not None:
        stage_local_vpp = [
            max(int(item), 1) for item in list(norm.partition_optimization.stage_local_vpp_vector or [])
        ]
    if not stage_local_vpp:
        stage_local_vpp = [max(int(item), 1) for item in (norm.stage_local_vpp or [])]
    if not stage_local_vpp:
        stage_local_vpp = [max(int(norm.parallel.vpp_degree or 1), 1) for _ in range(pp_degree)]
    if len(stage_local_vpp) < pp_degree:
        stage_local_vpp.extend([stage_local_vpp[-1] if stage_local_vpp else 1] * (pp_degree - len(stage_local_vpp)))

    grouped: Dict[int, List[LayerGroupSpec]] = {}
    for group in list(norm.layer_groups or []):
        grouped.setdefault(int(group.stage_id), []).append(group.normalized())

    max_stage_activation = 0.0
    for groups in grouped.values():
        max_stage_activation = max(max_stage_activation, sum(float(item.activation_size_mb) for item in groups))
    max_stage_activation = max(max_stage_activation, 1.0)

    virtual_chunks: List[VPPFlowVirtualChunkSpec] = []
    for stage_id in range(pp_degree):
        local_degree = max(int(stage_local_vpp[stage_id] if stage_id < len(stage_local_vpp) else stage_local_vpp[-1]), 1)
        buckets: Dict[int, List[LayerGroupSpec]] = {idx: [] for idx in range(local_degree)}
        for index, group in enumerate(grouped.get(stage_id, [])):
            buckets[index % local_degree].append(group)
        for local_vchunk_id, bucket in buckets.items():
            if not bucket:
                continue
            activation_mb = sum(float(item.activation_size_mb) for item in bucket)
            compute_ms = sum(
                float(item.fwd_time_ms) + float(item.bwd_input_time_ms) + float(item.bwd_weight_time_ms)
                for item in bucket
            )
            boundary_comm_ms = sum(float(item.comm_boundary_cost_ms) for item in bucket)
            reload_cost_ms = sum(float(item.reload_cost_ms) for item in bucket)
            offload_cost_ms = sum(float(item.offload_cost_ms) for item in bucket)
            memory_pressure = activation_mb / max_stage_activation
            virtual_chunks.append(
                VPPFlowVirtualChunkSpec(
                    chunk_id=f"s{stage_id}v{local_vchunk_id}",
                    stage_id=stage_id,
                    local_vchunk_id=local_vchunk_id,
                    layer_group_ids=[str(item.group_id) for item in bucket],
                    device_group=f"pp{stage_id}",
                    compute_ms=round(compute_ms, 4),
                    activation_mb=round(activation_mb, 4),
                    boundary_comm_ms=round(boundary_comm_ms, 4),
                    reload_cost_ms=round(reload_cost_ms, 4),
                    offload_cost_ms=round(offload_cost_ms, 4),
                    memory_pressure=round(memory_pressure, 4),
                    priority=0 if stage_id in {0, pp_degree - 1} else 1,
                ).normalized()
            )

    activation_lifecycle: List[VPPFlowActivationPolicySpec] = []
    allow_activation_offload = False
    if norm.schedule_ir is not None:
        allow_activation_offload = (
            str(norm.schedule_ir.memory_intents.offload_policy or "none").strip().lower()
            not in {"", "none", "off"}
        )
    memory_pressure_threshold = float((norm.metadata or {}).get("vpp_flow_memory_pressure_threshold", 0.34) or 0.34)
    for chunk in virtual_chunks:
        is_boundary_stage = int(chunk.stage_id) in {0, pp_degree - 1}
        reload_hidden = float(chunk.reload_cost_ms) <= max(float(chunk.compute_ms) * 0.20, 0.1)
        if is_boundary_stage:
            stable_policy = "resident"
            rationale = "boundary stage is tail/edge sensitive"
        elif allow_activation_offload and float(chunk.memory_pressure) >= memory_pressure_threshold and reload_hidden:
            stable_policy = "offload"
            rationale = "activation pressure is high and reload can fit the compute slack"
        elif float(chunk.memory_pressure) >= memory_pressure_threshold and not reload_hidden:
            stable_policy = "recompute"
            rationale = "reload is likely exposed, prefer recompute over tail stall"
        else:
            stable_policy = "resident"
            rationale = "activation pressure below policy threshold"
        activation_lifecycle.append(
            VPPFlowActivationPolicySpec(
                stage_id=int(chunk.stage_id),
                chunk_id=str(chunk.chunk_id),
                microbatch_class="stable",
                phase="steady",
                policy=stable_policy,
                prefetch_distance=max(int(norm.state_plan.reload_prefetch_window if norm.state_plan is not None else 1), 1),
                reload_deadline_slot=max(int(chunk.local_vchunk_id), 0),
                forbid_tail_reload=False,
                rationale=rationale,
            ).normalized()
        )
        edge_policy = "resident" if stable_policy in {"resident", "offload"} else "recompute"
        activation_lifecycle.append(
            VPPFlowActivationPolicySpec(
                stage_id=int(chunk.stage_id),
                chunk_id=str(chunk.chunk_id),
                microbatch_class="edge",
                phase="cooldown",
                policy=edge_policy,
                prefetch_distance=0,
                reload_deadline_slot=max(int(chunk.local_vchunk_id), 0),
                forbid_tail_reload=True,
                rationale="edge microbatches avoid reloads that cannot be hidden by following work",
            ).normalized()
        )

    flow_credit_policy = [
        VPPFlowCreditPolicySpec(
            resource="h2d",
            capacity=int((norm.metadata or {}).get("vpp_flow_h2d_credits", 1) or 1),
            priority_order=[
                "near_deadline_reload",
                "cross_node_boundary_send",
                "normal_reload",
                "normal_send",
                "d2h_offload",
            ],
            watermark_policy={"high_hbm_pressure_promotes": "d2h_offload"},
            notes="reload misses are treated as critical-path debt",
        ).normalized(),
        VPPFlowCreditPolicySpec(
            resource="d2h",
            capacity=int((norm.metadata or {}).get("vpp_flow_d2h_credits", 1) or 1),
            priority_order=[
                "hbm_watermark_offload",
                "stable_activation_offload",
                "optimizer_state_writeback",
                "background_offload",
            ],
            watermark_policy={"pause_when_h2d_backlog": True},
            notes="D2H work yields to near-deadline H2D reload",
        ).normalized(),
        VPPFlowCreditPolicySpec(
            resource="nic",
            capacity=int((norm.metadata or {}).get("vpp_flow_nic_credits", 1) or 1),
            priority_order=[
                "cross_node_boundary_send",
                "near_deadline_reload",
                "normal_send",
                "grad_reduce",
                "background_offload",
            ],
            watermark_policy={"boundary_chunks_preempt": True},
            notes="boundary activation sends are prioritized over background memory traffic",
        ).normalized(),
    ]

    stage_exposed_ms = max((float(item.compute_ms) for item in virtual_chunks), default=0.0)
    comm_exposed_ms = max((float(item.boundary_comm_ms) for item in virtual_chunks), default=0.0)
    reload_stall_ms = max(
        (
            float(item.reload_cost_ms)
            for item in virtual_chunks
            if any(
                policy.chunk_id == item.chunk_id and policy.policy == "offload"
                for policy in activation_lifecycle
            )
        ),
        default=0.0,
    )
    exposed_cost = VPPFlowExposedCostSpec(
        terms={
            "stage_exposed_ms": round(stage_exposed_ms, 4),
            "comm_exposed_ms": round(comm_exposed_ms, 4),
            "reload_stall_ms": round(reload_stall_ms, 4),
            "copy_stall_ms": round(max((float(item.offload_cost_ms) for item in virtual_chunks), default=0.0), 4),
            "uncovered_bubble_ms": float((norm.metadata or {}).get("uncovered_bubble_ms", 0.0) or 0.0),
            "optimizer_tail_ms": float((norm.metadata or {}).get("optimizer_tail_ms", 0.0) or 0.0),
            "straggler_penalty_ms": float((norm.metadata or {}).get("straggler_penalty_ms", 0.0) or 0.0),
        },
        weights=copy.deepcopy((norm.metadata or {}).get("vpp_flow_exposed_cost_weights") or {}),
    ).normalized()

    return VPPFlowPolicySpec(
        enabled=bool((norm.metadata or {}).get("enable_vpp_flow", True)),
        virtual_chunks=virtual_chunks,
        activation_lifecycle=activation_lifecycle,
        flow_credit_policy=flow_credit_policy,
        exposed_cost=exposed_cost,
        constraints={
            "stage_local_vpp_vector": [int(item) for item in stage_local_vpp[:pp_degree]],
            "allow_activation_offload": bool(allow_activation_offload),
            "memory_pressure_threshold": float(memory_pressure_threshold),
            "runtime_target": str(norm.cluster.target),
        },
        notes=[
            "virtual chunks are the joint unit for VPP partitioning, activation lifetime, and memory-flow priority",
            "edge microbatches use conservative lifecycle policies to avoid cooldown reload exposure",
        ],
    ).normalized()


def _derive_global_strategy_plan(program: MegatronProgram) -> GlobalStrategyPlanSpec:
    norm = copy.deepcopy(program)
    metadata = copy.deepcopy(norm.metadata or {})
    model_track = str(norm.model.track or "dense").strip().lower()
    run_target = str(norm.cluster.target or "single_g5").strip().lower()
    seq_len = int(metadata.get("seq_len", 1024) or 1024)
    activation_pressure = float(sum(float(item.activation_size_mb) for item in (norm.layer_groups or [])))
    memory_headroom_ratio = float(metadata.get("memory_headroom_ratio", 0.0) or 0.0)
    comm_exposure = float(metadata.get("comm_exposure_ratio", 0.0) or 0.0)
    stage_boundaries: List[List[int]] = []
    layer_group_to_stage: Dict[str, int] = {}
    stage_ranges: Dict[int, List[int]] = {}
    for group in (norm.layer_groups or []):
        stage_id = int(group.stage_id)
        layer_group_to_stage[str(group.group_id)] = stage_id
        if len(group.layer_range or []) == 2:
            entry = stage_ranges.setdefault(stage_id, [int(group.layer_range[0]), int(group.layer_range[1])])
            entry[0] = min(int(entry[0]), int(group.layer_range[0]))
            entry[1] = max(int(entry[1]), int(group.layer_range[1]))
    for stage_id in sorted(stage_ranges):
        stage_boundaries.append([int(stage_ranges[stage_id][0]), int(stage_ranges[stage_id][1])])
    activation_offload_enabled_groups = [
        str(item.owner_layer_group)
        for item in list((norm.state_plan.objects if norm.state_plan is not None else []) or [])
        if str(item.state_type) == "activation" and bool(item.offloadable)
    ]
    overlap_enabled_channels: List[str] = []
    if norm.overlap_policy is not None:
        if bool(norm.overlap_policy.enable_p2p_overlap):
            overlap_enabled_channels.append("p2p")
        if bool(norm.overlap_policy.enable_reload_overlap):
            overlap_enabled_channels.append("reload")
        if bool(norm.overlap_policy.enable_tp_comm_overlap):
            overlap_enabled_channels.append("tp_comm")
        if bool(norm.overlap_policy.enable_optimizer_tail_overlap):
            overlap_enabled_channels.append("optimizer_tail")
    primary_parallel_mode = "pp_vpp"
    selection_rationale = [
        f"target={run_target or 'single_g5'}",
        f"model_track={model_track or 'dense'}",
        f"seq_len={seq_len}",
    ]
    if run_target == "single_g5" and model_track == "dense":
        primary_parallel_mode = "pp_vpp"
        selection_rationale.append("single_g5_dense_prefers_pp_vpp")
    elif activation_pressure <= 0.0 and memory_headroom_ratio >= 0.25 and comm_exposure <= 0.06:
        primary_parallel_mode = "fsdp_zero"
        selection_rationale.append("low_activation_pressure_recommend_fsdp_zero")
    else:
        selection_rationale.append("stateful_pp_vpp_default")
    return GlobalStrategyPlanSpec(
        primary_parallel_mode=primary_parallel_mode,
        dp_degree=max(int(norm.cluster.world_size // max(int(norm.parallel.tp_degree * norm.parallel.pp_degree), 1)), 1),
        tp_degree=int(norm.parallel.tp_degree),
        pp_degree=int(norm.parallel.pp_degree),
        vpp_degree=int(norm.parallel.vpp_degree),
        stage_count=max(int(norm.partition.num_stages or norm.parallel.pp_degree), 1),
        stage_boundaries=stage_boundaries,
        layer_group_to_stage=layer_group_to_stage,
        activation_offload_enabled_groups=activation_offload_enabled_groups,
        overlap_enabled_channels=overlap_enabled_channels,
        selection_rationale=selection_rationale,
    ).normalized()


def _derive_rewrite_plan(program: MegatronProgram) -> RewriteExecutionPlanSpec:
    norm = copy.deepcopy(program)
    metadata = copy.deepcopy(norm.metadata or {})
    rewrite_actions: List[RewriteActionSpec] = []
    repair_assessment = dict(metadata.get("counterfactual_repair_assessment") or {})
    for item in list(metadata.get("rewrite_actions") or []):
        if isinstance(item, dict):
            rewrite_actions.append(RewriteActionSpec.from_dict(item).normalized())
    if not rewrite_actions and norm.applied_patch is not None:
        patch_family = str(norm.applied_patch.patch_family or "").strip()
        target_layer_groups = [
            str(item)
            for item in list((norm.applied_patch.expected_effects or {}).get("target_layer_groups") or [])
            if str(item).strip()
        ]
        target_state_objects = [
            str(item)
            for item in list((norm.applied_patch.expected_effects or {}).get("target_state_objects") or [])
            if str(item).strip()
        ]
        target_stage_ids = [
            int(item)
            for item in list((norm.metadata or {}).get("runtime_branch_target_stage_ids") or [])
            if str(item).strip()
        ]
        base_action_kwargs = {
            "bottleneck_match_score": float(repair_assessment.get("bottleneck_match_score") or 0.0),
            "target_compatibility_score": float(repair_assessment.get("target_compatibility_score") or 0.0),
            "rollback_risk": float(repair_assessment.get("rollback_risk") or 0.0),
            "expected_mfu_gain": float(repair_assessment.get("expected_mfu_gain") or metadata.get("expected_gain") or 0.0),
            "memory_safety_margin": float(repair_assessment.get("memory_safety_margin") or 0.0),
            "counterfactual_score": float(repair_assessment.get("counterfactual_score") or 0.0),
            "diagnostic_labels": [str(item) for item in (repair_assessment.get("dominant_diagnostics") or []) if str(item).strip()],
            "strategy_source": str(metadata.get("planner_backend") or metadata.get("planner_mode") or "heuristic"),
            "llm_rationale": metadata.get("llm_repair_rationale") or metadata.get("llm_rationale"),
        }
        if patch_family in {"reload_shift_patch", "activation_reload_shift_patch", "reload_prefetch_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="reload_shift",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("reload_shift_direction") or "hold"),
                    magnitude=float((norm.metadata or {}).get("reload_shift_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"adaptive_chunking_patch", "comm_chunk_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="adaptive_chunking",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("comm_chunk_direction") or "preserve"),
                    magnitude=float((norm.metadata or {}).get("comm_chunk_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"overlap_policy_patch", "enable_reload_overlap"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="overlap_window_switch",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("comm_chunk_direction") or "enable"),
                    magnitude=float((norm.metadata or {}).get("comm_chunk_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"tail_relief_patch", "enable_optimizer_tail_overlap"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="tail_optimizer_relief",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction="enable",
                    magnitude=float((norm.metadata or {}).get("optimizer_tail_relief_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"add_offload_policy", "offload_enable_patch", "activation_offload_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="cpu_offload_scope_switch",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction="expand",
                    magnitude=float((norm.metadata or {}).get("offload_scope_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"optimizer_offload_policy_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="optimizer_offload_policy_rewrite",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("optimizer_offload_policy_direction") or "tail_guarded_streaming"),
                    magnitude=float((norm.metadata or {}).get("optimizer_offload_policy_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"optimizer_state_partition_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="optimizer_state_partition_rewrite",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("optimizer_state_partition_direction") or "hierarchical_decoupled"),
                    magnitude=float((norm.metadata or {}).get("optimizer_state_partition_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"recompute_policy_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="recompute_policy_rewrite",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("runtime_recompute_policy") or "selective"),
                    magnitude=float((norm.metadata or {}).get("recompute_policy_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"change_schedule_family"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="schedule_family_switch",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("runtime_schedule_family") or norm.schedule.template or "preserve"),
                    magnitude=1.0,
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"pp_vpp_partition_patch"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="pp_vpp_partition_rewrite",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("runtime_partition_focus") or "stage_aware_nonuniform_vpp"),
                    magnitude=float((norm.metadata or {}).get("runtime_partition_shift", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
        elif patch_family in {"local_verticalization_patch", "layer_group_repack"}:
            rewrite_actions.append(
                RewriteActionSpec(
                    rewrite_type="local_verticalization",
                    target_stage_ids=target_stage_ids,
                    target_layer_group_ids=target_layer_groups,
                    target_state_ids=target_state_objects,
                    direction=str((norm.metadata or {}).get("local_verticalization_direction") or "enable"),
                    magnitude=float((norm.metadata or {}).get("local_verticalization_magnitude", 1.0) or 1.0),
                    expected_gain=float((norm.metadata or {}).get("expected_gain", 0.0) or 0.0),
                    **base_action_kwargs,
                    risk_flags=[str(item) for item in (norm.applied_patch.risk_flags or []) if str(item).strip()],
                ).normalized()
            )
    return RewriteExecutionPlanSpec(
        global_strategy=norm.global_strategy_plan if norm.global_strategy_plan is not None else _derive_global_strategy_plan(norm),
        rewrite_actions=rewrite_actions,
        telemetry_budget=norm.telemetry_budget if norm.telemetry_budget is not None else _derive_telemetry_budget(norm),
        window_reconfig=norm.window_reconfig if norm.window_reconfig is not None else _derive_window_reconfig(norm),
        version_tag=str(metadata.get("rewrite_plan_version", "v1") or "v1"),
    ).normalized()


def _derive_schedule_graph(program: MegatronProgram) -> tuple[List[ScheduleNodeSpec], List[ScheduleEdgeSpec]]:
    norm = copy.deepcopy(program)
    groups = list(norm.layer_groups or [])
    microbatch_count = _estimated_microbatch_count(norm)
    enable_offload = str(norm.schedule_ir.memory_intents.offload_policy or "none").strip().lower() not in {"", "none", "off"}
    enable_comm = int(norm.parallel.pp_degree) > 1
    enable_parameter_state_schedule = bool((norm.metadata or {}).get("enable_parameter_state_schedule", False))
    enable_optimizer_state_schedule = bool((norm.metadata or {}).get("enable_optimizer_state_schedule", False))
    nodes: List[ScheduleNodeSpec] = []
    edges: List[ScheduleEdgeSpec] = []
    previous_forward_by_key: Dict[tuple[int, int], str] = {}
    previous_backward_by_key: Dict[tuple[int, int], str] = {}
    for group in groups:
        activation_state_id = f"{group.group_id}:activation"
        for microbatch_id in range(microbatch_count):
            fwd_id = f"{group.group_id}:mb{microbatch_id}:fwd"
            bwd_in_id = f"{group.group_id}:mb{microbatch_id}:bwd_in"
            bwd_w_id = f"{group.group_id}:mb{microbatch_id}:bwd_w"
            nodes.append(
                ScheduleNodeSpec(
                    node_id=fwd_id,
                    node_type="forward",
                    stage_id=int(group.stage_id),
                    microbatch_id=microbatch_id,
                    layer_group_id=str(group.group_id),
                    lane_id=0,
                    chunk_id=0,
                    duration_hint_ms=float(group.fwd_time_ms),
                    state_refs=[activation_state_id],
                    resource_class="compute",
                ).normalized()
            )
            nodes.append(
                ScheduleNodeSpec(
                    node_id=bwd_in_id,
                    node_type="backward_input",
                    stage_id=int(group.stage_id),
                    microbatch_id=microbatch_id,
                    layer_group_id=str(group.group_id),
                    lane_id=0,
                    chunk_id=0,
                    duration_hint_ms=float(group.bwd_input_time_ms),
                    state_refs=[activation_state_id],
                    resource_class="compute",
                ).normalized()
            )
            nodes.append(
                ScheduleNodeSpec(
                    node_id=bwd_w_id,
                    node_type="backward_weight",
                    stage_id=int(group.stage_id),
                    microbatch_id=microbatch_id,
                    layer_group_id=str(group.group_id),
                    lane_id=0,
                    chunk_id=0,
                    duration_hint_ms=float(group.bwd_weight_time_ms),
                    state_refs=(
                        ([f"{group.group_id}:parameter"] if enable_parameter_state_schedule else [])
                        + ([f"{group.group_id}:optimizer"] if enable_optimizer_state_schedule else [])
                    ),
                    resource_class="compute",
                ).normalized()
            )
            edges.append(ScheduleEdgeSpec(src=fwd_id, dst=bwd_in_id, edge_type="data_dep", required=True).normalized())
            edges.append(ScheduleEdgeSpec(src=bwd_in_id, dst=bwd_w_id, edge_type="stage_order", required=True).normalized())
            previous_forward = previous_forward_by_key.get((int(group.stage_id), microbatch_id))
            if previous_forward:
                edges.append(
                    ScheduleEdgeSpec(
                        src=previous_forward,
                        dst=fwd_id,
                        edge_type="same_resource_mutex",
                        required=True,
                    ).normalized()
                )
            previous_forward_by_key[(int(group.stage_id), microbatch_id)] = fwd_id
            previous_backward = previous_backward_by_key.get((int(group.stage_id), microbatch_id))
            if previous_backward:
                edges.append(
                    ScheduleEdgeSpec(
                        src=previous_backward,
                        dst=bwd_in_id,
                        edge_type="same_resource_mutex",
                        required=True,
                    ).normalized()
                )
            previous_backward_by_key[(int(group.stage_id), microbatch_id)] = bwd_w_id
            if enable_offload:
                offload_id = f"{group.group_id}:mb{microbatch_id}:offload"
                reload_id = f"{group.group_id}:mb{microbatch_id}:reload"
                nodes.append(
                    ScheduleNodeSpec(
                        node_id=offload_id,
                        node_type="offload",
                        stage_id=int(group.stage_id),
                        microbatch_id=microbatch_id,
                        layer_group_id=str(group.group_id),
                        lane_id=1,
                        chunk_id=0,
                        duration_hint_ms=float(group.offload_cost_ms),
                        state_refs=[activation_state_id],
                        resource_class="memory_io",
                    ).normalized()
                )
                nodes.append(
                    ScheduleNodeSpec(
                        node_id=reload_id,
                        node_type="reload",
                        stage_id=int(group.stage_id),
                        microbatch_id=microbatch_id,
                        layer_group_id=str(group.group_id),
                        lane_id=1,
                        chunk_id=0,
                        duration_hint_ms=float(group.reload_cost_ms),
                        state_refs=[activation_state_id],
                        resource_class="memory_io",
                    ).normalized()
                )
                edges.append(ScheduleEdgeSpec(src=fwd_id, dst=offload_id, edge_type="state_dep", required=False).normalized())
                edges.append(ScheduleEdgeSpec(src=offload_id, dst=reload_id, edge_type="state_dep", required=False).normalized())
                edges.append(ScheduleEdgeSpec(src=reload_id, dst=bwd_in_id, edge_type="reload_before_use", required=False).normalized())
            if enable_comm:
                comm_id = f"{group.group_id}:mb{microbatch_id}:comm"
                nodes.append(
                    ScheduleNodeSpec(
                        node_id=comm_id,
                        node_type="comm_chunk",
                        stage_id=int(group.stage_id),
                        microbatch_id=microbatch_id,
                        layer_group_id=str(group.group_id),
                        lane_id=1,
                        chunk_id=0,
                        duration_hint_ms=float(group.comm_boundary_cost_ms),
                        state_refs=[activation_state_id],
                        resource_class="comm",
                    ).normalized()
                )
                edges.append(ScheduleEdgeSpec(src=fwd_id, dst=comm_id, edge_type="cross_stage_dep", required=False).normalized())
    return nodes, edges


def _derive_telemetry_budget(program: MegatronProgram) -> TelemetryBudgetSpec:
    metadata = copy.deepcopy(program.metadata or {})
    level = str(metadata.get("telemetry_budget_level") or metadata.get("runtime_trace_level") or "summary")
    return TelemetryBudgetSpec(
        level=level,
        max_trace_mb=int(metadata.get("telemetry_max_trace_mb", 128) or 128),
        max_events_per_rank=int(metadata.get("telemetry_max_events_per_rank", 20000) or 20000),
        sampled_windows=int(metadata.get("telemetry_sampled_windows", 2) or 2),
        emit_compare_svg=bool(metadata.get("telemetry_emit_compare_svg", False)),
    ).normalized()


def _derive_window_reconfig(program: MegatronProgram) -> WindowReconfigSpec:
    metadata = copy.deepcopy(program.metadata or {})
    return WindowReconfigSpec(
        window_steps=int(metadata.get("window_steps", 4) or 4),
        allowed_patch_categories=list(metadata.get("window_allowed_patch_categories") or ["schedule", "memory", "overlap", "partition"]),
        rollback_guard_steps=int(metadata.get("rollback_guard_steps", 1) or 1),
        promotion_threshold=float(metadata.get("window_promotion_threshold", 0.03) or 0.03),
        demotion_threshold=float(metadata.get("window_demotion_threshold", 0.05) or 0.05),
    ).normalized()


def _derive_program_patch(program: MegatronProgram) -> ProgramPatchSpec:
    metadata = copy.deepcopy(program.metadata or {})
    patch_payload = metadata.get("applied_patch")
    if isinstance(patch_payload, dict):
        return ProgramPatchSpec.from_dict(patch_payload).normalized()
    program_kind = str(metadata.get("program_kind") or "baseline")
    is_baseline = program_kind in {"baseline", "program"}
    lightweight_hash = _stable_hash(
        {
            "program_kind": program_kind,
            "pp_degree": int(program.parallel.pp_degree),
            "vpp_degree": int(program.parallel.vpp_degree),
            "schedule_template": str(program.schedule.template),
            "dispatch_order": str(program.schedule.dispatch_order),
        },
        "program_patch_v1",
    )
    return ProgramPatchSpec(
        patch_id=str(metadata.get("patch_id") or f"{program_kind}:{lightweight_hash[:8]}"),
        base_program_hash=str(metadata.get("base_program_hash") or ""),
        patch_family=str(metadata.get("patch_family") or ("baseline" if is_baseline else program_kind)),
        target_scope=str(metadata.get("patch_scope") or "program"),
        changes=copy.deepcopy(metadata.get("patch_changes") or {}),
        expected_effects=copy.deepcopy(metadata.get("patch_expected_effects") or {}),
        risk_flags=[str(item) for item in (metadata.get("patch_risk_flags") or [])],
        derived_program_hash=str(metadata.get("derived_program_hash") or "") or None,
    ).normalized()


def _schedule_ir_diverged(current: ScheduleIRSpec, derived: ScheduleIRSpec) -> bool:
    lhs = current.normalized().to_dict()
    rhs = derived.normalized().to_dict()
    keys = (
        "family",
        "skeleton",
        "microbatch_lanes",
        "microbatch_group_size_per_vp_stage",
        "dispatch_order",
        "warmup_policy",
        "steady_state_policy",
        "cooldown_policy",
        "weight_version_policy",
        "virtual_stage_grouping",
        "stage_semantics",
        "overlap_intents",
        "memory_intents",
        "execution_hints",
    )
    return any(copy.deepcopy(lhs.get(key)) != copy.deepcopy(rhs.get(key)) for key in keys)


def _partition_optimization_diverged(
    current: PartitionOptimizationSpec,
    derived: PartitionOptimizationSpec,
) -> bool:
    lhs = current.normalized().to_dict()
    rhs = derived.normalized().to_dict()
    return lhs != rhs


def _legacy_schedule_fields_override_schedule_ir(
    program: MegatronProgram,
    current: ScheduleIRSpec,
) -> bool:
    current_norm = current.normalized()
    schedule = program.schedule.normalized()
    pipe = program.strategy_ir.pipe.normalized()
    override_signals = 0
    if str(schedule.dispatch_order or "").strip() not in {"", "default"} and str(schedule.dispatch_order) != str(current_norm.dispatch_order):
        override_signals += 1
    if str(schedule.template or "").strip() not in {"", "fixed_1f1b"} and str(schedule.template) != str(current_norm.family):
        override_signals += 1
    if int(schedule.microbatch_group_size_per_vp_stage or 1) != int(current_norm.microbatch_group_size_per_vp_stage or 1):
        if int(schedule.microbatch_group_size_per_vp_stage or 1) > 1:
            override_signals += 1
    if str(pipe.warmup_policy or "").strip() not in {"", "default"} and str(pipe.warmup_policy) != str(current_norm.warmup_policy):
        override_signals += 1
    if str(pipe.cooldown_policy or "").strip() not in {"", "default"} and str(pipe.cooldown_policy) != str(current_norm.cooldown_policy):
        override_signals += 1
    return bool(override_signals > 0)


def _legacy_partition_fields_override_partition_optimization(
    program: MegatronProgram,
    current: PartitionOptimizationSpec,
    derived: PartitionOptimizationSpec,
) -> bool:
    current_norm = current.normalized()
    metadata = copy.deepcopy(program.metadata or {})
    stage_layer_counts = [int(stage.decoder_layers) for stage in (program.partition.stages or [])]
    stage_local_vpp_vector = [int(item) for item in list(metadata.get("stage_local_vpp_vector") or []) if int(item) > 0]
    stage_local_vpp_forward_vector = [
        int(item) for item in list(metadata.get("stage_local_vpp_forward_vector") or []) if int(item) > 0
    ]
    stage_local_vpp_backward_vector = [
        int(item) for item in list(metadata.get("stage_local_vpp_backward_vector") or []) if int(item) > 0
    ]
    partition_exposed_cost_weights = {
        str(key): float(value)
        for key, value in dict(
            metadata.get("partition_exposed_cost_weights")
            or metadata.get("runtime_partition_exposed_cost_weights")
            or {}
        ).items()
        if str(key).strip()
    }
    override_signals = 0
    if (
        stage_local_vpp_vector
        and stage_local_vpp_vector != list(current_norm.stage_local_vpp_vector or [])
    ):
        override_signals += 1
    if (
        stage_local_vpp_forward_vector
        and stage_local_vpp_forward_vector != list(current_norm.stage_local_vpp_forward_vector or [])
    ):
        override_signals += 1
    if (
        stage_local_vpp_backward_vector
        and stage_local_vpp_backward_vector != list(current_norm.stage_local_vpp_backward_vector or [])
    ):
        override_signals += 1
    if (
        partition_exposed_cost_weights
        and partition_exposed_cost_weights != dict(current_norm.exposed_cost_weights or {})
    ):
        override_signals += 1
    if len(stage_layer_counts) == len(current_norm.stage_layer_counts or []) and len(stage_layer_counts) > 1:
        if stage_layer_counts != list(current_norm.stage_layer_counts or []) and len(set(stage_layer_counts)) > 1:
            override_signals += 1
    if bool(program.search_space.allow_nonuniform_partition) and not bool(current_norm.allow_nonuniform_partition):
        override_signals += 1
    if (
        derived.normalized().to_dict() == current_norm.to_dict()
        and override_signals <= 0
    ):
        return False
    return bool(override_signals > 0)


def _backfill_legacy_policy_fields(program: MegatronProgram) -> None:
    schedule_ir = program.schedule_ir.normalized() if program.schedule_ir is not None else _derive_schedule_ir(program)
    partition_optimization = (
        program.partition_optimization.normalized()
        if program.partition_optimization is not None
        else _derive_partition_optimization(program)
    )
    program.schedule.skeleton = str(schedule_ir.skeleton or program.schedule.skeleton or "fixed_1f1b")
    if not str(program.schedule.template or "").strip() or program.schedule.template == "fixed_1f1b":
        program.schedule.template = str(schedule_ir.family or program.schedule.template or "fixed_1f1b")
    program.schedule.dispatch_order = str(schedule_ir.dispatch_order or program.schedule.dispatch_order or "default")
    if schedule_ir.microbatch_group_size_per_vp_stage is not None:
        program.schedule.microbatch_group_size_per_vp_stage = int(schedule_ir.microbatch_group_size_per_vp_stage)
    program.strategy_ir.pipe.template = str(schedule_ir.family or program.strategy_ir.pipe.template or "fixed_1f1b")
    program.strategy_ir.pipe.microbatch_order = str(schedule_ir.dispatch_order or program.strategy_ir.pipe.microbatch_order or "default")
    program.strategy_ir.pipe.steady_state_group_size = schedule_ir.microbatch_group_size_per_vp_stage
    program.strategy_ir.pipe.warmup_policy = str(schedule_ir.warmup_policy or program.strategy_ir.pipe.warmup_policy or "default")
    program.strategy_ir.pipe.cooldown_policy = str(schedule_ir.cooldown_policy or program.strategy_ir.pipe.cooldown_policy or "default")
    if partition_optimization.stage_local_vpp_vector:
        program.metadata["stage_local_vpp_vector"] = list(partition_optimization.stage_local_vpp_vector)
    if partition_optimization.stage_local_vpp_forward_vector:
        program.metadata["stage_local_vpp_forward_vector"] = list(partition_optimization.stage_local_vpp_forward_vector)
    if partition_optimization.stage_local_vpp_backward_vector:
        program.metadata["stage_local_vpp_backward_vector"] = list(partition_optimization.stage_local_vpp_backward_vector)
    if partition_optimization.exposed_cost_weights:
        program.metadata["partition_exposed_cost_weights"] = copy.deepcopy(partition_optimization.exposed_cost_weights)
    if partition_optimization.preferred_boundary_modules:
        program.metadata["boundary_semantic_focus"] = str(partition_optimization.preferred_boundary_modules[0])
    if partition_optimization.anti_boundary_modules:
        program.metadata["anti_boundary_module"] = str(partition_optimization.anti_boundary_modules[0])
    if partition_optimization.asymmetry_notes:
        program.metadata["partition_asymmetry_notes"] = list(partition_optimization.asymmetry_notes)
    if partition_optimization.partition_mode == "nonuniform":
        program.search_space.allow_nonuniform_partition = True
    stage_layer_counts = list(partition_optimization.stage_layer_counts or [])
    if stage_layer_counts and len(stage_layer_counts) == len(program.partition.stages or []):
        for stage, decoder_layers in zip(program.partition.stages, stage_layer_counts):
            stage.decoder_layers = int(decoder_layers)
    program.metadata["baseline_family"] = str(program.baseline_family or schedule_ir.family)
    program.metadata["policy_objective"] = str(
        program.policy_objective or "maximize_throughput_under_memory_and_legality_constraints"
    )
    if program.applied_patch is not None:
        program.metadata["applied_patch"] = program.applied_patch.to_dict()


def _derive_strategy_ir(program: MegatronProgram) -> StrategyIRSpec:
    norm = copy.deepcopy(program)
    strategy_ir = norm.strategy_ir.normalized() if hasattr(norm, "strategy_ir") else StrategyIRSpec()
    if (
        strategy_ir.apipe
        and strategy_ir.placement
        and strategy_ir.local_parallel
        and strategy_ir.morphable_pipe.units
    ):
        return strategy_ir.normalized()

    total_layers = max(int(norm.model.num_layers), 1)
    logical_group_size = max(int(norm.cluster.world_size) // max(int(norm.parallel.pp_degree), 1), 1)
    attention_boundary = max(total_layers // 3, 1)
    decoder_cursor = 0
    apipe: List[SubgraphSpec] = []
    placement: List[PlacementEntrySpec] = []
    local_parallel: List[LocalParallelSpec] = []
    morphable_units: List[MorphableUnitSpec] = []
    structure_edges: List[MorphableEdgeSpec] = []
    memory_edges: List[MorphableEdgeSpec] = []
    communication_edges: List[MorphableEdgeSpec] = []
    stage_families: List[MorphableStageFamilySpec] = []
    stage_device_counts = list(norm.layout.stage_device_counts or [])
    if len(stage_device_counts) < len(norm.partition.stages):
        stage_device_counts.extend([logical_group_size] * (len(norm.partition.stages) - len(stage_device_counts)))

    for stage_index, stage in enumerate(norm.partition.stages):
        decoder_layers = max(int(stage.decoder_layers), 0)
        decoder_start = decoder_cursor
        decoder_end = decoder_cursor + max(decoder_layers - 1, 0)
        decoder_cursor += decoder_layers
        has_embedding = "E" in set(stage.special_tokens or [])
        has_loss = "L" in set(stage.special_tokens or [])
        if has_embedding and has_loss:
            module_family = "embedding_decoder_loss"
            semantic_role = "embedding_loss_anchor"
        elif has_embedding:
            module_family = "embedding_decoder"
            semantic_role = "embedding_anchor"
        elif has_loss:
            module_family = "decoder_loss"
            semantic_role = "loss_anchor"
        else:
            module_family = "decoder"
            semantic_role = "decoder"
        stage_name = f"subg_stage_{stage_index}"
        attention_heavy = decoder_layers > 0 and decoder_start <= attention_boundary
        compute_weight = float(decoder_layers + (2 if has_embedding or has_loss else 0))
        memory_weight = float(decoder_layers + (1 if has_embedding or has_loss else 0))
        communication_weight = float(1 + (1 if stage_index in {0, len(norm.partition.stages) - 1} else 0))
        apipe.append(
            SubgraphSpec(
                name=stage_name,
                stage_index=stage_index,
                decoder_start=decoder_start,
                decoder_end=decoder_end,
                module_family=module_family,
                special_tokens=list(stage.special_tokens),
                attention_heavy=attention_heavy,
                loss_heavy=has_loss,
            )
        )
        stage_units: List[MorphableUnitSpec] = []
        if has_embedding:
            stage_units.append(
                MorphableUnitSpec(
                    name=f"{stage_name}.embedding_anchor",
                    semantic_role="embedding_anchor",
                    atom_kind="embedding_anchor",
                    parent_subgraph=stage_name,
                    stage_index=stage_index,
                    decoder_start=decoder_start,
                    decoder_end=decoder_start,
                    compute_weight=2.0,
                    memory_weight=1.5,
                    communication_weight=1.0,
                    boundary_cost=1.0,
                    liveness_weight=1.0,
                    special_tokens=list(stage.special_tokens),
                )
            )
        if decoder_layers > 0:
            attn_weight = max(float(decoder_layers) * 0.45, 1.0)
            mlp_weight = max(float(decoder_layers) * 0.40, 1.0)
            residual_weight = max(float(decoder_layers) * 0.15, 0.5)
            stage_units.extend(
                [
                    MorphableUnitSpec(
                        name=f"{stage_name}.attn_block",
                        semantic_role="attention_block",
                        atom_kind="attention_core",
                        parent_subgraph=stage_name,
                        stage_index=stage_index,
                        decoder_start=decoder_start,
                        decoder_end=decoder_end,
                        compute_weight=attn_weight,
                        memory_weight=max(attn_weight * 0.8, 0.5),
                        communication_weight=max(attn_weight * 0.12, 0.2),
                        boundary_cost=max(attn_weight * 0.10, 0.2),
                        liveness_weight=max(attn_weight * 0.9, 0.5),
                        special_tokens=list(stage.special_tokens),
                    ),
                    MorphableUnitSpec(
                        name=f"{stage_name}.mlp_block",
                        semantic_role="mlp_block",
                        atom_kind="mlp_core",
                        parent_subgraph=stage_name,
                        stage_index=stage_index,
                        decoder_start=decoder_start,
                        decoder_end=decoder_end,
                        compute_weight=mlp_weight,
                        memory_weight=max(mlp_weight * 0.85, 0.5),
                        communication_weight=max(mlp_weight * 0.06, 0.1),
                        boundary_cost=max(mlp_weight * 0.08, 0.1),
                        liveness_weight=max(mlp_weight * 0.95, 0.5),
                        special_tokens=list(stage.special_tokens),
                    ),
                    MorphableUnitSpec(
                        name=f"{stage_name}.residual_merge",
                        semantic_role="residual_merge",
                        atom_kind="residual_norm",
                        parent_subgraph=stage_name,
                        stage_index=stage_index,
                        decoder_start=decoder_start,
                        decoder_end=decoder_end,
                        compute_weight=residual_weight,
                        memory_weight=max(residual_weight * 0.75, 0.2),
                        communication_weight=max(residual_weight * 0.04, 0.05),
                        boundary_cost=max(residual_weight * 0.06, 0.05),
                        liveness_weight=max(residual_weight * 0.6, 0.2),
                        special_tokens=list(stage.special_tokens),
                    ),
                ]
            )
        if has_loss:
            stage_units.append(
                MorphableUnitSpec(
                    name=f"{stage_name}.loss_anchor",
                    semantic_role="loss_anchor",
                    atom_kind="loss_anchor",
                    parent_subgraph=stage_name,
                    stage_index=stage_index,
                    decoder_start=max(decoder_end, decoder_start),
                    decoder_end=max(decoder_end, decoder_start),
                    compute_weight=2.0,
                    memory_weight=1.5,
                    communication_weight=1.0,
                    boundary_cost=1.0,
                    liveness_weight=0.6,
                    special_tokens=list(stage.special_tokens),
                )
            )
        morphable_units.extend(stage_units)
        if stage_index > 0:
            prev_stage_name = f"subg_stage_{stage_index - 1}"
            structure_edges.append(
                MorphableEdgeSpec(
                    src=prev_stage_name,
                    dst=stage_name,
                    semantic="structure",
                    criticality=1.0,
                    cost=float(compute_weight),
                )
            )
            memory_edges.append(
                MorphableEdgeSpec(
                    src=prev_stage_name,
                    dst=stage_name,
                    semantic="memory",
                    criticality=max(memory_weight, 1.0),
                    cost=float(memory_weight),
                )
            )
            communication_edges.append(
                MorphableEdgeSpec(
                    src=prev_stage_name,
                    dst=stage_name,
                    semantic="communication",
                    criticality=max(communication_weight, 1.0),
                    cost=float(communication_weight),
                )
            )
        for local_index in range(len(stage_units) - 1):
            src_unit = stage_units[local_index]
            dst_unit = stage_units[local_index + 1]
            structure_edges.append(
                MorphableEdgeSpec(
                    src=src_unit.name,
                    dst=dst_unit.name,
                    semantic="structure",
                    criticality=max(float(dst_unit.compute_weight), 0.5),
                    cost=max(float(dst_unit.boundary_cost), 0.05),
                )
            )
            memory_edges.append(
                MorphableEdgeSpec(
                    src=src_unit.name,
                    dst=dst_unit.name,
                    semantic="memory",
                    criticality=max(float(dst_unit.liveness_weight), 0.2),
                    cost=max(float(dst_unit.memory_weight), 0.2),
                )
            )
            communication_edges.append(
                MorphableEdgeSpec(
                    src=src_unit.name,
                    dst=dst_unit.name,
                    semantic="communication",
                    criticality=max(float(dst_unit.communication_weight), 0.05),
                    cost=max(float(dst_unit.boundary_cost), 0.05),
                )
            )
        nodes = [str(norm.layout.stage_to_node[stage_index])] if stage_index < len(norm.layout.stage_to_node) else []
        placement.append(
            PlacementEntrySpec(
                subgraph=stage_name,
                nodes=nodes,
                device_group_size=max(int(stage_device_counts[stage_index]), 1),
                device_type=str(norm.machine_profile.device_class if norm.machine_profile is not None else "gpu"),
                topology_domain="intra_node" if len(set(nodes)) <= 1 else "cross_node_ib",
            )
        )
        local_parallel.append(
            LocalParallelSpec(
                subgraph=stage_name,
                vpp_degree=max(int(norm.parallel.vpp_degree), 1),
                cp_degree=max(int(norm.parallel.cp_degree), 1),
                fsdp_scope=str((norm.metadata or {}).get("local_fsdp_scope") or "none"),
                device_group_size=max(int(stage_device_counts[stage_index]), 1),
                device_group_type=str(norm.machine_profile.device_class if norm.machine_profile is not None else "gpu"),
            )
        )
        family = "balanced_interleave"
        dispatch_order = "default"
        warmup_policy = "default"
        cooldown_policy = "default"
        checkpoint_policy = None
        p2p_policy = None
        combined_policy = None
        recompute_modules: List[str] = []
        offload_modules: List[str] = []
        chunk_priority_hints: List[int] = []
        if semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"}:
            family = "critical_path_first"
            dispatch_order = "structure_aware_critical_first"
            warmup_policy = "balanced_fill"
            cooldown_policy = "opt_prioritized"
            chunk_priority_hints = [2] * max(int(norm.parallel.vpp_degree), 1)
        elif attention_heavy:
            family = "comm_guarded"
            dispatch_order = "balanced_round_robin"
            warmup_policy = "balanced_fill"
            cooldown_policy = "tail_min"
            p2p_policy = "serial"
            combined_policy = "serial"
            recompute_modules = ["core_attn"]
        elif decoder_layers >= max(total_layers // max(len(norm.partition.stages), 1), 1) + 1:
            family = "memory_guarded"
            dispatch_order = "middle_stage_relief"
            warmup_policy = "balanced_fill"
            cooldown_policy = "tail_min"
            checkpoint_policy = "selective"
            combined_policy = "serial"
            recompute_modules = ["core_attn", "mlp"]
        stage_families.append(
            MorphableStageFamilySpec(
                stage_index=stage_index,
                family=family,
                semantic_role=semantic_role,
                preferred_template=str(norm.schedule.template),
                dispatch_order=dispatch_order,
                warmup_policy=warmup_policy,
                cooldown_policy=cooldown_policy,
                checkpoint_policy=checkpoint_policy,
                p2p_policy=p2p_policy,
                combined_policy=combined_policy,
                recompute_modules=recompute_modules,
                offload_modules=offload_modules,
                chunk_priority_hints=chunk_priority_hints,
            )
        )

    strategy_ir = StrategyIRSpec(
        apipe=apipe,
        placement=placement,
        local_parallel=local_parallel,
        pipe=PipeRuntimeSpec(
            template=str(norm.schedule.template),
            microbatch_order=str(norm.schedule.dispatch_order),
            steady_state_group_size=norm.schedule.microbatch_group_size_per_vp_stage,
            warmup_policy="default",
            cooldown_policy="default",
        ),
        morphable_pipe=MorphablePipelineSpec(
            units=morphable_units,
            structure_edges=structure_edges,
            memory_edges=memory_edges,
            communication_edges=communication_edges,
            stage_families=stage_families,
            chunk_shape_vector=[max(int(norm.parallel.vpp_degree), 1) for _ in norm.partition.stages],
            legality_guards={
                "memory_budget_gb": float(norm.constraints.memory_budget_gb or norm.cluster.device_memory_gb or 0.0),
                "required_node_local_axes": list(norm.constraints.required_node_local_axes or []),
                "pp_degree": int(norm.parallel.pp_degree),
                "vpp_degree": int(norm.parallel.vpp_degree),
            },
            shape_signature=_stable_hash(
                {
                    "partition": norm.partition.to_dict(),
                    "layout": norm.layout.to_dict(),
                    "parallel": {
                        "pp_degree": int(norm.parallel.pp_degree),
                        "vpp_degree": int(norm.parallel.vpp_degree),
                    },
                    "families": [item.to_dict() for item in stage_families],
                },
                "morphable_pipeline_shape_v1",
            ),
        ),
    )
    return strategy_ir.normalized()


def _target_node_speed_map(target: str, nodes: List[str]) -> Dict[str, float]:
    if str(target) == "dual_g4_g5":
        return {str(node): (1.18 if str(node) == "g5" else 1.0) for node in nodes}
    return {str(node): 1.0 for node in nodes}


def default_grouped_stage_to_node(target: str, pp_degree: int) -> List[str]:
    target_name = str(target)
    degree = max(int(pp_degree), 1)
    if target_name == "dual_g4_g5":
        first_count = max(degree // 2, 1)
        return ["g4"] * first_count + ["g5"] * max(degree - first_count, 0)
    if target_name == "dual_g5_g5":
        first_count = max(degree // 2, 1)
        return ["g5_0"] * first_count + ["g5_1"] * max(degree - first_count, 0)
    node_name = "g4" if target_name == "single_g4" else "g5"
    return [node_name] * degree


def weighted_stage_layer_allocation(target: str, num_layers: int, stage_to_node: List[str]) -> List[int]:
    total_layers = max(int(num_layers), 0)
    stages = [str(node) for node in (stage_to_node or [])]
    if total_layers <= 0 or not stages:
        return []
    if total_layers < len(stages):
        layers = [1] * total_layers + [0] * (len(stages) - total_layers)
        return layers

    speed_map = _target_node_speed_map(str(target), stages)
    weights: List[float] = []
    last_index = len(stages) - 1
    for index, node in enumerate(stages):
        weight = max(float(speed_map.get(node, 1.0)), 0.1)
        if index == 0:
            weight *= 0.82
        elif index == last_index:
            weight *= 0.95
        elif len(stages) > 3 and index in {1, last_index - 1}:
            weight *= 1.04
        weights.append(weight)

    base = [1] * len(stages)
    remaining = total_layers - len(stages)
    if remaining <= 0:
        return base

    total_weight = max(sum(weights), 1e-6)
    fractional = [remaining * (weight / total_weight) for weight in weights]
    allocated = [int(value) for value in fractional]
    residual = remaining - sum(allocated)
    order = sorted(
        range(len(fractional)),
        key=lambda index: (fractional[index] - allocated[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:residual]:
        allocated[index] += 1
    return [base[index] + allocated[index] for index in range(len(stages))]


def default_cluster_spec(target: str) -> ClusterSpec:
    if str(target) == "dual_g5_g5":
        return ClusterSpec(target="dual_g5_g5", nodes=["g5_0", "g5_1"], gpus_per_node=8)
    if str(target) == "dual_g4_g5":
        return ClusterSpec(target="dual_g4_g5", nodes=["g4", "g5"], gpus_per_node=8)
    if str(target) == "single_g4":
        return ClusterSpec(target="single_g4", nodes=["g4"], gpus_per_node=8)
    return ClusterSpec(target="single_g5", nodes=["g5"], gpus_per_node=8)


def default_machine_profile(target: str) -> MachineProfile:
    if str(target) == "dual_g5_g5":
        return MachineProfile(
            name="dual_node_5090d_pair",
            device_class="consumer_gpu",
            device_memory_gb=32,
            interconnect_class="cross_node_consumer",
            communication_sensitivity="high",
            prefer_small_tp=True,
            prefer_pp_for_scaling=True,
            supports_te_path=True,
            notes="homogeneous dual-node 5090D profile with node-local TP preference and grouped PP scaling",
        )
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


def default_length_bucket_policies() -> List[LengthBucketPolicy]:
    return [
        LengthBucketPolicy(
            name="short",
            min_seq_len=1,
            max_seq_len=1024,
            preferred_program_kind="pp_first",
            cp_cap=1,
            micro_batch_cap=2,
            schedule_templates=["fixed_1f1b", "interleaved_grouped_g2"],
        ),
        LengthBucketPolicy(
            name="mid",
            min_seq_len=1025,
            max_seq_len=2048,
            preferred_program_kind="pp_vpp",
            cp_cap=1,
            micro_batch_cap=1,
            schedule_templates=["interleaved_grouped_g2", "pp4_frontload"],
        ),
        LengthBucketPolicy(
            name="long",
            min_seq_len=2049,
            max_seq_len=4096,
            preferred_program_kind="pp_vpp_cp",
            cp_cap=2,
            micro_batch_cap=1,
            schedule_templates=["interleaved_grouped_g4", "pp4_middle_relief"],
        ),
        LengthBucketPolicy(
            name="xlong",
            min_seq_len=4097,
            max_seq_len=None,
            preferred_program_kind="memory_relief",
            cp_cap=2,
            micro_batch_cap=1,
            schedule_templates=["pp4_middle_relief"],
        ),
    ]


def default_dense_program(target: str = "single_g5") -> MegatronProgram:
    cluster = default_cluster_spec(target)
    parallel = MegatronParallelSpec(tp_degree=2, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=False)
    memory_budget = float(default_machine_profile(target).device_memory_gb or 24)
    batch_plan = BatchPlanSpec(
        micro_batch_size=1,
        global_batch_size=16,
        grad_accum_steps=8,
        target_tokens_per_step=16 * 1024,
    )
    partition = PartitionSpec(stages=[PartitionStageSpec(decoder_layers=40, special_tokens=["E", "L"])])
    layout = LayoutSpec(stage_to_node=[cluster.nodes[-1]], vpp_degree=1)
    if target == "dual_g5_g5":
        parallel = MegatronParallelSpec(tp_degree=4, pp_degree=4, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=False)
        stage_to_node = default_grouped_stage_to_node(target, parallel.pp_degree)
        stage_layers = weighted_stage_layer_allocation(target, 40, stage_to_node)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=stage_layers[index], special_tokens=(["E"] if index == 0 else []) + (["L"] if index == len(stage_layers) - 1 else []))
                for index in range(len(stage_layers))
            ]
        )
        layout = LayoutSpec(stage_to_node=stage_to_node, vpp_degree=1)
    if target == "single_g5":
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=20, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=20, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g5", "g5"], vpp_degree=1)
    if target == "single_g4":
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=20, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=20, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g4"], vpp_degree=1)
    if target == "dual_g4_g5":
        parallel = MegatronParallelSpec(tp_degree=4, pp_degree=4, vpp_degree=1, cp_degree=1, ep_degree=1, sp_enabled=False)
        stage_to_node = default_grouped_stage_to_node(target, parallel.pp_degree)
        stage_layers = weighted_stage_layer_allocation(target, 40, stage_to_node)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=stage_layers[index], special_tokens=(["E"] if index == 0 else []) + (["L"] if index == len(stage_layers) - 1 else []))
                for index in range(len(stage_layers))
            ]
        )
        layout = LayoutSpec(stage_to_node=stage_to_node, vpp_degree=1)
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
        schedule=ScheduleSpec(template="fixed_1f1b"),
        batch_plan=batch_plan,
        constraints=ConstraintSpec(required_node_local_axes=["tp"], memory_budget_gb=memory_budget),
        search_space=SearchSpaceSpec(),
        length_bucket_policies=default_length_bucket_policies(),
        machine_profile=default_machine_profile(target),
        backend_caps=default_backend_caps("local"),
        metadata={"program_kind": "baseline_dense"},
    )


def default_moe_smoke_program(target: str = "single_g5") -> MegatronProgram:
    cluster = default_cluster_spec(target)
    parallel = MegatronParallelSpec(tp_degree=1, pp_degree=2, vpp_degree=1, cp_degree=1, ep_degree=2, expert_tp_degree=1, sp_enabled=False)
    memory_budget = float(default_machine_profile(target).device_memory_gb or 24)
    batch_plan = BatchPlanSpec(
        micro_batch_size=1,
        global_batch_size=8,
        grad_accum_steps=4,
        target_tokens_per_step=8 * 512,
    )
    partition = PartitionSpec(stages=[PartitionStageSpec(decoder_layers=8, special_tokens=["E", "L"])])
    layout = LayoutSpec(stage_to_node=[cluster.nodes[-1]], vpp_degree=1)
    if target == "dual_g5_g5":
        parallel = MegatronParallelSpec(tp_degree=2, pp_degree=4, vpp_degree=1, cp_degree=1, ep_degree=2, expert_tp_degree=1, sp_enabled=False)
        stage_to_node = default_grouped_stage_to_node(target, parallel.pp_degree)
        stage_layers = weighted_stage_layer_allocation(target, 8, stage_to_node)
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=stage_layers[index], special_tokens=(["E"] if index == 0 else []) + (["L"] if index == len(stage_layers) - 1 else []))
                for index in range(len(stage_layers))
            ]
        )
        layout = LayoutSpec(stage_to_node=stage_to_node, vpp_degree=1)
    if target == "single_g5":
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=4, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=4, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g5", "g5"], vpp_degree=1)
    if target == "single_g4":
        partition = PartitionSpec(
            stages=[
                PartitionStageSpec(decoder_layers=4, special_tokens=["E"]),
                PartitionStageSpec(decoder_layers=4, special_tokens=["L"]),
            ]
        )
        layout = LayoutSpec(stage_to_node=["g4", "g4"], vpp_degree=1)
    if target == "dual_g4_g5":
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
            enabled=False,
        ),
        schedule=ScheduleSpec(template="fixed_1f1b"),
        batch_plan=batch_plan,
        constraints=ConstraintSpec(required_node_local_axes=["tp", "ep"], memory_budget_gb=memory_budget),
        search_space=SearchSpaceSpec(),
        length_bucket_policies=default_length_bucket_policies(),
        machine_profile=default_machine_profile(target),
        backend_caps=default_backend_caps("local"),
        metadata={"program_kind": "baseline_moe_smoke"},
    )
