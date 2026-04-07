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
    shape_objective: str = "structure_memory_comm_coupled"
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
            str(norm.shape_objective or "structure_memory_comm_coupled").strip().lower()
            or "structure_memory_comm_coupled"
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
            shape_objective=str(payload.get("shape_objective", "structure_memory_comm_coupled")),
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
    legality: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    diagnosis: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    switch_cost: float = 0.0
    next_scope_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_legal": bool(self.is_legal),
            "legality": copy.deepcopy(self.legality),
            "cost": copy.deepcopy(self.cost),
            "diagnosis": list(self.diagnosis),
            "rejection_reason": self.rejection_reason,
            "switch_cost": float(self.switch_cost),
            "next_scope_hint": self.next_scope_hint,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerifierReport":
        return cls(
            is_legal=bool(payload.get("is_legal", True)),
            legality=copy.deepcopy(payload.get("legality") or {}),
            cost=copy.deepcopy(payload.get("cost") or {}),
            diagnosis=[str(item) for item in (payload.get("diagnosis") or [])],
            rejection_reason=payload.get("rejection_reason"),
            switch_cost=float(payload.get("switch_cost", 0.0) or 0.0),
            next_scope_hint=payload.get("next_scope_hint"),
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
