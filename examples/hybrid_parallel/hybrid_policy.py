from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _ensure_int(value: Any, context: str, *, min_value: int = 0) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{context} must be an integer; got {value!r}") from exc
    if out < int(min_value):
        raise ValueError(f"{context} must be >= {min_value}; got {out}")
    return out


def _coerce_bool_list(
    values: Optional[Sequence[Any]],
    *,
    length: int,
    context: str,
) -> Optional[List[bool]]:
    if values is None:
        return None
    if len(values) != int(length):
        raise ValueError(f"{context} must have len={length}; got {len(values)}")
    return [bool(x) for x in values]


def _coerce_str_list(
    values: Optional[Sequence[Any]],
    *,
    length: int,
    context: str,
) -> Optional[List[str]]:
    if values is None:
        return None
    if len(values) != int(length):
        raise ValueError(f"{context} must have len={length}; got {len(values)}")
    return [str(x) for x in values]


def _normalize_stage_ranges(stage_ranges: Sequence[Sequence[Any]], *, degree: int) -> List[List[int]]:
    if len(stage_ranges) != int(degree):
        raise ValueError(f"pipeline.stage_ranges must have len={degree}; got {len(stage_ranges)}")
    out: List[List[int]] = []
    prev_end = -1
    for idx, item in enumerate(stage_ranges):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"pipeline.stage_ranges[{idx}] must be [start, end]")
        start = _ensure_int(item[0], f"pipeline.stage_ranges[{idx}][0]")
        end = _ensure_int(item[1], f"pipeline.stage_ranges[{idx}][1]")
        if end < start:
            raise ValueError(f"pipeline.stage_ranges[{idx}] must satisfy start <= end")
        if idx > 0 and start != prev_end + 1:
            raise ValueError("pipeline.stage_ranges must be contiguous and non-overlapping")
        out.append([start, end])
        prev_end = end
    return out


def _default_stage_modules(stage_ranges: Sequence[Sequence[int]]) -> List[List[str]]:
    modules: List[List[str]] = []
    for idx, (start, end) in enumerate(stage_ranges):
        stage_modules: List[str] = []
        if idx == 0:
            stage_modules.append("tok_embeddings")
        stage_modules.extend([f"layers.{layer_idx}" for layer_idx in range(int(start), int(end) + 1)])
        if idx == len(stage_ranges) - 1:
            stage_modules.extend(["norm", "output"])
        modules.append(stage_modules)
    return modules


@dataclass
class PipelinePolicy:
    degree: int = 1
    vpp: int = 1
    microbatches: int = 1
    schedule: str = "1f1b"
    stage_ranges: Optional[List[List[int]]] = None
    stage_modules: Optional[List[List[str]]] = None
    mesh: Optional[List[List[List[int]]]] = None
    stage_to_node: Optional[List[str]] = None


@dataclass
class TensorParallelPolicy:
    enabled: bool = False
    degree: int = 1
    sequence_parallel: bool = False
    plan: str = "auto"
    use_local_output: Optional[bool] = None
    head_grouping: Optional[Dict[str, int]] = None


@dataclass
class ContextParallelPolicy:
    enabled: bool = False
    degree: int = 1


@dataclass
class ExpertParallelPolicy:
    enabled: bool = False
    degree: int = 1
    expert_tp_degree: int = 1


@dataclass
class Fsdp2Policy:
    enabled: bool = False
    param_dtype: str = "bf16"
    reduce_dtype: str = "bf16"
    reshard_after_forward: bool = True
    enabled_per_stage: Optional[List[bool]] = None
    reshard_after_forward_per_stage: Optional[List[bool]] = None


@dataclass
class RecomputePolicy:
    policy: str = "none"
    per_stage: Optional[List[str]] = None


@dataclass
class StagePolicy:
    stage_id: int
    node: Optional[str] = None
    layer_range: Optional[List[int]] = None
    modules: Optional[List[str]] = None
    fsdp_enabled: Optional[bool] = None
    reshard_after_forward: Optional[bool] = None
    recompute: Optional[str] = None
    vpp_chunks: Optional[int] = None
    tp_degree: Optional[int] = None
    cp_degree: Optional[int] = None
    ep_degree: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class ModulePolicy:
    pattern: str
    stage_ids: Optional[List[int]] = None
    placement_hint: Optional[str] = None
    tp_mode: Optional[str] = None
    cp_mode: Optional[str] = None
    ep_mode: Optional[str] = None
    fsdp_mode: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class PhasePolicy:
    name: str
    condition: str
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridParallelPolicy:
    name: str = "hybrid_policy"
    pipeline: PipelinePolicy = field(default_factory=PipelinePolicy)
    tensor_parallel: TensorParallelPolicy = field(default_factory=TensorParallelPolicy)
    context_parallel: ContextParallelPolicy = field(default_factory=ContextParallelPolicy)
    expert_parallel: ExpertParallelPolicy = field(default_factory=ExpertParallelPolicy)
    fsdp2: Fsdp2Policy = field(default_factory=Fsdp2Policy)
    recompute: RecomputePolicy = field(default_factory=RecomputePolicy)
    stage_policies: List[StagePolicy] = field(default_factory=list)
    module_policies: List[ModulePolicy] = field(default_factory=list)
    phase_policies: List[PhasePolicy] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HybridParallelPolicy":
        return cls(
            name=str(payload.get("name") or "hybrid_policy"),
            pipeline=PipelinePolicy(**(payload.get("pipeline") or {})),
            tensor_parallel=TensorParallelPolicy(**(payload.get("tensor_parallel") or {})),
            context_parallel=ContextParallelPolicy(**(payload.get("context_parallel") or {})),
            expert_parallel=ExpertParallelPolicy(**(payload.get("expert_parallel") or {})),
            fsdp2=Fsdp2Policy(**(payload.get("fsdp2") or {})),
            recompute=RecomputePolicy(**(payload.get("recompute") or {})),
            stage_policies=[StagePolicy(**item) for item in (payload.get("stage_policies") or [])],
            module_policies=[ModulePolicy(**item) for item in (payload.get("module_policies") or [])],
            phase_policies=[PhasePolicy(**item) for item in (payload.get("phase_policies") or [])],
            metadata=copy.deepcopy(payload.get("metadata") or {}),
            schema_version=int(payload.get("schema_version") or 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> "HybridParallelPolicy":
        self.pipeline.degree = _ensure_int(self.pipeline.degree, "pipeline.degree", min_value=1)
        self.pipeline.vpp = _ensure_int(self.pipeline.vpp, "pipeline.vpp", min_value=1)
        self.pipeline.microbatches = _ensure_int(
            self.pipeline.microbatches, "pipeline.microbatches", min_value=1
        )
        self.pipeline.schedule = str(self.pipeline.schedule or "1f1b").lower()

        if self.pipeline.stage_ranges is not None:
            self.pipeline.stage_ranges = _normalize_stage_ranges(
                self.pipeline.stage_ranges,
                degree=self.pipeline.degree,
            )
        if self.pipeline.stage_modules is not None and len(self.pipeline.stage_modules) != int(self.pipeline.degree):
            raise ValueError(
                f"pipeline.stage_modules must have len={self.pipeline.degree}; got {len(self.pipeline.stage_modules)}"
            )
        self.pipeline.stage_to_node = _coerce_str_list(
            self.pipeline.stage_to_node,
            length=self.pipeline.degree,
            context="pipeline.stage_to_node",
        )

        self.tensor_parallel.degree = _ensure_int(
            self.tensor_parallel.degree, "tensor_parallel.degree", min_value=1
        )
        self.tensor_parallel.enabled = bool(
            self.tensor_parallel.enabled or int(self.tensor_parallel.degree) > 1
        )
        self.tensor_parallel.sequence_parallel = bool(self.tensor_parallel.sequence_parallel)
        if not self.tensor_parallel.enabled:
            self.tensor_parallel.degree = 1

        self.context_parallel.degree = _ensure_int(
            self.context_parallel.degree, "context_parallel.degree", min_value=1
        )
        self.context_parallel.enabled = bool(
            self.context_parallel.enabled or int(self.context_parallel.degree) > 1
        )
        if not self.context_parallel.enabled:
            self.context_parallel.degree = 1

        self.expert_parallel.degree = _ensure_int(
            self.expert_parallel.degree, "expert_parallel.degree", min_value=1
        )
        self.expert_parallel.expert_tp_degree = _ensure_int(
            self.expert_parallel.expert_tp_degree,
            "expert_parallel.expert_tp_degree",
            min_value=1,
        )
        self.expert_parallel.enabled = bool(
            self.expert_parallel.enabled or int(self.expert_parallel.degree) > 1
        )
        if not self.expert_parallel.enabled:
            self.expert_parallel.degree = 1

        self.fsdp2.enabled = bool(self.fsdp2.enabled)
        self.fsdp2.enabled_per_stage = _coerce_bool_list(
            self.fsdp2.enabled_per_stage,
            length=self.pipeline.degree,
            context="fsdp2.enabled_per_stage",
        )
        self.fsdp2.reshard_after_forward_per_stage = _coerce_bool_list(
            self.fsdp2.reshard_after_forward_per_stage,
            length=self.pipeline.degree,
            context="fsdp2.reshard_after_forward_per_stage",
        )
        self.recompute.per_stage = _coerce_str_list(
            self.recompute.per_stage,
            length=self.pipeline.degree,
            context="recompute.per_stage",
        )

        for idx, stage_policy in enumerate(self.stage_policies):
            stage_policy.stage_id = _ensure_int(
                stage_policy.stage_id,
                f"stage_policies[{idx}].stage_id",
                min_value=0,
            )
            if int(stage_policy.stage_id) >= int(self.pipeline.degree):
                raise ValueError(
                    f"stage_policies[{idx}].stage_id must be < pipeline.degree ({self.pipeline.degree})"
                )
            if stage_policy.layer_range is not None:
                if len(stage_policy.layer_range) != 2:
                    raise ValueError(f"stage_policies[{idx}].layer_range must be [start, end]")
                stage_policy.layer_range = [
                    _ensure_int(stage_policy.layer_range[0], f"stage_policies[{idx}].layer_range[0]"),
                    _ensure_int(stage_policy.layer_range[1], f"stage_policies[{idx}].layer_range[1]"),
                ]
            if stage_policy.modules is not None:
                stage_policy.modules = [str(x) for x in stage_policy.modules]

        return self

    def _stage_ranges(self) -> Optional[List[List[int]]]:
        if self.pipeline.stage_ranges is not None:
            return copy.deepcopy(self.pipeline.stage_ranges)
        derived = [None] * int(self.pipeline.degree)
        for stage_policy in self.stage_policies:
            if stage_policy.layer_range is not None:
                derived[int(stage_policy.stage_id)] = list(stage_policy.layer_range)
        if all(item is not None for item in derived):
            return [list(item) for item in derived if item is not None]
        return None

    def _stage_modules(self, total_layers: Optional[int] = None) -> Optional[List[List[str]]]:
        if self.pipeline.stage_modules is not None:
            return copy.deepcopy(self.pipeline.stage_modules)
        stage_modules: List[Optional[List[str]]] = [None] * int(self.pipeline.degree)
        for stage_policy in self.stage_policies:
            if stage_policy.modules is not None:
                stage_modules[int(stage_policy.stage_id)] = list(stage_policy.modules)
        if all(item is not None for item in stage_modules):
            return [list(item) for item in stage_modules if item is not None]
        ranges = self._stage_ranges()
        if ranges is not None:
            return _default_stage_modules(ranges)
        if total_layers is not None:
            per_stage = max(1, int(total_layers) // int(self.pipeline.degree))
            ranges = []
            start = 0
            for stage_idx in range(int(self.pipeline.degree)):
                end = start + per_stage - 1
                if stage_idx == int(self.pipeline.degree) - 1:
                    end = int(total_layers) - 1
                ranges.append([start, end])
                start = end + 1
            return _default_stage_modules(ranges)
        return None

    def export_manual_runner(self) -> Tuple[Dict[str, Any], List[str]]:
        policy = copy.deepcopy(self).validate()
        warnings: List[str] = []

        stage_ranges = policy._stage_ranges()
        if stage_ranges is None:
            warnings.append("manual runner export needs pipeline.stage_ranges or per-stage layer_range; leaving pp.stages unchanged")

        fsdp_enabled_per_stage = (
            list(policy.fsdp2.enabled_per_stage)
            if policy.fsdp2.enabled_per_stage is not None
            else [bool(policy.fsdp2.enabled) for _ in range(int(policy.pipeline.degree))]
        )
        reshard_per_stage = (
            list(policy.fsdp2.reshard_after_forward_per_stage)
            if policy.fsdp2.reshard_after_forward_per_stage is not None
            else [bool(policy.fsdp2.reshard_after_forward) for _ in range(int(policy.pipeline.degree))]
        )
        recompute_per_stage = (
            list(policy.recompute.per_stage)
            if policy.recompute.per_stage is not None
            else [str(policy.recompute.policy or "none") for _ in range(int(policy.pipeline.degree))]
        )

        for stage_policy in policy.stage_policies:
            stage_id = int(stage_policy.stage_id)
            if stage_policy.fsdp_enabled is not None:
                fsdp_enabled_per_stage[stage_id] = bool(stage_policy.fsdp_enabled)
            if stage_policy.reshard_after_forward is not None:
                reshard_per_stage[stage_id] = bool(stage_policy.reshard_after_forward)
            if stage_policy.recompute is not None:
                recompute_per_stage[stage_id] = str(stage_policy.recompute)
            if stage_policy.tp_degree not in {None, policy.tensor_parallel.degree}:
                warnings.append(
                    f"stage {stage_id} requested tp_degree={stage_policy.tp_degree}, but train_manual_pp.py only supports uniform TP"
                )
            if stage_policy.cp_degree not in {None, policy.context_parallel.degree}:
                warnings.append(
                    f"stage {stage_id} requested cp_degree={stage_policy.cp_degree}, but train_manual_pp.py has no CP runtime path yet"
                )
            if stage_policy.ep_degree not in {None, policy.expert_parallel.degree}:
                warnings.append(
                    f"stage {stage_id} requested ep_degree={stage_policy.ep_degree}, but train_manual_pp.py has no EP runtime path yet"
                )
            if stage_policy.vpp_chunks not in {None, policy.pipeline.vpp}:
                warnings.append(
                    f"stage {stage_id} requested vpp_chunks={stage_policy.vpp_chunks}, but train_manual_pp.py only supports uniform VPP"
                )

        overrides = {
            "parallel": {
                "pp": {
                    "degree": int(policy.pipeline.degree),
                    "vpp": int(policy.pipeline.vpp),
                    "microbatches": int(policy.pipeline.microbatches),
                    "schedule": str(policy.pipeline.schedule),
                },
                "tp": {
                    "enabled": bool(policy.tensor_parallel.enabled),
                    "degree": int(policy.tensor_parallel.degree),
                    "plan": str(policy.tensor_parallel.plan),
                    "use_local_output": policy.tensor_parallel.use_local_output,
                    "head_grouping": copy.deepcopy(policy.tensor_parallel.head_grouping),
                },
                "sp": {
                    "enabled": bool(policy.tensor_parallel.sequence_parallel),
                },
                "cp": {
                    "enabled": bool(policy.context_parallel.enabled),
                    "degree": int(policy.context_parallel.degree),
                },
                "ep": {
                    "enabled": bool(policy.expert_parallel.enabled),
                    "degree": int(policy.expert_parallel.degree),
                    "expert_tp_degree": int(policy.expert_parallel.expert_tp_degree),
                },
                "fsdp2": {
                    "enabled": bool(policy.fsdp2.enabled or any(fsdp_enabled_per_stage)),
                    "param_dtype": str(policy.fsdp2.param_dtype),
                    "reduce_dtype": str(policy.fsdp2.reduce_dtype),
                    "reshard_after_forward": bool(policy.fsdp2.reshard_after_forward),
                    "enabled_per_stage": fsdp_enabled_per_stage,
                    "reshard_after_forward_per_stage": reshard_per_stage,
                },
                "recompute": {
                    "policy": str(policy.recompute.policy),
                    "per_stage": recompute_per_stage,
                },
            },
            "runtime": {
                "hybrid_policy_name": policy.name,
            },
            "hybrid_policy_meta": {
                "warnings": warnings,
                "module_policies": [asdict(item) for item in policy.module_policies],
                "phase_policies": [asdict(item) for item in policy.phase_policies],
            },
        }
        if stage_ranges is not None:
            overrides["parallel"]["pp"]["stages"] = stage_ranges
        if policy.pipeline.mesh is not None:
            overrides["parallel"]["pp"]["mesh"] = copy.deepcopy(policy.pipeline.mesh)
        if policy.pipeline.stage_to_node is not None:
            overrides["parallel"]["pp"]["stage_to_node"] = list(policy.pipeline.stage_to_node)
        return overrides, warnings

    def export_torchtitan(self, *, total_layers: Optional[int] = None) -> Tuple[Dict[str, Any], List[str]]:
        policy = copy.deepcopy(self).validate()
        warnings: List[str] = []
        stage_modules = policy._stage_modules(total_layers=total_layers)
        if stage_modules is None:
            warnings.append("TorchTitan export needs stage_modules or stage_ranges/total_layers to derive module_fqns_per_model_part")

        overrides: Dict[str, Any] = {
            "model.name": self.metadata.get("torchtitan_model_name", "qwen3"),
            "parallelism.pipeline_parallel_degree": int(policy.pipeline.degree),
            "parallelism.tensor_parallel_degree": int(policy.tensor_parallel.degree),
            "parallelism.context_parallel_degree": int(policy.context_parallel.degree),
            "parallelism.expert_parallel_degree": int(policy.expert_parallel.degree),
            "parallelism.pipeline_parallel_schedule": str(policy.pipeline.schedule),
            "parallelism.enable_loss_parallel": bool(policy.tensor_parallel.enabled),
            "training.mixed_precision_param": str(policy.fsdp2.param_dtype),
            "training.mixed_precision_reduce": str(policy.fsdp2.reduce_dtype),
            "training.seq_len": self.metadata.get("seq_len"),
            "parallelism.fsdp_reshard_after_forward": "always" if policy.fsdp2.reshard_after_forward else "never",
            "activation_checkpoint.mode": str(policy.recompute.policy or "none"),
        }
        if stage_modules is not None:
            overrides["parallelism.module_fqns_per_model_part"] = stage_modules
        if int(policy.pipeline.vpp) > 1:
            uniform_layers = None
            ranges = policy._stage_ranges()
            if ranges is not None:
                lengths = [(end - start + 1) for start, end in ranges]
                if lengths and len(set(lengths)) == 1:
                    uniform_layers = lengths[0]
            if uniform_layers is not None:
                overrides["parallelism.pipeline_parallel_layers_per_stage"] = int(uniform_layers)
            else:
                warnings.append("non-uniform stage sizes cannot be expressed as pipeline_parallel_layers_per_stage; keep module_fqns_per_model_part authoritative")

        if policy.pipeline.stage_to_node is not None:
            warnings.append("TorchTitan placement is fixed-rank; stage_to_node is kept as metadata only")
            overrides["experimental.stage_to_node"] = list(policy.pipeline.stage_to_node)

        if any(stage.tp_degree not in {None, policy.tensor_parallel.degree} for stage in policy.stage_policies):
            warnings.append("per-stage TP is not directly supported by TorchTitan's standard mesh layout")
        if any(stage.vpp_chunks not in {None, policy.pipeline.vpp} for stage in policy.stage_policies):
            warnings.append("per-stage asymmetric VPP is not directly supported by TorchTitan")

        overrides["hybrid_policy.module_policies"] = [asdict(item) for item in policy.module_policies]
        overrides["hybrid_policy.phase_policies"] = [asdict(item) for item in policy.phase_policies]
        return overrides, warnings

    @classmethod
    def from_plan_candidate(cls, candidate: Any, *, metadata: Optional[Dict[str, Any]] = None) -> "HybridParallelPolicy":
        stage_policies = [
            StagePolicy(
                stage_id=idx,
                node=(candidate.stage_to_node[idx] if idx < len(candidate.stage_to_node) else None),
                layer_range=list(candidate.stage_ranges[idx]) if idx < len(candidate.stage_ranges) else None,
                fsdp_enabled=(
                    bool(candidate.fsdp_enabled_per_stage[idx])
                    if idx < len(candidate.fsdp_enabled_per_stage)
                    else None
                ),
                reshard_after_forward=(
                    bool(candidate.reshard_after_forward_per_stage[idx])
                    if idx < len(candidate.reshard_after_forward_per_stage)
                    else None
                ),
                recompute=(
                    str(candidate.recompute_per_stage[idx])
                    if idx < len(candidate.recompute_per_stage)
                    else None
                ),
            )
            for idx in range(int(candidate.pp_degree))
        ]
        return cls(
            name="planner_candidate",
            pipeline=PipelinePolicy(
                degree=int(candidate.pp_degree),
                vpp=int(candidate.vpp),
                microbatches=int(candidate.microbatches),
                schedule=str(candidate.schedule),
                stage_ranges=copy.deepcopy(candidate.stage_ranges),
                mesh=copy.deepcopy(candidate.mesh),
                stage_to_node=list(candidate.stage_to_node),
            ),
            tensor_parallel=TensorParallelPolicy(
                enabled=bool(int(candidate.tp_degree) > 1),
                degree=int(candidate.tp_degree),
            ),
            fsdp2=Fsdp2Policy(
                enabled=bool(any(candidate.fsdp_enabled_per_stage)),
                enabled_per_stage=list(candidate.fsdp_enabled_per_stage),
                reshard_after_forward_per_stage=list(candidate.reshard_after_forward_per_stage),
            ),
            recompute=RecomputePolicy(
                policy="mixed",
                per_stage=list(candidate.recompute_per_stage),
            ),
            stage_policies=stage_policies,
            metadata=copy.deepcopy(metadata or {}),
        )


def load_hybrid_policy(cfg: Dict[str, Any], *, config_path: Optional[str] = None) -> Optional[HybridParallelPolicy]:
    raw = cfg.get("hybrid_policy")
    if not raw:
        return None
    if isinstance(raw, str):
        policy_path = raw
        inline = None
    else:
        policy_path = (raw.get("path") if isinstance(raw, dict) else None)
        inline = (raw.get("inline") if isinstance(raw, dict) else raw)
    if inline:
        return HybridParallelPolicy.from_dict(inline).validate()
    if policy_path:
        policy_file = Path(str(policy_path))
        if not policy_file.is_absolute() and config_path:
            policy_file = Path(config_path).resolve().parent / policy_file
        payload = json.loads(policy_file.read_text(encoding="utf-8"))
        return HybridParallelPolicy.from_dict(payload).validate()
    if isinstance(raw, dict):
        return HybridParallelPolicy.from_dict(raw).validate()
    raise ValueError("hybrid_policy must be a dict, a path string, or {path|inline}")


def apply_hybrid_policy_to_config(
    cfg: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    policy = load_hybrid_policy(cfg, config_path=config_path)
    if policy is None:
        return cfg, []
    overrides, warnings = policy.export_manual_runner()
    merged = _deep_merge(cfg, overrides)
    merged["hybrid_policy"] = policy.to_dict()
    return merged, warnings
