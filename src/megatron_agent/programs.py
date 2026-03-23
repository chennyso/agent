from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from megatron_agent.config import (
    ClusterSpec,
    ConstraintSpec,
    LayoutSpec,
    MegatronParallelSpec,
    MegatronProgram,
    MegatronStrategy,
    ModelSpec,
    PartitionSpec,
    ScheduleSpec,
    validate_strategy,
)

_DEFAULT_SCHEDULE_FAMILIES = {"fixed_1f1b", "interleaved_1f1b", "zero_bubble"}


@dataclass
class FamilyClassification:
    family_tags: List[str] = field(default_factory=list)
    is_family_outside: bool = False
    violated_priors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_tags": list(self.family_tags),
            "is_family_outside": bool(self.is_family_outside),
            "violated_priors": list(self.violated_priors),
        }


@dataclass
class ProgramLegalityReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "rejected_constraints": list(self.rejected_constraints),
        }


@dataclass
class CompiledProgram:
    strategy: MegatronStrategy
    launcher_env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    family: FamilyClassification = field(default_factory=FamilyClassification)
    legality: ProgramLegalityReport = field(default_factory=lambda: ProgramLegalityReport(is_valid=True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "launcher_env": dict(self.launcher_env),
            "extra_args": list(self.extra_args),
            "family": self.family.to_dict(),
            "legality": self.legality.to_dict(),
        }


def _copy_program(program: MegatronProgram) -> MegatronProgram:
    return MegatronProgram.from_dict(program.to_dict())


def classify_program_family(program: MegatronProgram) -> FamilyClassification:
    norm = program.normalized()
    tags: List[str] = []
    violated: List[str] = []

    stage_layers = [int(stage.decoder_layers) for stage in norm.partition.stages]
    if len(set(stage_layers)) <= 1:
        tags.append("uniform_pp")
    else:
        violated.append("uniform_pp")

    if not norm.plane_map.enabled:
        tags.append("single_plane")
    else:
        violated.append("single_plane")

    if int(norm.parallel.vpp_degree) <= 1 and not norm.layout.pipeline_layout:
        tags.append("symmetric_vpp")
    elif int(norm.parallel.vpp_degree) > 1 and not norm.layout.pipeline_layout:
        tags.append("symmetric_vpp")
    else:
        violated.append("symmetric_vpp")

    schedule_group = norm.schedule.microbatch_group_size_per_vp_stage
    if (
        norm.schedule.skeleton in _DEFAULT_SCHEDULE_FAMILIES
        and norm.schedule.dispatch_order == "default"
        and schedule_group in (None, 1)
    ):
        tags.append("fixed_schedule")
    else:
        violated.append("fixed_schedule")

    return FamilyClassification(
        family_tags=tags,
        is_family_outside=bool(violated),
        violated_priors=violated,
    )


def _build_layout_from_partition(partition: PartitionSpec) -> str:
    tokens_per_stage: List[str] = []
    for stage in partition.stages:
        prefix = "".join(token for token in stage.special_tokens if token == "E")
        middle = "t" * int(stage.decoder_layers)
        suffix = "".join(token for token in stage.special_tokens if token != "E")
        tokens = f"{prefix}{middle}{suffix}" or "t"
        tokens_per_stage.append(tokens)
    return "|".join(tokens_per_stage)


def _stage_nodes_match_target(cluster: ClusterSpec, layout: LayoutSpec) -> bool:
    known = set(cluster.nodes)
    return all(node in known for node in layout.stage_to_node)


def _validate_node_local_constraints(
    cluster: ClusterSpec,
    parallel: MegatronParallelSpec,
    constraints: ConstraintSpec,
) -> List[str]:
    errors: List[str] = []
    if cluster.target in {"single_g4", "dual_g4_g5"}:
        if "tp" in constraints.required_node_local_axes and int(parallel.tp_degree) > int(cluster.gpus_per_node):
            errors.append("tp degree exceeds per-node GPUs but tp is required to stay node-local")
        if "ep" in constraints.required_node_local_axes and int(parallel.ep_degree) > int(cluster.gpus_per_node):
            errors.append("ep degree exceeds per-node GPUs but ep is required to stay node-local")
        if "cp" in constraints.required_node_local_axes and int(parallel.cp_degree) > int(cluster.gpus_per_node):
            errors.append("cp degree exceeds per-node GPUs but cp is required to stay node-local")
    return errors


def check_program(program: MegatronProgram, target: Optional[str] = None) -> ProgramLegalityReport:
    norm = program.normalized()
    cluster = copy.deepcopy(norm.cluster)
    if target:
        cluster.target = str(target)
    errors: List[str] = []
    warnings: List[str] = []
    rejected: List[str] = []

    if norm.constraints.requires_runtime_pg_rebuild:
        rejected.append("runtime_pg_rebuild_not_supported_v1")
    if norm.constraints.requested_heterogeneous_apipe:
        rejected.append("heterogeneous_apipe_not_supported_v1")
    if norm.layout.stage_device_counts is not None and len(set(norm.layout.stage_device_counts)) > 1:
        rejected.append("heterogeneous_stage_device_counts_not_supported_v1")
    if norm.layout.submesh_hints:
        rejected.append("subgraph_submesh_not_supported_v1")
    if norm.search_space.allow_heterogeneous_apipe:
        warnings.append("search space allows heterogeneous apipe, but runtime support is disabled in v1")
    if norm.search_space.allow_subgraph_submeshes:
        warnings.append("search space allows subgraph submeshes, but runtime support is disabled in v1")

    if norm.partition.num_stages != int(norm.parallel.pp_degree):
        errors.append(
            f"partition stages={norm.partition.num_stages} must match pp_degree={int(norm.parallel.pp_degree)}"
        )
    if len(norm.layout.stage_to_node) != norm.partition.num_stages:
        errors.append("layout.stage_to_node length must match number of partition stages")
    if not _stage_nodes_match_target(cluster, norm.layout):
        errors.append("layout.stage_to_node contains nodes not present in cluster spec")
    if cluster.target == "single_g4" and any(node != "g4" for node in norm.layout.stage_to_node):
        errors.append("single_g4 target only allows stages on g4")
    if cluster.target == "single_g5" and any(node != "g5" for node in norm.layout.stage_to_node):
        errors.append("single_g5 target only allows stages on g5")
    if cluster.target == "dual_g4_g5" and sorted(set(norm.layout.stage_to_node)) not in (["g4"], ["g4", "g5"], ["g5"]):
        errors.append("dual_g4_g5 target only supports g4/g5 stage mapping")

    if norm.partition.total_decoder_layers != int(norm.model.num_layers):
        errors.append(
            f"partition total decoder layers={norm.partition.total_decoder_layers} "
            f"must equal model.num_layers={int(norm.model.num_layers)}"
        )

    if int(norm.parallel.vpp_degree) != int(norm.layout.vpp_degree):
        errors.append("parallel.vpp_degree must match layout.vpp_degree")

    product = (
        int(norm.parallel.tp_degree)
        * int(norm.parallel.pp_degree)
        * int(norm.parallel.cp_degree)
        * int(norm.parallel.ep_degree)
        * int(norm.parallel.expert_tp_degree)
    )
    if product <= 0:
        errors.append("parallel product must be positive")
    elif int(cluster.world_size) % product != 0:
        errors.append(f"cluster.world_size={cluster.world_size} is not divisible by tp*pp*cp*ep*expert_tp={product}")

    if int(norm.parallel.vpp_degree) > 1 and not norm.layout.pipeline_layout:
        total_virtual = int(norm.parallel.pp_degree) * int(norm.parallel.vpp_degree)
        if int(norm.model.num_layers) % total_virtual != 0:
            errors.append(f"model.num_layers={norm.model.num_layers} must be divisible by pp*vpp={total_virtual}")

    errors.extend(_validate_node_local_constraints(cluster, norm.parallel, norm.constraints))

    if norm.model.track != "moe" and norm.plane_map.enabled:
        warnings.append("plane_map is enabled for a non-MoE track; it will be compiled for diagnostics only")
    if norm.model.track == "moe" and not norm.plane_map.enabled:
        warnings.append("moe track is using single-plane mapping")

    if rejected:
        errors.extend(rejected)

    return ProgramLegalityReport(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        rejected_constraints=rejected,
    )


def program_to_strategy(program: MegatronProgram) -> MegatronStrategy:
    norm = program.normalized()
    metadata = copy.deepcopy(norm.metadata or {})
    default_micro = 1
    default_global = 16 if norm.model.track == "dense" else 8
    default_seq = 1024 if norm.model.track == "dense" else 512
    strategy = MegatronStrategy(
        parallel=norm.parallel,
        micro_batch_size=int(metadata.get("micro_batch_size", default_micro) or default_micro),
        global_batch_size=int(metadata.get("global_batch_size", default_global) or default_global),
        seq_len=int(metadata.get("seq_len", default_seq) or default_seq),
        use_bf16=bool(metadata.get("use_bf16", True)),
        use_fp16=bool(metadata.get("use_fp16", False)),
        recompute_granularity=metadata.get("recompute_granularity", "selective"),
        extra_args=list(metadata.get("extra_args") or []),
    )
    return validate_strategy(strategy)


def compile_program(program: MegatronProgram, target: Optional[str] = None) -> CompiledProgram:
    norm = _copy_program(program).normalized()
    if target:
        norm.cluster.target = str(target)
    family = classify_program_family(norm)
    legality = check_program(norm)
    strategy = program_to_strategy(norm)

    env: Dict[str, str] = {
        "RUN_TARGET": str(norm.cluster.target),
        "MODEL_TRACK": str(norm.model.track),
        "TP_SIZE": str(int(norm.parallel.tp_degree)),
        "PP_SIZE": str(int(norm.parallel.pp_degree)),
        "VPP_SIZE": str(int(norm.parallel.vpp_degree)),
        "CP_SIZE": str(int(norm.parallel.cp_degree)),
        "EP_SIZE": str(int(norm.parallel.ep_degree)),
        "EXPERT_TP_SIZE": str(int(norm.parallel.expert_tp_degree)),
        "MICRO_BATCH_SIZE": str(int(strategy.micro_batch_size)),
        "GLOBAL_BATCH_SIZE": str(int(strategy.global_batch_size)),
        "SEQ_LENGTH": str(int(strategy.seq_len)),
        "NUM_LAYERS": str(int(norm.model.num_layers)),
        "USE_BF16": "1" if bool(strategy.use_bf16) else "0",
        "USE_FP16": "1" if bool(strategy.use_fp16 and not strategy.use_bf16) else "0",
        "ENABLE_SP": "1" if bool(norm.parallel.sp_enabled and int(norm.parallel.tp_degree) > 1) else "0",
        "PROGRAM_SEMANTIC_HASH": norm.semantic_hash(),
        "PROGRAM_IS_FAMILY_OUTSIDE": "1" if family.is_family_outside else "0",
        "ALLOW_NONUNIFORM_PARTITION": "1" if norm.search_space.allow_nonuniform_partition else "0",
        "ALLOW_SINGLE_NODE_PP_SPLIT": "1" if norm.search_space.allow_single_node_pp_split else "0",
        "ALLOW_SEQUENCE_PARALLEL_TOGGLE": "1" if norm.search_space.allow_sequence_parallel_toggle else "0",
        "ALLOW_ASYMMETRIC_VPP": "1" if norm.search_space.allow_asymmetric_vpp else "0",
        "ALLOW_DUAL_PLANE": "1" if norm.search_space.allow_dual_plane else "0",
        "ALLOW_STAGE_AWARE_SCHEDULE": "1" if norm.search_space.allow_stage_aware_schedule else "0",
    }

    if norm.layout.pipeline_layout:
        pipeline_layout = norm.layout.pipeline_layout
    elif int(norm.parallel.vpp_degree) == 1 and (int(norm.parallel.pp_degree) > 1 or family.is_family_outside):
        pipeline_layout = _build_layout_from_partition(norm.partition)
    else:
        pipeline_layout = None
    if pipeline_layout:
        env["PIPELINE_LAYOUT"] = pipeline_layout

    if int(norm.parallel.vpp_degree) > 1 and not pipeline_layout:
        total_virtual = int(norm.parallel.pp_degree) * int(norm.parallel.vpp_degree)
        env["NUM_LAYERS_PER_VIRTUAL_STAGE"] = str(int(norm.model.num_layers) // total_virtual)

    if norm.schedule.microbatch_group_size_per_vp_stage:
        env["SCHEDULE_GROUP_SIZE"] = str(int(norm.schedule.microbatch_group_size_per_vp_stage))
    if norm.schedule.skeleton:
        env["SCHEDULE_SKELETON"] = str(norm.schedule.skeleton)
    if norm.schedule.dispatch_order:
        env["DISPATCH_ORDER"] = str(norm.schedule.dispatch_order)

    if norm.plane_map.enabled:
        env["ENABLE_PLANE_MAP"] = "1"
        env["ATTENTION_TP_SIZE"] = str(int(norm.plane_map.attention.tp_degree))
        env["ATTENTION_CP_SIZE"] = str(int(norm.plane_map.attention.cp_degree))
        if norm.plane_map.moe is not None:
            env["MOE_EP_SIZE"] = str(int(norm.plane_map.moe.ep_degree))
            env["MOE_EXPERT_TP_SIZE"] = str(int(norm.plane_map.moe.expert_tp_degree))
            env["MOE_ATTENTION_DECOUPLED"] = "1"

    if norm.model.track == "moe":
        env["MOE_NUM_EXPERTS"] = str(int(norm.model.num_experts or 4))
        env["MOE_LAYER_FREQ"] = str(int(norm.model.moe_layer_freq or 2))
        env["MOE_ROUTER_TOPK"] = str(int(norm.metadata.get("moe_router_topk", 2) or 2))

    for index, node in enumerate(norm.layout.stage_to_node):
        env[f"STAGE_{index}_NODE"] = str(node)

    extra_args = list(strategy.extra_args or [])
    return CompiledProgram(
        strategy=strategy,
        launcher_env=env,
        extra_args=extra_args,
        family=family,
        legality=legality,
    )
