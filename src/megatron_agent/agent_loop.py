from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from megatron_agent.config import (
    ConstraintRuleSpec,
    MegatronProgram,
    SearchSpaceSpec,
    default_dense_program,
    default_moe_smoke_program,
)
from megatron_agent.programs import check_program, classify_program_family, compile_program
from megatron_agent.trial_runner import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAUNCHER_SCRIPT,
    DEFAULT_MEGATRON_ROOT,
    DEFAULT_TOKENIZER_MODEL,
    add_observability_args,
    run_trial,
)


def _clone_program(program: MegatronProgram) -> MegatronProgram:
    return MegatronProgram.from_dict(program.to_dict())


def _score(metrics: Dict[str, Any]) -> float:
    if metrics.get("oom") or metrics.get("error_msg") or metrics.get("returncode") not in (0, None):
        return float("-inf")
    try:
        return float(metrics.get("throughput_tokens_per_s") or 0.0)
    except Exception:
        return 0.0


def _load_runtime_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _stage_load_variance(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not payload:
        return None
    existing = _safe_float(payload.get("stage_load_variance"))
    if existing is not None:
        return existing
    stage_summary = payload.get("stage_window_summary") or {}
    windows: List[float] = []
    for item in stage_summary.values():
        window_ms = _safe_float((item or {}).get("window_ms"))
        if window_ms is not None and window_ms > 0:
            windows.append(window_ms)
    if len(windows) < 2:
        return None
    mean = sum(windows) / float(len(windows))
    if mean <= 0:
        return 0.0
    return sum(((value / mean) - 1.0) ** 2 for value in windows) / float(len(windows))


def _observed_comm_ratio(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not payload:
        return None
    direct = _safe_float(payload.get("observed_comm_ratio"))
    if direct is not None:
        return direct
    return _safe_float(payload.get("comm_ratio_from_stages"))


def _first_observed_value(*sources: Optional[Dict[str, Any]], extractor) -> Optional[float]:
    for source in sources:
        value = extractor(source)
        if value is not None:
            return value
    return None


def _metric_delta(best_value: Optional[float], baseline_value: Optional[float]) -> Optional[float]:
    if best_value is None or baseline_value is None:
        return None
    return best_value - baseline_value


def _throughput_speedup(best_value: Optional[float], baseline_value: Optional[float]) -> Optional[float]:
    if best_value is None or baseline_value is None or baseline_value <= 0:
        return None
    return best_value / baseline_value


def _build_baseline_vs_best(
    baseline: MegatronProgram,
    baseline_metrics: Optional[Dict[str, Any]],
    best_program: Optional[MegatronProgram],
    best_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_throughput = _safe_float((baseline_metrics or {}).get("throughput_tokens_per_s"))
    best_throughput = _safe_float((best_metrics or {}).get("throughput_tokens_per_s"))
    baseline_step = _safe_float((baseline_metrics or {}).get("step_time_ms_p50"))
    best_step = _safe_float((best_metrics or {}).get("step_time_ms_p50"))
    baseline_bubble = _safe_float((baseline_metrics or {}).get("bubble_ratio"))
    best_bubble = _safe_float((best_metrics or {}).get("bubble_ratio"))
    baseline_comm = _observed_comm_ratio(baseline_metrics)
    best_comm = _observed_comm_ratio(best_metrics)
    baseline_stage_var = _stage_load_variance(baseline_metrics)
    best_stage_var = _stage_load_variance(best_metrics)
    baseline_hash = baseline.semantic_hash()
    best_hash = best_program.semantic_hash() if best_program is not None else None
    return {
        "available": baseline_metrics is not None and best_metrics is not None,
        "baseline_program_hash": baseline_hash,
        "best_program_hash": best_hash,
        "best_is_baseline": bool(best_hash is not None and best_hash == baseline_hash),
        "baseline_config_name": (baseline_metrics or {}).get("config_name", "baseline"),
        "best_config_name": (best_metrics or {}).get("config_name"),
        "throughput_tokens_per_s_delta": _metric_delta(best_throughput, baseline_throughput),
        "throughput_speedup": _throughput_speedup(best_throughput, baseline_throughput),
        "step_time_ms_p50_delta": _metric_delta(best_step, baseline_step),
        "bubble_ratio_delta": _metric_delta(best_bubble, baseline_bubble),
        "observed_comm_ratio_delta": _metric_delta(best_comm, baseline_comm),
        "stage_load_variance_delta": _metric_delta(best_stage_var, baseline_stage_var),
    }


def _slugify_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    compact = "_".join(part for part in safe.split("_") if part)
    return compact or "program"


def _export_programs(
    baseline: MegatronProgram,
    candidates: List[MegatronProgram],
    programs_dir: Path,
) -> List[Dict[str, Any]]:
    programs_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    items: List[Tuple[str, MegatronProgram]] = [("baseline", baseline)]
    for index, candidate in enumerate(candidates, start=1):
        label = str(candidate.metadata.get("program_kind") or f"candidate_{index:02d}")
        items.append((label, candidate))

    for execution_order, (config_name, program) in enumerate(items):
        filename = f"{execution_order:02d}_{_slugify_name(config_name)}.json"
        output_path = programs_dir / filename
        output_path.write_text(json.dumps(program.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        family = classify_program_family(program).to_dict()
        legality = check_program(program).to_dict()
        compile_success = False
        compile_error = None
        try:
            compile_program(program)
            compile_success = True
        except Exception as exc:
            compile_error = str(exc)
        manifest.append(
            {
                "config_name": config_name,
                "program_kind": str(program.metadata.get("program_kind") or config_name),
                "execution_order": int(execution_order),
                "is_baseline": bool(execution_order == 0),
                "program_hash": program.semantic_hash(),
                "program_path": str(output_path),
                "family": family,
                "legality": legality,
                "compile_success": compile_success,
                "compile_error": compile_error,
            }
        )
    return manifest


def _candidate_entries(candidate_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in candidate_manifest if not bool(entry.get("is_baseline"))]


def _build_summary_payload(
    *,
    export_only: bool,
    programs_dir: Path,
    runtime_summary: Dict[str, Any],
    rewrite: SearchSpaceSpec,
    baseline: MegatronProgram,
    baseline_metrics: Optional[Dict[str, Any]],
    best_program: Optional[MegatronProgram],
    best_metrics: Optional[Dict[str, Any]],
    tested: List[Dict[str, Any]],
    family_outside_trials: List[Dict[str, Any]],
    rejected_candidates: List[Dict[str, Any]],
    candidate_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_entries = _candidate_entries(candidate_manifest)
    compile_success_rate = None
    if candidate_entries:
        compile_success_rate = sum(1 for entry in candidate_entries if bool(entry.get("compile_success"))) / float(
            len(candidate_entries)
        )
    family_outside_ratio = 0.0
    if candidate_entries:
        family_outside_ratio = sum(
            1 for entry in candidate_entries if bool(((entry.get("family") or {}).get("is_family_outside")))
        ) / float(len(candidate_entries))

    summary = {
        "mode": "program_synthesis_export_only" if export_only else "program_synthesis_executor",
        "programs_dir": str(programs_dir),
        "runtime_summary": runtime_summary,
        "space_rewrite": rewrite.to_dict(),
        "baseline_program": baseline.to_dict(),
        "baseline_family": classify_program_family(baseline).to_dict(),
        "baseline_metrics": baseline_metrics,
        "best_program": best_program.to_dict() if best_program is not None else None,
        "best_metrics": best_metrics,
        "tested_trials": tested,
        "family_outside_trials": family_outside_trials,
        "rejected_candidates": rejected_candidates,
        "candidate_manifest": candidate_manifest,
        "recommended_execution_order": [entry["config_name"] for entry in candidate_manifest],
        "candidate_generation_count": len(candidate_entries),
        "candidate_execution_count": max(len(tested) - (1 if baseline_metrics is not None else 0), 0),
        "compile_success_rate": compile_success_rate,
        "family_outside_ratio": family_outside_ratio,
        "stage_load_variance": _first_observed_value(
            best_metrics,
            baseline_metrics,
            runtime_summary,
            extractor=_stage_load_variance,
        ),
        "observed_comm_ratio": _first_observed_value(
            best_metrics,
            baseline_metrics,
            runtime_summary,
            extractor=_observed_comm_ratio,
        ),
        "baseline_vs_best": _build_baseline_vs_best(
            baseline=baseline,
            baseline_metrics=baseline_metrics,
            best_program=best_program,
            best_metrics=best_metrics,
        ),
    }
    return summary


def _build_baseline_program(args: argparse.Namespace) -> MegatronProgram:
    program = default_moe_smoke_program(args.run_target) if args.model_track == "moe" else default_dense_program(args.run_target)
    program.parallel.tp_degree = int(args.tp or program.parallel.tp_degree)
    program.parallel.pp_degree = int(args.pp or program.parallel.pp_degree)
    program.parallel.vpp_degree = int(args.vpp or program.parallel.vpp_degree)
    program.parallel.ep_degree = int(args.ep or program.parallel.ep_degree)
    program.parallel.cp_degree = int(args.cp or program.parallel.cp_degree)
    program.parallel.expert_tp_degree = int(args.expert_tp or program.parallel.expert_tp_degree)
    program.parallel.sp_enabled = bool(int(program.parallel.tp_degree) > 1)
    program.layout.vpp_degree = int(program.parallel.vpp_degree)
    program.model.num_layers = int(args.num_layers or program.model.num_layers)
    if int(program.parallel.pp_degree) != program.partition.num_stages:
        decoder_layers = int(program.model.num_layers) // int(program.parallel.pp_degree)
        stages = []
        remaining = int(program.model.num_layers)
        for index in range(int(program.parallel.pp_degree)):
            stage_layers = decoder_layers
            if index == int(program.parallel.pp_degree) - 1:
                stage_layers = remaining
            remaining -= stage_layers
            special_tokens: List[str] = []
            if index == 0:
                special_tokens.append("E")
            if index == int(program.parallel.pp_degree) - 1:
                special_tokens.append("L")
            stages.append({"decoder_layers": stage_layers, "special_tokens": special_tokens})
        program.partition = program.partition.from_dict({"stages": stages})
        if args.run_target == "dual_g4_g5":
            midpoint = int(program.parallel.pp_degree) // 2
            program.layout.stage_to_node = ["g4"] * midpoint + ["g5"] * (int(program.parallel.pp_degree) - midpoint)
        else:
            program.layout.stage_to_node = [str(program.cluster.nodes[-1])] * int(program.parallel.pp_degree)
    program.metadata.update(
        {
            "micro_batch_size": int(args.micro_batch_size),
            "global_batch_size": int(args.global_batch_size),
            "seq_len": int(args.seq_len),
            "use_bf16": bool(args.bf16 or not args.fp16),
            "use_fp16": bool(args.fp16),
            "recompute_granularity": args.recompute_granularity,
            "program_kind": "baseline",
        }
    )
    program.search_space = SearchSpaceSpec()
    return program.normalized()


def _rewrite_space(program: MegatronProgram, runtime_summary: Dict[str, Any]) -> SearchSpaceSpec:
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    stage_spread = float(runtime_summary.get("stage_spread_ratio") or 0.0)
    cross_node_exposed = float(runtime_summary.get("cross_node_exposed_ratio") or runtime_summary.get("exposed_comm_ratio") or 0.0)
    is_dual = program.cluster.target == "dual_g4_g5"

    rules: List[ConstraintRuleSpec] = []
    required_local_axes = ["tp"] if is_dual else []
    if program.model.track == "moe":
        required_local_axes.append("ep")

    allow_nonuniform = bool(is_dual or stage_spread >= 0.08 or int(program.parallel.pp_degree) > 1)
    if allow_nonuniform:
        rules.append(
            ConstraintRuleSpec(
                name="relax_uniform_pp",
                rationale="stage spread or topology asymmetry indicates uniform stage prior is too restrictive",
                params={"stage_spread_ratio": stage_spread, "target": program.cluster.target},
            )
        )

    allow_single_node_pp_split = bool(
        program.cluster.target in {"single_g4", "single_g5"}
        and program.model.track == "dense"
        and int(program.parallel.pp_degree) == 1
        and int(program.parallel.tp_degree) >= 2
        and int(program.model.num_layers) >= 4
    )
    if allow_single_node_pp_split:
        rules.append(
            ConstraintRuleSpec(
                name="allow_single_node_pp_split",
                rationale="single-node dense baseline can be relaxed from tp-only into a 2-stage pp family-outside candidate",
                params={"tp_degree": int(program.parallel.tp_degree), "num_layers": int(program.model.num_layers)},
            )
        )

    allow_sequence_parallel_toggle = bool(int(program.parallel.tp_degree) > 1)
    if allow_sequence_parallel_toggle:
        rules.append(
            ConstraintRuleSpec(
                name="allow_sequence_parallel_toggle",
                rationale="Megatron sequence parallel is a TP-coupled optimization and can be toggled only when tp_degree > 1",
                params={
                    "tp_degree": int(program.parallel.tp_degree),
                    "baseline_sp_enabled": bool(program.parallel.sp_enabled),
                },
            )
        )

    allow_asymmetric_vpp = bool(
        int(program.parallel.pp_degree) > 1 and int(program.model.num_layers) % (int(program.parallel.pp_degree) * 2) == 0
    )
    if allow_asymmetric_vpp:
        rules.append(
            ConstraintRuleSpec(
                name="allow_asymmetric_vpp",
                rationale="model layers can be re-grouped into a vpp-aware schedule family",
                params={"num_layers": int(program.model.num_layers), "pp_degree": int(program.parallel.pp_degree)},
            )
        )

    allow_dual_plane = bool(program.model.track == "moe")
    if allow_dual_plane:
        rules.append(
            ConstraintRuleSpec(
                name="decouple_attention_and_moe_planes",
                rationale="MoE track benefits from separating attention TP/CP and expert EP/ETP decisions",
                params={"model_track": program.model.track},
            )
        )

    allow_stage_aware = bool(int(program.parallel.pp_degree) > 1 and (is_dual or bubble_ratio >= 0.03))
    if allow_stage_aware:
        rules.append(
            ConstraintRuleSpec(
                name="allow_stage_aware_schedule",
                rationale="bubble or dual-node boundary suggests fixed 1F1B skeleton may be suboptimal",
                params={"bubble_ratio": bubble_ratio},
            )
        )

    if cross_node_exposed > 0.0:
        rules.append(
            ConstraintRuleSpec(
                name="localize_high_frequency_axes",
                rationale="runtime logs show exposed communication on slow boundaries",
                params={"cross_node_exposed_ratio": cross_node_exposed},
            )
        )

    search_space = SearchSpaceSpec(
        allow_nonuniform_partition=allow_nonuniform,
        allow_single_node_pp_split=allow_single_node_pp_split,
        allow_sequence_parallel_toggle=allow_sequence_parallel_toggle,
        allow_asymmetric_vpp=allow_asymmetric_vpp,
        allow_dual_plane=allow_dual_plane,
        allow_stage_aware_schedule=allow_stage_aware,
        allow_subgraph_submeshes=False,
        allow_heterogeneous_apipe=False,
        max_tp_size=int(program.cluster.gpus_per_node) if is_dual else int(program.parallel.tp_degree),
        max_pp_size=min(int(program.cluster.world_size), max(int(program.parallel.pp_degree), 2)),
        max_ep_size=int(program.cluster.gpus_per_node) if is_dual else None,
        max_cp_size=int(program.cluster.gpus_per_node) if is_dual else None,
        max_vpp_size=2 if allow_asymmetric_vpp else 1,
        required_node_local_axes=required_local_axes,
        preferred_node_for_module={"embedding": "g4", "loss": "g5"} if is_dual else {"embedding": str(program.cluster.nodes[-1]), "loss": str(program.cluster.nodes[-1])},
        forbidden_axes_by_node={"g5": ["tp"]} if is_dual and cross_node_exposed >= 0.05 else {},
        allowed_schedule_skeletons=["fixed_1f1b", "stage_aware_grouped"] if allow_stage_aware else ["fixed_1f1b"],
        rewrite_rules=rules,
        notes="space rewrite derived from topology + runtime summary",
    )
    return search_space.normalized()


def _build_nonuniform_partition(program: MegatronProgram) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) < 2:
        return None
    candidate = _clone_program(program)
    stage_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
    if len(stage_layers) < 2 or min(stage_layers) <= 1:
        return None
    shift = min(2, stage_layers[-1] - 1)
    if shift <= 0:
        return None
    stage_layers[0] += shift
    stage_layers[-1] -= shift
    for index, stage in enumerate(candidate.partition.stages):
        stage.decoder_layers = stage_layers[index]
    candidate.layout.pipeline_layout = None
    candidate.metadata["program_kind"] = "candidate_nonuniform_partition"
    return candidate.normalized()


def _build_single_node_pipeline_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if program.cluster.target not in {"single_g4", "single_g5"} or int(program.parallel.pp_degree) != 1:
        return None
    if int(program.parallel.tp_degree) < 2 or int(program.model.num_layers) < 4:
        return None
    candidate = _clone_program(program)
    candidate.parallel.tp_degree = max(1, int(program.parallel.tp_degree) // 2)
    candidate.parallel.pp_degree = 2
    candidate.parallel.sp_enabled = bool(int(candidate.parallel.tp_degree) > 1)
    candidate.plane_map.attention.tp_degree = int(candidate.parallel.tp_degree)
    candidate.plane_map.attention.cp_degree = int(candidate.parallel.cp_degree)
    first = int(program.model.num_layers) // 2 + 2
    second = int(program.model.num_layers) - first
    if second <= 0:
        return None
    candidate.partition = candidate.partition.from_dict(
        {
            "stages": [
                {"decoder_layers": first, "special_tokens": ["E"]},
                {"decoder_layers": second, "special_tokens": ["L"]},
            ]
        }
    )
    node_name = str(candidate.cluster.nodes[-1])
    candidate.layout.stage_to_node = [node_name, node_name]
    candidate.metadata["program_kind"] = "candidate_single_node_pp_split"
    return candidate.normalized()


def _build_stage_aware_schedule(program: MegatronProgram) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    candidate = _clone_program(program)
    if int(candidate.parallel.vpp_degree) == 1:
        total_virtual = int(candidate.parallel.pp_degree) * 2
        if int(candidate.model.num_layers) % total_virtual != 0:
            return None
        candidate.parallel.vpp_degree = 2
        candidate.layout.vpp_degree = 2
        candidate.parallel.sp_enabled = bool(int(candidate.parallel.tp_degree) > 1)
    candidate.schedule.microbatch_group_size_per_vp_stage = 2
    candidate.schedule.skeleton = "stage_aware_grouped"
    candidate.schedule.dispatch_order = "frontload_forward"
    candidate.metadata["program_kind"] = "candidate_stage_aware_schedule"
    return candidate.normalized()


def _build_sequence_parallel_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if int(program.parallel.tp_degree) <= 1:
        return None
    candidate = _clone_program(program)
    candidate.parallel.sp_enabled = not bool(program.parallel.sp_enabled)
    candidate.metadata["program_kind"] = "candidate_sequence_parallel_toggle"
    candidate.metadata["sequence_parallel_target_state"] = bool(candidate.parallel.sp_enabled)
    return candidate.normalized()


def _build_dual_plane_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if program.model.track != "moe":
        return None
    candidate = _clone_program(program)
    if not candidate.plane_map.enabled:
        candidate.plane_map.enabled = True
    candidate.parallel.tp_degree = 1
    candidate.parallel.sp_enabled = False
    candidate.parallel.ep_degree = max(2, int(candidate.parallel.ep_degree))
    candidate.parallel.expert_tp_degree = max(1, int(candidate.parallel.expert_tp_degree))
    candidate.plane_map.attention.tp_degree = max(1, int(program.parallel.tp_degree))
    if candidate.plane_map.moe is not None:
        candidate.plane_map.moe.ep_degree = max(2, int(candidate.parallel.ep_degree))
        candidate.plane_map.moe.expert_tp_degree = max(1, int(candidate.parallel.expert_tp_degree))
    candidate.metadata["program_kind"] = "candidate_dual_plane"
    return candidate.normalized()


def _build_topology_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if program.cluster.target != "dual_g4_g5" or int(program.parallel.pp_degree) < 2:
        return None
    candidate = _clone_program(program)
    midpoint = max(int(candidate.parallel.pp_degree) // 2, 1)
    candidate.layout.stage_to_node = ["g4"] * midpoint + ["g5"] * (int(candidate.parallel.pp_degree) - midpoint)
    candidate.metadata["program_kind"] = "candidate_topology_layout"
    return candidate.normalized()


def _candidate_allowed_by_space(program: MegatronProgram, search_space: SearchSpaceSpec) -> Tuple[bool, str]:
    program_kind = str(program.metadata.get("program_kind") or "")
    is_single_node_pp_split = program_kind == "candidate_single_node_pp_split"
    is_sequence_parallel_toggle = program_kind == "candidate_sequence_parallel_toggle"
    if search_space.max_tp_size is not None and int(program.parallel.tp_degree) > int(search_space.max_tp_size):
        return False, f"tp_degree={program.parallel.tp_degree} exceeds search-space max_tp_size={search_space.max_tp_size}"
    if search_space.max_pp_size is not None and int(program.parallel.pp_degree) > int(search_space.max_pp_size):
        return False, f"pp_degree={program.parallel.pp_degree} exceeds search-space max_pp_size={search_space.max_pp_size}"
    if search_space.max_ep_size is not None and int(program.parallel.ep_degree) > int(search_space.max_ep_size):
        return False, f"ep_degree={program.parallel.ep_degree} exceeds search-space max_ep_size={search_space.max_ep_size}"
    if search_space.max_cp_size is not None and int(program.parallel.cp_degree) > int(search_space.max_cp_size):
        return False, f"cp_degree={program.parallel.cp_degree} exceeds search-space max_cp_size={search_space.max_cp_size}"
    if search_space.max_vpp_size is not None and int(program.parallel.vpp_degree) > int(search_space.max_vpp_size):
        return False, f"vpp_degree={program.parallel.vpp_degree} exceeds search-space max_vpp_size={search_space.max_vpp_size}"
    if not search_space.allow_dual_plane and program.plane_map.enabled:
        return False, "dual-plane mapping is not allowed in the current search space"
    if not search_space.allow_stage_aware_schedule and program.schedule.skeleton != "fixed_1f1b":
        return False, "stage-aware schedule is not allowed in the current search space"
    if program.schedule.skeleton not in set(search_space.allowed_schedule_skeletons):
        return False, f"schedule skeleton {program.schedule.skeleton} is outside allowed search-space skeletons"
    if not search_space.allow_asymmetric_vpp and int(program.parallel.vpp_degree) > 1:
        return False, "asymmetric VPP is not allowed in the current search space"
    if is_single_node_pp_split and not search_space.allow_single_node_pp_split:
        return False, "single-node PP split is not allowed in the current search space"
    if is_sequence_parallel_toggle and not search_space.allow_sequence_parallel_toggle:
        return False, "sequence parallel toggle is not allowed in the current search space"
    if bool(program.parallel.sp_enabled) and int(program.parallel.tp_degree) <= 1:
        return False, "sequence parallel requires tp_degree > 1"
    if not search_space.allow_nonuniform_partition:
        stage_layers = [int(stage.decoder_layers) for stage in program.partition.stages]
        if len(set(stage_layers)) > 1 and not (is_single_node_pp_split and search_space.allow_single_node_pp_split):
            return False, "nonuniform partition is not allowed in the current search space"
    return True, "allowed"


def _synthesize_programs(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    candidate_limit: int,
) -> Tuple[List[MegatronProgram], List[Dict[str, Any]]]:
    candidates: List[MegatronProgram] = []
    rejected: List[Dict[str, Any]] = []
    seen = {baseline.semantic_hash()}

    builders = []
    if rewrite.allow_single_node_pp_split:
        builders.append(_build_single_node_pipeline_candidate)
    if rewrite.allow_nonuniform_partition:
        builders.append(_build_nonuniform_partition)
    if rewrite.allow_stage_aware_schedule:
        builders.append(_build_stage_aware_schedule)
    if rewrite.allow_dual_plane:
        builders.append(_build_dual_plane_candidate)
    if baseline.cluster.target == "dual_g4_g5":
        builders.append(_build_topology_candidate)
    if rewrite.allow_sequence_parallel_toggle:
        builders.append(_build_sequence_parallel_candidate)

    for builder in builders:
        candidate = builder(baseline)
        if candidate is None:
            continue
        hash_value = candidate.semantic_hash()
        if hash_value in seen:
            continue
        seen.add(hash_value)
        candidate.search_space = rewrite.normalized()
        allowed, allowed_reason = _candidate_allowed_by_space(candidate, rewrite)
        if not allowed:
            rejected.append({"program": candidate.to_dict(), "reason": allowed_reason})
            continue
        legality = check_program(candidate)
        if not legality.is_valid:
            rejected.append({"program": candidate.to_dict(), "reason": legality.to_dict()})
            continue
        candidates.append(candidate)
        if len(candidates) >= int(candidate_limit):
            break

    return candidates, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron program synthesis bring-up runner.")
    parser.add_argument("--workdir", type=str, default="./runs_megatron_programs")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--programs-dir", type=str, default=None)
    parser.add_argument("--runtime-summary", type=str, default=None)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--run-target", type=str, choices=["single_g4", "single_g5", "dual_g4_g5"], default="single_g5")
    parser.add_argument("--model-track", type=str, choices=["dense", "moe"], default="dense")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=0)
    parser.add_argument("--pp", type=int, default=0)
    parser.add_argument("--vpp", type=int, default=0)
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--cp", type=int, default=0)
    parser.add_argument("--expert-tp", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--recompute-granularity", type=str, default="selective")
    parser.add_argument("--megatron-root", type=str, default=DEFAULT_MEGATRON_ROOT)
    parser.add_argument("--launcher-script", type=str, default=DEFAULT_LAUNCHER_SCRIPT)
    parser.add_argument("--megatron-entry", type=str, default="pretrain_gpt.py")
    parser.add_argument("--megatron-args", type=str, default=None)
    parser.add_argument("--megatron-args-file", type=str, default=None)
    parser.add_argument("--transformer-impl", type=str, default="local")
    parser.add_argument("--attention-backend", type=str, default="auto")
    parser.add_argument("--tokenizer-model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument("--enable-tp-comm-overlap", action="store_true")
    add_observability_args(parser)
    parser.add_argument("--run-root", type=str, default="./runs_megatron")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--ffn-hidden-size", type=int, default=17408)
    parser.add_argument("--num-attention-heads", type=int, default=40)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--kv-channels", type=int, default=128)
    parser.add_argument("--max-position-embeddings", type=int, default=40960)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--moe-hidden-size", type=int, default=1024)
    parser.add_argument("--moe-ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--moe-num-attention-heads", type=int, default=16)
    parser.add_argument("--moe-num-query-groups", type=int, default=4)
    parser.add_argument("--moe-kv-channels", type=int, default=64)
    parser.add_argument("--moe-max-position-embeddings", type=int, default=4096)
    parser.add_argument("--moe-vocab-size", type=int, default=32768)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--data-split", type=str, default="99,1,0")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--lr-decay-style", type=str, default="cosine")
    parser.add_argument("--lr-warmup-iters", type=int, default=5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--distributed-timeout-minutes", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_target == "dual_g4_g5":
        args.nnodes = max(int(args.nnodes), 2)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    programs_dir = Path(args.programs_dir) if args.programs_dir else workdir / "programs"

    runtime_summary = _load_runtime_summary(args.runtime_summary)
    baseline = _build_baseline_program(args)
    baseline_legality = check_program(baseline)
    if not baseline_legality.is_valid:
        raise ValueError(f"Baseline program is invalid: {json.dumps(baseline_legality.to_dict(), ensure_ascii=False)}")

    rewrite = _rewrite_space(baseline, runtime_summary)
    baseline.search_space = rewrite.normalized()
    candidates, rejected_candidates = _synthesize_programs(
        baseline,
        rewrite=rewrite,
        candidate_limit=int(args.candidate_limit),
    )
    candidate_manifest = _export_programs(baseline, candidates, programs_dir)

    tested: List[Dict[str, Any]] = []
    family_outside_trials: List[Dict[str, Any]] = []
    baseline_metrics: Optional[Dict[str, Any]] = None
    best_program: Optional[MegatronProgram] = None
    best_metrics: Optional[Dict[str, Any]] = None

    if not args.export_only:
        baseline_metrics = run_trial(args, baseline, trial_id=0)
        baseline_metrics["config_name"] = "baseline"
        tested.append(baseline_metrics)
        if bool((baseline_metrics.get("family") or {}).get("is_family_outside")):
            family_outside_trials.append(baseline_metrics)

        best_program = baseline
        best_metrics = baseline_metrics
        best_score = _score(baseline_metrics)

        for index, candidate in enumerate(candidates, start=1):
            metrics = run_trial(args, candidate, trial_id=index)
            metrics["config_name"] = candidate.metadata.get("program_kind", f"candidate_{index:02d}")
            tested.append(metrics)
            if bool((metrics.get("family") or {}).get("is_family_outside")):
                family_outside_trials.append(metrics)
            score = _score(metrics)
            if score > best_score:
                best_score = score
                best_program = candidate
                best_metrics = metrics

    summary = _build_summary_payload(
        export_only=bool(args.export_only),
        programs_dir=programs_dir,
        runtime_summary=runtime_summary,
        rewrite=rewrite,
        baseline=baseline,
        baseline_metrics=baseline_metrics,
        best_program=best_program,
        best_metrics=best_metrics,
        tested=tested,
        family_outside_trials=family_outside_trials,
        rejected_candidates=rejected_candidates,
        candidate_manifest=candidate_manifest,
    )
    summary_path = workdir / "summary_megatron.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[megatron_agent] summary written to {summary_path}")


if __name__ == "__main__":
    main()
