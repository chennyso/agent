from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from megatron_agent.config import (
    AgentObservation,
    AgentProposal,
    BackendCaps,
    BatchPlanSpec,
    ConstraintRuleSpec,
    ExperimentSpec,
    ProgramBank,
    ProgramTemplate,
    ReplanDecision,
    MachineProfile,
    MegatronProgram,
    SearchSpaceSpec,
    VerifierReport,
    default_backend_caps,
    default_dense_program,
    default_grouped_stage_to_node,
    default_length_bucket_policies,
    default_machine_profile,
    default_moe_smoke_program,
    weighted_stage_layer_allocation,
)
from megatron_agent.programs import (
    assess_vpp_comm_tradeoff,
    check_program,
    classify_program_family,
    compile_program,
    estimate_program_memory,
    verify_program,
)
from megatron_agent.trace_reducer import (
    build_agent_observation,
    build_context_record,
    build_trial_artifact,
    build_program_bank,
    classify_bottleneck,
    detect_failure_modes,
    reduce_trial_trace,
    select_program_templates,
)
from megatron_agent.trial_runner import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAUNCHER_SCRIPT,
    DEFAULT_MEGATRON_ROOT,
    DEFAULT_TOKENIZER_MODEL,
    add_observability_args,
    run_trial,
)


def _progress(message: str) -> None:
    print(f"[megatron_agent] {message}", flush=True)


def _llm_supervisor_enabled(llm_config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(llm_config, dict):
        return False
    if not bool(llm_config.get("enabled")):
        return False
    return bool(str(llm_config.get("endpoint") or "").strip() and str(llm_config.get("model") or "").strip())


def _agent_topology_summary(llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    llm_enabled = _llm_supervisor_enabled(llm_config)
    return {
        "planner_mode": "llm_http" if llm_enabled else "heuristic_only",
        "logical_roles": ["planner", "verifier", "executor"],
        "logical_agent_count": 3,
        "llm_planner_agents": 1 if llm_enabled else 0,
        "heuristic_planner_agents": 0 if llm_enabled else 1,
        "verifier_agents": 1,
        "executor_agents": 1,
        "llm_endpoint": str((llm_config or {}).get("endpoint") or "") if llm_enabled else None,
        "llm_model": str((llm_config or {}).get("model") or "") if llm_enabled else None,
    }


def _load_json_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_external_agent_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_structure_summary": _load_json_file(getattr(args, "model_structure_summary", None)),
        "hardware_topology_summary": _load_json_file(getattr(args, "hardware_topology_summary", None)),
        "profile_summary": _load_json_file(getattr(args, "profile_summary", None)),
        "baseline_catalog": _load_json_file(getattr(args, "baseline_catalog", None)),
    }


def _structure_summary_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    layers = list(payload.get("layers") or payload.get("layer_types") or [])
    type_counter: Dict[str, int] = {}
    for item in layers:
        if isinstance(item, dict):
            token = str(item.get("type") or item.get("module_type") or "unknown").strip().lower()
        else:
            token = str(item or "unknown").strip().lower()
        if not token:
            token = "unknown"
        type_counter[token] = int(type_counter.get(token, 0)) + 1
    total_layers = int(payload.get("num_layers") or len(layers) or 0)
    if total_layers <= 0:
        total_layers = sum(type_counter.values())
    attention_layers = sum(value for key, value in type_counter.items() if "attn" in key or "attention" in key)
    mlp_layers = sum(value for key, value in type_counter.items() if "mlp" in key or "ffn" in key)
    moe_layers = sum(value for key, value in type_counter.items() if "moe" in key or "expert" in key)
    vocab_layers = sum(value for key, value in type_counter.items() if key in {"embedding", "lm_head", "vocab", "loss"})
    return {
        "total_layers": int(total_layers),
        "attention_layers": int(attention_layers),
        "mlp_layers": int(mlp_layers),
        "moe_layers": int(moe_layers),
        "vocab_related_layers": int(vocab_layers),
        "layer_type_histogram": dict(type_counter),
        "has_moe": bool(moe_layers > 0),
        "edge_heavy_vocab": bool(vocab_layers > 0),
    }


def _topology_summary_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    links = list(payload.get("links") or payload.get("edges") or [])
    nvlink_links = 0
    pcie_links = 0
    ib_links = 0
    bandwidth_values: List[float] = []
    for item in links:
        token = str((item or {}).get("type") or (item or {}).get("fabric") or "").strip().lower()
        bandwidth = _safe_float((item or {}).get("bandwidth_gbps") or (item or {}).get("bandwidth"))
        if bandwidth is not None and bandwidth > 0.0:
            bandwidth_values.append(float(bandwidth))
        if "nvlink" in token:
            nvlink_links += 1
        elif "pcie" in token:
            pcie_links += 1
        elif "ib" in token or "infiniband" in token:
            ib_links += 1
    dominant = "unknown"
    dominant_count = max(nvlink_links, pcie_links, ib_links)
    if dominant_count > 0:
        dominant = (
            "nvlink"
            if nvlink_links == dominant_count
            else "pcie" if pcie_links == dominant_count else "infiniband"
        )
    return {
        "node_count": int(payload.get("node_count") or payload.get("num_nodes") or 0),
        "gpu_count": int(payload.get("gpu_count") or payload.get("world_size") or 0),
        "nvlink_links": int(nvlink_links),
        "pcie_links": int(pcie_links),
        "infiniband_links": int(ib_links),
        "dominant_fabric": dominant,
        "mean_bandwidth_gbps": round(sum(bandwidth_values) / float(len(bandwidth_values)), 4) if bandwidth_values else 0.0,
    }


def _profile_summary_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    layers = list(payload.get("layers") or payload.get("layer_profiles") or [])
    forward_values: List[float] = []
    backward_values: List[float] = []
    activation_values: List[float] = []
    comm_values: List[float] = []
    for item in layers:
        forward = _safe_float((item or {}).get("forward_ms"))
        backward = _safe_float((item or {}).get("backward_ms"))
        activation = _safe_float((item or {}).get("activation_mb") or (item or {}).get("activation_mib"))
        comm = _safe_float((item or {}).get("communication_ms") or (item or {}).get("comm_ms"))
        if forward is not None and forward > 0.0:
            forward_values.append(float(forward))
        if backward is not None and backward > 0.0:
            backward_values.append(float(backward))
        if activation is not None and activation > 0.0:
            activation_values.append(float(activation))
        if comm is not None and comm > 0.0:
            comm_values.append(float(comm))
    return {
        "layer_profile_count": len(layers),
        "mean_forward_ms": round(sum(forward_values) / float(len(forward_values)), 4) if forward_values else 0.0,
        "mean_backward_ms": round(sum(backward_values) / float(len(backward_values)), 4) if backward_values else 0.0,
        "peak_activation_mb": round(max(activation_values), 4) if activation_values else 0.0,
        "mean_comm_ms": round(sum(comm_values) / float(len(comm_values)), 4) if comm_values else 0.0,
        "peak_memory_gib": round(float(_safe_float(payload.get("peak_memory_gib")) or 0.0), 4),
    }


def _baseline_catalog_digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    baselines = list(payload.get("baselines") or payload.get("entries") or [])
    ranked: List[Dict[str, Any]] = []
    for item in baselines:
        step = _safe_float((item or {}).get("step_time_ms"))
        throughput = _safe_float((item or {}).get("throughput"))
        ranked.append(
            {
                "name": str((item or {}).get("name") or (item or {}).get("template") or "baseline"),
                "pp": int((item or {}).get("pp") or 1),
                "vpp": int((item or {}).get("vpp") or 1),
                "family": str((item or {}).get("family") or ""),
                "step_time_ms": float(step or 0.0),
                "throughput": float(throughput or 0.0),
            }
        )
    ranked.sort(key=lambda item: (item["step_time_ms"] <= 0.0, item["step_time_ms"] if item["step_time_ms"] > 0.0 else -item["throughput"]))
    return {
        "baseline_count": len(ranked),
        "best_baseline": ranked[0] if ranked else None,
        "ranked_baselines": ranked[:8],
    }


def _augment_context_with_external_inputs(
    context_record: Dict[str, Any],
    external_inputs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not external_inputs:
        return context_record
    merged = json.loads(json.dumps(context_record or {}, ensure_ascii=False))
    model_payload = dict(external_inputs.get("model_structure_summary") or {})
    topology_payload = dict(external_inputs.get("hardware_topology_summary") or {})
    profile_payload = dict(external_inputs.get("profile_summary") or {})
    baseline_payload = dict(external_inputs.get("baseline_catalog") or {})

    if model_payload:
        merged.setdefault("model_context", {})["external_structure_summary"] = model_payload
        merged["model_context"]["structure_digest"] = _structure_summary_digest(model_payload)
    if topology_payload:
        merged.setdefault("hardware_context", {})["external_topology_summary"] = topology_payload
        merged["hardware_context"]["topology_digest"] = _topology_summary_digest(topology_payload)
    if profile_payload:
        merged.setdefault("evidence_record", {})["external_profile_summary"] = profile_payload
        merged["evidence_record"]["profile_digest"] = _profile_summary_digest(profile_payload)
    if baseline_payload:
        merged.setdefault("evidence_record", {})["external_baseline_catalog"] = baseline_payload
        merged["evidence_record"]["baseline_catalog_digest"] = _baseline_catalog_digest(baseline_payload)
    return merged


def _batch_profile(context_record: Dict[str, Any]) -> str:
    workload = dict((context_record or {}).get("workload_context") or {})
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    model_context = dict((context_record or {}).get("model_context") or {})
    seq_len = int(workload.get("seq_len") or 1024)
    length_bucket = str(workload.get("length_bucket") or "default")
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    if str(model_context.get("track") or "") == "moe":
        return "moe_heavy"
    if seq_len >= 4096 or "long" in length_bucket:
        return "long_context"
    if peak_reserved_ratio >= 0.88:
        return "memory_constrained"
    return "normal"


def _call_llm_chat_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
) -> str:
    payload = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(prompt)},
        ],
        "temperature": float(temperature),
        "stream": False,
    }
    resp = requests.post(
        str(endpoint),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False),
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = list(data.get("choices") or [])
    if not choices:
        raise RuntimeError(f"unexpected LLM response: {data}")
    return str((((choices[0] or {}).get("message") or {}).get("content")) or "").strip()


def _robust_parse_llm_json(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```json\s*", "", str(text), flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _proposal_prompt_entry(proposal: AgentProposal) -> Dict[str, Any]:
    program = proposal.program.normalized()
    return {
        "proposal_id": str(proposal.proposal_id),
        "scope": str(proposal.scope),
        "program_kind": str(program.metadata.get("program_kind") or "candidate"),
        "template": str(program.schedule.template or "fixed_1f1b"),
        "dispatch_order": str(program.schedule.dispatch_order or "default"),
        "pp": int(program.parallel.pp_degree),
        "vpp": int(program.parallel.vpp_degree),
        "tp": int(program.parallel.tp_degree),
        "cp": int(program.parallel.cp_degree),
        "ep": int(program.parallel.ep_degree),
        "priority_rank": int(proposal.priority_rank),
        "rationale": str(proposal.rationale or ""),
        "memory_policy": str(program.metadata.get("runtime_memory_policy_mode") or "none"),
        "estimated_step_delta_ms": float(program.metadata.get("morphable_estimated_step_delta_ms") or 0.0),
    }


def _build_llm_supervisor_prompt(
    proposals: List[AgentProposal],
    *,
    context_record: Dict[str, Any],
    replan_decision: Dict[str, Any],
    candidate_limit: int,
) -> str:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    bottlenecks = list((context_record or {}).get("derived_bottlenecks") or [])
    payload = {
        "objective": "maximize throughput under memory budget",
        "hard_constraints": [
            "peak memory must remain within budget",
            "dependency legality must hold",
            "communication schedule must remain executable",
        ],
        "replan_decision": {
            "trigger": str(replan_decision.get("trigger") or "steady"),
            "scope": str(replan_decision.get("scope") or "none"),
            "rationale": str(replan_decision.get("rationale") or ""),
        },
        "runtime_evidence": {
            "bubble_ratio": float(runtime.get("bubble_ratio") or 0.0),
            "pipeline_wait_ratio": float(runtime.get("pipeline_wait_ratio") or 0.0),
            "optimizer_exposed_ratio": float(runtime.get("optimizer_exposed_ratio") or 0.0),
            "cross_node_exposed_ratio": float(runtime.get("cross_node_exposed_ratio") or 0.0),
            "peak_reserved_ratio": float(runtime.get("peak_reserved_ratio") or 0.0),
            "memory_budget_gb": float(runtime.get("memory_budget_gb") or runtime.get("memory_limit_gb") or 0.0),
        },
        "derived_bottlenecks": [
            {
                "label": str(item.get("label") or ""),
                "severity_label": str(item.get("severity") or ""),
                "severity_score": float(_safe_float(item.get("severity_score")) or 0.0),
            }
            for item in bottlenecks[:8]
        ],
        "candidate_limit": int(candidate_limit),
        "proposals": [_proposal_prompt_entry(item) for item in proposals[:12]],
        "output_schema": {
            "selected_proposal_ids": ["proposal_id_1", "proposal_id_2"],
            "rationales": {"proposal_id_1": "short reason"},
            "agent_topology": {
                "llm_planner_agents": 1,
                "verifier_agents": 1,
                "executor_agents": 1,
            },
            "notes": ["optional note"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _apply_llm_supervisor(
    proposals: List[AgentProposal],
    *,
    context_record: Dict[str, Any],
    replan_decision: Dict[str, Any],
    candidate_limit: int,
    llm_config: Optional[Dict[str, Any]],
) -> List[AgentProposal]:
    if not proposals or not _llm_supervisor_enabled(llm_config):
        return proposals
    system_prompt = (
        "You are the planner in a Megatron pipeline optimization agent. "
        "Select and rank candidate programs that best improve throughput under a hard memory budget. "
        "You may reorder proposals and provide concise rationales, but you must not invent new candidates. "
        "Return JSON only."
    )
    prompt = _build_llm_supervisor_prompt(
        proposals,
        context_record=context_record,
        replan_decision=replan_decision,
        candidate_limit=candidate_limit,
    )
    if bool((llm_config or {}).get("log_llm")):
        print("[megatron_agent] llm supervisor prompt >>>")
        print(prompt)
    try:
        reply = _call_llm_chat_completion(
            endpoint=str((llm_config or {}).get("endpoint") or ""),
            model=str((llm_config or {}).get("model") or ""),
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=float((llm_config or {}).get("temperature") or 0.0),
        )
        if bool((llm_config or {}).get("log_llm")):
            print("[megatron_agent] llm supervisor reply <<<")
            print(reply)
        plan = _robust_parse_llm_json(reply)
    except Exception:
        return proposals

    selected_ids = [str(item) for item in (plan.get("selected_proposal_ids") or []) if str(item).strip()]
    rationales = {
        str(key): str(value)
        for key, value in dict(plan.get("rationales") or {}).items()
        if str(key).strip() and str(value).strip()
    }
    topology = dict(plan.get("agent_topology") or {})
    ordered: List[AgentProposal] = []
    seen: set[str] = set()
    proposal_by_id = {str(item.proposal_id): item for item in proposals}
    for proposal_id in selected_ids:
        proposal = proposal_by_id.get(str(proposal_id))
        if proposal is None:
            continue
        seen.add(str(proposal_id))
        updated = proposal.normalized()
        updated.source = "llm_supervisor"
        updated.rationale = rationales.get(str(proposal_id), updated.rationale)
        updated.program.metadata["planner_backend"] = "llm_http"
        updated.program.metadata["planner_model"] = str((llm_config or {}).get("model") or "")
        updated.program.metadata["agent_topology"] = _agent_topology_summary(llm_config)
        if topology:
            updated.program.metadata["llm_declared_topology"] = topology
        ordered.append(updated.normalized())
    for proposal in proposals:
        if str(proposal.proposal_id) in seen:
            continue
        updated = proposal.normalized()
        if str(updated.source or "") == "heuristic_supervisor":
            updated.program.metadata["agent_topology"] = _agent_topology_summary(llm_config)
        ordered.append(updated.normalized())
    return ordered or proposals


def _clone_program(program: MegatronProgram) -> MegatronProgram:
    return MegatronProgram.from_dict(program.to_dict())


def _data_parallel_size(program: MegatronProgram) -> int:
    norm = program.normalized()
    product = (
        int(norm.parallel.tp_degree)
        * int(norm.parallel.pp_degree)
        * int(norm.parallel.cp_degree)
        * int(norm.parallel.ep_degree)
        * int(norm.parallel.expert_tp_degree)
    )
    if product <= 0:
        return 1
    return max(int(norm.cluster.world_size) // product, 1)


def _resolved_grad_accum_steps(program: MegatronProgram) -> int:
    norm = program.normalized()
    if norm.batch_plan.grad_accum_steps is not None:
        return max(int(norm.batch_plan.grad_accum_steps), 1)
    denom = max(int(norm.batch_plan.micro_batch_size) * _data_parallel_size(norm), 1)
    return max(int(norm.batch_plan.global_batch_size) // denom, 1)


def _sync_batch_plan_metadata(program: MegatronProgram) -> MegatronProgram:
    candidate = _clone_program(program)
    candidate.batch_plan.grad_accum_steps = _resolved_grad_accum_steps(candidate)
    candidate.metadata["micro_batch_size"] = int(candidate.batch_plan.micro_batch_size)
    candidate.metadata["global_batch_size"] = int(candidate.batch_plan.global_batch_size)
    candidate.metadata["grad_accum_steps"] = int(candidate.batch_plan.grad_accum_steps or 1)
    if candidate.batch_plan.target_tokens_per_step is None:
        candidate.batch_plan.target_tokens_per_step = (
            int(candidate.batch_plan.global_batch_size) * int(candidate.metadata.get("seq_len", 1024) or 1024)
        )
    candidate.metadata["target_tokens_per_step"] = int(candidate.batch_plan.target_tokens_per_step or 0)
    candidate.strategy_ir.pipe.template = str(candidate.schedule.template)
    candidate.strategy_ir.pipe.microbatch_order = str(candidate.schedule.dispatch_order)
    candidate.strategy_ir.pipe.steady_state_group_size = candidate.schedule.microbatch_group_size_per_vp_stage
    stage_local_vpp_by_stage: Dict[int, int] = {}
    raw_stage_local_vpp = candidate.metadata.get("stage_local_vpp_vector")
    if isinstance(raw_stage_local_vpp, dict):
        for key, value in raw_stage_local_vpp.items():
            stage_id = _safe_int(key)
            stage_vpp = _safe_int(value)
            if stage_id is None or stage_vpp is None:
                continue
            stage_local_vpp_by_stage[int(stage_id)] = max(int(stage_vpp), 1)
    elif isinstance(raw_stage_local_vpp, list):
        for stage_id, value in enumerate(raw_stage_local_vpp):
            stage_vpp = _safe_int(value)
            if stage_vpp is None:
                continue
            stage_local_vpp_by_stage[int(stage_id)] = max(int(stage_vpp), 1)
    preserve_stage_local_vpp = bool(candidate.metadata.get("preserve_stage_local_vpp", False))
    local_by_name = {entry.subgraph: entry for entry in (candidate.strategy_ir.local_parallel or [])}
    for subgraph in candidate.strategy_ir.apipe:
        entry = local_by_name.get(subgraph.name)
        if entry is None:
            continue
        stage_index = int(subgraph.stage_index)
        if stage_index in stage_local_vpp_by_stage:
            entry.vpp_degree = max(int(stage_local_vpp_by_stage[stage_index]), 1)
        elif not preserve_stage_local_vpp:
            entry.vpp_degree = max(int(candidate.parallel.vpp_degree), 1)
        else:
            entry.vpp_degree = max(int(entry.vpp_degree), 1)
        entry.cp_degree = max(entry.cp_degree, int(candidate.parallel.cp_degree))
    if stage_local_vpp_by_stage:
        max_stage_index = max(stage_local_vpp_by_stage)
        candidate.metadata["stage_local_vpp_vector"] = [
            int(stage_local_vpp_by_stage.get(index, max(int(candidate.parallel.vpp_degree), 1)))
            for index in range(max_stage_index + 1)
        ]
    return candidate.normalized()


def _virtual_stage_layout(decoder_layers: List[int]) -> str:
    tokens_per_stage: List[str] = []
    total = len(decoder_layers)
    for index, layers in enumerate(decoder_layers):
        prefix = "E" if index == 0 else ""
        suffix = "L" if index == total - 1 else ""
        tokens_per_stage.append(f"{prefix}{'t' * max(int(layers), 0)}{suffix}")
    return "|".join(tokens_per_stage)


def _stage_local_virtual_counts(
    stage_layers: List[int],
    stage_vpp_vector: List[int],
    *,
    global_vpp: int,
    focus: str = "balanced",
) -> List[int]:
    if global_vpp <= 1:
        return [max(int(value), 0) for value in stage_layers]

    stage_slot_counts: List[List[int]] = []
    for stage_id, layers in enumerate(stage_layers):
        local_vpp = 1
        if stage_id < len(stage_vpp_vector):
            local_vpp = max(min(int(stage_vpp_vector[stage_id]), int(global_vpp)), 1)
        layers = max(int(layers), 0)
        slots = [0 for _ in range(global_vpp)]
        if local_vpp == 1:
            slots[0] = layers
        else:
            base = layers // local_vpp
            rem = layers % local_vpp
            for slot_index in range(local_vpp):
                slots[slot_index] = base
            distribute_order = list(range(local_vpp))
            if focus == "tail-aware":
                distribute_order = list(reversed(distribute_order))
            for offset in range(rem):
                slots[distribute_order[offset % len(distribute_order)]] += 1
        stage_slot_counts.append(slots)

    counts: List[int] = []
    for virtual_slot in range(global_vpp):
        for stage_id in range(len(stage_layers)):
            counts.append(int(stage_slot_counts[stage_id][virtual_slot]))
    return counts


def _pp_vpp_layout_counts(pp_degree: int, num_layers: int, template: str) -> Optional[List[int]]:
    if int(pp_degree) == 2 and int(num_layers) == 40:
        if template == "interleaved_grouped_g2":
            return [8, 12, 12, 8]
        return [10, 10, 10, 10]
    if int(pp_degree) == 4 and int(num_layers) == 40:
        if template == "pp4_middle_relief":
            return [6, 4, 4, 6, 6, 4, 4, 6]
        if template in {"pp4_frontload", "interleaved_grouped_g4"}:
            return [4, 6, 6, 4, 4, 6, 6, 4]
        return [5, 5, 5, 5, 5, 5, 5, 5]
    total_virtual = int(pp_degree) * 2
    if total_virtual <= 0 or int(num_layers) % total_virtual != 0:
        return None
    return [int(num_layers) // total_virtual] * total_virtual


def _set_schedule_template(
    program: MegatronProgram,
    *,
    template: str,
    group_size: Optional[int],
    dispatch_order: str,
    skeleton: Optional[str] = None,
) -> MegatronProgram:
    candidate = _clone_program(program)
    candidate.schedule.template = str(template)
    candidate.schedule.skeleton = str(skeleton or ("fixed_1f1b" if template == "fixed_1f1b" else "stage_aware_grouped"))
    candidate.schedule.dispatch_order = str(dispatch_order)
    candidate.schedule.microbatch_group_size_per_vp_stage = group_size
    return candidate.normalized()


def _candidate_sort_key(program: MegatronProgram) -> tuple:
    template = str(program.schedule.template or "fixed_1f1b")
    evidence_score = _safe_float((program.metadata or {}).get("evidence_score")) or 0.0
    effective_rank = float(program.metadata.get("priority_rank", 0) or 0.0) - 20.0 * evidence_score
    return (
        1 if str((program.metadata or {}).get("vpp_veto_reason") or "") else 0,
        round(effective_rank, 4),
        {
            "fixed_1f1b": 0,
            "interleaved_grouped_g2": 1,
            "interleaved_grouped_g4": 2,
            "pp4_frontload": 3,
            "pp4_middle_relief": 4,
            "torchtitan_zero_bubble": 5,
            "torchtitan_dualpipev": 6,
        }.get(template, 99),
        str(program.metadata.get("program_kind") or ""),
    )


def _structural_program_key(program: MegatronProgram) -> str:
    norm = program.normalized()
    payload = {
        "cluster": norm.cluster.to_dict(),
        "model": norm.model.to_dict(),
        "parallel": norm.parallel.to_dict() if hasattr(norm.parallel, "to_dict") else {
            "tp_degree": int(norm.parallel.tp_degree),
            "pp_degree": int(norm.parallel.pp_degree),
            "vpp_degree": int(norm.parallel.vpp_degree),
            "ep_degree": int(norm.parallel.ep_degree),
            "cp_degree": int(norm.parallel.cp_degree),
            "expert_tp_degree": int(norm.parallel.expert_tp_degree),
            "sp_enabled": bool(norm.parallel.sp_enabled),
        },
        "partition": norm.partition.to_dict(),
        "layout": norm.layout.to_dict(),
        "plane_map": norm.plane_map.to_dict(),
        "schedule": norm.schedule.to_dict(),
        "batch_plan": norm.batch_plan.to_dict(),
        "constraints": norm.constraints.to_dict(),
        "micro_batch_size": int(norm.metadata.get("micro_batch_size", 1) or 1),
        "global_batch_size": int(norm.metadata.get("global_batch_size", 1) or 1),
        "seq_len": int(norm.metadata.get("seq_len", 1) or 1),
        "use_bf16": bool(norm.metadata.get("use_bf16", True)),
        "use_fp16": bool(norm.metadata.get("use_fp16", False)),
        "recompute_granularity": norm.metadata.get("recompute_granularity"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


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


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _median(values: List[float]) -> float:
    ordered = sorted(float(value) for value in values if float(value) > 0.0)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _relative_stage_spread(values: List[float]) -> float:
    center = _median(values)
    if center <= 0.0:
        return 0.0
    return max((max(values) - center) / center, 0.0)


def _edge_tail_delta(values: List[float]) -> float:
    center = _median(values)
    if center <= 0.0 or not values:
        return 0.0
    first_delta = max((float(values[0]) - center) / center, 0.0)
    last_delta = max((float(values[-1]) - center) / center, 0.0)
    return max(first_delta, last_delta)


def _stage_signal_series(context_record: Dict[str, Any], key: str, expected_count: int) -> List[float]:
    evidence = list((((context_record or {}).get("evidence_record") or {}).get("stage_evidence")) or [])
    by_stage: Dict[int, float] = {}
    for item in evidence:
        stage_id = _safe_int((item or {}).get("stage_id"))
        if stage_id is None:
            continue
        by_stage[stage_id] = float((item or {}).get(key) or 0.0)
    return [float(by_stage.get(index, 0.0) or 0.0) for index in range(max(expected_count, 0))]


def _stage_cost_entries(context_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    entries = list(evidence.get("stage_cost_model") or [])
    cleaned: List[Dict[str, Any]] = []
    for item in entries:
        stage_id = _safe_int((item or {}).get("stage_id"))
        if stage_id is None:
            continue
        cleaned.append(
            {
                "stage_id": int(stage_id),
                "T_stable_ms": float((item or {}).get("T_stable_ms") or 0.0),
                "delta_first_ms": float((item or {}).get("delta_first_ms") or 0.0),
                "delta_last_ms": float((item or {}).get("delta_last_ms") or 0.0),
                "boundary_exposed_ms": float((item or {}).get("boundary_exposed_ms") or 0.0),
                "fragmentation_ms": float((item or {}).get("fragmentation_ms") or 0.0),
                "memory_risk_ms": float((item or {}).get("memory_risk_ms") or 0.0),
                "total_cost_ms": float((item or {}).get("total_cost_ms") or 0.0),
            }
        )
    return sorted(cleaned, key=lambda item: int(item.get("stage_id") or 0))


def _boundary_semantic_entries(context_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    entries = list(evidence.get("boundary_semantics") or [])
    cleaned: List[Dict[str, Any]] = []
    for item in entries:
        boundary_id = str((item or {}).get("boundary_id") or "").strip()
        if not boundary_id:
            continue
        cleaned.append(
            {
                "boundary_id": boundary_id,
                "left_stage": int((item or {}).get("left_stage") or 0),
                "right_stage": int((item or {}).get("right_stage") or 0),
                "semantic": str((item or {}).get("semantic") or "normal").strip().lower(),
                "boundary_wait_ms": float((item or {}).get("boundary_wait_ms") or 0.0),
                "cross_node": bool((item or {}).get("cross_node")),
                "actions": [str(action) for action in ((item or {}).get("actions") or [])],
            }
        )
    return sorted(cleaned, key=lambda item: float(item.get("boundary_wait_ms") or 0.0), reverse=True)


def _local_parallel_by_stage(program: MegatronProgram) -> Dict[int, Any]:
    norm = program.normalized()
    local_by_name = {entry.subgraph: entry for entry in (norm.strategy_ir.local_parallel or [])}
    mapped: Dict[int, Any] = {}
    for subgraph in (norm.strategy_ir.apipe or []):
        entry = local_by_name.get(subgraph.name)
        if entry is None:
            continue
        mapped[int(subgraph.stage_index)] = entry
    return mapped


def _stage_family_hint_map(program: MegatronProgram) -> Dict[int, Dict[str, Any]]:
    norm = program.normalized()
    hint_map: Dict[int, Dict[str, Any]] = {}
    for item in list((norm.metadata or {}).get("morphable_stage_families") or []):
        stage_index = _safe_int((item or {}).get("stage_index"))
        if stage_index is None:
            continue
        hint_map[int(stage_index)] = dict(item or {})
    for subgraph in (norm.strategy_ir.apipe or []):
        stage_index = int(subgraph.stage_index)
        hint_map.setdefault(
            stage_index,
            {
                "stage_index": stage_index,
                "family": "balanced_interleave",
            },
        )
    return hint_map


def _sorted_stage_family_hints(hint_map: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(hint_map[stage_index])
        for stage_index in sorted(hint_map)
    ]


def _nonuniform_vpp_vector_from_evidence(
    context_record: Dict[str, Any],
    *,
    stage_count: int,
    fallback_vpp: int,
) -> Tuple[List[int], Dict[int, List[List[int]]]]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    payload = dict(evidence.get("nonuniform_vpp_shape") or {})
    per_stage = list(payload.get("per_stage_candidates") or [])
    vector = [max(int(fallback_vpp), 1) for _ in range(max(stage_count, 0))]
    chunk_shapes: Dict[int, List[List[int]]] = {}
    for item in per_stage:
        stage_id = _safe_int((item or {}).get("stage_id"))
        if stage_id is None or stage_id < 0 or stage_id >= stage_count:
            continue
        legal_values = [max(int(value), 1) for value in ((item or {}).get("currently_executable_values") or []) if _safe_int(value) is not None]
        recommended = max(int((item or {}).get("recommended_v") or 1), 1)
        if legal_values:
            if recommended in set(legal_values):
                target_v = recommended
            else:
                target_v = max(legal_values)
        else:
            target_v = min(recommended, 2)
        vector[int(stage_id)] = max(int(target_v), 1)
        raw_shapes = list((item or {}).get("candidate_chunk_shapes") or [])
        cleaned_shapes: List[List[int]] = []
        for shape in raw_shapes:
            if not isinstance(shape, list) or not shape:
                continue
            parsed = [max(int(value), 1) for value in shape]
            cleaned_shapes.append(parsed)
        if cleaned_shapes:
            chunk_shapes[int(stage_id)] = cleaned_shapes
    return vector, chunk_shapes


def _stage_cost_breakdown(
    baseline: MegatronProgram,
    candidate: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    stage_costs = _stage_cost_entries(context_record)
    if not stage_costs:
        return None
    baseline_norm = baseline.normalized()
    candidate_norm = candidate.normalized()
    baseline_locals = _local_parallel_by_stage(baseline_norm)
    candidate_locals = _local_parallel_by_stage(candidate_norm)
    template = str(candidate_norm.schedule.template or "fixed_1f1b")
    dispatch = str(candidate_norm.schedule.dispatch_order or "default").lower()
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    bubble_ratio = float(runtime.get("bubble_ratio") or 0.0)

    stable_factor = 1.0
    first_factor = 1.0
    last_factor = 1.0
    boundary_factor = 1.0
    frag_factor = 1.0
    memory_factor = 1.0
    if template == "interleaved_grouped_g2":
        stable_factor *= 0.96
        first_factor *= 0.90
        last_factor *= 0.90
        frag_factor *= 1.05
    elif template in {"interleaved_grouped_g4", "pp4_frontload", "pp4_middle_relief"}:
        stable_factor *= 0.93
        first_factor *= 0.84
        last_factor *= 0.84
        boundary_factor *= 0.92
        frag_factor *= 1.10
    elif template == "torchtitan_zero_bubble":
        stable_factor *= 0.90
        first_factor *= 0.76
        last_factor *= 0.76
        boundary_factor *= 0.82
        frag_factor *= 1.16
    elif template == "torchtitan_dualpipev":
        stable_factor *= 0.92
        first_factor *= 0.80
        last_factor *= 0.80
        boundary_factor *= 0.86
        frag_factor *= 1.13
    if "frontload" in dispatch or "tail" in dispatch:
        first_factor *= 0.86
        last_factor *= 0.92
    if "middle" in dispatch:
        stable_factor *= 0.95
    if "boundary" in dispatch or "comm" in dispatch:
        boundary_factor *= 0.86

    semantic = str((candidate_norm.metadata or {}).get("boundary_semantic_focus") or "").strip().lower()
    boundary_id = str((candidate_norm.metadata or {}).get("boundary_semantic_boundary") or "")
    boundary_stage_ids: List[int] = []
    if "->" in boundary_id:
        left, _, right = boundary_id.partition("->")
        left_id = _safe_int(left)
        right_id = _safe_int(right)
        if left_id is not None:
            boundary_stage_ids.append(int(left_id))
        if right_id is not None:
            boundary_stage_ids.append(int(right_id))
    if semantic == "comm-aware":
        boundary_factor *= 0.82
    elif semantic == "tail-aware":
        first_factor *= 0.80
        last_factor *= 0.80
    elif semantic == "memory-aware":
        memory_factor *= 0.78
        frag_factor *= 0.95

    policy_stage_ids = {
        int(item.get("stage_id"))
        for item in list((candidate_norm.metadata or {}).get("stage_local_memory_policy") or [])
        if _safe_int((item or {}).get("stage_id")) is not None
    }

    baseline_components = {
        "T_stable_ms": 0.0,
        "delta_first_ms": 0.0,
        "delta_last_ms": 0.0,
        "boundary_exposed_ms": 0.0,
        "fragmentation_ms": 0.0,
        "memory_risk_ms": 0.0,
    }
    candidate_components = {
        "T_stable_ms": 0.0,
        "delta_first_ms": 0.0,
        "delta_last_ms": 0.0,
        "boundary_exposed_ms": 0.0,
        "fragmentation_ms": 0.0,
        "memory_risk_ms": 0.0,
    }

    max_stage_id = max((int(item.get("stage_id") or 0) for item in stage_costs), default=0)
    for item in stage_costs:
        stage_id = int(item.get("stage_id") or 0)
        is_edge = stage_id in {0, max_stage_id}
        stage_stable = stable_factor
        stage_first = first_factor
        stage_last = last_factor
        stage_boundary = boundary_factor
        stage_frag = frag_factor
        stage_memory = memory_factor
        if is_edge and bubble_ratio >= 0.08:
            stage_first *= 0.94
            stage_last *= 0.94
        if stage_id in boundary_stage_ids and semantic == "comm-aware":
            stage_boundary *= 0.80
        if stage_id in policy_stage_ids:
            stage_memory *= 0.84
            stage_stable *= 1.03

        baseline_local = baseline_locals.get(stage_id)
        candidate_local = candidate_locals.get(stage_id)
        base_cp = int(baseline_local.cp_degree) if baseline_local is not None else int(baseline_norm.parallel.cp_degree)
        cand_cp = int(candidate_local.cp_degree) if candidate_local is not None else int(candidate_norm.parallel.cp_degree)
        cp_delta = max(cand_cp - base_cp, 0)
        if cp_delta > 0:
            stage_memory *= max(0.65, 1.0 - 0.12 * float(cp_delta))
            stage_boundary *= max(0.80, 1.0 - 0.05 * float(cp_delta))
        base_vpp = int(baseline_local.vpp_degree) if baseline_local is not None else int(baseline_norm.parallel.vpp_degree)
        cand_vpp = int(candidate_local.vpp_degree) if candidate_local is not None else int(candidate_norm.parallel.vpp_degree)
        vpp_delta = cand_vpp - base_vpp
        if vpp_delta > 0:
            stage_stable *= max(0.80, 1.0 - 0.04 * float(vpp_delta) * max(1.0 + bubble_ratio, 1.0))
            stage_frag *= 1.0 + 0.11 * float(vpp_delta)
        elif vpp_delta < 0:
            stage_frag *= max(0.75, 1.0 + 0.06 * float(vpp_delta))
        base_sharded = bool(
            baseline_local is not None
            and (
                str(baseline_local.shard_strategy or "none") in {"fsdp", "hsdp"}
                or str(baseline_local.fsdp_scope or "none") not in {"none", "off"}
            )
        )
        cand_sharded = bool(
            candidate_local is not None
            and (
                str(candidate_local.shard_strategy or "none") in {"fsdp", "hsdp"}
                or str(candidate_local.fsdp_scope or "none") not in {"none", "off"}
            )
        )
        if cand_sharded and not base_sharded:
            stage_memory *= 0.78
        if candidate_local is not None and str(candidate_local.reshard_policy or "default") not in {"default", "none"}:
            stage_memory *= 0.93

        baseline_components["T_stable_ms"] += float(item.get("T_stable_ms") or 0.0)
        baseline_components["delta_first_ms"] += float(item.get("delta_first_ms") or 0.0)
        baseline_components["delta_last_ms"] += float(item.get("delta_last_ms") or 0.0)
        baseline_components["boundary_exposed_ms"] += float(item.get("boundary_exposed_ms") or 0.0)
        baseline_components["fragmentation_ms"] += float(item.get("fragmentation_ms") or 0.0)
        baseline_components["memory_risk_ms"] += float(item.get("memory_risk_ms") or 0.0)

        candidate_components["T_stable_ms"] += float(item.get("T_stable_ms") or 0.0) * stage_stable
        candidate_components["delta_first_ms"] += float(item.get("delta_first_ms") or 0.0) * stage_first
        candidate_components["delta_last_ms"] += float(item.get("delta_last_ms") or 0.0) * stage_last
        candidate_components["boundary_exposed_ms"] += float(item.get("boundary_exposed_ms") or 0.0) * stage_boundary
        candidate_components["fragmentation_ms"] += float(item.get("fragmentation_ms") or 0.0) * stage_frag
        candidate_components["memory_risk_ms"] += float(item.get("memory_risk_ms") or 0.0) * stage_memory

    def _objective(components: Dict[str, float]) -> float:
        return (
            float(components.get("T_stable_ms") or 0.0)
            + 0.55 * float(components.get("delta_first_ms") or 0.0)
            + 0.55 * float(components.get("delta_last_ms") or 0.0)
            + 0.90 * float(components.get("boundary_exposed_ms") or 0.0)
            + 0.70 * float(components.get("fragmentation_ms") or 0.0)
            + 1.20 * float(components.get("memory_risk_ms") or 0.0)
        )

    baseline_objective = _objective(baseline_components)
    candidate_objective = _objective(candidate_components)
    if baseline_objective <= 0.0:
        return None
    normalized_improvement = (baseline_objective - candidate_objective) / baseline_objective
    return {
        "candidate_objective": round(float(candidate_objective), 4),
        "baseline_objective": round(float(baseline_objective), 4),
        "improvement_ratio": round(float(normalized_improvement), 4),
        "components": {
            key: round(float(candidate_components.get(key) or 0.0), 4) for key in candidate_components
        },
    }


def _project_stage_signal(
    current_stage_layers: List[int],
    observed_stage_values: List[float],
    candidate_stage_layers: List[int],
    *,
    first_stage_bias: float = 0.0,
    last_stage_bias: float = 0.0,
) -> List[float]:
    expanded: List[float] = []
    total_layers = sum(max(int(layers), 0) for layers in current_stage_layers)
    for index, layers in enumerate(current_stage_layers):
        stage_layers = max(int(layers), 1)
        observed = max(float(observed_stage_values[index] if index < len(observed_stage_values) else 0.0), 0.0)
        fixed_bias = 0.0
        if index == 0:
            fixed_bias += first_stage_bias
        if index == len(current_stage_layers) - 1:
            fixed_bias += last_stage_bias
        per_layer = max(observed - fixed_bias, 0.0) / float(stage_layers)
        expanded.extend([per_layer] * stage_layers)
    if len(expanded) < total_layers:
        expanded.extend([0.0] * (total_layers - len(expanded)))
    projected: List[float] = []
    cursor = 0
    last_index = len(candidate_stage_layers) - 1
    for index, layers in enumerate(candidate_stage_layers):
        stage_layers = max(int(layers), 1)
        window = sum(expanded[cursor : cursor + stage_layers])
        if index == 0:
            window += first_stage_bias
        if index == last_index:
            window += last_stage_bias
        projected.append(float(window))
        cursor += stage_layers
    return projected


def _tail_aware_partition_breakdown(
    baseline: MegatronProgram,
    candidate: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    baseline_layers = [int(stage.decoder_layers) for stage in baseline.partition.stages]
    candidate_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
    if not baseline_layers or not candidate_layers or sum(baseline_layers) != sum(candidate_layers):
        return None
    completion = _stage_signal_series(context_record, "completion_ms", len(baseline_layers))
    if not any(value > 0.0 for value in completion):
        return None
    memory = _stage_signal_series(context_record, "peak_reserved_gib", len(baseline_layers))
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    mem_skew_ratio = float(runtime.get("mem_skew_ratio") or 0.0)
    comm_exposure_ratio = float(runtime.get("comm_exposure_ratio") or 0.0)
    cross_node_ratio = float(runtime.get("cross_node_exposed_ratio") or 0.0)
    first_stage_bias = 0.10 * float(completion[0] if completion else 0.0)
    last_stage_bias = 0.08 * float(completion[-1] if completion else 0.0)
    first_memory_bias = 0.12 * float(memory[0] if memory else 0.0)
    last_memory_bias = 0.08 * float(memory[-1] if memory else 0.0)
    predicted_completion = _project_stage_signal(
        baseline_layers,
        completion,
        candidate_layers,
        first_stage_bias=first_stage_bias,
        last_stage_bias=last_stage_bias,
    )
    predicted_memory = _project_stage_signal(
        baseline_layers,
        memory,
        candidate_layers,
        first_stage_bias=first_memory_bias,
        last_stage_bias=last_memory_bias,
    ) if any(value > 0.0 for value in memory) else []
    baseline_completion = _project_stage_signal(
        baseline_layers,
        completion,
        baseline_layers,
        first_stage_bias=first_stage_bias,
        last_stage_bias=last_stage_bias,
    )
    baseline_memory = _project_stage_signal(
        baseline_layers,
        memory,
        baseline_layers,
        first_stage_bias=first_memory_bias,
        last_stage_bias=last_memory_bias,
    ) if any(value > 0.0 for value in memory) else []
    candidate_virtual_edges = max(int(candidate.parallel.pp_degree) * max(int(candidate.parallel.vpp_degree), 1) - 1, 0)
    baseline_virtual_edges = max(int(baseline.parallel.pp_degree) * max(int(baseline.parallel.vpp_degree), 1) - 1, 0)
    boundary_growth = max(candidate_virtual_edges - baseline_virtual_edges, 0) / float(max(baseline_virtual_edges, 1))
    comm_penalty = boundary_growth * (comm_exposure_ratio + 0.50 * cross_node_ratio)
    predicted_mem_skew = _relative_stage_spread(predicted_memory) if predicted_memory else 0.0
    baseline_mem_skew = _relative_stage_spread(baseline_memory) if baseline_memory else 0.0
    stable_stage_time = _relative_stage_spread(predicted_completion)
    baseline_stable_stage_time = _relative_stage_spread(baseline_completion)
    edge_tail_delta = _edge_tail_delta(predicted_completion)
    baseline_edge_tail_delta = _edge_tail_delta(baseline_completion)
    candidate_objective = (
        0.45 * stable_stage_time
        + 0.20 * edge_tail_delta
        + 0.20 * comm_penalty
        + 0.15 * (predicted_mem_skew * max(mem_skew_ratio, 0.10))
    )
    baseline_objective = (
        0.45 * baseline_stable_stage_time
        + 0.20 * baseline_edge_tail_delta
        + 0.15 * (baseline_mem_skew * max(mem_skew_ratio, 0.10))
    )
    return {
        "candidate_objective": round(float(candidate_objective), 4),
        "baseline_objective": round(float(baseline_objective), 4),
        "improvement": round(float(baseline_objective - candidate_objective), 4),
        "stable_stage_time": round(float(stable_stage_time), 4),
        "edge_tail_delta": round(float(edge_tail_delta), 4),
        "comm_penalty": round(float(comm_penalty), 4),
        "predicted_mem_skew": round(float(predicted_mem_skew), 4),
    }


def _cross_node_boundary_count(stage_to_node: List[str]) -> int:
    boundaries = 0
    for index in range(1, len(stage_to_node)):
        if str(stage_to_node[index]) != str(stage_to_node[index - 1]):
            boundaries += 1
    return boundaries


def _dual_node_placement_breakdown(
    baseline: MegatronProgram,
    candidate: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    if not _is_dual_target(candidate.cluster.target):
        return None
    candidate_nodes = [str(node) for node in (candidate.layout.stage_to_node or [])]
    baseline_nodes = [str(node) for node in (baseline.layout.stage_to_node or [])]
    if not candidate_nodes or not baseline_nodes:
        return None
    if sum(int(stage.decoder_layers) for stage in candidate.partition.stages) <= 0:
        return None

    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    bubble_ratio = float(runtime.get("bubble_ratio") or 0.0)
    stage_tail_ratio = float(runtime.get("stage_tail_ratio") or 0.0)
    cross_node_ratio = float(runtime.get("cross_node_exposed_ratio") or 0.0)
    comm_exposure_ratio = float(runtime.get("comm_exposure_ratio") or 0.0)
    speed_map = _node_speed_map(candidate)
    fastest_node = _fastest_node(candidate)

    def _objective(program: MegatronProgram) -> Dict[str, float]:
        stage_nodes = [str(node) for node in (program.layout.stage_to_node or [])]
        total_layers = max(sum(int(stage.decoder_layers) for stage in program.partition.stages), 1)
        node_loads: Dict[str, float] = {}
        for stage, node in zip(program.partition.stages, stage_nodes):
            node_loads[node] = node_loads.get(node, 0.0) + float(int(stage.decoder_layers))
        total_speed = max(sum(float(speed_map.get(node, 1.0)) for node in set(stage_nodes)), 1e-6)
        share_mismatch = 0.0
        for node in set(stage_nodes):
            ideal = float(speed_map.get(node, 1.0)) / total_speed
            actual = float(node_loads.get(node, 0.0)) / float(total_layers)
            share_mismatch += abs(actual - ideal)
        share_mismatch *= 0.5

        boundary_penalty = float(max(_cross_node_boundary_count(stage_nodes) - 1, 0)) * (
            cross_node_ratio + 0.50 * comm_exposure_ratio
        )
        depth_gain = max(int(program.parallel.pp_degree) - int(baseline.parallel.pp_degree), 0) / float(
            max(int(program.parallel.pp_degree), 1)
        )
        depth_bonus = depth_gain * (0.35 * bubble_ratio + 0.15 * stage_tail_ratio)
        placement_bonus = 0.0
        if fastest_node is not None and stage_nodes:
            if stage_nodes[-1] == fastest_node:
                placement_bonus += 0.04 + 0.18 * stage_tail_ratio
            if str(program.cluster.target) == "dual_g4_g5" and stage_nodes[0] != fastest_node:
                placement_bonus += 0.02
        objective = share_mismatch + boundary_penalty - placement_bonus - depth_bonus
        return {
            "objective": round(float(objective), 4),
            "share_mismatch": round(float(share_mismatch), 4),
            "boundary_penalty": round(float(boundary_penalty), 4),
            "placement_bonus": round(float(placement_bonus), 4),
            "depth_bonus": round(float(depth_bonus), 4),
        }

    candidate_metrics = _objective(candidate)
    baseline_metrics = _objective(baseline)
    return {
        "candidate_objective": float(candidate_metrics["objective"]),
        "baseline_objective": float(baseline_metrics["objective"]),
        "improvement": round(float(baseline_metrics["objective"] - candidate_metrics["objective"]), 4),
        "share_mismatch": float(candidate_metrics["share_mismatch"]),
        "boundary_penalty": float(candidate_metrics["boundary_penalty"]),
        "placement_bonus": float(candidate_metrics["placement_bonus"]),
        "depth_bonus": float(candidate_metrics["depth_bonus"]),
    }


def _hybrid_shard_breakdown(candidate: MegatronProgram, context_record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    shard_modes = [str(item.shard_strategy or "none") for item in (candidate.strategy_ir.local_parallel or [])]
    if not any(mode in {"fsdp", "hsdp"} for mode in shard_modes):
        return None
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    mem_skew_ratio = float(runtime.get("mem_skew_ratio") or 0.0)
    comm_exposure_ratio = float(runtime.get("comm_exposure_ratio") or 0.0)
    cross_node_ratio = float(runtime.get("cross_node_exposed_ratio") or 0.0)
    hsdp_fraction = sum(1.0 for mode in shard_modes if mode == "hsdp") / float(len(shard_modes) or 1)
    relief_score = (0.45 * peak_reserved_ratio + 0.35 * mem_skew_ratio + 0.10 * hsdp_fraction)
    comm_penalty = (0.20 * comm_exposure_ratio + 0.10 * cross_node_ratio) * (1.0 - 0.35 * hsdp_fraction)
    return {
        "relief_score": round(float(relief_score), 4),
        "comm_penalty": round(float(comm_penalty), 4),
        "improvement": round(float(relief_score - comm_penalty), 4),
        "hsdp_fraction": round(float(hsdp_fraction), 4),
    }


def _annotate_candidate_runtime_evidence(
    baseline: MegatronProgram,
    candidate: MegatronProgram,
    context_record: Dict[str, Any],
) -> MegatronProgram:
    annotated = _sync_batch_plan_metadata(candidate)
    for key in (
        "tail_partition_score",
        "tail_partition_objective",
        "tail_partition_components",
        "dual_node_score",
        "dual_node_objective",
        "dual_node_components",
        "hybrid_shard_score",
        "hybrid_shard_components",
        "stage_cost_score",
        "stage_cost_objective",
        "stage_cost_components",
        "stage_cost_improvement_ratio",
        "vpp_tradeoff",
        "vpp_veto_reason",
        "evidence_score",
    ):
        annotated.metadata.pop(key, None)
    evidence_score = 0.0
    partition_changed = (
        [int(stage.decoder_layers) for stage in annotated.partition.stages]
        != [int(stage.decoder_layers) for stage in baseline.partition.stages]
        or int(annotated.parallel.pp_degree) != int(baseline.parallel.pp_degree)
    )
    if partition_changed:
        breakdown = _tail_aware_partition_breakdown(baseline, annotated, context_record)
        if breakdown is not None:
            annotated.metadata["tail_partition_score"] = float(breakdown.get("improvement") or 0.0)
            annotated.metadata["tail_partition_objective"] = float(breakdown.get("candidate_objective") or 0.0)
            annotated.metadata["tail_partition_components"] = {
                "stable_stage_time": float(breakdown.get("stable_stage_time") or 0.0),
                "edge_tail_delta": float(breakdown.get("edge_tail_delta") or 0.0),
                "comm_penalty": float(breakdown.get("comm_penalty") or 0.0),
                "predicted_mem_skew": float(breakdown.get("predicted_mem_skew") or 0.0),
            }
            evidence_score += float(breakdown.get("improvement") or 0.0)
    dual_node_breakdown = _dual_node_placement_breakdown(baseline, annotated, context_record)
    if dual_node_breakdown is not None:
        annotated.metadata["dual_node_score"] = float(dual_node_breakdown.get("improvement") or 0.0)
        annotated.metadata["dual_node_objective"] = float(dual_node_breakdown.get("candidate_objective") or 0.0)
        annotated.metadata["dual_node_components"] = {
            "share_mismatch": float(dual_node_breakdown.get("share_mismatch") or 0.0),
            "boundary_penalty": float(dual_node_breakdown.get("boundary_penalty") or 0.0),
            "placement_bonus": float(dual_node_breakdown.get("placement_bonus") or 0.0),
            "depth_bonus": float(dual_node_breakdown.get("depth_bonus") or 0.0),
        }
        evidence_score += float(dual_node_breakdown.get("improvement") or 0.0)
    hybrid_shard = _hybrid_shard_breakdown(annotated, context_record)
    if hybrid_shard is not None:
        annotated.metadata["hybrid_shard_score"] = float(hybrid_shard.get("improvement") or 0.0)
        annotated.metadata["hybrid_shard_components"] = {
            "relief_score": float(hybrid_shard.get("relief_score") or 0.0),
            "comm_penalty": float(hybrid_shard.get("comm_penalty") or 0.0),
            "hsdp_fraction": float(hybrid_shard.get("hsdp_fraction") or 0.0),
        }
        evidence_score += float(hybrid_shard.get("improvement") or 0.0)
    stage_cost = _stage_cost_breakdown(baseline, annotated, context_record)
    if stage_cost is not None:
        annotated.metadata["stage_cost_score"] = float(stage_cost.get("improvement_ratio") or 0.0)
        annotated.metadata["stage_cost_objective"] = float(stage_cost.get("candidate_objective") or 0.0)
        annotated.metadata["stage_cost_components"] = dict(stage_cost.get("components") or {})
        annotated.metadata["stage_cost_improvement_ratio"] = float(stage_cost.get("improvement_ratio") or 0.0)
        evidence_score += float(stage_cost.get("improvement_ratio") or 0.0)
    if int(annotated.parallel.vpp_degree) > 1:
        tradeoff = assess_vpp_comm_tradeoff(
            annotated,
            runtime_summary=dict((context_record or {}).get("runtime_evidence") or {}),
        )
        annotated.metadata["vpp_tradeoff"] = tradeoff
        evidence_score += float(tradeoff.get("bubble_relief_score") or 0.0) - float(tradeoff.get("comm_pressure_score") or 0.0)
        if bool(tradeoff.get("should_veto")):
            annotated.metadata["vpp_veto_reason"] = str(tradeoff.get("reason") or "comm-exposure-aware VPP veto")
    annotated.metadata["evidence_score"] = round(float(evidence_score), 4)
    return annotated.normalized()


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


def _stage_window_values(payload: Optional[Dict[str, Any]], expected_count: Optional[int] = None) -> List[float]:
    if not payload:
        return []
    stage_summary = payload.get("stage_window_summary") or {}
    indexed: List[Tuple[int, float]] = []
    for key, item in stage_summary.items():
        window_ms = _safe_float((item or {}).get("window_ms"))
        if window_ms is None or window_ms <= 0:
            continue
        try:
            stage_id = int(str(key))
        except Exception:
            stage_id = len(indexed)
        indexed.append((stage_id, window_ms))
    indexed.sort(key=lambda pair: pair[0])
    values = [value for _, value in indexed]
    if expected_count is not None and expected_count > 0:
        if len(values) < expected_count:
            values.extend([0.0] * (expected_count - len(values)))
        elif len(values) > expected_count:
            values = values[:expected_count]
    return values


def _dominant_stage_indices(payload: Optional[Dict[str, Any]], expected_count: int) -> Tuple[Optional[int], Optional[int], List[float]]:
    values = _stage_window_values(payload, expected_count=expected_count)
    if len(values) < 2 or max(values) <= 0:
        return None, None, values
    slow_idx = max(range(len(values)), key=lambda idx: values[idx])
    fast_idx = min(range(len(values)), key=lambda idx: values[idx])
    return slow_idx, fast_idx, values


def _stage_window_component_values(
    payload: Optional[Dict[str, Any]],
    metric: str,
    expected_count: int,
) -> List[float]:
    if not payload:
        return [0.0] * max(int(expected_count), 0)
    stage_summary = payload.get("stage_window_summary") or {}
    indexed: List[Tuple[int, float]] = []
    for key, item in stage_summary.items():
        value = _safe_float((item or {}).get(metric))
        if value is None or value < 0.0:
            value = 0.0
        try:
            stage_id = int(str(key))
        except Exception:
            stage_id = len(indexed)
        indexed.append((stage_id, float(value)))
    indexed.sort(key=lambda pair: pair[0])
    values = [value for _, value in indexed]
    if len(values) < expected_count:
        values.extend([0.0] * (expected_count - len(values)))
    elif len(values) > expected_count:
        values = values[:expected_count]
    return values


def _position_aware_partition_focus(runtime_summary: Dict[str, Any]) -> str:
    optimizer_exposed_ratio = float(runtime_summary.get("optimizer_exposed_ratio") or 0.0)
    pipeline_wait_ratio = float(runtime_summary.get("pipeline_wait_ratio") or 0.0)
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    if optimizer_exposed_ratio >= max(0.18, pipeline_wait_ratio + 0.03):
        return "tail-aware"
    if pipeline_wait_ratio >= 0.10 or bubble_ratio >= 0.10:
        return "comm-aware"
    return "position-aware"


def _runtime_guided_partition_plan(
    stage_layers: List[int],
    runtime_summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    expected_count = len(stage_layers)
    if expected_count < 2:
        return None
    window_ms = _stage_window_values(runtime_summary, expected_count=expected_count)
    if len(window_ms) < 2 or max(window_ms) <= 0.0:
        return None
    compute_ms = _stage_window_component_values(runtime_summary, "compute_ms", expected_count)
    comm_ms = _stage_window_component_values(runtime_summary, "comm_ms", expected_count)
    bubble_ms = _stage_window_component_values(runtime_summary, "bubble_ms", expected_count)
    peak_reserved_gib = _stage_window_component_values(runtime_summary, "peak_reserved_gib", expected_count)
    pipeline_wait_ratio = float(runtime_summary.get("pipeline_wait_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_summary.get("optimizer_exposed_ratio") or 0.0)
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    stage_skew = float(runtime_summary.get("stage_skew") or 0.0)
    focus = _position_aware_partition_focus(runtime_summary)
    mean_window = sum(value for value in window_ms if value > 0.0) / float(max(sum(1 for value in window_ms if value > 0.0), 1))

    stage_objectives: List[float] = []
    last_index = expected_count - 1
    for idx in range(expected_count):
        compute = float(compute_ms[idx] or 0.0)
        comm = float(comm_ms[idx] or 0.0)
        bubble = float(bubble_ms[idx] or 0.0)
        memory = float(peak_reserved_gib[idx] or 0.0)
        objective = compute + 0.85 * comm + 0.65 * bubble
        edge_wait_penalty = mean_window * pipeline_wait_ratio * (0.85 if idx in {0, last_index} else 0.35)
        tail_penalty = 0.0
        if idx == last_index:
            tail_penalty = mean_window * optimizer_exposed_ratio * 0.95
        elif idx == 0:
            tail_penalty = mean_window * optimizer_exposed_ratio * 0.35
        memory_penalty = mean_window * 0.02 * max(memory - _median(peak_reserved_gib), 0.0)
        if focus == "comm-aware":
            objective += edge_wait_penalty + 0.35 * mean_window * bubble_ratio * (1.0 if idx in {0, last_index} else 0.5)
        elif focus == "tail-aware":
            objective += 0.60 * edge_wait_penalty + tail_penalty
        else:
            objective += 0.75 * edge_wait_penalty + 0.45 * tail_penalty
        objective += memory_penalty
        stage_objectives.append(float(objective))

    donor = max(range(expected_count), key=lambda idx: stage_objectives[idx])
    receiver_candidates = sorted(range(expected_count), key=lambda idx: stage_objectives[idx])
    receiver = None
    min_objective = stage_objectives[receiver_candidates[0]]
    relaxed_threshold = min_objective * 1.05 if min_objective > 0.0 else min_objective + 1.0
    for idx in receiver_candidates:
        if idx == donor:
            continue
        if stage_objectives[idx] <= relaxed_threshold:
            if receiver is None or abs(idx - donor) < abs(receiver - donor):
                receiver = idx
    if receiver is None:
        receiver = min((idx for idx in range(expected_count) if idx != donor), key=lambda idx: stage_objectives[idx], default=None)
    if receiver is None:
        return None
    donor_value = float(stage_objectives[donor] or 0.0)
    receiver_value = float(stage_objectives[receiver] or 0.0)
    if donor_value <= 0.0:
        return None
    spread_ratio = max((donor_value - receiver_value) / donor_value, 0.0)
    if spread_ratio < 0.08 and stage_skew < 1.08:
        return None
    if stage_layers[donor] <= 1:
        return None
    shift = 1
    if spread_ratio >= 0.16:
        shift += 1
    if spread_ratio >= 0.28:
        shift += 1
    if donor in {0, last_index} and pipeline_wait_ratio >= 0.10:
        shift += 1
    if donor == last_index and optimizer_exposed_ratio >= 0.18:
        shift += 1
    shift = max(1, min(shift, stage_layers[donor] - 1, 3))
    return {
        "donor_stage_index": int(donor),
        "receiver_stage_index": int(receiver),
        "shift_layers": int(shift),
        "focus": focus,
        "spread_ratio": round(float(spread_ratio), 4),
        "stage_objectives": [round(float(value), 4) for value in stage_objectives],
        "pipeline_wait_ratio": round(float(pipeline_wait_ratio), 4),
        "optimizer_exposed_ratio": round(float(optimizer_exposed_ratio), 4),
        "stage_skew": round(float(stage_skew), 4),
    }


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


def _normalize_lower_is_better(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _rank_trials(tested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_trials = [
        trial for trial in tested if trial.get("returncode") in (0, None) and not trial.get("oom")
    ]
    if not valid_trials:
        return tested

    step_map: Dict[str, float] = {}
    peak_map: Dict[str, float] = {}
    bubble_map: Dict[str, float] = {}
    stall_map: Dict[str, float] = {}
    for trial in valid_trials:
        key = str(trial.get("config_name") or trial.get("program_hash") or len(step_map))
        trace_summary = trial.get("trace_summary") or {}
        step_map[key] = float(
            trace_summary.get("steady_state_step_time_ms_p50")
            or trial.get("step_time_ms_p50")
            or 0.0
        )
        peak_map[key] = float(trace_summary.get("peak_reserved_ratio") or 0.0)
        bubble_map[key] = float(trace_summary.get("bubble_ratio") or trial.get("bubble_ratio") or 0.0)
        stall_map[key] = float(trace_summary.get("stall_ratio") or 0.0)

    step_norm = _normalize_lower_is_better(step_map)
    peak_norm = _normalize_lower_is_better(peak_map)
    bubble_norm = _normalize_lower_is_better(bubble_map)
    stall_norm = _normalize_lower_is_better(stall_map)

    for trial in valid_trials:
        key = str(trial.get("config_name") or trial.get("program_hash") or "")
        trial["selection_score"] = 1.0 - (
            0.50 * step_norm.get(key, 0.0)
            + 0.20 * peak_norm.get(key, 0.0)
            + 0.20 * bubble_norm.get(key, 0.0)
            + 0.10 * stall_norm.get(key, 0.0)
        )
    for trial in tested:
        if trial not in valid_trials:
            trial["selection_score"] = float("-inf")
    return sorted(tested, key=lambda item: float(item.get("selection_score", float("-inf"))), reverse=True)


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
    *,
    context_record: Optional[Dict[str, Any]] = None,
    previous_program: Optional[MegatronProgram] = None,
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
        verifier_report = verify_program(
            program,
            observation=context_record,
            previous_program=(previous_program if execution_order > 0 else None),
        ).to_dict()
        legality = dict(verifier_report.get("legality") or {})
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
                "verifier_report": verifier_report,
                "compile_success": compile_success,
                "compile_error": compile_error,
            }
        )
    return manifest


def _candidate_entries(candidate_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in candidate_manifest if not bool(entry.get("is_baseline"))]


def _select_first_program(
    proposals: List[AgentProposal],
    preferred_kinds: List[str],
) -> Optional[Tuple[AgentProposal, MegatronProgram]]:
    for kind in preferred_kinds:
        for proposal in proposals:
            program = proposal.program.normalized()
            if str(program.metadata.get("program_kind") or "") == str(kind):
                return proposal, program
    return None


def _strategy_template_record(
    *,
    template_id: str,
    label: str,
    description: str,
    category: str,
    trigger_conditions: List[str],
    supported_batch_profiles: List[str],
    proposal: AgentProposal,
    program: MegatronProgram,
) -> Dict[str, Any]:
    verifier_report = dict(proposal.verifier_report or {})
    return {
        "template_id": str(template_id),
        "label": str(label),
        "category": str(category),
        "description": str(description),
        "trigger_conditions": [str(item) for item in (trigger_conditions or [])],
        "supported_batch_profiles": [str(item) for item in (supported_batch_profiles or [])],
        "program_kind": str(program.metadata.get("program_kind") or ""),
        "program_hash": program.semantic_hash(),
        "schedule_template": str(program.schedule.template or "fixed_1f1b"),
        "dispatch_order": str(program.schedule.dispatch_order or "default"),
        "pp_degree": int(program.parallel.pp_degree),
        "vpp_degree": int(program.parallel.vpp_degree),
        "cp_degree": int(program.parallel.cp_degree),
        "memory_policy_mode": str(program.metadata.get("runtime_memory_policy_mode") or "none"),
        "planner_backend": str(program.metadata.get("planner_backend") or "heuristic"),
        "verifier_legal": bool(verifier_report.get("is_legal", False)),
        "verifier_report": verifier_report,
        "rationale": str(proposal.rationale or ""),
    }


def _build_verified_strategy_template_library(
    baseline: MegatronProgram,
    proposals: List[AgentProposal],
    context_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    library: List[Dict[str, Any]] = []
    definitions = [
        {
            "template_id": "A_pp_comm_balanced",
            "label": "PP communication-balanced",
            "category": "partition",
            "description": "ordinary PP with stage cuts biased toward communication-balanced boundaries",
            "trigger_conditions": ["communication_drag", "comm_exposed", "topology_mismatch", "compute_imbalance"],
            "supported_batch_profiles": ["normal", "memory_constrained"],
            "preferred_kinds": ["candidate_runtime_guided_partition", "candidate_nonuniform_partition", "candidate_stage_aware_schedule"],
        },
        {
            "template_id": "B_middle_stage_vpp",
            "label": "Middle-stage selective VPP",
            "category": "vpp",
            "description": "edge stages stay conservative while middle stages use finer virtual pipeline chunks",
            "trigger_conditions": ["schedule_coupling", "bubble", "pipeline_wait"],
            "supported_batch_profiles": ["normal"],
            "preferred_kinds": ["candidate_nonuniform_vpp_shape", "candidate_morphable_pipeline", "candidate_stage_aware_schedule"],
        },
        {
            "template_id": "C_memory_guarded_offload",
            "label": "Selective memory relief",
            "category": "memory",
            "description": "high-memory stages use selective recompute or offload while preserving critical-path stages",
            "trigger_conditions": ["memory_hotspot", "memory_skew", "memory_constrained"],
            "supported_batch_profiles": ["memory_constrained", "long_context", "moe_heavy"],
            "preferred_kinds": ["candidate_offload_first_refinement", "candidate_stage_local_memory_policy", "candidate_morphable_pipeline", "candidate_memory_relief"],
        },
        {
            "template_id": "D_long_context_conservative",
            "label": "Long-context conservative VPP",
            "category": "batch_routing",
            "description": "long-sequence batches prefer coarser chunking and conservative overlap with CP relief",
            "trigger_conditions": ["long_context", "communication_drag"],
            "supported_batch_profiles": ["long_context"],
            "preferred_kinds": ["candidate_long_context_cp_relief", "candidate_stage_local_memory_policy", "candidate_stage_aware_schedule"],
        },
        {
            "template_id": "E_vocab_edge_conservative",
            "label": "Vocab-edge conservative partition",
            "category": "boundary",
            "description": "embedding and lm_head edge stages avoid over-fragmentation and expensive boundary activations",
            "trigger_conditions": ["tail_heavy", "memory_hotspot", "comm_exposed"],
            "supported_batch_profiles": ["normal", "memory_constrained", "long_context"],
            "preferred_kinds": ["candidate_boundary_semantic_memory", "candidate_boundary_semantic_tail", "candidate_morphable_pipeline"],
        },
        {
            "template_id": "F_optimizer_exposure_relief",
            "label": "Optimizer-aware relief",
            "category": "execution",
            "description": "reduce optimizer exposure on the critical path with tail-guarded overlap and flush control",
            "trigger_conditions": ["optimizer_exposed", "tail_heavy", "pipeline_wait"],
            "supported_batch_profiles": ["normal", "memory_constrained"],
            "preferred_kinds": ["candidate_optimizer_aware_pipeline", "candidate_tail_aware_execution", "candidate_morphable_pipeline"],
        },
        {
            "template_id": "G_tail_stage_heterogeneous",
            "label": "Tail-stage heterogeneous execution",
            "category": "tail",
            "description": "tail stages use more conservative VPP, checkpoint, and cooldown policies than middle stages",
            "trigger_conditions": ["tail_heavy", "tail_jitter", "pipeline_wait"],
            "supported_batch_profiles": ["normal", "memory_constrained", "long_context"],
            "preferred_kinds": ["candidate_tail_aware_execution", "candidate_nonuniform_vpp_shape", "candidate_runtime_guided_partition"],
        },
        {
            "template_id": "H_checkpoint_boundary_joint",
            "label": "Checkpoint-boundary joint refinement",
            "category": "checkpoint",
            "description": "checkpoint and recompute boundaries are refined jointly with PP/VPP to expand the feasible region",
            "trigger_conditions": ["memory_hotspot", "tail_heavy", "optimizer_exposed"],
            "supported_batch_profiles": ["memory_constrained", "long_context"],
            "preferred_kinds": ["candidate_checkpoint_boundary_refinement", "candidate_offload_first_refinement", "candidate_stage_local_memory_policy"],
        },
    ]
    for definition in definitions:
        selected = _select_first_program(proposals, list(definition.get("preferred_kinds") or []))
        if selected is None:
            continue
        proposal, program = selected
        library.append(
            _strategy_template_record(
                template_id=str(definition["template_id"]),
                label=str(definition["label"]),
                description=str(definition["description"]),
                category=str(definition["category"]),
                trigger_conditions=list(definition.get("trigger_conditions") or []),
                supported_batch_profiles=list(definition.get("supported_batch_profiles") or []),
                proposal=proposal,
                program=program,
            )
        )
    if not library:
        baseline_program = baseline.normalized()
        pseudo_proposal = AgentProposal(
            proposal_id="baseline_template",
            scope="pipe",
            program=baseline_program,
            rationale="fallback baseline template when no verified alternatives are available",
            priority_rank=999,
            source="fallback",
            verifier_report={"is_legal": True, "legality": {"fallback": True}},
        ).normalized()
        library.append(
            _strategy_template_record(
                template_id="baseline_safe",
                label="Baseline safe template",
                description="fallback template that preserves the baseline configuration",
                category="fallback",
                trigger_conditions=["fallback"],
                supported_batch_profiles=["normal", "memory_constrained", "long_context", "moe_heavy"],
                proposal=pseudo_proposal,
                program=baseline_program,
            )
        )
    return library


def _select_verified_strategy_template(
    template_library: List[Dict[str, Any]],
    context_record: Dict[str, Any],
) -> Dict[str, Any]:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    failures = list((context_record or {}).get("failure_modes") or [])
    batch_profile = _batch_profile(context_record)
    active_labels = {str(item.get("label") or "") for item in failures}
    bubble_ratio = float(runtime.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    stage_tail_ratio = float(runtime.get("stage_tail_ratio") or 0.0)
    comm_exposure_ratio = float(runtime.get("comm_exposure_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime.get("optimizer_exposed_ratio") or 0.0)
    scores: List[Dict[str, Any]] = []
    for item in template_library:
        score = 0.0
        reasons: List[str] = []
        supported_profiles = set(str(value) for value in (item.get("supported_batch_profiles") or []))
        if batch_profile in supported_profiles:
            score += 3.0
            reasons.append(f"batch profile {batch_profile} matches template support")
        if "memory_hotspot" in active_labels or peak_reserved_ratio >= 0.88:
            if str(item.get("category") or "") in {"memory", "boundary"}:
                score += 2.5
                reasons.append("memory pressure favors memory or boundary guarded templates")
        if ("memory_hotspot" in active_labels or peak_reserved_ratio >= 0.86) and stage_tail_ratio >= 0.10:
            if str(item.get("template_id") or "") == "C_memory_guarded_offload":
                score += 3.0
                reasons.append("tail-heavy memory pressure favors offload-first local refinement before global rewrites")
        if "communication_drag" in active_labels or comm_exposure_ratio >= 0.12:
            if str(item.get("category") or "") in {"partition", "boundary"}:
                score += 2.0
                reasons.append("communication exposure favors communication-aware templates")
        if batch_profile == "long_context":
            if str(item.get("template_id") or "") == "D_long_context_conservative":
                score += 4.0
                reasons.append("long-context batches should use conservative VPP or CP relief")
        if optimizer_exposed_ratio >= 0.18:
            if str(item.get("category") or "") in {"execution", "tail", "checkpoint"}:
                score += 2.5
                reasons.append("optimizer exposure favors execution-semantic relief over pure bubble tuning")
        if bubble_ratio >= 0.10:
            if str(item.get("category") or "") in {"vpp", "partition"}:
                score += 1.5
                reasons.append("bubble pressure favors VPP or partition refinement")
        if stage_tail_ratio >= 0.12:
            if str(item.get("category") or "") in {"boundary", "tail", "execution", "checkpoint"}:
                score += 1.5
                reasons.append("tail-heavy execution favors conservative edge-stage handling and checkpoint control")
        scores.append(
            {
                "template_id": str(item.get("template_id") or ""),
                "program_kind": str(item.get("program_kind") or ""),
                "score": round(float(score), 4),
                "reasons": reasons,
            }
        )
    ranked = sorted(scores, key=lambda entry: (float(entry["score"]), str(entry["template_id"])), reverse=True)
    selected = ranked[0] if ranked else {}
    return {
        "batch_profile": batch_profile,
        "active_failure_labels": sorted(active_labels),
        "ranked_templates": ranked,
        "selected_template_id": str(selected.get("template_id") or ""),
        "selected_program_kind": str(selected.get("program_kind") or ""),
        "selector_mode": "verified_template_switch_only",
        "notes": "selector only switches among verifier-approved templates and does not rewrite the runtime graph online",
    }


def _proposal_scope_fallback(scope: str) -> str:
    current = str(scope or "none")
    if current == "pipe":
        return "local_parallel"
    if current == "local_parallel":
        return "skeleton"
    return "none"


def _make_agent_observation(
    program: MegatronProgram,
    *,
    runtime_summary: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    trace_summary: Optional[Dict[str, Any]] = None,
    motivation_evidence_manifest: Optional[List[Dict[str, Any]]] = None,
) -> AgentObservation:
    return build_agent_observation(
        program,
        runtime_summary=runtime_summary,
        metrics=metrics,
        trace_summary=trace_summary,
        motivation_evidence_manifest=motivation_evidence_manifest,
    )


def _build_agent_proposal(
    program: MegatronProgram,
    *,
    scope: str,
    rationale: str,
    source: str,
) -> AgentProposal:
    norm = _sync_batch_plan_metadata(program)
    norm.metadata["replan_scope"] = str(scope)
    return AgentProposal(
        proposal_id=str(norm.metadata.get("program_kind") or f"{scope}_proposal"),
        scope=str(scope),
        program=norm,
        rationale=str(rationale),
        priority_rank=int(norm.metadata.get("priority_rank", 0) or 0),
        source=str(source),
    ).normalized()


def _annotate_local_parallel(program: MegatronProgram, context_record: Dict[str, Any]) -> MegatronProgram:
    candidate = _clone_program(program)
    preserve_stage_local_vpp = bool(candidate.metadata.get("preserve_stage_local_vpp", False))
    hot_subgraphs = {
        str(item.get("anchor")): str(item.get("label"))
        for item in (context_record.get("failure_modes") or [])
        if str(item.get("anchor") or "")
    }
    for local in candidate.strategy_ir.local_parallel:
        label = hot_subgraphs.get(local.subgraph)
        if label == "schedule_coupling" and not preserve_stage_local_vpp:
            local.vpp_degree = max(int(local.vpp_degree), 2)
        elif label == "memory_hotspot":
            local.cp_degree = max(int(local.cp_degree), 2)
            local.fsdp_scope = "selective"
        elif label == "compute_imbalance" and not preserve_stage_local_vpp:
            local.vpp_degree = max(int(local.vpp_degree), int(candidate.parallel.vpp_degree))
    candidate.strategy_ir.pipe.template = str(candidate.schedule.template)
    candidate.strategy_ir.pipe.microbatch_order = str(candidate.schedule.dispatch_order)
    candidate.strategy_ir.pipe.steady_state_group_size = candidate.schedule.microbatch_group_size_per_vp_stage
    return candidate.normalized()


def _build_evidence_matrix(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    evidence_programs: List[MegatronProgram] = []
    baseline_fixed = _clone_program(baseline)
    baseline_fixed.metadata["program_kind"] = "evidence_pp_fixed_pipe"
    evidence_programs.append(_annotate_local_parallel(_sync_batch_plan_metadata(baseline_fixed), context_record))

    stage_aware = _build_stage_aware_schedule(baseline)
    if stage_aware is not None:
        stage_aware.metadata["program_kind"] = "evidence_pp_vpp"
        evidence_programs.append(_annotate_local_parallel(stage_aware, context_record))

    cp_probe = _build_long_context_cp_candidate(stage_aware or baseline)
    if cp_probe is not None:
        cp_probe.metadata["program_kind"] = "evidence_pp_vpp_cp"
        evidence_programs.append(_annotate_local_parallel(cp_probe, context_record))

    fsdp_probe = _clone_program(cp_probe or stage_aware or baseline)
    if fsdp_probe.strategy_ir.local_parallel:
        for local in fsdp_probe.strategy_ir.local_parallel:
            if local.cp_degree > 1 or int(fsdp_probe.parallel.pp_degree) > 1:
                local.fsdp_scope = "selective"
        fsdp_probe.metadata["program_kind"] = "evidence_pp_vpp_cp_fsdp_scope"
        evidence_programs.append(_sync_batch_plan_metadata(fsdp_probe))

    if rewrite.allow_stage_aware_schedule:
        drift_probe = _build_runtime_guided_schedule(stage_aware or baseline, runtime_summary)
        if drift_probe is not None:
            drift_probe.metadata["program_kind"] = "evidence_dynamic_drift_probe"
            evidence_programs.append(_annotate_local_parallel(drift_probe, context_record))

    ordered: List[MegatronProgram] = []
    seen: set[str] = set()
    for program in evidence_programs:
        key = _structural_program_key(program)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(program)
    return ordered


def _resolve_replan_decision(
    baseline: MegatronProgram,
    context_record: Dict[str, Any],
    previous_context: Optional[Dict[str, Any]] = None,
) -> ReplanDecision:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    workload = dict((context_record or {}).get("workload_context") or {})
    optimization_hints = list((context_record or {}).get("optimization_hints") or [])
    previous_workload = dict((previous_context or {}).get("workload_context") or {})
    previous_runtime = dict((previous_context or {}).get("runtime_evidence") or {})

    seq_changed = str(workload.get("length_bucket") or "") != str(previous_workload.get("length_bucket") or "")
    cross_node_worsened = float(runtime.get("cross_node_exposed_ratio") or 0.0) > float(previous_runtime.get("cross_node_exposed_ratio") or 0.0) + 0.05
    memory_worsened = float(runtime.get("peak_reserved_ratio") or 0.0) > float(previous_runtime.get("peak_reserved_ratio") or 0.0) + 0.08
    bubble_worsened = float(runtime.get("bubble_ratio") or 0.0) > max(float(previous_runtime.get("bubble_ratio") or 0.0), 0.08)
    stage_imbalance = any(str(item.get("label")) == "compute_imbalance" for item in (context_record.get("failure_modes") or []))
    memory_hotspot = any(str(item.get("label")) == "memory_hotspot" for item in (context_record.get("failure_modes") or []))
    tail_heavy = any(str(item.get("label")) == "tail_heavy" for item in (context_record.get("derived_bottlenecks") or []))
    comm_exposed = any(str(item.get("label")) in {"comm_exposed", "topology_mismatch"} for item in (context_record.get("derived_bottlenecks") or []))

    scope = "none"
    trigger = "steady"
    rationale = "current context does not require replanning"
    expected_switch_cost = 0.0
    if seq_changed or bubble_worsened:
        scope = "pipe"
        trigger = "workload_drift" if seq_changed else "bubble_spike"
        rationale = "prefer low-cost pipe adaptation before touching local or global structure"
        expected_switch_cost = 0.06
    if memory_worsened or memory_hotspot:
        scope = "local_parallel"
        trigger = "memory_pressure"
        rationale = "promote CP/FSDP relief on hotspot subgraphs before changing PP skeleton"
        expected_switch_cost = 0.14
    if cross_node_worsened or stage_imbalance:
        scope = "skeleton" if cross_node_worsened else "local_parallel"
        trigger = "topology_shift" if cross_node_worsened else "stage_imbalance"
        rationale = "communication drift or persistent imbalance requires broader repartitioning"
        expected_switch_cost = 0.28 if cross_node_worsened else 0.14
    if tail_heavy and memory_hotspot:
        scope = "local_parallel"
        trigger = "tail_memory_hotspot"
        rationale = "tail-heavy execution under memory pressure should first try offload/checkpoint refinement before global PP rewrites"
        expected_switch_cost = 0.12
    elif tail_heavy:
        scope = "skeleton"
        trigger = "tail_drift"
        rationale = "tail-heavy execution suggests PP boundaries or virtual chunks should be rebalanced before smaller tweaks"
        expected_switch_cost = 0.22
    if comm_exposed and scope == "none":
        scope = "local_parallel"
        trigger = "comm_exposure"
        rationale = "communication exposure is better handled with placement and VPP chunk tuning before changing the full skeleton"
        expected_switch_cost = 0.14
    if optimization_hints:
        top_scope = str((optimization_hints[0] or {}).get("scope") or scope or "none")
        if scope == "none" and top_scope in {"pipe", "local_parallel", "skeleton"}:
            scope = top_scope
            trigger = "agent_hint"
            rationale = str((optimization_hints[0] or {}).get("rationale") or rationale)
            expected_switch_cost = {"pipe": 0.06, "local_parallel": 0.14, "skeleton": 0.22}.get(scope, expected_switch_cost)

    return ReplanDecision(
        trigger=trigger,
        scope=scope,
        rationale=rationale,
        expected_switch_cost=expected_switch_cost,
        fallback_if_rejected=_proposal_scope_fallback(scope),
        failure_modes=list(context_record.get("failure_modes") or []),
    ).normalized()


def _build_replan_decision(
    baseline: MegatronProgram,
    context_record: Dict[str, Any],
    previous_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _resolve_replan_decision(
        baseline,
        context_record,
        previous_context=previous_context,
    ).to_dict()


def _build_experiment_specs(
    baseline: MegatronProgram,
    evidence_programs: List[MegatronProgram],
    candidates: List[MegatronProgram],
) -> List[ExperimentSpec]:
    baseline_kind = str(baseline.metadata.get("program_kind") or "baseline")
    evidence_kinds = [str(program.metadata.get("program_kind") or f"evidence_{index:02d}") for index, program in enumerate(evidence_programs)]
    candidate_kinds = [str(program.metadata.get("program_kind") or f"candidate_{index:02d}") for index, program in enumerate(candidates)]
    dynamic_kinds = [kind for kind in evidence_kinds + candidate_kinds if "drift" in kind or "runtime_guided" in kind or "schedule" in kind]
    ablation_kinds = [kind for kind in candidate_kinds if any(token in kind for token in ("cp", "fsdp", "vpp", "schedule"))]
    return [
        ExperimentSpec(
            experiment_id="A_problem_existence",
            category="A",
            label="problem_existence",
            objective="show that fixed or restricted strategy spaces miss stable failure modes",
            program_kinds=[baseline_kind] + evidence_kinds,
        ).normalized(),
        ExperimentSpec(
            experiment_id="B_overall_effect",
            category="B",
            label="overall_effect",
            objective="compare static baseline against verifier-guided candidates",
            program_kinds=[baseline_kind] + candidate_kinds,
        ).normalized(),
        ExperimentSpec(
            experiment_id="C_ablation",
            category="C",
            label="ablation",
            objective="measure which local policy and pipe components drive gains",
            program_kinds=[baseline_kind] + (ablation_kinds or candidate_kinds),
        ).normalized(),
        ExperimentSpec(
            experiment_id="D_dynamic_replanning",
            category="D",
            label="dynamic_scene",
            objective="measure low-cost replanning under workload drift and topology changes",
            program_kinds=dynamic_kinds or evidence_kinds,
        ).normalized(),
    ]


def _build_summary_payload(
    *,
    export_only: bool,
    programs_dir: Path,
    runtime_summary: Dict[str, Any],
    runtime_signature: Dict[str, Any],
    context_record: Dict[str, Any],
    replan_decision: Dict[str, Any],
    bottleneck_signature: Dict[str, Any],
    rewrite: SearchSpaceSpec,
    baseline: MegatronProgram,
    baseline_metrics: Optional[Dict[str, Any]],
    best_program: Optional[MegatronProgram],
    best_metrics: Optional[Dict[str, Any]],
    tested: List[Dict[str, Any]],
    family_outside_trials: List[Dict[str, Any]],
    rejected_candidates: List[Dict[str, Any]],
    candidate_manifest: List[Dict[str, Any]],
    program_bank: ProgramBank,
    evidence_manifest: List[Dict[str, Any]],
    experiment_specs: Optional[List[ExperimentSpec]] = None,
    paper_artifacts: Optional[List[Dict[str, Any]]] = None,
    agent_proposals: Optional[List[Dict[str, Any]]] = None,
    agent_topology: Optional[Dict[str, Any]] = None,
    external_inputs: Optional[Dict[str, Any]] = None,
    strategy_template_library: Optional[List[Dict[str, Any]]] = None,
    template_selection_decision: Optional[Dict[str, Any]] = None,
    second_stage_runtime_summary: Optional[Dict[str, Any]] = None,
    second_stage_context_record: Optional[Dict[str, Any]] = None,
    second_stage_replan_decision: Optional[Dict[str, Any]] = None,
    second_stage_bottleneck_signature: Optional[Dict[str, Any]] = None,
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
        "runtime_signature": runtime_signature,
        "context_record": context_record,
        "failure_modes": list(context_record.get("failure_modes") or []),
        "derived_bottlenecks": list(context_record.get("derived_bottlenecks") or []),
        "optimization_hints": list(context_record.get("optimization_hints") or []),
        "replan_decision": replan_decision,
        "bottleneck_signature": bottleneck_signature,
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
        "motivation_evidence_manifest": evidence_manifest,
        "experiment_specs": [item.to_dict() for item in (experiment_specs or [])],
        "paper_artifacts": list(paper_artifacts or []),
        "agent_proposals": list(agent_proposals or []),
        "agent_topology": dict(agent_topology or {}),
        "external_inputs": dict(external_inputs or {}),
        "strategy_template_library": list(strategy_template_library or []),
        "template_selection_decision": dict(template_selection_decision or {}),
        "second_stage_runtime_summary": dict(second_stage_runtime_summary or {}),
        "second_stage_context_record": dict(second_stage_context_record or {}),
        "second_stage_replan_decision": dict(second_stage_replan_decision or {}),
        "second_stage_bottleneck_signature": dict(second_stage_bottleneck_signature or {}),
        "recommended_execution_order": [entry["config_name"] for entry in candidate_manifest],
        "program_bank": program_bank.to_dict(),
        "candidate_generation_count": len(candidate_entries),
        "candidate_execution_count": max(len(tested) - (1 if baseline_metrics is not None else 0), 0),
        "compile_success_rate": compile_success_rate,
        "family_outside_ratio": family_outside_ratio,
        "baseline_estimated_memory": estimate_program_memory(baseline).to_dict(),
        "best_estimated_memory": estimate_program_memory(best_program).to_dict() if best_program is not None else None,
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




def _profile_context(program: MegatronProgram) -> Tuple[MachineProfile, BackendCaps]:
    machine_profile = (program.machine_profile or default_machine_profile(str(program.cluster.target))).normalized()
    backend_caps = (program.backend_caps or default_backend_caps("local")).normalized()
    return machine_profile, backend_caps


def _profile_prefers_small_tp(profile: MachineProfile) -> bool:
    return bool(profile.prefer_small_tp or profile.communication_sensitivity in {"high", "very_high"})


def _is_dual_target(target: Any) -> bool:
    return str(target or "") in {"dual_g4_g5", "dual_g5_g5"}


def _node_speed_map(program: MegatronProgram) -> Dict[str, float]:
    nodes = [str(node) for node in (program.cluster.nodes or [])]
    if str(program.cluster.target) == "dual_g4_g5":
        return {node: (1.18 if node == "g5" else 1.0) for node in nodes}
    return {node: 1.0 for node in nodes}


def _fastest_node(program: MegatronProgram) -> Optional[str]:
    speeds = _node_speed_map(program)
    if not speeds:
        return None
    return max(speeds.items(), key=lambda item: (item[1], item[0]))[0]


def _preferred_module_nodes(program: MegatronProgram) -> Dict[str, str]:
    if str(program.cluster.target) == "dual_g4_g5":
        return {"embedding": "g4", "loss": "g5"}
    if str(program.cluster.target) == "dual_g5_g5":
        return {"embedding": "g5_0", "loss": "g5_1"}
    node_name = str(program.cluster.nodes[-1]) if program.cluster.nodes else "g5"
    return {"embedding": node_name, "loss": node_name}


def _execution_backend_family(program: MegatronProgram, context_record: Optional[Dict[str, Any]] = None) -> str:
    if context_record is not None:
        backend_family = str(((context_record.get("backend_context") or {}).get("backend_family")) or "").strip().lower()
        if backend_family:
            return backend_family
    hint = str(
        (program.metadata or {}).get("execution_backend")
        or (program.metadata or {}).get("planner_backend")
        or "megatron_core"
    ).strip().lower()
    return "torchtitan" if "torchtitan" in hint else "megatron_core"


def _profile_max_tp(program: MegatronProgram, profile: MachineProfile) -> int:
    cluster_limit = max(int(program.cluster.gpus_per_node), 1)
    baseline_tp = max(int(program.parallel.tp_degree), 1)
    if _profile_prefers_small_tp(profile):
        return max(1, min(cluster_limit, baseline_tp))
    return cluster_limit


def _profile_max_pp(program: MegatronProgram, profile: MachineProfile) -> int:
    base = max(int(program.parallel.pp_degree), 1)
    if _is_dual_target(program.cluster.target):
        preferred = 4
        if program.model.track == "dense" and int(program.model.num_layers) >= 16:
            preferred = 8 if int(program.cluster.world_size) >= 16 else 4
        return min(int(program.cluster.world_size), max(base, preferred))
    if bool(profile.prefer_pp_for_scaling):
        preferred = 2
        if program.model.track == "dense" and int(program.model.num_layers) >= 8:
            preferred = 4
        return min(int(program.cluster.world_size), max(base, preferred))
    return min(int(program.cluster.world_size), base)


def _build_baseline_program(args: argparse.Namespace) -> MegatronProgram:
    program = default_moe_smoke_program(args.run_target) if args.model_track == "moe" else default_dense_program(args.run_target)
    explicit_tp = int(args.tp or 0) > 0
    explicit_pp = int(args.pp or 0) > 0
    explicit_cp = int(args.cp or 0) > 0
    explicit_ep = int(args.ep or 0) > 0
    explicit_expert_tp = int(args.expert_tp or 0) > 0
    program.parallel.tp_degree = int(args.tp or program.parallel.tp_degree)
    program.parallel.pp_degree = int(args.pp or program.parallel.pp_degree)
    program.parallel.vpp_degree = int(args.vpp or program.parallel.vpp_degree)
    program.parallel.ep_degree = int(args.ep or program.parallel.ep_degree)
    program.parallel.cp_degree = int(args.cp or program.parallel.cp_degree)
    program.parallel.expert_tp_degree = int(args.expert_tp or program.parallel.expert_tp_degree)
    program.parallel.sp_enabled = False
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
        if _is_dual_target(args.run_target):
            program.layout.stage_to_node = default_grouped_stage_to_node(args.run_target, int(program.parallel.pp_degree))
        else:
            program.layout.stage_to_node = [str(program.cluster.nodes[-1])] * int(program.parallel.pp_degree)
    requested_impl = str(getattr(args, "transformer_impl", "auto") or "auto").strip().lower()
    program.machine_profile = default_machine_profile(args.run_target)
    program.backend_caps = default_backend_caps("local" if requested_impl == "auto" else requested_impl)
    program.parallel.sp_enabled = bool(int(program.parallel.tp_degree) > 1 and program.backend_caps.supports_sequence_parallel)

    if args.model_track == "dense":
        memory_budget = float(program.constraints.memory_budget_gb or program.machine_profile.device_memory_gb or 24.0)
        program.constraints.memory_budget_gb = memory_budget
        if (
            not explicit_cp
            and int(args.seq_len) >= 2048
            and int(program.parallel.cp_degree) == 1
        ):
            product_without_cp = (
                int(program.parallel.tp_degree)
                * int(program.parallel.pp_degree)
                * int(program.parallel.ep_degree)
                * int(program.parallel.expert_tp_degree)
            )
            if int(program.cluster.world_size) % max(product_without_cp * 2, 1) == 0:
                program.parallel.cp_degree = 2
                program.plane_map.attention.cp_degree = 2

        if (
            not explicit_tp
            and not explicit_pp
            and args.run_target in {"single_g4", "single_g5"}
            and int(program.parallel.pp_degree) == 1
        ):
            program.parallel.tp_degree = 2
            program.parallel.pp_degree = 2
            program.parallel.vpp_degree = 1
            program.parallel.sp_enabled = bool(int(program.parallel.tp_degree) > 1 and program.backend_caps.supports_sequence_parallel)
            first = int(program.model.num_layers) // 2
            second = int(program.model.num_layers) - first
            node_name = str(program.cluster.nodes[-1])
            program.partition = program.partition.from_dict(
                {
                    "stages": [
                        {"decoder_layers": first, "special_tokens": ["E"]},
                        {"decoder_layers": second, "special_tokens": ["L"]},
                    ]
                }
            )
            program.layout.stage_to_node = [node_name, node_name]
            program.layout.vpp_degree = 1

    if args.model_track == "moe":
        if not explicit_ep:
            program.parallel.ep_degree = max(2, int(program.parallel.ep_degree))
        if not explicit_expert_tp:
            program.parallel.expert_tp_degree = max(1, int(program.parallel.expert_tp_degree))

    program.batch_plan = BatchPlanSpec(
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
        grad_accum_steps=None,
        target_tokens_per_step=int(args.global_batch_size) * int(args.seq_len),
    )
    program.length_bucket_policies = default_length_bucket_policies()
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
    return _sync_batch_plan_metadata(program)


def _rewrite_space(program: MegatronProgram, runtime_summary: Dict[str, Any]) -> SearchSpaceSpec:
    trace_summary = reduce_trial_trace(program, runtime_summary=runtime_summary)
    bottleneck = classify_bottleneck(program, trace_summary)
    context_record = build_context_record(program, runtime_summary=runtime_summary)
    backend_family = _execution_backend_family(program, context_record)
    failure_labels = {str(item.get("label")) for item in (context_record.get("failure_modes") or [])}
    runtime_evidence = dict(context_record.get("runtime_evidence") or {})
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or trace_summary.get("bubble_ratio") or runtime_summary.get("bubble_ratio") or 0.0)
    stage_spread = float(
        runtime_evidence.get("stage_load_variance")
        or runtime_summary.get("stage_spread_ratio")
        or trace_summary.get("stage_load_variance")
        or 0.0
    )
    cross_node_exposed = float(
        runtime_evidence.get("cross_node_exposed_ratio")
        or trace_summary.get("cross_node_exposed_ratio")
        or runtime_summary.get("cross_node_exposed_ratio")
        or runtime_summary.get("exposed_comm_ratio")
        or 0.0
    )
    oom_detected = bool(runtime_summary.get("oom") or runtime_summary.get("last_trial_oom") or runtime_summary.get("baseline_oom"))
    peak_memory_ratio = float(
        runtime_evidence.get("peak_reserved_ratio")
        or trace_summary.get("peak_reserved_ratio")
        or runtime_summary.get("peak_memory_ratio")
        or runtime_summary.get("memory_utilization_ratio")
        or runtime_summary.get("memory_pressure_ratio")
        or 0.0
    )
    is_dual = _is_dual_target(program.cluster.target)
    machine_profile, backend_caps = _profile_context(program)
    baseline_memory = estimate_program_memory(program)
    current_micro_batch = max(int(program.metadata.get("micro_batch_size", 1) or 1), 1)
    current_seq_len = max(int(program.metadata.get("seq_len", 1024) or 1024), 1)
    length_bucket_policy = trace_summary.get("length_bucket_policy") or {}

    rules: List[ConstraintRuleSpec] = []
    required_local_axes = ["tp"] if is_dual else []
    if program.model.track == "moe":
        required_local_axes.append("ep")

    allow_nonuniform = bool(
        is_dual
        or int(program.parallel.pp_degree) > 1
        or stage_spread >= 0.08
        or "compute_imbalance" in failure_labels
        or machine_profile.communication_sensitivity in {"high", "very_high"}
    )
    if allow_nonuniform:
        rules.append(
            ConstraintRuleSpec(
                name="relax_uniform_pp",
                rationale="profile or runtime asymmetry indicates uniform stage prior is too restrictive",
                params={
                    "stage_spread_ratio": stage_spread,
                    "target": program.cluster.target,
                    "communication_sensitivity": machine_profile.communication_sensitivity,
                },
            )
        )

    allow_single_node_pp_split = bool(
        program.cluster.target in {"single_g4", "single_g5"}
        and program.model.track == "dense"
        and int(program.parallel.pp_degree) == 1
        and int(program.parallel.tp_degree) >= 2
        and int(program.model.num_layers) >= 4
        and bool(machine_profile.prefer_pp_for_scaling)
    )
    if allow_single_node_pp_split:
        rules.append(
            ConstraintRuleSpec(
                name="allow_single_node_pp_split",
                rationale="single-node consumer dense profile prefers PP scaling over larger TP",
                params={
                    "tp_degree": int(program.parallel.tp_degree),
                    "num_layers": int(program.model.num_layers),
                    "profile": machine_profile.name,
                },
            )
        )

    allow_sequence_parallel_toggle = bool(
        int(program.parallel.tp_degree) > 1 and backend_caps.supports_sequence_parallel
    )
    if allow_sequence_parallel_toggle:
        rules.append(
            ConstraintRuleSpec(
                name="allow_sequence_parallel_toggle",
                rationale="sequence parallel toggles are allowed only when backend capabilities support the intended path",
                params={
                    "tp_degree": int(program.parallel.tp_degree),
                    "baseline_sp_enabled": bool(program.parallel.sp_enabled),
                    "transformer_impl": backend_caps.transformer_impl,
                },
            )
        )
    elif int(program.parallel.tp_degree) > 1:
        rules.append(
            ConstraintRuleSpec(
                name="suppress_sequence_parallel_toggle",
                rationale="sequence parallel toggle suppressed because backend capabilities do not support it",
                params={"transformer_impl": backend_caps.transformer_impl},
            )
        )

    allow_asymmetric_vpp = bool(
        int(program.parallel.pp_degree) > 1 and int(program.model.num_layers) % (int(program.parallel.pp_degree) * 2) == 0
    )
    if oom_detected or baseline_memory.pressure_score >= 1.0 or peak_memory_ratio >= 0.92:
        allow_asymmetric_vpp = False
    if allow_asymmetric_vpp:
        rules.append(
            ConstraintRuleSpec(
                name="allow_asymmetric_vpp",
                rationale="model layers can be regrouped into a VPP-aware schedule family",
                params={"num_layers": int(program.model.num_layers), "pp_degree": int(program.parallel.pp_degree)},
            )
        )

    allow_dual_plane = bool(program.model.track == "moe" and backend_caps.supports_dual_plane)
    if allow_dual_plane:
        rules.append(
            ConstraintRuleSpec(
                name="decouple_attention_and_moe_planes",
                rationale="dual-plane candidate retained only for MoE profiles with backend support",
                params={"model_track": program.model.track, "transformer_impl": backend_caps.transformer_impl},
            )
        )
    elif program.model.track == "moe":
        rules.append(
            ConstraintRuleSpec(
                name="suppress_dual_plane",
                rationale="dual-plane candidate suppressed because backend capabilities do not advertise support",
                params={"transformer_impl": backend_caps.transformer_impl},
            )
        )

    allow_stage_aware = bool(
        int(program.parallel.pp_degree) > 1
        and (
            is_dual
            or bubble_ratio >= 0.03
            or stage_spread >= 0.08
            or str((bottleneck or {}).get("dominant_label")) in {"tp_overpartitioned", "stage_imbalanced", "memory_underfilled"}
            or "schedule_coupling" in failure_labels
            or machine_profile.communication_sensitivity in {"high", "very_high"}
            or machine_profile.prefer_pp_for_scaling
        )
    )
    if allow_stage_aware:
        rules.append(
            ConstraintRuleSpec(
                name="allow_stage_aware_schedule",
                rationale="bubble, stage spread, or communication-sensitive profile suggests grouped stage-aware scheduling",
                params={
                    "bubble_ratio": bubble_ratio,
                    "stage_spread_ratio": stage_spread,
                    "communication_sensitivity": machine_profile.communication_sensitivity,
                },
            )
        )
    allow_morphable_pipeline = bool(
        int(program.parallel.pp_degree) > 1
        and (
            allow_nonuniform
            or allow_asymmetric_vpp
            or stage_spread >= 0.06
            or bubble_ratio >= 0.08
            or cross_node_exposed >= 0.08
            or peak_memory_ratio >= 0.84
            or "compute_imbalance" in failure_labels
            or "memory_hotspot" in failure_labels
            or "communication_drag" in failure_labels
        )
    )
    if allow_morphable_pipeline:
        rules.append(
            ConstraintRuleSpec(
                name="allow_morphable_pipeline",
                rationale="joint structure-memory-communication evidence justifies regrouping stage/chunk shape as a first-class search object",
                params={
                    "stage_spread_ratio": stage_spread,
                    "bubble_ratio": bubble_ratio,
                    "peak_memory_ratio": peak_memory_ratio,
                    "cross_node_exposed_ratio": cross_node_exposed,
                },
            )
        )

    allow_torchtitan_schedule_sandbox = bool(
        backend_family == "torchtitan"
        and int(program.parallel.pp_degree) > 1
        and (
            bubble_ratio >= 0.08
            or stage_spread >= 0.08
            or "schedule_coupling" in failure_labels
            or "tail_heavy" in {str(item.get("label")) for item in (context_record.get("derived_bottlenecks") or [])}
        )
    )
    if allow_torchtitan_schedule_sandbox:
        rules.append(
            ConstraintRuleSpec(
                name="allow_torchtitan_schedule_sandbox",
                rationale="torchtitan backend can probe richer zero-bubble or DualPipe-style schedule families when bubble/tail evidence is persistent",
                params={"backend_family": backend_family, "bubble_ratio": bubble_ratio},
            )
        )

    max_tp_size = _profile_max_tp(program, machine_profile)
    if max_tp_size < int(program.cluster.gpus_per_node):
        rules.append(
            ConstraintRuleSpec(
                name="tighten_tp_bound",
                rationale="TP bound tightened for communication-sensitive consumer profile",
                params={"max_tp_size": max_tp_size, "profile": machine_profile.name},
            )
        )

    max_pp_size = _profile_max_pp(program, machine_profile)
    if max_pp_size > int(program.parallel.pp_degree):
        rules.append(
            ConstraintRuleSpec(
                name="expand_pp_bound",
                rationale="PP bound opened because the machine profile prefers PP for scaling",
                params={"max_pp_size": max_pp_size, "profile": machine_profile.name},
            )
        )

    if cross_node_exposed > 0.0:
        rules.append(
            ConstraintRuleSpec(
                name="localize_high_frequency_axes",
                rationale="runtime logs show exposed communication on slower boundaries",
                params={"cross_node_exposed_ratio": cross_node_exposed},
            )
        )
    if "communication_drag" in failure_labels:
        rules.append(
            ConstraintRuleSpec(
                name="promote_local_parallel_replan",
                rationale="communication drag suggests higher-frequency adjustments should stay local to hot subgraphs first",
                params={"failure_mode": "communication_drag"},
            )
        )

    prefer_memory_relief = bool(
        oom_detected
        or baseline_memory.pressure_score >= 1.0
        or peak_memory_ratio >= 0.92
        or "memory_hotspot" in failure_labels
        or (machine_profile.device_memory_gb or 0) <= 24 and current_seq_len >= 2048
    )
    max_micro_batch_size = current_micro_batch
    max_estimated_memory_pressure = 1.15
    if prefer_memory_relief:
        max_micro_batch_size = 1
        max_estimated_memory_pressure = 1.60 if oom_detected else 1.12
        rules.append(
            ConstraintRuleSpec(
                name="tighten_memory_budget",
                rationale="OOM or high memory pressure indicates the search must stay inside a smaller memory envelope",
                params={
                    "oom_detected": oom_detected,
                    "peak_memory_ratio": peak_memory_ratio,
                    "baseline_pressure_score": round(float(baseline_memory.pressure_score), 4),
                    "budget_gb": round(float(baseline_memory.budget_gb), 3),
                },
            )
        )

    if prefer_memory_relief and int(program.parallel.pp_degree) == 1 and program.cluster.target in {"single_g4", "single_g5"}:
        allow_nonuniform = True
        allow_single_node_pp_split = bool(int(program.model.num_layers) >= 4)
        rules.append(
            ConstraintRuleSpec(
                name="prefer_pp_for_memory_relief",
                rationale="single-node OOM should first try to split layers across PP stages before exploring higher-risk schedules",
                params={"pp_degree": int(program.parallel.pp_degree), "num_layers": int(program.model.num_layers)},
            )
        )

    if prefer_memory_relief and current_seq_len >= 2048 and int(program.parallel.cp_degree) == 1:
        rules.append(
            ConstraintRuleSpec(
                name="allow_cp_for_long_context_memory_relief",
                rationale="long-context profile may use CP candidates to reduce attention memory pressure",
                params={"seq_len": current_seq_len},
            )
        )

    allow_hybrid_shard = bool(
        backend_family == "torchtitan"
        and (
            prefer_memory_relief
            or is_dual
            or current_seq_len >= 2048
            or "memory_hotspot" in failure_labels
            or "memory_bound" in failure_labels
        )
    )
    if allow_hybrid_shard:
        rules.append(
            ConstraintRuleSpec(
                name="allow_hybrid_shard",
                rationale="torchtitan sandbox can probe HSDP/FSDP2 mesh and reshard policies when memory pressure or dual-node asymmetry is present",
                params={"backend_family": backend_family, "seq_len": current_seq_len},
            )
        )

    search_space = SearchSpaceSpec(
        allow_nonuniform_partition=allow_nonuniform,
        allow_single_node_pp_split=allow_single_node_pp_split,
        allow_sequence_parallel_toggle=allow_sequence_parallel_toggle,
        allow_asymmetric_vpp=allow_asymmetric_vpp,
        allow_dual_plane=allow_dual_plane,
        allow_stage_aware_schedule=allow_stage_aware,
        allow_hybrid_shard=allow_hybrid_shard,
        allow_torchtitan_schedule_sandbox=allow_torchtitan_schedule_sandbox,
        allow_subgraph_submeshes=False,
        allow_heterogeneous_apipe=False,
        allow_morphable_pipeline=allow_morphable_pipeline,
        max_tp_size=max_tp_size,
        max_pp_size=max_pp_size,
        max_ep_size=int(program.cluster.gpus_per_node) if is_dual else None,
        max_cp_size=min(
            int(program.cluster.gpus_per_node),
            int(length_bucket_policy.get("cp_cap") or (int(program.cluster.gpus_per_node) if (is_dual or current_seq_len >= 2048) else 1)),
        ),
        max_vpp_size=2 if allow_asymmetric_vpp else 1,
        max_shard_group_size=min(int(program.cluster.gpus_per_node), 8) if allow_hybrid_shard else None,
        max_replicate_group_size=max(len(program.cluster.nodes), 1) if allow_hybrid_shard else None,
        max_micro_batch_size=min(max_micro_batch_size, int(length_bucket_policy.get("micro_batch_cap") or max_micro_batch_size)),
        max_estimated_memory_pressure=max_estimated_memory_pressure,
        prefer_memory_relief=prefer_memory_relief,
        required_node_local_axes=required_local_axes,
        preferred_node_for_module=_preferred_module_nodes(program),
        forbidden_axes_by_node={"g5": ["tp"]} if str(program.cluster.target) == "dual_g4_g5" and cross_node_exposed >= 0.05 else {},
        allowed_schedule_skeletons=["fixed_1f1b", "stage_aware_grouped"] if allow_stage_aware else ["fixed_1f1b"],
        allowed_schedule_templates=(
            (
                list(length_bucket_policy.get("schedule_templates") or [])
                + (["fixed_1f1b"] if "fixed_1f1b" not in set(length_bucket_policy.get("schedule_templates") or []) else [])
                + (
                    ["torchtitan_zero_bubble", "torchtitan_dualpipev"]
                    if allow_torchtitan_schedule_sandbox
                    else []
                )
            )
            if allow_stage_aware
            else (["fixed_1f1b"] + (["torchtitan_zero_bubble", "torchtitan_dualpipev"] if allow_torchtitan_schedule_sandbox else []))
        ),
        rewrite_rules=rules,
        notes="space rewrite derived from explicit profile/capability priors plus runtime summary and memory envelope",
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
    return _sync_batch_plan_metadata(candidate)


def _partition_from_stage_layers(stage_layers: List[int]) -> Dict[str, Any]:
    stages: List[Dict[str, Any]] = []
    last_index = len(stage_layers) - 1
    for index, stage_layers_count in enumerate(stage_layers):
        special_tokens: List[str] = []
        if index == 0:
            special_tokens.append("E")
        if index == last_index:
            special_tokens.append("L")
        stages.append({"decoder_layers": int(stage_layers_count), "special_tokens": special_tokens})
    return {"stages": stages}


def _grouped_stage_nodes(program: MegatronProgram, pp_degree: int) -> List[str]:
    return default_grouped_stage_to_node(str(program.cluster.target), int(pp_degree))


def _apply_grouped_pp_layout(
    candidate: MegatronProgram,
    *,
    target_tp: int,
    target_pp: int,
) -> Optional[MegatronProgram]:
    world_size = int(candidate.cluster.world_size)
    product = (
        int(target_tp)
        * int(target_pp)
        * int(candidate.parallel.cp_degree)
        * int(candidate.parallel.ep_degree)
        * int(candidate.parallel.expert_tp_degree)
    )
    if product <= 0 or world_size % product != 0:
        return None
    stage_to_node = _grouped_stage_nodes(candidate, int(target_pp))
    stage_layers = weighted_stage_layer_allocation(str(candidate.cluster.target), int(candidate.model.num_layers), stage_to_node)
    if sum(stage_layers) != int(candidate.model.num_layers):
        return None
    candidate.parallel.tp_degree = int(target_tp)
    candidate.parallel.pp_degree = int(target_pp)
    candidate.parallel.vpp_degree = 1
    candidate.parallel.sp_enabled = bool(int(target_tp) > 1 and candidate.backend_caps.supports_sequence_parallel)
    candidate.partition = candidate.partition.from_dict(_partition_from_stage_layers(stage_layers))
    candidate.layout.stage_to_node = list(stage_to_node)
    candidate.layout.vpp_degree = 1
    candidate.layout.pipeline_layout = _virtual_stage_layout(stage_layers)
    candidate.schedule.template = "fixed_1f1b"
    candidate.schedule.skeleton = "fixed_1f1b"
    candidate.schedule.dispatch_order = "default"
    candidate.schedule.microbatch_group_size_per_vp_stage = None
    candidate.plane_map.attention.tp_degree = int(target_tp)
    candidate.plane_map.attention.cp_degree = int(candidate.parallel.cp_degree)
    return candidate


def _build_dual_node_pp4_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if not _is_dual_target(program.cluster.target):
        return None
    candidate = _clone_program(program)
    target_pp = 4
    product_without_tp = (
        target_pp
        * int(candidate.parallel.cp_degree)
        * int(candidate.parallel.ep_degree)
        * int(candidate.parallel.expert_tp_degree)
    )
    if product_without_tp <= 0 or int(candidate.cluster.world_size) % product_without_tp != 0:
        return None
    target_tp = int(candidate.cluster.world_size) // product_without_tp
    if target_tp <= 0 or target_tp > int(candidate.cluster.gpus_per_node):
        return None
    desired_stage_to_node = _grouped_stage_nodes(candidate, target_pp)
    desired_stage_layers = weighted_stage_layer_allocation(str(candidate.cluster.target), int(candidate.model.num_layers), desired_stage_to_node)
    if (
        int(program.parallel.tp_degree) == target_tp
        and int(program.parallel.pp_degree) == target_pp
        and list(program.layout.stage_to_node) == list(desired_stage_to_node)
        and [int(stage.decoder_layers) for stage in program.partition.stages] == list(desired_stage_layers)
    ):
        return None
    candidate = _apply_grouped_pp_layout(candidate, target_tp=target_tp, target_pp=target_pp)
    if candidate is None:
        return None
    candidate.metadata["program_kind"] = "candidate_dual_node_pp4_node_aware"
    candidate.metadata["priority_rank"] = 12
    return _sync_batch_plan_metadata(candidate)


def _build_runtime_guided_partition(program: MegatronProgram, runtime_summary: Dict[str, Any]) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) < 2:
        return None
    candidate = _clone_program(program)
    stage_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
    plan = _runtime_guided_partition_plan(stage_layers, runtime_summary)
    if plan is None:
        return None
    slow_idx = int(plan["donor_stage_index"])
    fast_idx = int(plan["receiver_stage_index"])
    shift = int(plan["shift_layers"])
    if slow_idx >= len(stage_layers) or fast_idx >= len(stage_layers) or slow_idx == fast_idx:
        return None
    stage_layers[slow_idx] -= shift
    stage_layers[fast_idx] += shift
    for index, stage in enumerate(candidate.partition.stages):
        stage.decoder_layers = stage_layers[index]
    candidate.layout.pipeline_layout = None
    candidate.metadata["program_kind"] = "candidate_runtime_guided_partition"
    candidate.metadata["slow_stage_index"] = int(slow_idx)
    candidate.metadata["fast_stage_index"] = int(fast_idx)
    candidate.metadata["stage_spread_ratio"] = float(plan["spread_ratio"])
    candidate.metadata["runtime_partition_focus"] = str(plan["focus"])
    candidate.metadata["runtime_partition_shift"] = int(shift)
    candidate.metadata["runtime_partition_stage_objectives"] = list(plan["stage_objectives"])
    candidate.metadata["boundary_semantic_focus"] = str(plan["focus"])
    candidate.metadata["pipeline_wait_ratio"] = float(plan["pipeline_wait_ratio"])
    candidate.metadata["optimizer_exposed_ratio"] = float(plan["optimizer_exposed_ratio"])
    candidate.metadata["stage_skew"] = float(plan["stage_skew"])
    return _sync_batch_plan_metadata(candidate)


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
    candidate.schedule.template = "fixed_1f1b"
    candidate.metadata["program_kind"] = "candidate_single_node_pp_split"
    return _sync_batch_plan_metadata(candidate)


def _build_pp_scaleout_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    world_size = int(program.cluster.world_size)
    current_pp = int(program.parallel.pp_degree)
    current_tp = int(program.parallel.tp_degree)
    if world_size < 4 or current_pp >= 4 or current_tp <= 1:
        return None
    target_pp = 4
    target_tp = max(1, current_tp // 2)
    candidate = _clone_program(program)
    candidate = _apply_grouped_pp_layout(candidate, target_tp=target_tp, target_pp=target_pp)
    if candidate is None:
        return None
    candidate.metadata["program_kind"] = "candidate_pp_scaleout"
    candidate.metadata["priority_rank"] = 24
    return _sync_batch_plan_metadata(candidate)


def _build_stage_aware_schedule(program: MegatronProgram) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    candidate = _clone_program(program)
    if int(candidate.parallel.vpp_degree) == 1:
        counts = _pp_vpp_layout_counts(int(candidate.parallel.pp_degree), int(candidate.model.num_layers), "interleaved_grouped_g2")
        if counts is None:
            return None
        candidate.parallel.vpp_degree = 2
        candidate.layout.vpp_degree = 2
        candidate.parallel.sp_enabled = bool(int(candidate.parallel.tp_degree) > 1)
        candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
    candidate.schedule.microbatch_group_size_per_vp_stage = 2
    candidate.schedule.skeleton = "stage_aware_grouped"
    candidate.schedule.template = "interleaved_grouped_g2"
    candidate.schedule.dispatch_order = "frontload_forward"
    candidate.metadata["program_kind"] = "candidate_stage_aware_schedule"
    candidate.metadata["priority_rank"] = 20
    return _sync_batch_plan_metadata(candidate)


def _build_runtime_guided_schedule(program: MegatronProgram, runtime_summary: Dict[str, Any]) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    slow_idx, _, stage_windows = _dominant_stage_indices(runtime_summary, int(program.parallel.pp_degree))
    if bubble_ratio < 0.03 and slow_idx is None:
        return None
    candidate = _clone_program(program)
    if int(candidate.parallel.vpp_degree) == 1:
        counts = _pp_vpp_layout_counts(int(candidate.parallel.pp_degree), int(candidate.model.num_layers), "interleaved_grouped_g2")
        if counts is not None:
            candidate.parallel.vpp_degree = 2
            candidate.layout.vpp_degree = 2
            candidate.parallel.sp_enabled = bool(int(candidate.parallel.tp_degree) > 1)
            candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
    group_size = 2
    template = "interleaved_grouped_g2"
    if bubble_ratio >= 0.15:
        group_size = 4
        template = "interleaved_grouped_g4"
    elif bubble_ratio >= 0.08:
        group_size = 3
    dispatch_order = "default"
    if slow_idx is not None and stage_windows:
        if slow_idx >= max(len(stage_windows) - 1, 0):
            dispatch_order = "frontload_forward"
        elif slow_idx == 0:
            dispatch_order = "balanced_round_robin"
        else:
            dispatch_order = "middle_stage_relief"
    candidate.schedule.microbatch_group_size_per_vp_stage = group_size
    candidate.schedule.skeleton = "stage_aware_grouped"
    candidate.schedule.template = template
    candidate.schedule.dispatch_order = dispatch_order
    candidate.metadata["program_kind"] = "candidate_runtime_guided_schedule"
    candidate.metadata["bubble_ratio"] = round(float(bubble_ratio), 4)
    if slow_idx is not None:
        candidate.metadata["slow_stage_index"] = int(slow_idx)
    candidate.metadata["priority_rank"] = 25
    return _sync_batch_plan_metadata(candidate)


def _build_pp_vpp_scaleout_candidate(program: MegatronProgram, runtime_summary: Dict[str, Any]) -> Optional[MegatronProgram]:
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    base = _build_pp_scaleout_candidate(program)
    if base is None:
        return None
    template = "pp4_middle_relief" if bubble_ratio >= 0.08 else "pp4_frontload"
    counts = _pp_vpp_layout_counts(int(base.parallel.pp_degree), int(base.model.num_layers), template)
    if counts is None:
        return None
    base.parallel.vpp_degree = 2
    base.layout.vpp_degree = 2
    base.parallel.sp_enabled = bool(int(base.parallel.tp_degree) > 1 and program.backend_caps.supports_sequence_parallel)
    base.layout.pipeline_layout = _virtual_stage_layout(counts)
    base.schedule.microbatch_group_size_per_vp_stage = 4 if bubble_ratio >= 0.12 else 2
    base.schedule.skeleton = "stage_aware_grouped"
    base.schedule.template = template if bubble_ratio >= 0.08 else "interleaved_grouped_g4"
    if bubble_ratio < 0.08:
        base.schedule.dispatch_order = "balanced_round_robin"
    else:
        base.schedule.dispatch_order = "frontload_forward" if template == "pp4_frontload" else "middle_stage_relief"
    base.metadata["program_kind"] = "candidate_pp_vpp_scaleout"
    base.metadata["bubble_ratio"] = round(float(bubble_ratio), 4)
    base.metadata["priority_rank"] = 40
    return _sync_batch_plan_metadata(base)


def _build_memory_relief_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    candidate = _clone_program(program)
    current_micro = max(int(candidate.batch_plan.micro_batch_size or candidate.metadata.get("micro_batch_size", 1) or 1), 1)
    current_seq = max(int(candidate.metadata.get("seq_len", 1024) or 1024), 1)
    changed = False

    if current_micro > 1:
        candidate.batch_plan.micro_batch_size = max(1, current_micro // 2)
        changed = True

    product_without_cp = (
        int(candidate.parallel.tp_degree)
        * int(candidate.parallel.pp_degree)
        * int(candidate.parallel.ep_degree)
        * int(candidate.parallel.expert_tp_degree)
    )
    if current_seq >= 2048 and int(candidate.parallel.cp_degree) == 1:
        if int(candidate.cluster.world_size) % max(product_without_cp * 2, 1) == 0:
            candidate.parallel.cp_degree = 2
            candidate.plane_map.attention.cp_degree = 2
            changed = True

    if (
        candidate.cluster.target in {"single_g4", "single_g5"}
        and int(candidate.parallel.pp_degree) == 1
        and int(candidate.model.num_layers) >= 4
        and int(candidate.model.num_layers) % 2 == 0
    ):
        candidate.parallel.tp_degree = max(1, int(candidate.parallel.tp_degree) // 2)
        candidate.parallel.pp_degree = 2
        candidate.parallel.vpp_degree = 1
        candidate.parallel.sp_enabled = bool(int(candidate.parallel.tp_degree) > 1)
        first = int(candidate.model.num_layers) // 2
        second = int(candidate.model.num_layers) - first
        node_name = str(candidate.cluster.nodes[-1])
        candidate.partition = candidate.partition.from_dict(
            {
                "stages": [
                    {"decoder_layers": first, "special_tokens": ["E"]},
                    {"decoder_layers": second, "special_tokens": ["L"]},
                ]
            }
        )
        candidate.layout.stage_to_node = [node_name, node_name]
        candidate.layout.vpp_degree = 1
        changed = True

    if not changed:
        return None

    candidate.metadata["program_kind"] = "candidate_memory_relief"
    candidate.metadata["priority_rank"] = 10
    return _sync_batch_plan_metadata(candidate)


def _build_sequence_parallel_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if int(program.parallel.tp_degree) <= 1:
        return None
    candidate = _clone_program(program)
    candidate.parallel.sp_enabled = not bool(program.parallel.sp_enabled)
    candidate.metadata["program_kind"] = "candidate_sequence_parallel_toggle"
    candidate.metadata["sequence_parallel_target_state"] = bool(candidate.parallel.sp_enabled)
    return _sync_batch_plan_metadata(candidate)


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
    return _sync_batch_plan_metadata(candidate)


def _apply_torchtitan_hybrid_shard_policy(
    program: MegatronProgram,
    *,
    selective_only: bool,
    hot_subgraphs: Optional[Dict[str, str]] = None,
) -> MegatronProgram:
    candidate = _clone_program(program)
    candidate.metadata["execution_backend"] = "torchtitan"
    shard_group_size = min(int(candidate.cluster.gpus_per_node), 4 if _is_dual_target(candidate.cluster.target) else 8)
    replicate_group_size = max(len(candidate.cluster.nodes), 1) if _is_dual_target(candidate.cluster.target) else 1
    hot_subgraphs = dict(hot_subgraphs or {})
    for local in candidate.strategy_ir.local_parallel:
        is_hot = not selective_only or str(local.subgraph) in hot_subgraphs
        if not is_hot:
            continue
        local.fsdp_scope = "selective" if selective_only else "full"
        local.shard_strategy = "hsdp" if replicate_group_size > 1 else "fsdp"
        local.reshard_policy = "intra_node" if replicate_group_size > 1 else "full"
        local.shard_group_size = shard_group_size
        local.replicate_group_size = replicate_group_size
        local.offload_policy = "none"
        local.reduce_dtype = "bf16"
    return _sync_batch_plan_metadata(candidate)


def _build_torchtitan_hsdp_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if _execution_backend_family(program, context_record) != "torchtitan":
        return None
    failure_modes = list((context_record or {}).get("failure_modes") or [])
    hot_subgraphs = {
        str(item.get("anchor")): str(item.get("label"))
        for item in failure_modes
        if str(item.get("anchor") or "").startswith("subg_stage_")
    }
    if not hot_subgraphs and not any(str(item.get("label")) in {"memory_hotspot", "memory_skew"} for item in failure_modes):
        return None
    candidate = _apply_torchtitan_hybrid_shard_policy(program, selective_only=True, hot_subgraphs=hot_subgraphs)
    candidate.metadata["program_kind"] = "candidate_torchtitan_hsdp_mesh"
    candidate.metadata["priority_rank"] = 14
    return _sync_batch_plan_metadata(candidate)


def _build_pp_hsdp_hybrid_candidate(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if _execution_backend_family(program, context_record) != "torchtitan":
        return None
    if int(program.parallel.pp_degree) <= 1:
        return None
    if float(runtime_summary.get("bubble_ratio") or 0.0) < 0.03 and not any(
        str(item.get("label")) in {"memory_hotspot", "memory_skew"} for item in (context_record.get("failure_modes") or [])
    ):
        return None
    base = _clone_program(program)
    if _is_dual_target(base.cluster.target) and int(base.parallel.pp_degree) < 4:
        promoted = _build_dual_node_pp4_candidate(base)
        if promoted is not None:
            base = promoted
    hot_subgraphs = {
        str(item.get("anchor")): str(item.get("label"))
        for item in (context_record.get("failure_modes") or [])
        if str(item.get("anchor") or "").startswith("subg_stage_")
    }
    candidate = _apply_torchtitan_hybrid_shard_policy(base, selective_only=True, hot_subgraphs=hot_subgraphs)
    candidate.metadata["program_kind"] = "candidate_pp_hsdp_hybrid"
    candidate.metadata["priority_rank"] = 19
    return _sync_batch_plan_metadata(candidate)


def _build_torchtitan_zero_bubble_schedule_candidate(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if _execution_backend_family(program, context_record) != "torchtitan":
        return None
    if int(program.parallel.pp_degree) <= 1:
        return None
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    if bubble_ratio < 0.08:
        return None
    candidate = _clone_program(program)
    candidate.schedule.skeleton = "stage_aware_grouped"
    candidate.schedule.template = "torchtitan_zero_bubble"
    candidate.schedule.dispatch_order = "zero_bubble_greedy"
    candidate.schedule.microbatch_group_size_per_vp_stage = 4 if bubble_ratio >= 0.15 else 2
    candidate.strategy_ir.pipe.template = "torchtitan_zero_bubble"
    candidate.strategy_ir.pipe.microbatch_order = "zero_bubble_greedy"
    candidate.strategy_ir.pipe.warmup_policy = "max_forward_fill"
    candidate.strategy_ir.pipe.cooldown_policy = "drain_with_w"
    candidate.metadata["execution_backend"] = "torchtitan"
    candidate.metadata["program_kind"] = "candidate_torchtitan_zero_bubble_schedule"
    candidate.metadata["priority_rank"] = 17
    return _sync_batch_plan_metadata(candidate)


def _build_torchtitan_dualpipev_schedule_candidate(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if _execution_backend_family(program, context_record) != "torchtitan":
        return None
    if int(program.parallel.pp_degree) <= 1:
        return None
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    comm_exposure_ratio = float(((context_record.get("runtime_evidence") or {}).get("comm_exposure_ratio")) or 0.0)
    if bubble_ratio < 0.12 or comm_exposure_ratio >= 0.18:
        return None
    candidate = _clone_program(program)
    if int(candidate.parallel.vpp_degree) == 1:
        counts = _pp_vpp_layout_counts(int(candidate.parallel.pp_degree), int(candidate.model.num_layers), "interleaved_grouped_g2")
        if counts is None:
            return None
        candidate.parallel.vpp_degree = 2
        candidate.layout.vpp_degree = 2
        candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
    candidate.schedule.skeleton = "stage_aware_grouped"
    candidate.schedule.template = "torchtitan_dualpipev"
    candidate.schedule.dispatch_order = "dualpipe_overlap"
    candidate.schedule.microbatch_group_size_per_vp_stage = 2
    candidate.strategy_ir.pipe.template = "torchtitan_dualpipev"
    candidate.strategy_ir.pipe.microbatch_order = "dualpipe_overlap"
    candidate.strategy_ir.pipe.warmup_policy = "prefill_overlap"
    candidate.strategy_ir.pipe.cooldown_policy = "staggered_wgrad"
    candidate.metadata["execution_backend"] = "torchtitan"
    candidate.metadata["program_kind"] = "candidate_torchtitan_dualpipev_schedule"
    candidate.metadata["priority_rank"] = 21
    return _sync_batch_plan_metadata(candidate)


def _build_dual_node_pp8_scaleout_candidate(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if not _is_dual_target(program.cluster.target):
        return None
    target_pp = 8
    if int(program.cluster.world_size) < target_pp or int(program.parallel.pp_degree) >= target_pp:
        return None
    if int(program.model.num_layers) < target_pp:
        return None
    bubble_ratio = float(runtime_summary.get("bubble_ratio") or 0.0)
    if bubble_ratio < 0.04 and int(program.parallel.pp_degree) >= 4:
        return None
    candidate = _clone_program(program)
    product_without_tp = (
        target_pp
        * int(candidate.parallel.cp_degree)
        * int(candidate.parallel.ep_degree)
        * int(candidate.parallel.expert_tp_degree)
    )
    if product_without_tp <= 0 or int(candidate.cluster.world_size) % product_without_tp != 0:
        return None
    target_tp = int(candidate.cluster.world_size) // product_without_tp
    if target_tp <= 0 or target_tp > int(candidate.cluster.gpus_per_node):
        return None
    candidate = _apply_grouped_pp_layout(candidate, target_tp=target_tp, target_pp=target_pp)
    if candidate is None:
        return None
    candidate.metadata["program_kind"] = "candidate_dual_node_pp8_scaleout"
    candidate.metadata["bubble_ratio"] = round(float(bubble_ratio), 4)
    candidate.metadata["priority_rank"] = 22 if bubble_ratio >= 0.08 else 24
    return _sync_batch_plan_metadata(candidate)


def _build_dual_node_orientation_candidate(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if str(program.cluster.target) != "dual_g4_g5" or int(program.parallel.pp_degree) < 4:
        return None
    slow_idx, _, stage_windows = _dominant_stage_indices(runtime_summary, int(program.parallel.pp_degree))
    if slow_idx is None or not stage_windows:
        return None
    midpoint = max(len(stage_windows) // 2, 1)
    first_half = stage_windows[:midpoint]
    second_half = stage_windows[midpoint:]
    if not first_half or not second_half:
        return None
    first_avg = sum(first_half) / float(len(first_half))
    second_avg = sum(second_half) / float(len(second_half))
    if first_avg <= second_avg * 1.08 and slow_idx >= midpoint:
        return None
    candidate = _clone_program(program)
    stage_to_node = ["g5"] * midpoint + ["g4"] * max(int(candidate.parallel.pp_degree) - midpoint, 0)
    if stage_to_node == list(candidate.layout.stage_to_node):
        return None
    stage_layers = weighted_stage_layer_allocation(str(candidate.cluster.target), int(candidate.model.num_layers), stage_to_node)
    if sum(stage_layers) != int(candidate.model.num_layers):
        return None
    candidate.partition = candidate.partition.from_dict(_partition_from_stage_layers(stage_layers))
    candidate.layout.stage_to_node = list(stage_to_node)
    candidate.layout.pipeline_layout = _virtual_stage_layout(stage_layers)
    candidate.metadata["program_kind"] = "candidate_dual_node_orientation_flip"
    candidate.metadata["slow_stage_index"] = int(slow_idx)
    candidate.metadata["priority_rank"] = 16
    return _sync_batch_plan_metadata(candidate)


def _build_topology_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    if not _is_dual_target(program.cluster.target) or int(program.parallel.pp_degree) < 2:
        return None
    candidate = _clone_program(program)
    candidate.layout.stage_to_node = _grouped_stage_nodes(candidate, int(candidate.parallel.pp_degree))
    if list(candidate.layout.stage_to_node) == list(program.layout.stage_to_node):
        return None
    candidate.metadata["program_kind"] = "candidate_topology_layout"
    return _sync_batch_plan_metadata(candidate)


def _build_batch_plan_fill_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    candidate = _clone_program(program)
    current_micro = max(int(candidate.batch_plan.micro_batch_size), 1)
    current_global = max(int(candidate.batch_plan.global_batch_size), 1)
    current_grad_accum = max(int(candidate.batch_plan.grad_accum_steps or _resolved_grad_accum_steps(candidate)), 1)
    if current_micro > 1:
        return None
    candidate.batch_plan.grad_accum_steps = min(current_grad_accum * 2, current_grad_accum + 8)
    candidate.batch_plan.global_batch_size = int(candidate.batch_plan.micro_batch_size) * _data_parallel_size(candidate) * int(candidate.batch_plan.grad_accum_steps)
    if int(candidate.batch_plan.global_batch_size) <= current_global:
        return None
    candidate.metadata["program_kind"] = "candidate_batch_plan_fill"
    candidate.metadata["priority_rank"] = 15
    return _sync_batch_plan_metadata(candidate)


def _build_long_context_cp_candidate(program: MegatronProgram) -> Optional[MegatronProgram]:
    candidate = _clone_program(program)
    current_seq = max(int(candidate.metadata.get("seq_len", 1024) or 1024), 1)
    if current_seq < 2048 or int(candidate.parallel.cp_degree) > 1:
        return None
    product_without_cp = (
        int(candidate.parallel.tp_degree)
        * int(candidate.parallel.pp_degree)
        * int(candidate.parallel.ep_degree)
        * int(candidate.parallel.expert_tp_degree)
    )
    if int(candidate.cluster.world_size) % max(product_without_cp * 2, 1) != 0:
        return None
    candidate.parallel.cp_degree = 2
    candidate.plane_map.attention.cp_degree = 2
    if str(candidate.schedule.template or "") == "fixed_1f1b" and int(candidate.parallel.vpp_degree) > 1:
        candidate.schedule.template = "interleaved_grouped_g4"
        candidate.schedule.skeleton = "stage_aware_grouped"
        candidate.schedule.microbatch_group_size_per_vp_stage = 4
    candidate.metadata["program_kind"] = "candidate_long_context_cp_relief"
    candidate.metadata["priority_rank"] = 50
    return _sync_batch_plan_metadata(candidate)


def _build_stage_local_vpp_shape_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    stage_count = int(program.partition.num_stages)
    target_vector, chunk_shapes = _nonuniform_vpp_vector_from_evidence(
        context_record,
        stage_count=stage_count,
        fallback_vpp=int(program.parallel.vpp_degree),
    )
    if not target_vector:
        return None
    baseline_by_stage = _local_parallel_by_stage(program)
    baseline_vector = [
        int((baseline_by_stage.get(stage_id) or {}).vpp_degree if stage_id in baseline_by_stage else int(program.parallel.vpp_degree))
        for stage_id in range(stage_count)
    ]
    if list(target_vector) == list(baseline_vector) and not chunk_shapes:
        return None

    candidate = _clone_program(program)
    candidate.metadata["stage_local_vpp_vector"] = [int(value) for value in target_vector]
    candidate.metadata["preserve_stage_local_vpp"] = True
    if chunk_shapes:
        candidate.metadata["stage_local_chunk_shapes"] = {str(key): value for key, value in chunk_shapes.items()}

    local_by_name = {entry.subgraph: entry for entry in (candidate.strategy_ir.local_parallel or [])}
    for subgraph in (candidate.strategy_ir.apipe or []):
        entry = local_by_name.get(subgraph.name)
        if entry is None:
            continue
        stage_id = int(subgraph.stage_index)
        if stage_id < len(target_vector):
            entry.vpp_degree = max(int(target_vector[stage_id]), 1)

    if any(int(value) > 1 for value in target_vector):
        target_global_vpp = min(max(max(int(value) for value in target_vector), int(candidate.parallel.vpp_degree)), 2)
        candidate.parallel.vpp_degree = int(target_global_vpp)
        candidate.layout.vpp_degree = int(target_global_vpp)
        if int(target_global_vpp) > 1:
            counts = _pp_vpp_layout_counts(
                int(candidate.parallel.pp_degree),
                int(candidate.model.num_layers),
                "interleaved_grouped_g2",
            )
            if counts is None:
                total_virtual = int(candidate.parallel.pp_degree) * int(target_global_vpp)
                if total_virtual > 0 and int(candidate.model.num_layers) % total_virtual == 0:
                    counts = [int(candidate.model.num_layers) // total_virtual] * total_virtual
            if counts is not None:
                candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
            candidate.schedule.skeleton = "stage_aware_grouped"
            if str(candidate.schedule.template or "fixed_1f1b") == "fixed_1f1b":
                candidate.schedule.template = "interleaved_grouped_g2"
            candidate.schedule.dispatch_order = "stage_local_nonuniform_vpp"
            candidate.schedule.microbatch_group_size_per_vp_stage = max(
                int(candidate.schedule.microbatch_group_size_per_vp_stage or 1),
                2,
            )

    candidate.metadata["program_kind"] = "candidate_nonuniform_vpp_shape"
    candidate.metadata["priority_rank"] = 21
    return _sync_batch_plan_metadata(candidate)


def _build_stage_local_memory_policy_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    local_memory = dict(evidence.get("local_memory_search_space") or {})
    policies = list(local_memory.get("per_stage_policy") or [])
    runtime_policy = dict(local_memory.get("runtime_policy") or {})
    if not policies:
        return None
    policy_by_stage: Dict[int, Dict[str, Any]] = {}
    for item in policies:
        stage_id = _safe_int((item or {}).get("stage_id"))
        if stage_id is None:
            continue
        policy_by_stage[int(stage_id)] = dict(item or {})
    if not policy_by_stage:
        return None

    candidate = _clone_program(program)
    selected_policies: List[Dict[str, Any]] = []
    touched = False
    for subgraph in (candidate.strategy_ir.apipe or []):
        stage_id = int(subgraph.stage_index)
        policy = dict(policy_by_stage.get(stage_id) or {})
        if not policy:
            continue
        checkpoint_policy = str(policy.get("checkpoint_policy") or "keep").strip().lower()
        remat_policy = str(policy.get("remat_policy") or "off").strip().lower()
        prefetch_policy = str(policy.get("prefetch_policy") or "default").strip().lower()
        if checkpoint_policy == "keep" and remat_policy in {"off", "none"} and prefetch_policy == "default":
            continue
        touched = True
        selected_policies.append(
            {
                "stage_id": int(stage_id),
                "subgraph": str(subgraph.name),
                "checkpoint_policy": checkpoint_policy,
                "remat_policy": remat_policy,
                "prefetch_policy": prefetch_policy,
                "runtime_recompute_modules": list(policy.get("runtime_recompute_modules") or []),
                "runtime_offload_modules": list(policy.get("runtime_offload_modules") or []),
                "reason": str(policy.get("reason") or ""),
            }
        )
    runtime_recompute_modules = [
        str(item)
        for item in list(runtime_policy.get("recompute_modules") or [])
        if str(item).strip()
    ]
    runtime_offload_modules = [
        str(item)
        for item in list(runtime_policy.get("offload_modules") or [])
        if str(item).strip()
    ]
    if runtime_recompute_modules:
        candidate.metadata["runtime_recompute_granularity"] = str(
            runtime_policy.get("recompute_granularity") or "selective"
        )
        candidate.metadata["runtime_enable_recompute_activations"] = bool(
            runtime_policy.get("enable_recompute_activations", True)
        )
        candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
        candidate.metadata["schedule_warmup_checkpoint_policy"] = str(
            runtime_policy.get("warmup_checkpoint_policy") or "full"
        )
        candidate.metadata["schedule_steady_checkpoint_policy"] = str(
            runtime_policy.get("steady_checkpoint_policy") or "default"
        )
        candidate.metadata["schedule_warmup_combined_policy"] = str(
            runtime_policy.get("warmup_combined_policy") or "serial"
        )
        touched = True
    if runtime_offload_modules:
        candidate.metadata["runtime_enable_fine_grained_activation_offloading"] = bool(
            runtime_policy.get("fine_grained_activation_offloading", True)
        )
        candidate.metadata["runtime_offload_modules"] = runtime_offload_modules
        touched = True
    if not touched:
        return None
    candidate.metadata["stage_local_memory_policy"] = selected_policies
    candidate.metadata["runtime_memory_policy_mode"] = "module_level_activation_relief"
    candidate.metadata["runtime_memory_expected_effect"] = str(
        runtime_policy.get("expected_effect") or ""
    )
    candidate.metadata["runtime_memory_performance_hypothesis"] = str(
        runtime_policy.get("performance_hypothesis") or ""
    )
    candidate.metadata["program_kind"] = "candidate_stage_local_memory_policy"
    candidate.metadata["priority_rank"] = 42
    return _sync_batch_plan_metadata(candidate)


def _build_offload_first_refinement_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    failures = list((context_record or {}).get("failure_modes") or [])
    failure_labels = {str(item.get("label") or "") for item in failures}
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    optimizer_ratio = float(runtime.get("optimizer_ratio") or 0.0)
    tail_jitter_ratio = float(runtime.get("tail_step_jitter_ratio") or 0.0)
    if (
        "memory_hotspot" not in failure_labels
        and peak_reserved_ratio < 0.84
        and optimizer_ratio < 0.40
    ):
        return None

    candidate = _clone_program(program)
    stage_ids: List[int] = []
    if candidate.strategy_ir.apipe:
        stage_ids.append(int(candidate.strategy_ir.apipe[0].stage_index))
        if len(candidate.strategy_ir.apipe) > 1:
            stage_ids.append(int(candidate.strategy_ir.apipe[-1].stage_index))
    stage_ids = sorted(set(stage_ids))
    if not stage_ids:
        stage_ids = [0]

    local_by_subgraph = {str(item.subgraph): item for item in (candidate.strategy_ir.local_parallel or [])}
    selected_policies: List[Dict[str, Any]] = []
    runtime_offload_modules: List[str] = []
    runtime_recompute_modules: List[str] = []
    touched = False
    for subgraph in (candidate.strategy_ir.apipe or []):
        stage_id = int(subgraph.stage_index)
        if stage_id not in stage_ids:
            continue
        local = local_by_subgraph.get(str(subgraph.name))
        if local is None:
            continue
        local.fsdp_scope = "selective"
        local.offload_policy = "selective_overlap_aware"
        local.reshard_policy = "intra_node"
        if int(local.vpp_degree) > 1:
            local.vpp_degree = 1
        touched = True
        stage_offload_modules = ["core_attn", "attn_proj"]
        if stage_id == stage_ids[-1] or tail_jitter_ratio >= 0.20:
            stage_offload_modules.append("mlp")
        stage_recompute_modules = ["core_attn"]
        if peak_reserved_ratio >= 0.88:
            stage_recompute_modules.append("mlp")
        for module in stage_offload_modules:
            if module not in runtime_offload_modules:
                runtime_offload_modules.append(module)
        for module in stage_recompute_modules:
            if module not in runtime_recompute_modules:
                runtime_recompute_modules.append(module)
        selected_policies.append(
            {
                "stage_id": stage_id,
                "subgraph": str(subgraph.name),
                "checkpoint_policy": "selective",
                "remat_policy": "selective",
                "prefetch_policy": "guarded",
                "runtime_recompute_modules": list(stage_recompute_modules),
                "runtime_offload_modules": list(stage_offload_modules),
                "reason": "offload-first refinement protects hotspot and tail-sensitive stages before broader PP/VPP rewrites",
            }
        )

    if not touched:
        return None

    if int(candidate.parallel.vpp_degree) > 1 and peak_reserved_ratio >= 0.86:
        candidate.parallel.vpp_degree = 1
        candidate.layout.vpp_degree = 1
        candidate.layout.pipeline_layout = None
        candidate.schedule.template = "fixed_1f1b"
        candidate.schedule.skeleton = "fixed_1f1b"
        candidate.schedule.microbatch_group_size_per_vp_stage = None

    candidate.metadata["stage_local_memory_policy"] = selected_policies
    candidate.metadata["runtime_enable_fine_grained_activation_offloading"] = True
    candidate.metadata["runtime_offload_modules"] = runtime_offload_modules
    candidate.metadata["runtime_recompute_granularity"] = "selective"
    candidate.metadata["runtime_enable_recompute_activations"] = True
    candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
    candidate.metadata["schedule_warmup_checkpoint_policy"] = "full"
    candidate.metadata["schedule_steady_checkpoint_policy"] = "selective"
    candidate.metadata["schedule_warmup_combined_policy"] = "serial"
    candidate.metadata["schedule_steady_combined_policy"] = "serial"
    candidate.metadata["schedule_cooldown_p2p_policy"] = "serial"
    candidate.metadata["schedule_cooldown_combined_policy"] = "serial"
    candidate.metadata["runtime_memory_policy_mode"] = "offload_first_hotspot_relief"
    candidate.metadata["runtime_memory_expected_effect"] = (
        "reduce edge-stage activation pressure and optimizer exposure before global PP/VPP reshaping"
    )
    candidate.metadata["runtime_checkpoint_boundary_mode"] = "edge_guarded_selective"
    candidate.metadata["program_kind"] = "candidate_offload_first_refinement"
    candidate.metadata["priority_rank"] = 14
    return _sync_batch_plan_metadata(candidate)


def _build_optimizer_aware_pipeline_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    optimizer_exposed_ratio = float(runtime.get("optimizer_exposed_ratio") or 0.0)
    optimizer_ratio = float(runtime.get("optimizer_ratio") or 0.0)
    tail_ratio = max(
        float(runtime.get("stage_tail_ratio") or 0.0),
        float(runtime.get("tail_step_jitter_ratio") or 0.0),
    )
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    if optimizer_exposed_ratio < 0.18 and optimizer_ratio < 0.45:
        return None

    candidate = _clone_program(program)
    candidate.schedule.dispatch_order = "optimizer_tail_guarded"
    candidate.strategy_ir.pipe.warmup_policy = "balanced_fill"
    candidate.strategy_ir.pipe.cooldown_policy = "optimizer_tail_hide"
    candidate.metadata["flush_order_policy"] = "optimizer_tail_hide"
    candidate.metadata["schedule_cooldown_p2p_policy"] = "serial"
    candidate.metadata["schedule_cooldown_combined_policy"] = "serial"
    if optimizer_exposed_ratio >= 0.24:
        candidate.metadata["schedule_steady_combined_policy"] = "serial"

    runtime_recompute_modules: List[str] = []
    if peak_reserved_ratio >= 0.84 or tail_ratio >= 0.10:
        runtime_recompute_modules.append("core_attn")
        if peak_reserved_ratio >= 0.88:
            runtime_recompute_modules.append("mlp")
        candidate.metadata["schedule_warmup_checkpoint_policy"] = "full"
        candidate.metadata["schedule_steady_checkpoint_policy"] = "guarded_selective"
        candidate.metadata["runtime_enable_recompute_activations"] = True
        candidate.metadata["runtime_recompute_granularity"] = "selective"
        candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
        candidate.metadata["runtime_checkpoint_boundary_mode"] = "optimizer_tail_guarded"

    hint_map = _stage_family_hint_map(candidate)
    tail_stage = max(hint_map) if hint_map else 0
    tail_hint = hint_map.setdefault(tail_stage, {"stage_index": tail_stage, "family": "balanced_interleave"})
    tail_hint.update(
        {
            "family": "optimizer_guarded_tail",
            "dispatch_order": "optimizer_tail_guarded",
            "warmup_policy": "balanced_fill",
            "cooldown_policy": "optimizer_tail_hide",
            "checkpoint_policy": "guarded_selective" if runtime_recompute_modules else "tail_selective",
            "p2p_policy": "serial" if optimizer_exposed_ratio >= 0.22 else "",
            "combined_policy": "serial",
            "chunk_priority_hints": [4, 2] if int(candidate.parallel.vpp_degree) > 1 else [4],
        }
    )
    candidate.metadata["morphable_stage_families"] = _sorted_stage_family_hints(hint_map)
    candidate.metadata["runtime_optimizer_policy_mode"] = "tail_hidden_overlap"
    candidate.metadata["runtime_optimizer_expected_effect"] = (
        "reduce optimizer exposure on the tail-stage critical path before larger PP/VPP repartitioning"
    )
    candidate.metadata["runtime_optimizer_target_exposed_ratio"] = round(
        max(optimizer_exposed_ratio - 0.08, optimizer_exposed_ratio * 0.70),
        4,
    )
    candidate.metadata["program_kind"] = "candidate_optimizer_aware_pipeline"
    candidate.metadata["priority_rank"] = 12
    return _sync_batch_plan_metadata(candidate)


def _build_tail_aware_execution_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    failure_labels = {
        str(item.get("label") or "")
        for item in list((context_record or {}).get("failure_modes") or [])
    }
    derived_labels = {
        str(item.get("label") or "")
        for item in list((context_record or {}).get("derived_bottlenecks") or [])
    }
    bubble_ratio = float(runtime.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime.get("optimizer_exposed_ratio") or 0.0)
    tail_ratio = max(
        float(runtime.get("stage_tail_ratio") or 0.0),
        float(runtime.get("tail_step_jitter_ratio") or 0.0),
    )
    if "tail_heavy" not in failure_labels and "tail_heavy" not in derived_labels and tail_ratio < 0.12:
        return None

    candidate = _clone_program(program)
    stage_count = int(candidate.partition.num_stages)
    local_by_stage = _local_parallel_by_stage(candidate)
    base_vector = [
        int(((local_by_stage.get(stage_id)).vpp_degree if stage_id in local_by_stage else int(candidate.parallel.vpp_degree)))
        for stage_id in range(stage_count)
    ]
    target_vector = list(base_vector)
    if stage_count >= 1:
        target_vector[-1] = 1
    if stage_count >= 2 and bubble_ratio >= 0.10 and peak_reserved_ratio < 0.86:
        middle_stage_ids = list(range(1, max(stage_count - 1, 1)))
        if not middle_stage_ids:
            middle_stage_ids = [0]
        for stage_id in middle_stage_ids:
            target_vector[stage_id] = max(int(target_vector[stage_id]), 2)

    if target_vector != base_vector and any(int(value) > 1 for value in target_vector):
        candidate.metadata["stage_local_vpp_vector"] = [int(value) for value in target_vector]
        candidate.metadata["preserve_stage_local_vpp"] = True
        target_global_vpp = min(max(max(int(value) for value in target_vector), int(candidate.parallel.vpp_degree)), 2)
        candidate.parallel.vpp_degree = int(target_global_vpp)
        candidate.layout.vpp_degree = int(target_global_vpp)
        if int(target_global_vpp) > 1:
            stage_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
            counts = _stage_local_virtual_counts(
                stage_layers,
                target_vector,
                global_vpp=int(target_global_vpp),
                focus="tail-aware",
            )
            candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
            candidate.schedule.skeleton = "stage_aware_grouped"
            if str(candidate.schedule.template or "fixed_1f1b") == "fixed_1f1b":
                candidate.schedule.template = "interleaved_grouped_g2"
            candidate.schedule.microbatch_group_size_per_vp_stage = max(
                int(candidate.schedule.microbatch_group_size_per_vp_stage or 1),
                2,
            )
    candidate.schedule.dispatch_order = "tail_boundary_rewrite"
    candidate.strategy_ir.pipe.warmup_policy = "balanced_fill"
    candidate.strategy_ir.pipe.cooldown_policy = "tail_checkpoint_guard"
    candidate.metadata["flush_order_policy"] = "tail_checkpoint_guard"

    runtime_recompute_modules = ["core_attn"]
    if peak_reserved_ratio >= 0.88:
        runtime_recompute_modules.append("mlp")
    candidate.metadata["runtime_enable_recompute_activations"] = True
    candidate.metadata["runtime_recompute_granularity"] = "selective"
    candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
    candidate.metadata["schedule_warmup_checkpoint_policy"] = "full"
    candidate.metadata["schedule_steady_checkpoint_policy"] = "tail_selective"
    if peak_reserved_ratio >= 0.84 or optimizer_exposed_ratio >= 0.18:
        candidate.metadata["schedule_cooldown_combined_policy"] = "serial"
    candidate.metadata["runtime_checkpoint_boundary_mode"] = "tail_stage_guarded"
    candidate.metadata["runtime_memory_policy_mode"] = "tail_guarded_selective_recompute"

    hint_map = _stage_family_hint_map(candidate)
    tail_stage = max(hint_map) if hint_map else 0
    tail_hint = hint_map.setdefault(tail_stage, {"stage_index": tail_stage, "family": "balanced_interleave"})
    tail_hint.update(
        {
            "family": "tail_guarded",
            "dispatch_order": "tail_boundary_rewrite",
            "warmup_policy": "balanced_fill",
            "cooldown_policy": "tail_checkpoint_guard",
            "checkpoint_policy": "tail_selective",
            "combined_policy": "serial" if peak_reserved_ratio >= 0.84 or optimizer_exposed_ratio >= 0.18 else "",
            "chunk_priority_hints": [1],
        }
    )
    for stage_id in range(1, max(stage_count - 1, 1)):
        if stage_id >= tail_stage:
            continue
        if stage_id < len(target_vector) and int(target_vector[stage_id]) > 1:
            hint_map.setdefault(stage_id, {"stage_index": stage_id, "family": "balanced_interleave"}).update(
                {
                    "family": "heterogeneous_middle_relief",
                    "dispatch_order": "middle_stage_relief",
                    "warmup_policy": "balanced_fill",
                    "cooldown_policy": "tail_min",
                    "chunk_priority_hints": [3, 1],
                }
            )
    candidate.metadata["morphable_stage_families"] = _sorted_stage_family_hints(hint_map)
    candidate.metadata["program_kind"] = "candidate_tail_aware_execution"
    candidate.metadata["priority_rank"] = 11
    return _sync_batch_plan_metadata(candidate)


def _build_checkpoint_boundary_refinement_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    failure_labels = {
        str(item.get("label") or "")
        for item in list((context_record or {}).get("failure_modes") or [])
    }
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    tail_ratio = max(
        float(runtime.get("stage_tail_ratio") or 0.0),
        float(runtime.get("tail_step_jitter_ratio") or 0.0),
    )
    if "memory_hotspot" not in failure_labels and peak_reserved_ratio < 0.84 and tail_ratio < 0.10:
        return None

    candidate = _clone_program(program)
    stage_windows = {
        int(stage_id): dict(values or {})
        for stage_id, values in dict(runtime.get("stage_window_summary") or {}).items()
        if _safe_int(stage_id) is not None
    }
    hot_stage_ids = sorted(
        stage_windows,
        key=lambda stage_id: float((stage_windows.get(stage_id) or {}).get("peak_reserved_gib") or 0.0),
        reverse=True,
    )[:2]
    if candidate.strategy_ir.apipe:
        tail_stage = int(candidate.strategy_ir.apipe[-1].stage_index)
        if tail_stage not in hot_stage_ids:
            hot_stage_ids.append(tail_stage)
    hot_stage_ids = sorted(set(int(stage_id) for stage_id in hot_stage_ids))
    if not hot_stage_ids:
        hot_stage_ids = [0]

    selected_policies: List[Dict[str, Any]] = []
    runtime_recompute_modules: List[str] = []
    runtime_offload_modules: List[str] = []
    hint_map = _stage_family_hint_map(candidate)
    for stage_id in hot_stage_ids:
        local_ratio = 0.0
        stage_window = dict(stage_windows.get(int(stage_id)) or {})
        peak_reserved_gib = float(stage_window.get("peak_reserved_gib") or 0.0)
        if float(candidate.constraints.memory_budget_gb or candidate.cluster.device_memory_gb or 0.0) > 0.0:
            local_ratio = peak_reserved_gib / float(candidate.constraints.memory_budget_gb or candidate.cluster.device_memory_gb or 1.0)
        recompute_modules = ["core_attn"]
        if peak_reserved_ratio >= 0.88 or local_ratio >= 0.88:
            recompute_modules.append("mlp")
        offload_modules = ["core_attn"] if peak_reserved_ratio >= 0.90 or local_ratio >= 0.90 else []
        for module in recompute_modules:
            if module not in runtime_recompute_modules:
                runtime_recompute_modules.append(module)
        for module in offload_modules:
            if module not in runtime_offload_modules:
                runtime_offload_modules.append(module)
        selected_policies.append(
            {
                "stage_id": int(stage_id),
                "checkpoint_policy": "guarded_selective",
                "remat_policy": "selective",
                "prefetch_policy": "guarded",
                "runtime_recompute_modules": list(recompute_modules),
                "runtime_offload_modules": list(offload_modules),
                "reason": "checkpoint boundaries should be tightened around hotspot and tail-sensitive stages before widening VPP",
            }
        )
        hint_map.setdefault(int(stage_id), {"stage_index": int(stage_id), "family": "balanced_interleave"}).update(
            {
                "family": "checkpoint_guarded",
                "dispatch_order": "tail_boundary_rewrite" if int(stage_id) == max(hot_stage_ids) else "middle_stage_relief",
                "warmup_policy": "balanced_fill",
                "cooldown_policy": "tail_checkpoint_guard" if int(stage_id) == max(hot_stage_ids) else "tail_min",
                "checkpoint_policy": "guarded_selective",
                "combined_policy": "serial",
            }
        )

    candidate.metadata["stage_local_memory_policy"] = selected_policies
    candidate.metadata["runtime_enable_recompute_activations"] = True
    candidate.metadata["runtime_recompute_granularity"] = "selective"
    candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
    if runtime_offload_modules:
        candidate.metadata["runtime_enable_fine_grained_activation_offloading"] = True
        candidate.metadata["runtime_offload_modules"] = runtime_offload_modules
    candidate.metadata["schedule_warmup_checkpoint_policy"] = "full"
    candidate.metadata["schedule_steady_checkpoint_policy"] = "guarded_selective"
    candidate.metadata["schedule_warmup_combined_policy"] = "serial"
    candidate.metadata["schedule_steady_combined_policy"] = "serial"
    candidate.metadata["runtime_memory_policy_mode"] = "checkpoint_boundary_joint_refinement"
    candidate.metadata["runtime_checkpoint_boundary_mode"] = "hotspot_tail_staggered"
    candidate.metadata["morphable_stage_families"] = _sorted_stage_family_hints(hint_map)
    candidate.metadata["program_kind"] = "candidate_checkpoint_boundary_refinement"
    candidate.metadata["priority_rank"] = 13
    return _sync_batch_plan_metadata(candidate)


def _build_morphable_pipeline_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    plan = dict(evidence.get("morphable_pipeline_plan") or {})
    families = list(plan.get("stage_families") or [])
    runtime_memory_policy = dict(plan.get("runtime_memory_policy") or {})
    objective = dict(plan.get("objective") or {})
    chunk_shape_vector = [max(int(item), 1) for item in list(plan.get("chunk_shape_vector") or []) if _safe_int(item) is not None]
    regroup_actions = list(plan.get("regroup_actions") or [])
    if not families and not chunk_shape_vector and not regroup_actions:
        return None

    candidate = _clone_program(program)
    stage_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
    if stage_layers and regroup_actions:
        for action in regroup_actions:
            left_stage = _safe_int((action or {}).get("left_stage_id"))
            right_stage = _safe_int((action or {}).get("right_stage_id"))
            shift_blocks = max(_safe_int((action or {}).get("shift_blocks")) or 1, 1)
            direction = str((action or {}).get("direction") or "left_to_right")
            if left_stage is None or right_stage is None:
                continue
            if left_stage < 0 or right_stage < 0 or left_stage >= len(stage_layers) or right_stage >= len(stage_layers):
                continue
            if direction == "left_to_right":
                movable = min(max(stage_layers[left_stage] - 1, 0), shift_blocks)
                if movable > 0:
                    stage_layers[left_stage] -= movable
                    stage_layers[right_stage] += movable
            else:
                movable = min(max(stage_layers[right_stage] - 1, 0), shift_blocks)
                if movable > 0:
                    stage_layers[right_stage] -= movable
                    stage_layers[left_stage] += movable
        candidate.partition = candidate.partition.from_dict(_partition_from_stage_layers(stage_layers))
        candidate.layout.pipeline_layout = None

    stage_count = int(candidate.partition.num_stages)
    if chunk_shape_vector:
        if len(chunk_shape_vector) < stage_count:
            fill = max(int(candidate.parallel.vpp_degree), 1)
            chunk_shape_vector = chunk_shape_vector + [fill] * (stage_count - len(chunk_shape_vector))
        else:
            chunk_shape_vector = chunk_shape_vector[:stage_count]
    else:
        chunk_shape_vector = [max(int(candidate.parallel.vpp_degree), 1) for _ in range(stage_count)]
    candidate.metadata["stage_local_vpp_vector"] = [int(value) for value in chunk_shape_vector]
    candidate.metadata["preserve_stage_local_vpp"] = True
    candidate.metadata["morphable_chunk_shape_vector"] = [int(value) for value in chunk_shape_vector]
    candidate.metadata["morphable_regroup_actions"] = list(regroup_actions)
    candidate.metadata["morphable_shape_signature"] = str(plan.get("shape_signature") or "")
    if objective:
        candidate.metadata["morphable_objective_type"] = str(objective.get("type") or "")
        candidate.metadata["morphable_estimated_step_time_ms"] = float(objective.get("estimated_step_time_ms") or 0.0)
        candidate.metadata["morphable_estimated_step_delta_ms"] = float(objective.get("estimated_step_delta_ms") or 0.0)
    target_global_vpp = max([max(int(candidate.parallel.vpp_degree), 1)] + [int(value) for value in chunk_shape_vector])
    candidate.parallel.vpp_degree = int(target_global_vpp)
    candidate.layout.vpp_degree = int(target_global_vpp)
    if int(target_global_vpp) > 1:
        focus = "balanced"
        dominant_family = str(plan.get("dominant_family") or "")
        if dominant_family == "critical_path_first":
            focus = "left_heavy"
        elif dominant_family == "memory_guarded":
            focus = "balanced"
        stage_virtual_counts = _stage_local_virtual_counts(
            stage_layers,
            [int(value) for value in chunk_shape_vector],
            global_vpp=int(target_global_vpp),
            focus=focus,
        )
        if stage_virtual_counts and sum(stage_virtual_counts) == int(candidate.model.num_layers):
            candidate.layout.pipeline_layout = _virtual_stage_layout(stage_virtual_counts)
        candidate.schedule.skeleton = "stage_aware_grouped"
        if str(candidate.schedule.template or "fixed_1f1b") == "fixed_1f1b":
            candidate.schedule.template = "interleaved_grouped_g2" if int(target_global_vpp) >= 2 else "fixed_1f1b"
        candidate.schedule.microbatch_group_size_per_vp_stage = max(
            int(candidate.schedule.microbatch_group_size_per_vp_stage or 1),
            4 if int(target_global_vpp) >= 2 else 1,
        )

    family_hints: List[Dict[str, Any]] = []
    runtime_recompute_modules: List[str] = []
    runtime_offload_modules: List[str] = []
    for family in families:
        stage_index = _safe_int((family or {}).get("stage_index"))
        if stage_index is None:
            continue
        hint = {
            "stage_index": int(stage_index),
            "family": str((family or {}).get("family") or "balanced_interleave"),
            "preferred_template": str((family or {}).get("preferred_template") or "").strip(),
            "dispatch_order": str((family or {}).get("dispatch_order") or "default"),
            "warmup_policy": str((family or {}).get("warmup_policy") or "default"),
            "cooldown_policy": str((family or {}).get("cooldown_policy") or "default"),
            "checkpoint_policy": str((family or {}).get("checkpoint_policy") or "").strip(),
            "p2p_policy": str((family or {}).get("p2p_policy") or "").strip(),
            "combined_policy": str((family or {}).get("combined_policy") or "").strip(),
            "chunk_priority_hints": [int(item) for item in list((family or {}).get("chunk_priority_hints") or []) if _safe_int(item) is not None],
        }
        family_hints.append(hint)
        for module in list((family or {}).get("recompute_modules") or []):
            token = str(module).strip()
            if token and token not in runtime_recompute_modules:
                runtime_recompute_modules.append(token)
        for module in list((family or {}).get("offload_modules") or []):
            token = str(module).strip()
            if token and token not in runtime_offload_modules:
                runtime_offload_modules.append(token)

    if family_hints:
        candidate.metadata["morphable_stage_families"] = family_hints
        family_by_stage = {int(item["stage_index"]): item for item in family_hints}
        ordered_families = [family_by_stage[index] for index in sorted(family_by_stage)]
        dominant = ordered_families[0]
        candidate.schedule.dispatch_order = str(dominant.get("dispatch_order") or candidate.schedule.dispatch_order or "default")
        candidate.strategy_ir.pipe.warmup_policy = str(dominant.get("warmup_policy") or candidate.strategy_ir.pipe.warmup_policy or "default")
        candidate.strategy_ir.pipe.cooldown_policy = str(dominant.get("cooldown_policy") or candidate.strategy_ir.pipe.cooldown_policy or "default")
        if any(str(item.get("dispatch_order") or "") == "structure_aware_critical_first" for item in ordered_families):
            candidate.schedule.dispatch_order = "structure_aware_critical_first"
        if runtime_recompute_modules:
            candidate.metadata["runtime_recompute_granularity"] = "selective"
            candidate.metadata["runtime_enable_recompute_activations"] = True
            candidate.metadata["runtime_recompute_modules"] = runtime_recompute_modules
        if runtime_offload_modules:
            candidate.metadata["runtime_enable_fine_grained_activation_offloading"] = True
            candidate.metadata["runtime_offload_modules"] = runtime_offload_modules
        if any(str(item.get("checkpoint_policy") or "") for item in ordered_families):
            candidate.metadata["schedule_warmup_checkpoint_policy"] = str(
                next((item["checkpoint_policy"] for item in ordered_families if item.get("checkpoint_policy")), "selective")
            )
        if any(str(item.get("p2p_policy") or "") for item in ordered_families):
            candidate.metadata["schedule_cooldown_p2p_policy"] = str(
                next((item["p2p_policy"] for item in ordered_families if item.get("p2p_policy")), "serial")
            )
        if any(str(item.get("combined_policy") or "") for item in ordered_families):
            policy = str(next((item["combined_policy"] for item in ordered_families if item.get("combined_policy")), "serial"))
            candidate.metadata["schedule_warmup_combined_policy"] = policy
            candidate.metadata["schedule_steady_combined_policy"] = policy
            candidate.metadata["schedule_cooldown_combined_policy"] = policy
    if runtime_memory_policy:
        recompute_modules = [
            str(item).strip()
            for item in list(runtime_memory_policy.get("recompute_modules") or [])
            if str(item).strip()
        ]
        offload_modules = [
            str(item).strip()
            for item in list(runtime_memory_policy.get("offload_modules") or [])
            if str(item).strip()
        ]
        if recompute_modules:
            candidate.metadata["runtime_recompute_granularity"] = str(
                runtime_memory_policy.get("recompute_granularity") or "selective"
            )
            candidate.metadata["runtime_enable_recompute_activations"] = bool(
                runtime_memory_policy.get("enable_recompute_activations", True)
            )
            candidate.metadata["runtime_recompute_modules"] = recompute_modules
            candidate.metadata["schedule_warmup_checkpoint_policy"] = str(
                runtime_memory_policy.get("warmup_checkpoint_policy") or "full"
            )
            candidate.metadata["schedule_steady_checkpoint_policy"] = str(
                runtime_memory_policy.get("steady_checkpoint_policy") or "default"
            )
            candidate.metadata["schedule_warmup_combined_policy"] = str(
                runtime_memory_policy.get("warmup_combined_policy") or "serial"
            )
        if offload_modules:
            candidate.metadata["runtime_enable_fine_grained_activation_offloading"] = bool(
                runtime_memory_policy.get("fine_grained_activation_offloading", True)
            )
            candidate.metadata["runtime_offload_modules"] = offload_modules
        if str(runtime_memory_policy.get("cooldown_p2p_policy") or "").strip():
            candidate.metadata["schedule_cooldown_p2p_policy"] = str(
                runtime_memory_policy.get("cooldown_p2p_policy") or "serial"
            )
        if str(runtime_memory_policy.get("cooldown_combined_policy") or "").strip():
            candidate.metadata["schedule_cooldown_combined_policy"] = str(
                runtime_memory_policy.get("cooldown_combined_policy") or "serial"
            )
        if str(runtime_memory_policy.get("steady_combined_policy") or "").strip():
            candidate.metadata["schedule_steady_combined_policy"] = str(
                runtime_memory_policy.get("steady_combined_policy") or "combined"
            )
        candidate.metadata["runtime_memory_policy_mode"] = str(
            runtime_memory_policy.get("policy_mode")
            or runtime_memory_policy.get("offload_policy")
            or "budgeted_joint_runtime_policy"
        )
        candidate.metadata["runtime_memory_expected_effect"] = str(
            runtime_memory_policy.get("expected_effect") or ""
        )

    candidate.metadata["program_kind"] = "candidate_morphable_pipeline"
    candidate.metadata["priority_rank"] = 16
    return _sync_batch_plan_metadata(candidate)


def _build_boundary_semantic_schedule_candidates(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return []
    boundaries = _boundary_semantic_entries(context_record)
    if not boundaries:
        return []
    candidates: List[MegatronProgram] = []
    seen_semantics: set[str] = set()
    for boundary in boundaries:
        semantic = str(boundary.get("semantic") or "normal").strip().lower()
        if semantic in {"", "normal"} or semantic in seen_semantics:
            continue
        seen_semantics.add(semantic)
        candidate = _clone_program(program)
        candidate.metadata["boundary_semantic_focus"] = semantic
        candidate.metadata["boundary_semantic_boundary"] = str(boundary.get("boundary_id") or "")
        candidate.metadata["boundary_semantic_actions"] = list(boundary.get("actions") or [])
        if semantic in {"comm-aware", "tail-aware"}:
            if str(candidate.schedule.template or "fixed_1f1b") == "fixed_1f1b":
                candidate.schedule.template = "interleaved_grouped_g2"
            candidate.schedule.skeleton = "stage_aware_grouped"
            candidate.schedule.microbatch_group_size_per_vp_stage = max(
                int(candidate.schedule.microbatch_group_size_per_vp_stage or 1),
                2,
            )
        if semantic == "comm-aware":
            candidate.schedule.dispatch_order = "boundary_comm_aware"
            candidate.strategy_ir.pipe.warmup_policy = "comm_shy"
            candidate.strategy_ir.pipe.cooldown_policy = "late_wait"
            candidate.metadata["program_kind"] = "candidate_boundary_semantic_comm"
            candidate.metadata["priority_rank"] = 23
        elif semantic == "tail-aware":
            if int(candidate.parallel.vpp_degree) == 1:
                counts = _pp_vpp_layout_counts(
                    int(candidate.parallel.pp_degree),
                    int(candidate.model.num_layers),
                    "interleaved_grouped_g2",
                )
                if counts is not None:
                    candidate.parallel.vpp_degree = 2
                    candidate.layout.vpp_degree = 2
                    candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
            candidate.schedule.dispatch_order = "tail_boundary_rewrite"
            candidate.strategy_ir.pipe.warmup_policy = "tail_prefill"
            candidate.strategy_ir.pipe.cooldown_policy = "tail_drain"
            candidate.metadata["flush_order_policy"] = "reverse_last_group"
            candidate.metadata["schedule_cooldown_p2p_policy"] = "serial"
            candidate.metadata["schedule_cooldown_combined_policy"] = "serial"
            candidate.metadata["program_kind"] = "candidate_boundary_semantic_tail"
            candidate.metadata["priority_rank"] = 24
        elif semantic == "memory-aware":
            left_stage = int(boundary.get("left_stage") or 0)
            right_stage = int(boundary.get("right_stage") or 0)
            local_by_name = {entry.subgraph: entry for entry in (candidate.strategy_ir.local_parallel or [])}
            for subgraph in (candidate.strategy_ir.apipe or []):
                if int(subgraph.stage_index) not in {left_stage, right_stage}:
                    continue
                entry = local_by_name.get(subgraph.name)
                if entry is None:
                    continue
                entry.cp_degree = max(int(entry.cp_degree), 2)
                entry.fsdp_scope = "selective"
                entry.vpp_degree = 1
            if int(candidate.parallel.vpp_degree) > 1:
                candidate.parallel.vpp_degree = 1
                candidate.layout.vpp_degree = 1
                candidate.layout.pipeline_layout = None
            candidate.schedule.template = "fixed_1f1b"
            candidate.schedule.skeleton = "fixed_1f1b"
            candidate.schedule.dispatch_order = "memory_boundary_guard"
            candidate.schedule.microbatch_group_size_per_vp_stage = None
            candidate.strategy_ir.pipe.warmup_policy = "default"
            candidate.strategy_ir.pipe.cooldown_policy = "default"
            candidate.metadata["program_kind"] = "candidate_boundary_semantic_memory"
            candidate.metadata["priority_rank"] = 21
        else:
            continue
        candidates.append(_sync_batch_plan_metadata(candidate))
    return candidates


def _build_pipe_search_space_candidates(
    program: MegatronProgram,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return []
    evidence = dict((context_record or {}).get("evidence_record") or {})
    pipe_space = dict(evidence.get("pipe_search_space") or {})
    variants = list(pipe_space.get("variants") or [])
    if not variants:
        return []
    backend_family = _execution_backend_family(program, context_record)
    candidates: List[MegatronProgram] = []
    for variant in variants:
        name = str((variant or {}).get("name") or "").strip().lower()
        status = str((variant or {}).get("status") or "").strip().lower()
        if not name or status not in {"executable_now", "sandbox_now"}:
            continue
        if name in {"fixed_1f1b", "stage_aware_grouped"}:
            continue
        candidate: Optional[MegatronProgram] = None
        if name == "fixed_1f1b":
            if str(program.schedule.template or "fixed_1f1b") != "fixed_1f1b":
                candidate = _set_schedule_template(
                    program,
                    template="fixed_1f1b",
                    group_size=None,
                    dispatch_order="default",
                    skeleton="fixed_1f1b",
                )
        elif name == "stage_aware_grouped":
            candidate = _build_stage_aware_schedule(program)
        elif name == "zero_bubble_family":
            if backend_family == "torchtitan":
                candidate = _build_torchtitan_zero_bubble_schedule_candidate(program, runtime_summary, context_record)
            if candidate is None:
                candidate = _clone_program(program)
                if int(candidate.parallel.vpp_degree) == 1:
                    counts = _pp_vpp_layout_counts(
                        int(candidate.parallel.pp_degree),
                        int(candidate.model.num_layers),
                        "interleaved_grouped_g4",
                    )
                    if counts is not None:
                        candidate.parallel.vpp_degree = 2
                        candidate.layout.vpp_degree = 2
                        candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
                candidate.schedule.skeleton = "stage_aware_grouped"
                candidate.schedule.template = "interleaved_grouped_g4"
                candidate.schedule.dispatch_order = "zero_bubble_proxy"
                candidate.schedule.microbatch_group_size_per_vp_stage = 4
                candidate = _sync_batch_plan_metadata(candidate)
        elif name == "comm_aware_boundary_schedule":
            candidate = _clone_program(program)
            candidate.schedule.skeleton = "stage_aware_grouped"
            candidate.schedule.template = "interleaved_grouped_g2"
            candidate.schedule.dispatch_order = "boundary_localized"
            candidate.schedule.microbatch_group_size_per_vp_stage = max(
                int(candidate.schedule.microbatch_group_size_per_vp_stage or 1),
                2,
            )
            candidate = _sync_batch_plan_metadata(candidate)
        if candidate is None:
            continue
        tagged = _clone_program(candidate)
        tagged.strategy_ir.pipe.warmup_policy = str((variant or {}).get("warmup") or tagged.strategy_ir.pipe.warmup_policy or "default")
        tagged.strategy_ir.pipe.cooldown_policy = str((variant or {}).get("cooldown") or tagged.strategy_ir.pipe.cooldown_policy or "default")
        flush_order = str((variant or {}).get("flush_order") or "").strip()
        if flush_order:
            tagged.metadata["flush_order_policy"] = flush_order
        tagged.metadata["pipe_variant_name"] = name
        tagged.metadata["pipe_variant_status"] = status
        tagged.metadata["pipe_variant_issue_wait"] = str((variant or {}).get("issue_wait") or "")
        tagged.metadata["program_kind"] = f"candidate_pipe_variant_{name}"
        tagged.metadata["priority_rank"] = 46
        candidates.append(_sync_batch_plan_metadata(tagged))
    return candidates


def _build_apipe_pipe_heuristic_candidate(
    program: MegatronProgram,
    context_record: Dict[str, Any],
) -> Optional[MegatronProgram]:
    if int(program.parallel.pp_degree) <= 1:
        return None
    evidence = dict((context_record or {}).get("evidence_record") or {})
    plan = dict(evidence.get("apipe_heuristic_plan") or {})
    runtime_controls = dict(plan.get("runtime_controls") or {})
    actions = list(plan.get("actions") or [])
    if not runtime_controls and not actions:
        return None

    candidate = _clone_program(program)
    stage_layers = [int(stage.decoder_layers) for stage in candidate.partition.stages]
    if not stage_layers:
        return None

    for action in actions:
        name = str((action or {}).get("name") or "").strip().lower()
        status = str((action or {}).get("status") or "").strip().lower()
        if not status.startswith("executable"):
            continue
        if name == "move_boundary":
            donor = _safe_int((action or {}).get("donor_stage"))
            receiver = _safe_int((action or {}).get("receiver_stage"))
            shift = max(int((action or {}).get("shift_blocks") or 1), 1)
            if (
                donor is None
                or receiver is None
                or donor < 0
                or receiver < 0
                or donor >= len(stage_layers)
                or receiver >= len(stage_layers)
                or donor == receiver
            ):
                continue
            transferable = max(stage_layers[donor] - 1, 0)
            shift = min(shift, transferable)
            if shift <= 0:
                continue
            stage_layers[donor] -= shift
            stage_layers[receiver] += shift
        elif name == "set_local_vpp":
            target_vector = [max(int(value), 1) for value in ((action or {}).get("vpp_vector") or [])]
            if not target_vector:
                continue
            global_vpp = max(
                min(int((action or {}).get("global_vpp_cap") or 1), 2),
                max(int(candidate.parallel.vpp_degree), max(target_vector)),
            )
            candidate.metadata["stage_local_vpp_vector"] = [int(value) for value in target_vector]
            candidate.metadata["preserve_stage_local_vpp"] = True
            candidate.parallel.vpp_degree = max(int(global_vpp), 1)
            candidate.layout.vpp_degree = max(int(global_vpp), 1)
            local_by_name = {entry.subgraph: entry for entry in (candidate.strategy_ir.local_parallel or [])}
            for subgraph in (candidate.strategy_ir.apipe or []):
                entry = local_by_name.get(subgraph.name)
                if entry is None:
                    continue
                stage_id = int(subgraph.stage_index)
                if stage_id < len(target_vector):
                    entry.vpp_degree = max(int(target_vector[stage_id]), 1)
            if int(candidate.parallel.vpp_degree) > 1:
                focus = str(
                    (next((item for item in actions if str((item or {}).get("name") or "").strip().lower() == "move_boundary"), {}) or {}).get("focus")
                    or candidate.metadata.get("boundary_semantic_focus")
                    or "balanced"
                )
                counts = _stage_local_virtual_counts(
                    stage_layers,
                    target_vector,
                    global_vpp=int(candidate.parallel.vpp_degree),
                    focus=str(focus),
                )
                candidate.layout.pipeline_layout = _virtual_stage_layout(counts)
        elif name == "reorder_flush_microbatches":
            policy = str((action or {}).get("policy") or "").strip()
            if policy:
                candidate.metadata["flush_order_policy"] = policy

    for index, stage in enumerate(candidate.partition.stages):
        stage.decoder_layers = int(stage_layers[index])

    template = str(runtime_controls.get("template") or candidate.schedule.template or "fixed_1f1b")
    dispatch_order = str(
        runtime_controls.get("dispatch_order") or candidate.schedule.dispatch_order or "default"
    )
    group_size = runtime_controls.get("steady_state_group_size")
    candidate.schedule.template = template
    candidate.schedule.skeleton = "fixed_1f1b" if template == "fixed_1f1b" else "stage_aware_grouped"
    candidate.schedule.dispatch_order = dispatch_order
    candidate.schedule.microbatch_group_size_per_vp_stage = (
        max(int(group_size), 1) if group_size is not None else candidate.schedule.microbatch_group_size_per_vp_stage
    )
    candidate.strategy_ir.pipe.warmup_policy = str(
        runtime_controls.get("warmup_policy") or candidate.strategy_ir.pipe.warmup_policy or "default"
    )
    candidate.strategy_ir.pipe.cooldown_policy = str(
        runtime_controls.get("cooldown_policy") or candidate.strategy_ir.pipe.cooldown_policy or "default"
    )
    flush_order_policy = str(runtime_controls.get("flush_order_policy") or "").strip()
    if flush_order_policy:
        candidate.metadata["flush_order_policy"] = flush_order_policy
        candidate.metadata["schedule_cooldown_p2p_policy"] = "serial"
        candidate.metadata["schedule_cooldown_combined_policy"] = "serial"

    candidate.metadata["program_kind"] = "candidate_apipe_pipe_heuristic_v1"
    candidate.metadata["priority_rank"] = 47
    candidate.metadata["apipe_pipe_heuristic_actions"] = actions
    return _sync_batch_plan_metadata(candidate)


def _runtime_branch_plan(context_record: Dict[str, Any]) -> Dict[str, Any]:
    evidence = dict((context_record or {}).get("evidence_record") or {})
    return dict(evidence.get("runtime_branch_plan") or {})


def _build_runtime_branch_candidates(
    baseline: MegatronProgram,
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    branch_plan = _runtime_branch_plan(context_record)
    branches = list(branch_plan.get("branches") or [])
    if not branches:
        return []
    candidates: List[MegatronProgram] = []
    for branch in branches:
        if not bool((branch or {}).get("active")):
            continue
        branch_id = str((branch or {}).get("branch_id") or "").strip()
        builder = str((branch or {}).get("builder") or "").strip().lower()
        priority_rank = int((branch or {}).get("priority_rank") or 40)
        candidate: Optional[MegatronProgram] = None
        if builder == "stage_local_vpp_shape":
            continue
        elif builder == "stage_local_memory_policy":
            candidate = _build_stage_local_memory_policy_candidate(baseline, context_record)
            if candidate is not None and branch_id == "branch_moe_skew_memory_policy":
                candidate.metadata["runtime_memory_policy_mode"] = "moe_skew_relief"
        elif builder == "apipe_pipe_heuristic":
            candidate = _build_apipe_pipe_heuristic_candidate(baseline, context_record)
        elif builder == "morphable_pipeline_candidate":
            candidate = _build_morphable_pipeline_candidate(baseline, context_record)
        if candidate is None:
            continue
        candidate.metadata["runtime_branch_id"] = branch_id
        candidate.metadata["runtime_branch_scope"] = str((branch or {}).get("scope") or "")
        candidate.metadata["runtime_branch_label"] = str((branch or {}).get("label") or branch_id)
        candidate.metadata["runtime_branch_trigger_rule_id"] = str((branch or {}).get("trigger_rule_id") or "")
        candidate.metadata["runtime_branch_target_stage_ids"] = [
            int(item) for item in list((branch or {}).get("target_stage_ids") or []) if _safe_int(item) is not None
        ]
        candidate.metadata["runtime_branch_status"] = str((branch or {}).get("status") or "")
        candidate.metadata["priority_rank"] = min(int(candidate.metadata.get("priority_rank") or priority_rank), priority_rank)
        if branch_id:
            candidate.metadata["program_kind"] = f"candidate_{branch_id}"
        candidates.append(_annotate_local_parallel(candidate, context_record))
    return candidates


def _candidate_allowed_by_space(program: MegatronProgram, search_space: SearchSpaceSpec) -> Tuple[bool, str]:
    program_kind = str(program.metadata.get("program_kind") or "")
    is_single_node_pp_split = program_kind == "candidate_single_node_pp_split"
    is_sequence_parallel_toggle = program_kind == "candidate_sequence_parallel_toggle"
    is_dual_plane_candidate = program_kind == "candidate_dual_plane"
    uses_hybrid_shard = any(
        str(item.shard_strategy or "none") in {"fsdp", "hsdp"}
        for item in (program.strategy_ir.local_parallel or [])
    )
    uses_torchtitan_schedule = str(program.schedule.template or "") in {"torchtitan_zero_bubble", "torchtitan_dualpipev"}
    uses_morphable_pipeline = str(program.metadata.get("program_kind") or "") == "candidate_morphable_pipeline"
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
    candidate_micro_batch = max(int(program.metadata.get("micro_batch_size", 1) or 1), 1)
    if search_space.max_micro_batch_size is not None and candidate_micro_batch > int(search_space.max_micro_batch_size):
        return (
            False,
            f"micro_batch_size={candidate_micro_batch} exceeds search-space max_micro_batch_size={search_space.max_micro_batch_size}",
        )
    memory_estimate = estimate_program_memory(program)
    if (
        search_space.max_estimated_memory_pressure is not None
        and float(memory_estimate.pressure_score) > float(search_space.max_estimated_memory_pressure)
    ):
        return (
            False,
            "estimated memory pressure "
            f"{memory_estimate.pressure_score:.2f} exceeds search-space ceiling "
            f"{float(search_space.max_estimated_memory_pressure):.2f}",
        )
    if is_dual_plane_candidate and not search_space.allow_dual_plane:
        return False, "dual-plane mapping is not allowed in the current search space"
    if not search_space.allow_stage_aware_schedule and program.schedule.skeleton != "fixed_1f1b":
        return False, "stage-aware schedule is not allowed in the current search space"
    if uses_hybrid_shard and not search_space.allow_hybrid_shard:
        return False, "hybrid shard candidates are not allowed in the current search space"
    if uses_torchtitan_schedule and not search_space.allow_torchtitan_schedule_sandbox:
        return False, "torchtitan schedule sandbox is not allowed in the current search space"
    if uses_morphable_pipeline and not search_space.allow_morphable_pipeline:
        return False, "morphable pipeline candidates are not allowed in the current search space"
    if program.schedule.skeleton not in set(search_space.allowed_schedule_skeletons):
        return False, f"schedule skeleton {program.schedule.skeleton} is outside allowed search-space skeletons"
    if program.schedule.template not in set(search_space.allowed_schedule_templates):
        return False, f"schedule template {program.schedule.template} is outside allowed search-space templates"
    if not search_space.allow_asymmetric_vpp and int(program.parallel.vpp_degree) > 1:
        return False, "asymmetric VPP is not allowed in the current search space"
    if search_space.max_shard_group_size is not None:
        for local in (program.strategy_ir.local_parallel or []):
            if local.shard_group_size is not None and int(local.shard_group_size) > int(search_space.max_shard_group_size):
                return False, f"shard_group_size={local.shard_group_size} exceeds search-space max_shard_group_size={search_space.max_shard_group_size}"
    if search_space.max_replicate_group_size is not None:
        for local in (program.strategy_ir.local_parallel or []):
            if local.replicate_group_size is not None and int(local.replicate_group_size) > int(search_space.max_replicate_group_size):
                return False, f"replicate_group_size={local.replicate_group_size} exceeds search-space max_replicate_group_size={search_space.max_replicate_group_size}"
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


def _build_skeleton_candidates(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    runtime_summary: Dict[str, Any],
) -> List[MegatronProgram]:
    candidates: List[MegatronProgram] = []
    if _is_dual_target(baseline.cluster.target):
        dual_pp4 = _build_dual_node_pp4_candidate(baseline)
        if dual_pp4 is not None:
            candidates.append(dual_pp4)
    if rewrite.prefer_memory_relief:
        memory_relief = _build_memory_relief_candidate(baseline)
        if memory_relief is not None:
            candidates.append(memory_relief)
    if rewrite.allow_single_node_pp_split:
        single_node = _build_single_node_pipeline_candidate(baseline)
        if single_node is not None:
            candidates.append(single_node)
    if rewrite.allow_nonuniform_partition:
        nonuniform = _build_nonuniform_partition(baseline)
        if nonuniform is not None:
            candidates.append(nonuniform)
        runtime_guided = _build_runtime_guided_partition(baseline, runtime_summary)
        if runtime_guided is not None:
            candidates.append(runtime_guided)
    if rewrite.max_pp_size is not None and int(rewrite.max_pp_size) >= 4:
        pp4 = _build_pp_scaleout_candidate(baseline)
        if pp4 is not None:
            candidates.append(pp4)
    if rewrite.max_pp_size is not None and int(rewrite.max_pp_size) >= 8:
        pp8 = _build_dual_node_pp8_scaleout_candidate(baseline, runtime_summary)
        if pp8 is not None:
            candidates.append(pp8)
    if _is_dual_target(baseline.cluster.target):
        topology = _build_topology_candidate(baseline)
        if topology is not None:
            candidates.append(topology)
    if str(baseline.cluster.target) == "dual_g4_g5":
        orientation = _build_dual_node_orientation_candidate(baseline, runtime_summary)
        if orientation is not None:
            candidates.append(orientation)
    return candidates


def _build_local_parallel_candidates(
    baseline: MegatronProgram,
    skeletons: List[MegatronProgram],
    rewrite: SearchSpaceSpec,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    trace_summary = reduce_trial_trace(baseline, runtime_summary=runtime_summary)
    bottleneck = classify_bottleneck(baseline, trace_summary)
    failure_labels = {str(item.get("label")) for item in (context_record.get("failure_modes") or [])}
    candidates: List[MegatronProgram] = []
    seeds = [baseline] + list(skeletons)

    if "memory_underfilled" in set(bottleneck.get("labels") or []):
        for seed in seeds:
            filled = _build_batch_plan_fill_candidate(seed)
            if filled is not None:
                candidates.append(_annotate_local_parallel(filled, context_record))

    if rewrite.max_cp_size is not None and int(rewrite.max_cp_size) > 1 and (
        "memory_hotspot" in failure_labels or "long_context_attention_heavy" in set(bottleneck.get("labels") or [])
    ):
        for seed in seeds:
            cp_candidate = _build_long_context_cp_candidate(seed)
            if cp_candidate is not None:
                candidates.append(_annotate_local_parallel(cp_candidate, context_record))

    if "memory_hotspot" in failure_labels:
        for seed in seeds:
            fsdp_candidate = _clone_program(seed)
            touched = False
            for local in fsdp_candidate.strategy_ir.local_parallel:
                if int(local.cp_degree) > 1 or int(fsdp_candidate.parallel.pp_degree) > 1:
                    local.fsdp_scope = "selective"
                    touched = True
            if touched:
                fsdp_candidate.metadata["program_kind"] = "candidate_local_fsdp_scope"
                fsdp_candidate.metadata["priority_rank"] = 18
                candidates.append(_sync_batch_plan_metadata(fsdp_candidate))

    if rewrite.allow_dual_plane:
        dual_plane = _build_dual_plane_candidate(baseline)
        if dual_plane is not None:
            candidates.append(_annotate_local_parallel(dual_plane, context_record))

    if rewrite.allow_sequence_parallel_toggle:
        sp_candidate = _build_sequence_parallel_candidate(baseline)
        if sp_candidate is not None:
            candidates.append(_annotate_local_parallel(sp_candidate, context_record))

    if rewrite.allow_hybrid_shard:
        hsdp_candidate = _build_torchtitan_hsdp_candidate(baseline, context_record)
        if hsdp_candidate is not None:
            candidates.append(_annotate_local_parallel(hsdp_candidate, context_record))
        for seed in seeds:
            hybrid_candidate = _build_pp_hsdp_hybrid_candidate(seed, runtime_summary, context_record)
            if hybrid_candidate is not None:
                candidates.append(_annotate_local_parallel(hybrid_candidate, context_record))

    if rewrite.allow_asymmetric_vpp or rewrite.allow_stage_aware_schedule:
        stage_local_vpp = _build_stage_local_vpp_shape_candidate(baseline, context_record)
        if stage_local_vpp is not None:
            candidates.append(_annotate_local_parallel(stage_local_vpp, context_record))
        tail_aware = _build_tail_aware_execution_candidate(baseline, context_record)
        if tail_aware is not None:
            candidates.append(_annotate_local_parallel(tail_aware, context_record))

    if rewrite.prefer_memory_relief or "memory_hotspot" in failure_labels or "memory_skew" in failure_labels:
        for seed in seeds:
            checkpoint_refinement = _build_checkpoint_boundary_refinement_candidate(seed, context_record)
            if checkpoint_refinement is not None:
                candidates.append(_annotate_local_parallel(checkpoint_refinement, context_record))
            offload_first = _build_offload_first_refinement_candidate(seed, context_record)
            if offload_first is not None:
                candidates.append(_annotate_local_parallel(offload_first, context_record))
            local_memory_policy = _build_stage_local_memory_policy_candidate(seed, context_record)
            if local_memory_policy is not None:
                candidates.append(_annotate_local_parallel(local_memory_policy, context_record))

    return candidates


def _build_schedule_candidates(
    baseline: MegatronProgram,
    skeletons: List[MegatronProgram],
    rewrite: SearchSpaceSpec,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
) -> List[MegatronProgram]:
    if not rewrite.allow_stage_aware_schedule:
        return []
    allowed_templates = set(rewrite.allowed_schedule_templates or ["fixed_1f1b"])
    candidates: List[MegatronProgram] = []
    stage_aware = _build_stage_aware_schedule(baseline)
    if stage_aware is not None:
        candidates.append(_annotate_local_parallel(stage_aware, context_record))
    runtime_guided = _build_runtime_guided_schedule(baseline, runtime_summary)
    if runtime_guided is not None:
        candidates.append(_annotate_local_parallel(runtime_guided, context_record))
    optimizer_aware = _build_optimizer_aware_pipeline_candidate(baseline, context_record)
    if optimizer_aware is not None:
        candidates.append(_annotate_local_parallel(optimizer_aware, context_record))
    if rewrite.allow_torchtitan_schedule_sandbox:
        zero_bubble = _build_torchtitan_zero_bubble_schedule_candidate(baseline, runtime_summary, context_record)
        if zero_bubble is not None:
            candidates.append(_annotate_local_parallel(zero_bubble, context_record))
        dualpipev = _build_torchtitan_dualpipev_schedule_candidate(baseline, runtime_summary, context_record)
        if dualpipev is not None:
            candidates.append(_annotate_local_parallel(dualpipev, context_record))
    apipe_heuristic = _build_apipe_pipe_heuristic_candidate(baseline, context_record)
    if (
        apipe_heuristic is not None
        and str(apipe_heuristic.schedule.template or "fixed_1f1b") in allowed_templates
    ):
        candidates.append(_annotate_local_parallel(apipe_heuristic, context_record))
    if rewrite.allow_morphable_pipeline:
        morphable_candidate = _build_morphable_pipeline_candidate(baseline, context_record)
        if (
            morphable_candidate is not None
            and str(morphable_candidate.schedule.template or "fixed_1f1b") in allowed_templates
        ):
            candidates.append(_annotate_local_parallel(morphable_candidate, context_record))
    for skeleton in skeletons:
        if int(skeleton.parallel.pp_degree) >= 4:
            scaled = _build_pp_vpp_scaleout_candidate(skeleton, runtime_summary)
            if scaled is not None:
                candidates.append(_annotate_local_parallel(scaled, context_record))
    for boundary_candidate in _build_boundary_semantic_schedule_candidates(baseline, context_record):
        if str(boundary_candidate.schedule.template or "") not in allowed_templates:
            continue
        candidates.append(_annotate_local_parallel(boundary_candidate, context_record))
    for pipe_candidate in _build_pipe_search_space_candidates(baseline, runtime_summary, context_record):
        if str(pipe_candidate.schedule.template or "") not in allowed_templates:
            continue
        candidates.append(_annotate_local_parallel(pipe_candidate, context_record))
    return candidates


def _build_agent_proposals(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    runtime_summary: Dict[str, Any],
    context_record: Dict[str, Any],
    replan_decision: Dict[str, Any],
) -> List[AgentProposal]:
    optimization_hints = list((context_record or {}).get("optimization_hints") or [])
    hinted_rationales: Dict[str, str] = {}
    for item in optimization_hints:
        scope = str((item or {}).get("scope") or "")
        rationale = str((item or {}).get("rationale") or "")
        if scope and rationale and scope not in hinted_rationales:
            hinted_rationales[scope] = rationale
    skeletons = _build_skeleton_candidates(baseline, rewrite, runtime_summary)
    skeletons = [
        _annotate_candidate_runtime_evidence(
            baseline,
            _annotate_local_parallel(program, context_record),
            context_record,
        )
        for program in skeletons
    ]
    runtime_branch_candidates = [
        _annotate_candidate_runtime_evidence(baseline, program, context_record)
        for program in _build_runtime_branch_candidates(baseline, context_record)
        if _candidate_allowed_by_space(program, rewrite)[0]
    ]
    local_parallel = [
        _annotate_candidate_runtime_evidence(baseline, program, context_record)
        for program in _build_local_parallel_candidates(baseline, skeletons, rewrite, runtime_summary, context_record)
    ]
    schedules = [
        _annotate_candidate_runtime_evidence(baseline, program, context_record)
        for program in _build_schedule_candidates(baseline, skeletons, rewrite, runtime_summary, context_record)
    ]

    if str(replan_decision.get("scope")) == "pipe":
        ordered_pool = (
            [(program, "pipe", hinted_rationales.get("pipe", "active runtime branch indicates a low-cost schedule-local intervention is worth trying first")) for program in runtime_branch_candidates if str((program.metadata or {}).get("runtime_branch_scope") or "") == "pipe"]
            + [(program, "pipe", hinted_rationales.get("pipe", "runtime evidence prefers schedule adaptation before touching local or global structure")) for program in schedules]
            + [(program, "local_parallel", hinted_rationales.get("local_parallel", "active runtime branch says a local relief branch is already justified")) for program in runtime_branch_candidates if str((program.metadata or {}).get("runtime_branch_scope") or "") == "local_parallel"]
            + [(program, "local_parallel", hinted_rationales.get("local_parallel", "schedule fix is insufficient; escalate to local subgraph policy")) for program in local_parallel]
            + [(program, "skeleton", hinted_rationales.get("skeleton", "schedule/local adaptations are exhausted; escalate to PP skeleton change")) for program in skeletons]
        )
    elif str(replan_decision.get("scope")) == "local_parallel":
        ordered_pool = (
            [(program, "local_parallel", hinted_rationales.get("local_parallel", "memory or hotspot evidence prefers CP/VPP/FSDP relief first")) for program in local_parallel]
            + [(program, "local_parallel", hinted_rationales.get("local_parallel", "active runtime branch says local policy relief is worth trying before broader search")) for program in runtime_branch_candidates if str((program.metadata or {}).get("runtime_branch_scope") or "") == "local_parallel"]
            + [(program, "pipe", hinted_rationales.get("pipe", "if local relief is insufficient, try lower-cost runtime pipe changes")) for program in runtime_branch_candidates if str((program.metadata or {}).get("runtime_branch_scope") or "") == "pipe"]
            + [(program, "pipe", hinted_rationales.get("pipe", "if local relief is insufficient, try lower-cost runtime pipe changes")) for program in schedules]
            + [(program, "skeleton", hinted_rationales.get("skeleton", "persistent local failures justify skeleton repartitioning")) for program in skeletons]
        )
    else:
        ordered_pool = (
            [(program, str((program.metadata or {}).get("runtime_branch_scope") or "local_parallel"), hinted_rationales.get(str((program.metadata or {}).get("runtime_branch_scope") or "local_parallel"), "active runtime branch offers the cheapest next intervention")) for program in runtime_branch_candidates]
            + [(program, "skeleton", hinted_rationales.get("skeleton", "topology or persistent imbalance indicates broader stage repartitioning")) for program in skeletons]
            + [(program, "local_parallel", hinted_rationales.get("local_parallel", "after skeleton changes, refine local CP/VPP/FSDP policy")) for program in local_parallel]
            + [(program, "pipe", hinted_rationales.get("pipe", "pipe is last-mile tuning after broader structural changes")) for program in schedules]
        )
    proposals: List[AgentProposal] = []
    for program, scope, rationale in sorted(ordered_pool, key=lambda item: _candidate_sort_key(item[0])):
        proposals.append(_build_agent_proposal(program, scope=scope, rationale=rationale, source="heuristic_supervisor"))
    return proposals


def _verify_agent_proposals(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    proposals: List[AgentProposal],
    *,
    observation: Dict[str, Any],
    candidate_limit: int,
) -> Tuple[List[AgentProposal], List[Dict[str, Any]]]:
    accepted: List[AgentProposal] = []
    rejected: List[Dict[str, Any]] = []
    seen = {_structural_program_key(baseline)}
    for proposal in proposals:
        candidate = proposal.program.normalized()
        hash_value = _structural_program_key(candidate)
        if hash_value in seen:
            continue
        seen.add(hash_value)
        candidate.search_space = rewrite.normalized()
        allowed, allowed_reason = _candidate_allowed_by_space(candidate, rewrite)
        if not allowed:
            rejected.append({"proposal": proposal.to_dict(), "reason": allowed_reason})
            continue
        report = verify_program(candidate, observation=observation, previous_program=baseline)
        proposal.verifier_report = report.to_dict()
        if not report.is_legal:
            rejected.append({"proposal": proposal.to_dict(), "reason": report.to_dict()})
            continue
        if len(accepted) < int(candidate_limit):
            accepted.append(proposal.normalized())
    return accepted, rejected


def _synthesize_proposals(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    runtime_summary: Optional[Dict[str, Any]] = None,
    context_record: Optional[Dict[str, Any]] = None,
    replan_decision: Optional[Dict[str, Any]] = None,
    candidate_limit: int = 4,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[AgentProposal], List[Dict[str, Any]]]:
    runtime_summary = runtime_summary or {}
    context_record = context_record or build_context_record(baseline, runtime_summary=runtime_summary)
    replan_decision = replan_decision or _build_replan_decision(baseline, context_record)
    proposals = _build_agent_proposals(
        baseline,
        rewrite,
        runtime_summary=runtime_summary,
        context_record=context_record,
        replan_decision=replan_decision,
    )
    proposals = _apply_llm_supervisor(
        proposals,
        context_record=context_record,
        replan_decision=replan_decision,
        candidate_limit=candidate_limit,
        llm_config=llm_config,
    )
    return _verify_agent_proposals(
        baseline,
        rewrite,
        proposals,
        observation=context_record,
        candidate_limit=candidate_limit,
    )


def _synthesize_programs(
    baseline: MegatronProgram,
    rewrite: SearchSpaceSpec,
    runtime_summary: Optional[Dict[str, Any]] = None,
    context_record: Optional[Dict[str, Any]] = None,
    replan_decision: Optional[Dict[str, Any]] = None,
    candidate_limit: int = 4,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[MegatronProgram], List[Dict[str, Any]]]:
    proposals, rejected = _synthesize_proposals(
        baseline,
        rewrite,
        runtime_summary=runtime_summary,
        context_record=context_record,
        replan_decision=replan_decision,
        candidate_limit=candidate_limit,
        llm_config=llm_config,
    )
    return [proposal.program for proposal in proposals], rejected


def _real_trial_runtime_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    trace_summary = dict(metrics.get("trace_summary") or {})
    if trace_summary:
        return trace_summary
    summary: Dict[str, Any] = {}
    for key in (
        "steady_state_step_time_ms_p50",
        "steady_state_step_time_ms_p95",
        "bubble_ratio",
        "comm_exposure_ratio",
        "peak_reserved_ratio",
        "peak_reserved_gib",
        "optimizer_ratio",
        "optimizer_exposed_ratio",
        "stage_tail_ratio",
        "mem_skew_ratio",
        "stall_ratio",
        "tail_step_jitter_ratio",
    ):
        if key in metrics:
            summary[key] = metrics.get(key)
    return summary


def _second_stage_runtime_inputs(
    baseline: MegatronProgram,
    baseline_metrics: Dict[str, Any],
    fallback_context: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    trial_runtime_summary = _real_trial_runtime_summary(baseline_metrics)
    trial_context = dict(baseline_metrics.get("context_record") or {})
    if not trial_context:
        baseline_observation = _make_agent_observation(
            baseline,
            metrics=baseline_metrics,
            trace_summary=trial_runtime_summary,
        )
        trial_context = baseline_observation.to_dict()
    previous_context = fallback_context or {}
    second_stage_replan = _resolve_replan_decision(
        baseline,
        trial_context,
        previous_context=previous_context,
    ).to_dict()
    second_stage_bottleneck = classify_bottleneck(baseline, trial_runtime_summary)
    return trial_runtime_summary, trial_context, second_stage_replan, second_stage_bottleneck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron program synthesis bring-up runner.")
    parser.add_argument("--workdir", type=str, default="./runs_megatron_programs")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--programs-dir", type=str, default=None)
    parser.add_argument("--runtime-summary", type=str, default=None)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--enable-llm-supervisor", action="store_true")
    parser.add_argument("--llm-endpoint", type=str, default="http://10.100.1.93:12365/v1/chat/completions")
    parser.add_argument("--llm-model", type=str, default="/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--log-llm", action="store_true")
    parser.add_argument("--model-structure-summary", type=str, default=None)
    parser.add_argument("--hardware-topology-summary", type=str, default=None)
    parser.add_argument("--profile-summary", type=str, default=None)
    parser.add_argument("--baseline-catalog", type=str, default=None)
    parser.add_argument("--selector-only", action="store_true")
    parser.add_argument("--run-target", type=str, choices=["single_g4", "single_g5", "dual_g4_g5", "dual_g5_g5"], default="single_g5")
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
    parser.add_argument("--transformer-impl", type=str, default="auto")
    parser.add_argument("--attention-backend", type=str, default="auto")
    parser.add_argument("--tokenizer-model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument("--enable-tp-comm-overlap", action="store_true")
    add_observability_args(parser)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--min-free-gpu-memory-mib", type=int, default=28672)
    parser.add_argument("--busy-gpu-memory-threshold-mib", type=int, default=2048)
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
    if args.run_target in {"dual_g4_g5", "dual_g5_g5"}:
        args.nnodes = max(int(args.nnodes), 2)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    programs_dir = Path(args.programs_dir) if args.programs_dir else workdir / "programs"

    runtime_summary = _load_runtime_summary(args.runtime_summary)
    llm_config = {
        "enabled": bool(args.enable_llm_supervisor),
        "endpoint": str(args.llm_endpoint),
        "model": str(args.llm_model),
        "temperature": float(args.llm_temperature),
        "log_llm": bool(args.log_llm),
    }
    external_inputs = _load_external_agent_inputs(args)
    baseline = _build_baseline_program(args)
    runtime_signature = reduce_trial_trace(baseline, runtime_summary=runtime_summary)
    observation = _make_agent_observation(baseline, runtime_summary=runtime_summary)
    context_record = _augment_context_with_external_inputs(observation.to_dict(), external_inputs)
    previous_context = dict(runtime_summary.get("previous_context_record") or {}) if isinstance(runtime_summary, dict) else {}
    replan_decision_obj = _resolve_replan_decision(baseline, context_record, previous_context=previous_context or None)
    replan_decision = replan_decision_obj.to_dict()
    bottleneck_signature = classify_bottleneck(baseline, runtime_signature)
    baseline_legality = check_program(baseline)
    if not baseline_legality.is_valid:
        raise ValueError(f"Baseline program is invalid: {json.dumps(baseline_legality.to_dict(), ensure_ascii=False)}")

    rewrite = _rewrite_space(baseline, runtime_signature)
    baseline.search_space = rewrite.normalized()
    evidence_programs = _build_evidence_matrix(baseline, rewrite, runtime_signature, context_record)
    proposal_pool, rejected_candidates = _synthesize_proposals(
        baseline,
        rewrite=rewrite,
        runtime_summary=runtime_signature,
        context_record=context_record,
        replan_decision=replan_decision,
        candidate_limit=int(args.candidate_limit),
        llm_config=llm_config,
    )
    strategy_template_library = _build_verified_strategy_template_library(
        baseline,
        proposal_pool,
        context_record,
    )
    template_selection_decision = _select_verified_strategy_template(
        strategy_template_library,
        context_record,
    )
    candidates = [proposal.program for proposal in proposal_pool]
    if bool(args.selector_only):
        selected_kind = str(template_selection_decision.get("selected_program_kind") or "")
        if selected_kind:
            selected_candidates = [program for program in candidates if str(program.metadata.get("program_kind") or "") == selected_kind]
            if selected_candidates:
                candidates = selected_candidates[:1]
                proposal_pool = [
                    proposal
                    for proposal in proposal_pool
                    if str(proposal.program.metadata.get("program_kind") or "") == selected_kind
                ]
    experiment_specs = _build_experiment_specs(baseline, evidence_programs, candidates)
    initial_bank = build_program_bank(
        [baseline] + evidence_programs + candidates,
        trace_summaries={str(baseline.metadata.get("program_kind")): runtime_signature},
    )
    ordered_templates = select_program_templates(
        initial_bank,
        run_target=str(args.run_target),
        model_track=str(args.model_track),
        length_bucket=str(runtime_signature.get("length_bucket") or "default"),
        bottleneck_signature=bottleneck_signature,
    )
    ordered_programs: List[MegatronProgram] = []
    kind_to_program = {
        str(program.metadata.get("program_kind") or "program"): program
        for program in [baseline] + evidence_programs + candidates
    }
    for template in ordered_templates:
        program = kind_to_program.get(template.name)
        if program is None or program is baseline or program in evidence_programs:
            continue
        ordered_programs.append(program)
    ordered_candidates = ordered_programs + [program for program in candidates if program not in ordered_programs]
    candidates = ordered_candidates[: int(args.candidate_limit)]
    proposal_by_kind = {proposal.proposal_id: proposal for proposal in proposal_pool}
    candidate_manifest = _export_programs(
        baseline,
        candidates,
        programs_dir,
        context_record=context_record,
        previous_program=baseline,
    )
    _progress(
        "prepared "
        f"{1 + len(candidates)} executable programs "
        f"(baseline + {len(candidates)} candidates); "
        f"rejected={len(rejected_candidates)}"
    )
    evidence_manifest = []
    for index, program in enumerate(evidence_programs):
        program_kind = str(program.metadata.get("program_kind") or f"evidence_{index:02d}")
        verifier_report = verify_program(program, observation=context_record, previous_program=baseline)
        evidence_manifest.append(
            {
                "config_name": program_kind,
                "program_hash": program.semantic_hash(),
                "family": classify_program_family(program).to_dict(),
                "legality": dict(verifier_report.legality or {}),
                "verifier_report": verifier_report.to_dict(),
                "experiment_ids": [
                    spec.experiment_id for spec in experiment_specs if program_kind in set(spec.program_kinds)
                ],
            }
        )
    observation.motivation_evidence_manifest = list(evidence_manifest)
    context_record = observation.to_dict()

    tested: List[Dict[str, Any]] = []
    paper_artifacts: List[Dict[str, Any]] = []
    family_outside_trials: List[Dict[str, Any]] = []
    baseline_metrics: Optional[Dict[str, Any]] = None
    best_program: Optional[MegatronProgram] = None
    best_metrics: Optional[Dict[str, Any]] = None
    second_stage_runtime_summary: Optional[Dict[str, Any]] = None
    second_stage_context_record: Optional[Dict[str, Any]] = None
    second_stage_replan_decision: Optional[Dict[str, Any]] = None
    second_stage_bottleneck_signature: Optional[Dict[str, Any]] = None

    if not args.export_only:
        _progress("starting baseline trial")
        baseline_metrics = run_trial(args, baseline, trial_id=0)
        baseline_metrics["config_name"] = "baseline"
        baseline_metrics["trace_summary"] = reduce_trial_trace(baseline, metrics=baseline_metrics, runtime_summary=runtime_signature)
        baseline_metrics["bottleneck_signature"] = classify_bottleneck(baseline, baseline_metrics["trace_summary"])
        baseline_observation = _make_agent_observation(baseline, trace_summary=baseline_metrics["trace_summary"], motivation_evidence_manifest=evidence_manifest)
        baseline_metrics["context_record"] = baseline_observation.to_dict()
        baseline_metrics["trial_artifact"] = build_trial_artifact(
            baseline,
            baseline_observation,
            bottleneck_signature=baseline_metrics["bottleneck_signature"],
            experiment=next(
                (
                    spec
                    for spec in experiment_specs
                    if str(baseline.metadata.get("program_kind") or "baseline") in set(spec.program_kinds)
                ),
                None,
            ),
        )
        tested.append(baseline_metrics)
        paper_artifacts.append(dict(baseline_metrics.get("trial_artifact") or {}))
        if bool((baseline_metrics.get("family") or {}).get("is_family_outside")):
            family_outside_trials.append(baseline_metrics)
        _progress(
            "baseline trial finished "
            f"returncode={int(baseline_metrics.get('returncode') or 0)} "
            f"step_time_ms_p50={baseline_metrics.get('step_time_ms_p50')} "
            f"throughput={baseline_metrics.get('throughput_tokens_per_s') or baseline_metrics.get('throughput_effective_tokens_per_s')}"
        )

        if int(baseline_metrics.get("returncode") or 0) == 0:
            (
                second_stage_runtime_summary,
                second_stage_context_record,
                second_stage_replan_decision,
                second_stage_bottleneck_signature,
            ) = _second_stage_runtime_inputs(
                baseline,
                baseline_metrics,
                context_record,
            )
            second_stage_proposals, second_stage_rejected = _synthesize_proposals(
                baseline,
                rewrite=rewrite,
                runtime_summary=second_stage_runtime_summary,
                context_record=second_stage_context_record,
                replan_decision=second_stage_replan_decision,
                candidate_limit=int(args.candidate_limit),
                llm_config=llm_config,
            )
            if second_stage_proposals:
                proposal_pool = second_stage_proposals
                rejected_candidates = second_stage_rejected
                strategy_template_library = _build_verified_strategy_template_library(
                    baseline,
                    proposal_pool,
                    second_stage_context_record,
                )
                template_selection_decision = _select_verified_strategy_template(
                    strategy_template_library,
                    second_stage_context_record,
                )
                candidates = [proposal.program for proposal in proposal_pool]
                if bool(args.selector_only):
                    selected_kind = str(template_selection_decision.get("selected_program_kind") or "")
                    if selected_kind:
                        selected_candidates = [
                            program
                            for program in candidates
                            if str(program.metadata.get("program_kind") or "") == selected_kind
                        ]
                        if selected_candidates:
                            candidates = selected_candidates[:1]
                            proposal_pool = [
                                proposal
                                for proposal in proposal_pool
                                if str(proposal.program.metadata.get("program_kind") or "") == selected_kind
                            ]
                experiment_specs = _build_experiment_specs(baseline, evidence_programs, candidates)
                candidate_manifest = _export_programs(
                    baseline,
                    candidates,
                    programs_dir,
                    context_record=second_stage_context_record,
                    previous_program=baseline,
                )
                _progress(
                    "second-stage replan refreshed candidates "
                    f"count={len(candidates)} selected_template={template_selection_decision.get('selected_template_id')}"
                )

        trial_queue: List[Tuple[str, MegatronProgram, Optional[ExperimentSpec]]] = []
        for program in evidence_programs:
            kind = str(program.metadata.get("program_kind") or "evidence")
            trial_queue.append(
                (
                    kind,
                    program,
                    next((spec for spec in experiment_specs if kind in set(spec.program_kinds)), None),
                )
            )
        for candidate in candidates:
            kind = str(candidate.metadata.get("program_kind") or "candidate")
            trial_queue.append(
                (
                    kind,
                    candidate,
                    next((spec for spec in experiment_specs if kind in set(spec.program_kinds)), None),
                )
            )

        _progress(f"starting candidate trials: total={len(trial_queue)}")
        for index, (config_name, candidate, experiment_spec) in enumerate(trial_queue, start=1):
            _progress(f"trial {index}/{len(trial_queue)} starting: {config_name}")
            metrics = run_trial(args, candidate, trial_id=index)
            metrics["config_name"] = config_name
            metrics["trace_summary"] = reduce_trial_trace(candidate, metrics=metrics)
            metrics["bottleneck_signature"] = classify_bottleneck(candidate, metrics["trace_summary"])
            candidate_observation = _make_agent_observation(
                candidate,
                trace_summary=metrics["trace_summary"],
                motivation_evidence_manifest=evidence_manifest,
            )
            metrics["context_record"] = candidate_observation.to_dict()
            metrics["trial_artifact"] = build_trial_artifact(
                candidate,
                candidate_observation,
                bottleneck_signature=metrics["bottleneck_signature"],
                experiment=experiment_spec,
            )
            tested.append(metrics)
            paper_artifacts.append(dict(metrics.get("trial_artifact") or {}))
            if bool((metrics.get("family") or {}).get("is_family_outside")):
                family_outside_trials.append(metrics)
            _progress(
                f"trial {index}/{len(trial_queue)} finished: {config_name} "
                f"returncode={int(metrics.get('returncode') or 0)} "
                f"step_time_ms_p50={metrics.get('step_time_ms_p50')} "
                f"throughput={metrics.get('throughput_tokens_per_s') or metrics.get('throughput_effective_tokens_per_s')}"
            )

        ranked_trials = _rank_trials(tested)
        best_metrics = ranked_trials[0] if ranked_trials else baseline_metrics
        selected_hash = str(best_metrics.get("program_hash") or baseline.semantic_hash()) if best_metrics is not None else baseline.semantic_hash()
        all_programs = [baseline] + evidence_programs + candidates
        best_program = next((program for program in all_programs if program.semantic_hash() == selected_hash), baseline)
        _progress(
            "trial ranking complete "
            f"best={str((best_program.metadata if best_program else {}).get('program_kind') or 'baseline')}"
        )
    else:
        ranked_trials = []
        paper_artifacts = [
            build_trial_artifact(
                program,
                context_record,
                bottleneck_signature=bottleneck_signature,
                experiment=next(
                    (spec for spec in experiment_specs if str(program.metadata.get("program_kind") or "baseline") in set(spec.program_kinds)),
                    None,
                ),
            )
            for program in [baseline] + evidence_programs + candidates
        ]

    trace_summaries = {
        str(item.get("config_name") or item.get("program_hash") or "program"): dict(item.get("trace_summary") or {})
        for item in tested
    }
    selection_scores = {
        str(item.get("config_name") or item.get("program_hash") or "program"): float(item.get("selection_score") or 0.0)
        for item in tested
        if item.get("selection_score") is not None
    }
    program_bank = build_program_bank(
        [baseline] + evidence_programs + candidates,
        trace_summaries=trace_summaries,
        selection_scores=selection_scores,
    )

    summary = _build_summary_payload(
        export_only=bool(args.export_only),
        programs_dir=programs_dir,
        runtime_summary=runtime_summary,
        runtime_signature=runtime_signature,
        context_record=context_record,
        replan_decision=replan_decision,
        bottleneck_signature=bottleneck_signature,
        rewrite=rewrite,
        baseline=baseline,
        baseline_metrics=baseline_metrics,
        best_program=best_program,
        best_metrics=best_metrics,
        tested=tested,
        family_outside_trials=family_outside_trials,
        rejected_candidates=rejected_candidates,
        candidate_manifest=candidate_manifest,
        program_bank=program_bank,
        evidence_manifest=evidence_manifest,
        experiment_specs=experiment_specs,
        paper_artifacts=paper_artifacts,
        agent_proposals=[proposal.to_dict() for proposal in proposal_pool],
        agent_topology=_agent_topology_summary(llm_config),
        external_inputs=external_inputs,
        strategy_template_library=strategy_template_library,
        template_selection_decision=template_selection_decision,
        second_stage_runtime_summary=second_stage_runtime_summary,
        second_stage_context_record=second_stage_context_record,
        second_stage_replan_decision=second_stage_replan_decision,
        second_stage_bottleneck_signature=second_stage_bottleneck_signature,
    )
    summary_path = workdir / "summary_megatron.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[megatron_agent] summary written to {summary_path}")


if __name__ == "__main__":
    main()
