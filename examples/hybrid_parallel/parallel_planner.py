from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hybrid_policy import HybridParallelPolicy


@dataclass
class HardwareNode:
    name: str
    rank_start: int
    gpus: int
    mem_gb: float
    perf: float


@dataclass
class PlanCandidate:
    pp_degree: int
    tp_degree: int
    dp_degree: int
    vpp: int
    microbatches: int
    schedule: str
    stage_ranges: List[List[int]]
    stage_to_node: List[str]
    mesh: List[List[List[int]]]
    score: float
    stage_costs: List[float]
    stage_times: List[float]
    fsdp_enabled_per_stage: List[bool]
    reshard_after_forward_per_stage: List[bool]
    recompute_per_stage: List[str]
    reasons: List[str]

    def _expand_virtual(self, values: Sequence[Any]) -> List[Any]:
        if int(self.vpp) <= 1:
            return list(values)
        out: List[Any] = []
        for value in values:
            out.extend([value] * int(self.vpp))
        return out

    def to_config_overrides(self) -> Dict[str, Any]:
        return {
            "parallel": {
                "pp": {
                    "degree": int(self.pp_degree),
                    "vpp": int(self.vpp),
                    "microbatches": int(self.microbatches),
                    "schedule": self.schedule,
                    "stages": copy.deepcopy(self.stage_ranges),
                    "mesh": copy.deepcopy(self.mesh),
                    "stage_to_node": list(self.stage_to_node),
                },
                "tp": {
                    "degree": int(self.tp_degree),
                    "enabled": bool(self.tp_degree > 1),
                },
                "fsdp2": {
                    "enabled": bool(any(self.fsdp_enabled_per_stage)),
                    "enabled_per_stage": self._expand_virtual(self.fsdp_enabled_per_stage),
                    "reshard_after_forward_per_stage": self._expand_virtual(
                        self.reshard_after_forward_per_stage
                    ),
                },
                "recompute": {
                    "per_stage": self._expand_virtual(self.recompute_per_stage),
                },
            },
            "planner": {
                "selected_summary": {
                    "score": float(self.score),
                    "stage_costs": [float(x) for x in self.stage_costs],
                    "stage_times": [float(x) for x in self.stage_times],
                    "reasons": list(self.reasons),
                }
            },
        }

    def to_hybrid_policy(self, *, metadata: Optional[Dict[str, Any]] = None) -> HybridParallelPolicy:
        return HybridParallelPolicy.from_plan_candidate(self, metadata=metadata)


def _extract_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find transformer layers (model.layers / transformer.h)")


def _layer_param_bytes_from_meta(model: nn.Module) -> List[int]:
    layers = _extract_transformer_layers(model)
    out: List[int] = []
    for layer in layers:
        total = 0
        for p in layer.parameters(recurse=True):
            try:
                total += int(p.numel()) * int(p.element_size())
            except Exception:
                total += int(p.numel()) * 2
        out.append(int(total))
    return out


def _normalize_nodes(
    *,
    cluster_cfg: Dict[str, Any],
    world_size: int,
) -> List[HardwareNode]:
    raw_nodes = cluster_cfg.get("nodes") or []
    if not raw_nodes:
        return [
            HardwareNode(
                name="node0",
                rank_start=0,
                gpus=int(world_size),
                mem_gb=24.0,
                perf=1.0,
            )
        ]

    nodes: List[HardwareNode] = []
    rank_start = 0
    for idx, item in enumerate(raw_nodes):
        gpus = int(item.get("gpus") or 0)
        if gpus <= 0:
            raise ValueError(f"cluster.nodes[{idx}].gpus must be > 0")
        nodes.append(
            HardwareNode(
                name=str(item.get("name") or f"node{idx}"),
                rank_start=rank_start,
                gpus=gpus,
                mem_gb=float(item.get("mem_gb") or 24.0),
                perf=float(item.get("perf") or 1.0),
            )
        )
        rank_start += gpus

    if rank_start != int(world_size):
        raise ValueError(
            f"cluster node gpu counts sum to {rank_start}, expected world_size={world_size}"
        )
    return nodes


def _boundary_positions(
    n_layers: int,
    align_to: int,
    min_layers_per_stage: int,
) -> List[int]:
    align = max(1, int(align_to))
    min_layers = max(1, int(min_layers_per_stage))
    positions = sorted(
        {
            cut
            for cut in range(min_layers, n_layers)
            if cut % align == 0 or cut == n_layers - min_layers
        }
    )
    if not positions:
        positions = list(range(min_layers, n_layers))
    return positions


def _solve_contiguous_partition(
    *,
    layer_costs: Sequence[float],
    stage_perf: Sequence[float],
    first_stage_extra: float,
    last_stage_extra: float,
    align_to: int,
    min_layers_per_stage: int,
) -> Tuple[List[List[int]], List[float], List[float]]:
    n_layers = len(layer_costs)
    n_stages = len(stage_perf)
    if n_stages <= 0:
        raise ValueError("n_stages must be > 0")
    if n_layers < n_stages:
        raise ValueError("number of stages exceeds number of layers")

    prefix = [0.0]
    for cost in layer_costs:
        prefix.append(prefix[-1] + float(cost))

    def raw_sum(start: int, end: int) -> float:
        total = prefix[end + 1] - prefix[start]
        if start == 0:
            total += float(first_stage_extra)
        if end == n_layers - 1:
            total += float(last_stage_extra)
        return total

    def normalized_cost(start: int, end: int, stage_idx: int) -> float:
        return raw_sum(start, end) / max(1e-6, float(stage_perf[stage_idx]))

    valid_cuts = set(
        _boundary_positions(
            n_layers=n_layers,
            align_to=align_to,
            min_layers_per_stage=min_layers_per_stage,
        )
    )

    inf = float("inf")
    dp = [[inf for _ in range(n_layers)] for _ in range(n_stages)]
    prev = [[-1 for _ in range(n_layers)] for _ in range(n_stages)]

    for end in range(n_layers):
        if end + 1 >= min_layers_per_stage:
            dp[0][end] = normalized_cost(0, end, 0)

    for stage_idx in range(1, n_stages):
        remain = n_stages - stage_idx - 1
        for end in range(stage_idx, n_layers):
            best_cost = inf
            best_split = -1
            split_min = stage_idx * min_layers_per_stage - 1
            split_max = end - min_layers_per_stage
            for split in range(split_min, split_max + 1):
                if split < 0:
                    continue
                if split != split_min and split + 1 not in valid_cuts:
                    continue
                if n_layers - (split + 1) < remain * min_layers_per_stage:
                    continue
                candidate = max(
                    dp[stage_idx - 1][split],
                    normalized_cost(split + 1, end, stage_idx),
                )
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = split
            dp[stage_idx][end] = best_cost
            prev[stage_idx][end] = best_split

    end = n_layers - 1
    stage_ranges: List[List[int]] = []
    for stage_idx in range(n_stages - 1, -1, -1):
        split = prev[stage_idx][end]
        start = split + 1
        stage_ranges.append([int(start), int(end)])
        end = split
    stage_ranges.reverse()

    stage_costs = [raw_sum(start, end) for start, end in stage_ranges]
    stage_times = [
        float(stage_costs[idx]) / max(1e-6, float(stage_perf[idx]))
        for idx in range(n_stages)
    ]
    return stage_ranges, stage_costs, stage_times


def _build_stage_slots_for_nodes(
    *,
    nodes: Sequence[HardwareNode],
    stage_world_size: int,
    pp_degree: int,
    prefer_contiguous: bool,
) -> Optional[List[int]]:
    if stage_world_size <= 0:
        return None
    slots: List[int] = []
    for node_idx, node in enumerate(nodes):
        if node.gpus % stage_world_size != 0:
            return None
        slots.extend([node_idx] * (node.gpus // stage_world_size))
    if len(slots) != int(pp_degree):
        return None
    if prefer_contiguous:
        return slots
    reordered: List[int] = []
    left = 0
    right = len(slots) - 1
    flip = False
    while left <= right:
        if flip:
            reordered.append(slots[right])
            right -= 1
        else:
            reordered.append(slots[left])
            left += 1
        flip = not flip
    return reordered


def _mesh_from_stage_slots(
    *,
    nodes: Sequence[HardwareNode],
    stage_slots: Sequence[int],
    dp_degree: int,
    tp_degree: int,
) -> List[List[List[int]]]:
    stage_world_size = int(dp_degree) * int(tp_degree)
    mesh: List[List[List[int]]] = []
    node_offsets = {idx: 0 for idx in range(len(nodes))}
    for node_idx in stage_slots:
        node = nodes[int(node_idx)]
        offset = node_offsets[int(node_idx)]
        stage_ranks = list(
            range(
                int(node.rank_start) + int(offset),
                int(node.rank_start) + int(offset) + int(stage_world_size),
            )
        )
        node_offsets[int(node_idx)] += int(stage_world_size)
        mesh.append(
            [
                stage_ranks[dp_idx * int(tp_degree) : (dp_idx + 1) * int(tp_degree)]
                for dp_idx in range(int(dp_degree))
            ]
        )
    return mesh


def _schedule_bubble_factor(schedule: str, *, num_virtual: int, microbatches: int) -> float:
    mb = max(1, int(microbatches))
    if schedule == "gpipe":
        return 1.0 + max(0, num_virtual - 1) / float(mb)
    if schedule == "1f1b":
        return 1.0 + max(0, num_virtual - 1) / float(2 * mb)
    if schedule == "interleaved1f1b":
        return 1.0 + max(0, num_virtual - 1) / float(3 * mb)
    return 1.0 + max(0, num_virtual - 1) / float(mb)


def _infer_stage_policies(
    *,
    stage_costs: Sequence[float],
    stage_times: Sequence[float],
    stage_to_node: Sequence[HardwareNode],
    dp_degree: int,
    planner_cfg: Dict[str, Any],
) -> Tuple[List[bool], List[bool], List[str]]:
    fsdp_pref = bool((planner_cfg.get("policy") or {}).get("fsdp_enabled", True))
    recompute_threshold = float(
        ((planner_cfg.get("policy") or {}).get("recompute_pressure_threshold") or 0.82)
    )
    reshard_threshold = float(
        ((planner_cfg.get("policy") or {}).get("reshard_pressure_threshold") or 0.90)
    )
    max_stage_time = max(float(x) for x in stage_times) if stage_times else 1.0

    fsdp_enabled_per_stage: List[bool] = []
    reshard_per_stage: List[bool] = []
    recompute_per_stage: List[str] = []

    for idx, node in enumerate(stage_to_node):
        mem_bytes = float(node.mem_gb) * (1024.0**3)
        pressure = float(stage_costs[idx]) / max(1.0, mem_bytes)
        slowdown = float(stage_times[idx]) / max(1e-6, max_stage_time)

        fsdp_enabled = bool(fsdp_pref and int(dp_degree) > 1)
        reshard_enabled = bool(fsdp_enabled and pressure >= reshard_threshold)
        recompute_policy = "full" if (pressure >= recompute_threshold and slowdown > 0.75) else "none"

        fsdp_enabled_per_stage.append(fsdp_enabled)
        reshard_per_stage.append(reshard_enabled)
        recompute_per_stage.append(recompute_policy)

    return fsdp_enabled_per_stage, reshard_per_stage, recompute_per_stage


def _score_candidate(
    *,
    schedule: str,
    microbatches: int,
    stage_times: Sequence[float],
    stage_slots: Sequence[int],
    seq_len: int,
    hidden_size: int,
    global_batch_size: int,
    dp_degree: int,
    tp_degree: int,
    vpp: int,
) -> Tuple[float, List[str]]:
    num_virtual = len(stage_times) * max(1, int(vpp))
    bubble = _schedule_bubble_factor(
        schedule,
        num_virtual=num_virtual,
        microbatches=microbatches,
    )
    max_stage_time = max(float(x) for x in stage_times) if stage_times else 0.0
    mean_stage_time = (
        sum(float(x) for x in stage_times) / max(1, len(stage_times))
        if stage_times
        else 0.0
    )
    imbalance = max_stage_time / max(1e-6, mean_stage_time) if mean_stage_time > 0 else 1.0
    cross_node_edges = sum(
        1 for left, right in zip(stage_slots[:-1], stage_slots[1:]) if int(left) != int(right)
    )
    per_dp_batch = int(math.ceil(global_batch_size / float(max(1, dp_degree))))
    activation_mb = (
        float(per_dp_batch)
        * float(seq_len)
        * float(hidden_size)
        * 2.0
        / float(1024**2)
    )
    comm_penalty = float(cross_node_edges) * activation_mb / max(1.0, float(microbatches))
    layout_penalty = 0.0
    if int(tp_degree) > 1 and int(vpp) > 1:
        layout_penalty += 0.06 * float(len(stage_times))
    if schedule == "interleaved1f1b" and int(vpp) <= 1:
        layout_penalty += 0.25

    score = max_stage_time * bubble * (1.0 + 0.15 * (imbalance - 1.0))
    score += 0.02 * comm_penalty
    score += float(layout_penalty)

    reasons = [
        f"bubble={bubble:.3f}",
        f"imbalance={imbalance:.3f}",
        f"cross_node_edges={cross_node_edges}",
        f"act_penalty_mb={comm_penalty:.1f}",
    ]
    return float(score), reasons


def _candidate_search_space(
    *,
    planner_cfg: Dict[str, Any],
    world_size: int,
) -> Dict[str, List[int] | List[str]]:
    search_cfg = planner_cfg.get("search") or {}
    pp_values = [int(x) for x in (search_cfg.get("pp_degrees") or [2])]
    tp_values = [int(x) for x in (search_cfg.get("tp_degrees") or [2])]
    vpp_values = [int(x) for x in (search_cfg.get("vpp_values") or [1])]
    microbatch_values = [
        int(x) for x in (search_cfg.get("microbatch_values") or [1, 2, 4, 8])
    ]
    schedule_values = [
        str(x).lower()
        for x in (search_cfg.get("schedules") or ["1f1b", "gpipe", "interleaved1f1b"])
    ]
    return {
        "pp": [x for x in pp_values if x > 0 and world_size % x == 0],
        "tp": [x for x in tp_values if x > 0],
        "vpp": [x for x in vpp_values if x > 0],
        "microbatches": [x for x in microbatch_values if x > 0],
        "schedules": schedule_values,
    }


def build_execution_plans(
    *,
    cfg: Dict[str, Any],
    model: nn.Module,
    world_size: int,
) -> List[PlanCandidate]:
    planner_cfg = cfg.get("planner") or {}
    cluster_cfg = cfg.get("cluster") or {}
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    search_space = _candidate_search_space(planner_cfg=planner_cfg, world_size=world_size)
    nodes = _normalize_nodes(cluster_cfg=cluster_cfg, world_size=world_size)

    layer_costs = [float(x) for x in _layer_param_bytes_from_meta(model)]
    hidden_size = int(
        getattr(getattr(model, "config", None), "hidden_size", 0)
        or getattr(getattr(model, "config", None), "n_embd", 0)
        or 0
    )
    seq_len = int(
        model_cfg.get("seq_len")
        or getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
        or 1024
    )
    global_batch_size = int(train_cfg.get("global_batch_size") or 8)

    embed_weight = float(
        ((planner_cfg.get("cost_model") or {}).get("embed_extra_layers") or 1.4)
    )
    head_weight = float(
        ((planner_cfg.get("cost_model") or {}).get("lm_head_extra_layers") or 1.8)
    )
    layer_mean = sum(layer_costs) / max(1, len(layer_costs))
    first_stage_extra = embed_weight * layer_mean
    last_stage_extra = head_weight * layer_mean

    align_to = int(((planner_cfg.get("constraints") or {}).get("align_layers_to") or 2))
    min_layers_per_stage = int(
        ((planner_cfg.get("constraints") or {}).get("min_layers_per_stage") or 1)
    )
    prefer_contiguous = bool(
        ((planner_cfg.get("constraints") or {}).get("prefer_contiguous_nodes") or True)
    )

    candidates: List[PlanCandidate] = []
    for pp_degree in search_space["pp"]:
        ranks_per_stage = world_size // int(pp_degree)
        for tp_degree in search_space["tp"]:
            if ranks_per_stage % int(tp_degree) != 0:
                continue
            dp_degree = ranks_per_stage // int(tp_degree)
            stage_slots = _build_stage_slots_for_nodes(
                nodes=nodes,
                stage_world_size=ranks_per_stage,
                pp_degree=int(pp_degree),
                prefer_contiguous=prefer_contiguous,
            )
            if stage_slots is None:
                continue

            stage_perf = [nodes[idx].perf for idx in stage_slots]
            stage_ranges, stage_costs, stage_times = _solve_contiguous_partition(
                layer_costs=layer_costs,
                stage_perf=stage_perf,
                first_stage_extra=first_stage_extra,
                last_stage_extra=last_stage_extra,
                align_to=align_to,
                min_layers_per_stage=min_layers_per_stage,
            )
            mesh = _mesh_from_stage_slots(
                nodes=nodes,
                stage_slots=stage_slots,
                dp_degree=dp_degree,
                tp_degree=int(tp_degree),
            )
            stage_nodes = [nodes[idx] for idx in stage_slots]
            fsdp_enabled_per_stage, reshard_per_stage, recompute_per_stage = _infer_stage_policies(
                stage_costs=stage_costs,
                stage_times=stage_times,
                stage_to_node=stage_nodes,
                dp_degree=dp_degree,
                planner_cfg=planner_cfg,
            )

            for vpp in search_space["vpp"]:
                for microbatches in search_space["microbatches"]:
                    if microbatches > int(math.ceil(global_batch_size / float(max(1, dp_degree)))):
                        continue
                    for schedule in search_space["schedules"]:
                        if schedule == "interleaved1f1b" and int(vpp) <= 1:
                            continue
                        if schedule in {"1f1b", "interleaved1f1b"} and microbatches < int(pp_degree) * int(vpp):
                            continue
                        score, reasons = _score_candidate(
                            schedule=schedule,
                            microbatches=microbatches,
                            stage_times=stage_times,
                            stage_slots=stage_slots,
                            seq_len=seq_len,
                            hidden_size=hidden_size,
                            global_batch_size=global_batch_size,
                            dp_degree=dp_degree,
                            tp_degree=int(tp_degree),
                            vpp=int(vpp),
                        )
                        candidates.append(
                            PlanCandidate(
                                pp_degree=int(pp_degree),
                                tp_degree=int(tp_degree),
                                dp_degree=int(dp_degree),
                                vpp=int(vpp),
                                microbatches=int(microbatches),
                                schedule=str(schedule),
                                stage_ranges=copy.deepcopy(stage_ranges),
                                stage_to_node=[node.name for node in stage_nodes],
                                mesh=copy.deepcopy(mesh),
                                score=float(score),
                                stage_costs=[float(x) for x in stage_costs],
                                stage_times=[float(x) for x in stage_times],
                                fsdp_enabled_per_stage=list(fsdp_enabled_per_stage),
                                reshard_after_forward_per_stage=list(reshard_per_stage),
                                recompute_per_stage=list(recompute_per_stage),
                                reasons=list(reasons),
                            )
                        )

    candidates.sort(key=lambda item: item.score)
    top_k = int(((planner_cfg.get("selection") or {}).get("top_k") or 8))
    return candidates[: max(1, top_k)]


def select_execution_plan(
    *,
    cfg: Dict[str, Any],
    model: nn.Module,
    world_size: int,
) -> Tuple[PlanCandidate, List[PlanCandidate]]:
    candidates = build_execution_plans(cfg=cfg, model=model, world_size=world_size)
    selection_cfg = (cfg.get("planner") or {}).get("selection") or {}
    candidate_index = int(selection_cfg.get("candidate_index") or 0)
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError(
            f"planner.selection.candidate_index={candidate_index} is out of range for {len(candidates)} candidates"
        )
    return candidates[candidate_index], candidates


def summarize_candidates(candidates: Sequence[PlanCandidate]) -> str:
    rows = []
    for idx, item in enumerate(candidates):
        rows.append(
            {
                "idx": idx,
                "score": round(float(item.score), 4),
                "pp": int(item.pp_degree),
                "tp": int(item.tp_degree),
                "dp": int(item.dp_degree),
                "vpp": int(item.vpp),
                "mb": int(item.microbatches),
                "schedule": item.schedule,
                "stages": copy.deepcopy(item.stage_ranges),
                "nodes": list(item.stage_to_node),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def export_candidates(
    *,
    candidates: Sequence[PlanCandidate],
    path: str,
) -> None:
    payload = [asdict(item) for item in candidates]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
