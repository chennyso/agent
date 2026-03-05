from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fsdp_agent.config import Fsdp2Layout, Fsdp2Strategy, ParallelSpec


def load_strategy_catalog() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "rag" / "strategy_catalog.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _select_family(catalog: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    families = catalog.get("families") or []
    hint = str(model_name or "").lower()
    for fam in families:
        family_id = str(fam.get("family_id") or "")
        aliases = [family_id] + list(fam.get("aliases") or [])
        for alias in aliases:
            if alias and str(alias).lower() in hint:
                return fam
    return None


def _pick_degrees(values: List[Any], *, cap: Optional[int], max_count: int) -> List[int]:
    cleaned = sorted({int(v) for v in values if int(v) >= 1})
    if cap is not None and int(cap) > 0:
        cleaned = [v for v in cleaned if v <= int(cap)]
    if not cleaned:
        return [1]
    selected = cleaned[-max(int(max_count), 1) :]
    if 1 not in selected:
        selected = [1] + selected
    return selected


def _ensure_degree(values: List[int], base_val: int) -> List[int]:
    out = list(values)
    if base_val > 0 and base_val not in out:
        out.insert(0, int(base_val))
    return out


def _collect_named_targets(
    semantic_state: Dict[str, Any],
    *,
    keep_embeddings: bool,
    keep_lm_head: bool,
) -> List[str]:
    if not keep_embeddings and not keep_lm_head:
        return []
    anatomy = semantic_state.get("model_anatomy") or {}
    comm = anatomy.get("comm_hotspots") or {}
    allowed_keys = list(comm.get("named_override_keys") or [])
    allowed_paths = list(comm.get("paths") or [])
    if not allowed_keys and not allowed_paths:
        return []
    candidates = list(dict.fromkeys(allowed_paths + allowed_keys))
    hits: List[str] = []
    for item in candidates:
        low = str(item).lower()
        if keep_embeddings and "embed" in low:
            hits.append(str(item))
            continue
        if keep_lm_head and ("lm_head" in low or "lmhead" in low):
            hits.append(str(item))
            continue
    return sorted(set(hits))


def _build_keep_dp_overrides(
    base: Fsdp2Strategy,
    semantic_state: Dict[str, Any],
    *,
    keep_embeddings: bool,
    keep_lm_head: bool,
) -> Tuple[Dict[str, Fsdp2Layout], List[str]]:
    targets = _collect_named_targets(
        semantic_state,
        keep_embeddings=keep_embeddings,
        keep_lm_head=keep_lm_head,
    )
    if not targets:
        return {}, []
    layout = Fsdp2Layout(**asdict(base.global_layout))
    layout.sharding_strategy = "NO"
    overrides = dict(base.named_overrides)
    for name in targets:
        overrides[str(name)] = layout
    return overrides, targets


def _clone_strategy(base: Fsdp2Strategy) -> Fsdp2Strategy:
    return Fsdp2Strategy.from_dict(base.to_dict())


def _suggest_pp_microbatches(
    *,
    pp_degree: int,
    base_microbatches: int,
    global_batch_size: Optional[int],
    world_size: int,
    tp_degree: int,
    ep_degree: int,
    cp_degree: int,
) -> int:
    if int(pp_degree) <= 1:
        return 1
    target = max(int(base_microbatches), 4 * int(pp_degree))
    product = max(int(tp_degree), 1) * max(int(pp_degree), 1) * max(int(ep_degree), 1) * max(int(cp_degree), 1)
    dp_world = int(world_size) // product if product > 0 and int(world_size) % product == 0 else int(world_size)
    per_rank_batch = int(global_batch_size) if global_batch_size is not None else 0
    if dp_world > 0 and per_rank_batch > 0:
        per_rank_batch = int((per_rank_batch + dp_world - 1) / dp_world)
    if per_rank_batch <= 0:
        per_rank_batch = 1
    return max(1, min(int(target), int(per_rank_batch)))


def build_catalog_candidates(
    base: Fsdp2Strategy,
    *,
    model_name: str,
    semantic_state: Dict[str, Any],
    num_layers_hint: Optional[int],
    gpus_per_node: Optional[int],
) -> List[Tuple[str, Fsdp2Strategy, str]]:
    catalog = load_strategy_catalog()
    if not catalog:
        return []
    family = _select_family(catalog, model_name)
    if not family:
        return []

    parallel_space = semantic_state.get("parallel_search_space") or {}
    degrees = list(parallel_space.get("degree_candidates") or [])
    world_size = int(parallel_space.get("world_size") or 1)
    train_hyper = semantic_state.get("train_hyper") or {}
    global_batch_size = train_hyper.get("global_batch_size")

    tp_candidates = _pick_degrees(degrees, cap=gpus_per_node, max_count=2)
    pp_candidates = _pick_degrees(degrees, cap=num_layers_hint, max_count=2)
    base_p = getattr(base, "parallel", ParallelSpec())
    tp_candidates = _ensure_degree(tp_candidates, int(getattr(base_p, "tp_degree", 1) or 1))
    pp_candidates = _ensure_degree(pp_candidates, int(getattr(base_p, "pp_degree", 1) or 1))

    def _can_factor(tp: int, pp: int, ep: int, cp: int) -> bool:
        product = max(int(tp), 1) * max(int(pp), 1) * max(int(ep), 1) * max(int(cp), 1)
        return int(world_size) % product == 0

    family_id = str(family.get("family_id") or "unknown")
    out: List[Tuple[str, Fsdp2Strategy, str]] = []
    for tpl in family.get("strategies") or []:
        strategy_id = str(tpl.get("strategy_id") or "strategy")
        description = str(tpl.get("description") or "")
        tp_plan = tpl.get("tp_plan")
        pp_degree = tpl.get("pp_degree")
        sp_enabled = bool(tpl.get("sp_enabled", False))
        pp_schedule = tpl.get("pp_schedule")
        keep_embeddings_dp = bool(tpl.get("keep_embeddings_dp", False))
        keep_lm_head_dp = bool(tpl.get("keep_lm_head_dp", False))

        if tp_plan:
            for tp in tp_candidates:
                if int(tp) <= 1:
                    continue
                if sp_enabled and int(tp) <= 1:
                    continue
                if not _can_factor(int(tp), int(base_p.pp_degree), int(base_p.ep_degree), int(base_p.cp_degree)):
                    continue
                spec = ParallelSpec(**asdict(base_p))
                spec.tp_degree = int(tp)
                spec.tp_plan = str(tp_plan)
                spec.sp_enabled = bool(sp_enabled)
                if tpl.get("tp_head_grouping") is not None:
                    spec.tp_head_grouping = tpl.get("tp_head_grouping")
                strat = _clone_strategy(base)
                strat.parallel = spec
                note = f"catalog:{family_id}:{strategy_id} {description}".strip()
                if keep_embeddings_dp or keep_lm_head_dp:
                    overrides, targets = _build_keep_dp_overrides(
                        strat,
                        semantic_state,
                        keep_embeddings=keep_embeddings_dp,
                        keep_lm_head=keep_lm_head_dp,
                    )
                    if overrides:
                        strat.named_overrides = overrides
                    else:
                        note = f"{note} (keep_dp_targets_missing)"
                name = f"catalog_{family_id}_{strategy_id}_tp{int(tp)}"
                out.append((name, strat, note))

        if pp_degree:
            if str(pp_degree).lower() == "auto":
                pp_choices = [v for v in pp_candidates if int(v) > 1]
            else:
                try:
                    pp_choices = [int(pp_degree)]
                except Exception:
                    pp_choices = []
            for pp in pp_choices:
                if int(pp) <= 1:
                    continue
                if num_layers_hint is not None and int(pp) > int(num_layers_hint):
                    continue
                if not _can_factor(int(base_p.tp_degree), int(pp), int(base_p.ep_degree), int(base_p.cp_degree)):
                    continue
                spec = ParallelSpec(**asdict(base_p))
                spec.pp_degree = int(pp)
                if pp_schedule:
                    spec.pp_schedule = str(pp_schedule)
                spec.pp_microbatches = _suggest_pp_microbatches(
                    pp_degree=int(pp),
                    base_microbatches=int(getattr(base_p, "pp_microbatches", 1) or 1),
                    global_batch_size=int(global_batch_size) if global_batch_size is not None else None,
                    world_size=world_size,
                    tp_degree=int(base_p.tp_degree),
                    ep_degree=int(base_p.ep_degree),
                    cp_degree=int(base_p.cp_degree),
                )
                strat = _clone_strategy(base)
                strat.parallel = spec
                note = f"catalog:{family_id}:{strategy_id} {description}".strip()
                name = f"catalog_{family_id}_{strategy_id}_pp{int(pp)}"
                out.append((name, strat, note))

    return out[:20]
