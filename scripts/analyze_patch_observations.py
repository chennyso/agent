from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(median(list(values)))


def _p75(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    index = min(max(int(round((len(ordered) - 1) * 0.75)), 0), len(ordered) - 1)
    return float(ordered[index])


def _discover_summary_paths(inputs: Sequence[str]) -> List[Path]:
    discovered: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.name == "summary_megatron.json":
            discovered.append(path)
            continue
        if path.is_dir():
            direct = path / "summary_megatron.json"
            if direct.exists():
                discovered.append(direct)
                continue
            discovered.extend(sorted(path.rglob("summary_megatron.json")))
    seen = set()
    unique_paths: List[Path] = []
    for path in discovered:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _scenario_label(summary: Dict[str, Any]) -> str:
    signature = dict(summary.get("bottleneck_signature") or {})
    return (
        str(signature.get("canonical_dominant_label") or "").strip()
        or str(signature.get("dominant_label") or "").strip()
        or "unknown"
    )


def _baseline_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline = dict(summary.get("baseline_metrics") or {})
    if baseline:
        return baseline
    for trial in list(summary.get("tested_trials") or []):
        record = dict(trial or {})
        if str(record.get("config_name") or "") == "baseline":
            return record
    return {}


def _baseline_family(summary: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> str:
    family = dict(summary.get("baseline_family") or {})
    token = str(family.get("runtime_schedule_family") or family.get("name") or "").strip()
    if token:
        return token
    artifact = dict(baseline_metrics.get("trial_artifact") or {})
    return str(artifact.get("schedule_template") or "").strip() or "baseline"


def _candidate_family(record: Dict[str, Any]) -> str:
    artifact = dict(record.get("trial_artifact") or {})
    token = (
        str(artifact.get("schedule_template") or "").strip()
        or str(((record.get("family") or {}).get("runtime_schedule_family") or "")).strip()
        or str(((record.get("family") or {}).get("name") or "")).strip()
    )
    return token or "unknown"


def _bottleneck_label(record: Dict[str, Any], summary: Dict[str, Any]) -> str:
    signature = dict(record.get("bottleneck_signature") or {})
    return (
        str(signature.get("canonical_dominant_label") or "").strip()
        or str(signature.get("dominant_label") or "").strip()
        or _scenario_label(summary)
    )


def _trial_metric(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
    trace_summary = dict(record.get("trace_summary") or {})
    trial_artifact = dict(record.get("trial_artifact") or {})
    decomposition = dict(trial_artifact.get("decomposition") or {})
    return _safe_float(
        trace_summary.get(key, decomposition.get(key, record.get(key, default))),
        default,
    )


def _successful_trial(record: Dict[str, Any]) -> bool:
    return _safe_int(record.get("returncode"), 1) == 0 and not _safe_bool(record.get("oom"))


def _harmful_trial(record: Dict[str, Any], step_gain_ratio: float, throughput_gain_ratio: float) -> bool:
    if not _successful_trial(record):
        return True
    return throughput_gain_ratio < -0.01 or step_gain_ratio < -0.01


def _trial_analysis_paths(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record.get("analysis_artifact_paths") or {})


def _preferred_visual_path(paths: Dict[str, Any]) -> str:
    for key in ("compare_pipeline", "compare_pipeline_svg", "pipeline_projection", "pipeline_projection_svg"):
        token = str(paths.get(key) or "").strip()
        if token:
            return token
    return ""


def collect_patch_observations(inputs: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries = _discover_summary_paths(inputs)
    rows: List[Dict[str, Any]] = []
    run_metadata: List[Dict[str, Any]] = []
    for summary_path in summaries:
        summary = _load_json(summary_path)
        run_id = summary_path.parent.name
        baseline = _baseline_metrics(summary)
        baseline_step = _safe_float(
            dict(baseline.get("trace_summary") or {}).get("steady_state_step_time_ms_p50")
            or baseline.get("step_time_ms_p50"),
            0.0,
        )
        baseline_tp = _safe_float(
            baseline.get("throughput_tokens_per_s") or baseline.get("throughput_effective_tokens_per_s"),
            0.0,
        )
        run_metadata.append(
            {
                "run_id": run_id,
                "summary_path": str(summary_path),
                "scenario_label": _scenario_label(summary),
                "baseline_family": _baseline_family(summary, baseline),
                "search_unit": str(summary.get("search_unit") or "patch"),
                "patch_memory_enabled": bool(summary.get("patch_memory_enabled", True)),
            }
        )
        for trial in list(summary.get("tested_trials") or []):
            record = dict(trial or {})
            trial_id = _safe_int(record.get("trial_id"), -1)
            step_time_ms = _safe_float(
                dict(record.get("trace_summary") or {}).get("steady_state_step_time_ms_p50")
                or record.get("step_time_ms_p50"),
                0.0,
            )
            throughput = _safe_float(
                record.get("throughput_tokens_per_s") or record.get("throughput_effective_tokens_per_s"),
                0.0,
            )
            step_gain_ratio = ((baseline_step - step_time_ms) / baseline_step) if baseline_step > 0.0 and step_time_ms > 0.0 else 0.0
            throughput_gain_ratio = ((throughput - baseline_tp) / baseline_tp) if baseline_tp > 0.0 and throughput > 0.0 else 0.0
            row = {
                "run_id": run_id,
                "trial_id": trial_id,
                "trial_index": trial_id,
                "config_name": str(record.get("config_name") or ""),
                "returncode": _safe_int(record.get("returncode"), 1),
                "oom": _safe_bool(record.get("oom")),
                "search_unit": str(record.get("search_unit") or summary.get("search_unit") or "patch"),
                "patch_memory_enabled": bool(record.get("patch_memory_enabled", summary.get("patch_memory_enabled", True))),
                "patch_family": str(record.get("patch_family") or dict(record.get("trial_artifact") or {}).get("patch_family") or ""),
                "patch_category": str(record.get("patch_category") or dict(record.get("trial_artifact") or {}).get("patch_category") or ""),
                "patch_count": _safe_int(record.get("patch_count"), _safe_int(dict(record.get("trial_artifact") or {}).get("patch_count"), 0)),
                "baseline_family": _baseline_family(summary, baseline),
                "candidate_family": _candidate_family(record),
                "bottleneck_label": _bottleneck_label(record, summary),
                "scenario_label": _scenario_label(summary),
                "step_time_ms": round(step_time_ms, 4),
                "throughput_tokens_per_s": round(throughput, 4),
                "step_gain_ratio": round(step_gain_ratio, 6),
                "throughput_gain_ratio": round(throughput_gain_ratio, 6),
                "bubble_ratio": round(_trial_metric(record, "bubble_ratio"), 6),
                "stage_skew_ratio": round(_trial_metric(record, "stage_load_variance"), 6),
                "memory_skew_ratio": round(_trial_metric(record, "mem_skew_ratio"), 6),
                "tail_ratio": round(_trial_metric(record, "stage_tail_ratio"), 6),
                "optimizer_exposed_ratio": round(_trial_metric(record, "optimizer_exposed_ratio"), 6),
                "success": _successful_trial(record),
                "harmful": False,
                "summary_path": str(summary_path),
                "analysis_artifact_paths": _trial_analysis_paths(record),
                "baseline_analysis_artifact_paths": _trial_analysis_paths(baseline),
            }
            row["harmful"] = _harmful_trial(record, step_gain_ratio, throughput_gain_ratio)
            rows.append(row)
    return rows, run_metadata


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _aggregate_bottleneck_success(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _safe_int(row.get("patch_count"), 0) <= 0:
            continue
        key = (
            str(row.get("bottleneck_label") or ""),
            str(row.get("patch_family") or ""),
            str(row.get("patch_category") or ""),
        )
        grouped[key].append(row)
    output: List[Dict[str, Any]] = []
    for (bottleneck_label, patch_family, patch_category), items in sorted(grouped.items()):
        attempts = len(items)
        successes = sum(1 for item in items if bool(item.get("success")) and not bool(item.get("harmful")))
        harmful = sum(1 for item in items if bool(item.get("harmful")))
        output.append(
            {
                "bottleneck_label": bottleneck_label,
                "patch_family": patch_family,
                "patch_category": patch_category,
                "attempts": attempts,
                "successes": successes,
                "harmful": harmful,
                "success_rate": round(successes / float(attempts), 6) if attempts else 0.0,
                "harmful_rate": round(harmful / float(attempts), 6) if attempts else 0.0,
            }
        )
    return output


def _aggregate_bottleneck_gain(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _safe_int(row.get("patch_count"), 0) <= 0:
            continue
        key = (
            str(row.get("bottleneck_label") or ""),
            str(row.get("patch_family") or ""),
            str(row.get("patch_category") or ""),
        )
        grouped[key].append(row)
    output: List[Dict[str, Any]] = []
    for (bottleneck_label, patch_family, patch_category), items in sorted(grouped.items()):
        step_gains = [_safe_float(item.get("step_gain_ratio")) for item in items if bool(item.get("success"))]
        throughput_gains = [_safe_float(item.get("throughput_gain_ratio")) for item in items if bool(item.get("success"))]
        output.append(
            {
                "bottleneck_label": bottleneck_label,
                "patch_family": patch_family,
                "patch_category": patch_category,
                "median_step_gain_ratio": round(_median(step_gains), 6),
                "median_throughput_gain_ratio": round(_median(throughput_gains), 6),
                "p75_throughput_gain_ratio": round(_p75(throughput_gains), 6),
            }
        )
    return output


def _aggregate_search_ablation(rows: List[Dict[str, Any]], run_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    meta_by_run = {str(item.get("run_id") or ""): dict(item) for item in run_metadata}
    for row in rows:
        if str(row.get("config_name") or "") == "baseline":
            continue
        key = (
            str(row.get("run_id") or ""),
            str(row.get("search_unit") or "patch"),
            str(bool(row.get("patch_memory_enabled"))),
        )
        grouped[key].append(row)
    output: List[Dict[str, Any]] = []
    for (run_id, search_unit, patch_memory_enabled), items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda item: _safe_int(item.get("trial_index"), 0))
        successful = [item for item in sorted_items if bool(item.get("success"))]
        best_throughput = max((_safe_float(item.get("throughput_tokens_per_s")) for item in successful), default=0.0)
        near_best_threshold = 0.98 * best_throughput if best_throughput > 0.0 else 0.0
        best_so_far = 0.0
        trials_to_near_best = 0
        for index, item in enumerate(sorted_items, start=1):
            best_so_far = max(best_so_far, _safe_float(item.get("throughput_tokens_per_s")))
            if near_best_threshold > 0.0 and best_so_far >= near_best_threshold:
                trials_to_near_best = index
                break
        best_step_time = min((_safe_float(item.get("step_time_ms")) for item in successful if _safe_float(item.get("step_time_ms")) > 0.0), default=0.0)
        harmful_ratio = (
            sum(1 for item in sorted_items if bool(item.get("harmful"))) / float(len(sorted_items))
            if sorted_items
            else 0.0
        )
        output.append(
            {
                "run_id": run_id,
                "scenario_label": str(meta_by_run.get(run_id, {}).get("scenario_label") or "unknown"),
                "search_unit": search_unit,
                "patch_memory_enabled": _safe_bool(patch_memory_enabled),
                "trial_count": len(sorted_items),
                "harmful_trial_ratio": round(harmful_ratio, 6),
                "best_step_time_ms": round(best_step_time, 4),
                "best_throughput_tokens_per_s": round(best_throughput, 4),
                "trials_to_near_best": int(trials_to_near_best or 0),
                "near_best_threshold": round(near_best_threshold, 4),
            }
        )
    return output


def _select_case_study(rows: List[Dict[str, Any]], label: str, topk: int) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("bottleneck_label") or "") == label
        and bool(row.get("success"))
        and _safe_int(row.get("patch_count"), 0) > 0
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: (_safe_float(item.get("throughput_gain_ratio")), _safe_float(item.get("step_gain_ratio"))), reverse=True)
    inspected = ranked[: max(int(topk), 1)]
    for row in inspected:
        if _preferred_visual_path(dict(row.get("analysis_artifact_paths") or {})):
            return row
    return ranked[0]


def _build_case_study_manifest(rows: List[Dict[str, Any]], *, case_study_topk: int) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for label in ("bubble_bound", "memory_bound"):
        selected = _select_case_study(rows, label, case_study_topk)
        if selected is None:
            continue
        baseline_paths = dict(selected.get("baseline_analysis_artifact_paths") or {})
        candidate_paths = dict(selected.get("analysis_artifact_paths") or {})
        cases.append(
            {
                "scenario_label": label,
                "run_id": str(selected.get("run_id") or ""),
                "config_name": str(selected.get("config_name") or ""),
                "baseline": {
                    "config_name": "baseline",
                    "family": str(selected.get("baseline_family") or ""),
                    "visual_path": _preferred_visual_path(baseline_paths),
                    "analysis_artifact_paths": baseline_paths,
                },
                "candidate": {
                    "config_name": str(selected.get("config_name") or ""),
                    "family": str(selected.get("candidate_family") or ""),
                    "patch_family": str(selected.get("patch_family") or ""),
                    "patch_category": str(selected.get("patch_category") or ""),
                    "visual_path": _preferred_visual_path(candidate_paths),
                    "analysis_artifact_paths": candidate_paths,
                },
                "metrics": {
                    "step_time_ms": round(_safe_float(selected.get("step_time_ms")), 4),
                    "throughput_tokens_per_s": round(_safe_float(selected.get("throughput_tokens_per_s")), 4),
                    "step_gain_ratio": round(_safe_float(selected.get("step_gain_ratio")), 6),
                    "throughput_gain_ratio": round(_safe_float(selected.get("throughput_gain_ratio")), 6),
                    "bubble_ratio": round(_safe_float(selected.get("bubble_ratio")), 6),
                    "stage_skew_ratio": round(_safe_float(selected.get("stage_skew_ratio")), 6),
                    "memory_skew_ratio": round(_safe_float(selected.get("memory_skew_ratio")), 6),
                    "tail_ratio": round(_safe_float(selected.get("tail_ratio")), 6),
                    "optimizer_exposed_ratio": round(_safe_float(selected.get("optimizer_exposed_ratio")), 6),
                },
            }
        )
    return {
        "format": "patch_case_study_manifest",
        "cases": cases,
    }


def analyze_runs(inputs: Sequence[str], out_dir: Path, case_study_topk: int = 2) -> Dict[str, Path]:
    rows, run_metadata = collect_patch_observations(inputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_observations_path = out_dir / "patch_observations.csv"
    bottleneck_patch_success_path = out_dir / "bottleneck_patch_success.csv"
    bottleneck_patch_gain_path = out_dir / "bottleneck_patch_gain.csv"
    search_ablation_path = out_dir / "search_ablation.csv"
    case_study_manifest_path = out_dir / "case_study_manifest.json"

    csv_rows = [
        {
            "run_id": row["run_id"],
            "trial_id": row["trial_id"],
            "config_name": row["config_name"],
            "returncode": row["returncode"],
            "oom": row["oom"],
            "search_unit": row["search_unit"],
            "patch_memory_enabled": row["patch_memory_enabled"],
            "patch_family": row["patch_family"],
            "patch_category": row["patch_category"],
            "patch_count": row["patch_count"],
            "baseline_family": row["baseline_family"],
            "candidate_family": row["candidate_family"],
            "bottleneck_label": row["bottleneck_label"],
            "step_time_ms": row["step_time_ms"],
            "throughput_tokens_per_s": row["throughput_tokens_per_s"],
            "step_gain_ratio": row["step_gain_ratio"],
            "throughput_gain_ratio": row["throughput_gain_ratio"],
            "bubble_ratio": row["bubble_ratio"],
            "stage_skew_ratio": row["stage_skew_ratio"],
            "memory_skew_ratio": row["memory_skew_ratio"],
            "tail_ratio": row["tail_ratio"],
            "optimizer_exposed_ratio": row["optimizer_exposed_ratio"],
        }
        for row in rows
    ]
    _write_csv(
        patch_observations_path,
        csv_rows,
        [
            "run_id",
            "trial_id",
            "config_name",
            "returncode",
            "oom",
            "search_unit",
            "patch_memory_enabled",
            "patch_family",
            "patch_category",
            "patch_count",
            "baseline_family",
            "candidate_family",
            "bottleneck_label",
            "step_time_ms",
            "throughput_tokens_per_s",
            "step_gain_ratio",
            "throughput_gain_ratio",
            "bubble_ratio",
            "stage_skew_ratio",
            "memory_skew_ratio",
            "tail_ratio",
            "optimizer_exposed_ratio",
        ],
    )
    _write_csv(
        bottleneck_patch_success_path,
        _aggregate_bottleneck_success(rows),
        [
            "bottleneck_label",
            "patch_family",
            "patch_category",
            "attempts",
            "successes",
            "harmful",
            "success_rate",
            "harmful_rate",
        ],
    )
    _write_csv(
        bottleneck_patch_gain_path,
        _aggregate_bottleneck_gain(rows),
        [
            "bottleneck_label",
            "patch_family",
            "patch_category",
            "median_step_gain_ratio",
            "median_throughput_gain_ratio",
            "p75_throughput_gain_ratio",
        ],
    )
    _write_csv(
        search_ablation_path,
        _aggregate_search_ablation(rows, run_metadata),
        [
            "run_id",
            "scenario_label",
            "search_unit",
            "patch_memory_enabled",
            "trial_count",
            "harmful_trial_ratio",
            "best_step_time_ms",
            "best_throughput_tokens_per_s",
            "trials_to_near_best",
            "near_best_threshold",
        ],
    )
    case_study_manifest = _build_case_study_manifest(rows, case_study_topk=case_study_topk)
    case_study_manifest_path.write_text(json.dumps(case_study_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "patch_observations": patch_observations_path,
        "bottleneck_patch_success": bottleneck_patch_success_path,
        "bottleneck_patch_gain": bottleneck_patch_gain_path,
        "search_ablation": search_ablation_path,
        "case_study_manifest": case_study_manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate patch-aware PP/VPP observations for paper figures.")
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--case-study-topk", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze_runs(args.runs, Path(args.out_dir), case_study_topk=int(args.case_study_topk))
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
