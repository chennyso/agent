from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9l9mEAAAAASUVORK5CYII="
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_blank_png(path: Path) -> None:
    path.write_bytes(_BLANK_PNG)


def _load_plotting_backend():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        return plt, np
    except Exception:
        return None, None


def _scatter_patch_sparsity(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    xs = [_safe_float(row.get("patch_count")) for row in rows if _safe_float(row.get("patch_count")) > 0]
    ys = [_safe_float(row.get("throughput_gain_ratio")) for row in rows if _safe_float(row.get("patch_count")) > 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.75, color="#2f6fed", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Patch Count")
    ax.set_ylabel("Throughput Gain Ratio")
    ax.set_title("Patch Sparsity vs Throughput Gain")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _hist_high_gain_patch_count(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    values = [
        int(_safe_float(row.get("patch_count")))
        for row in rows
        if _safe_float(row.get("throughput_gain_ratio")) >= 0.03 and _safe_float(row.get("patch_count")) > 0
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = range(0, max(values + [1]) + 2)
    ax.hist(values or [0], bins=bins, color="#ff9f1c", edgecolor="white")
    ax.set_xlabel("Patch Count")
    ax.set_ylabel("High-Gain Trial Count")
    ax.set_title("Patch Count Distribution of High-Gain Trials")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _heatmap(rows: List[Dict[str, Any]], value_key: str, title: str, out_path: Path) -> None:
    plt, np = _load_plotting_backend()
    if plt is None or np is None:
        _write_blank_png(out_path)
        return
    bottlenecks = sorted({str(row.get("bottleneck_label") or "") for row in rows})
    patch_families = sorted({str(row.get("patch_family") or "") for row in rows})
    if not bottlenecks or not patch_families:
        _write_blank_png(out_path)
        return
    matrix = np.zeros((len(bottlenecks), len(patch_families)))
    for row in rows:
        i = bottlenecks.index(str(row.get("bottleneck_label") or ""))
        j = patch_families.index(str(row.get("patch_family") or ""))
        matrix[i, j] = _safe_float(row.get(value_key))
    fig, ax = plt.subplots(figsize=(max(7, len(patch_families) * 0.8), max(4, len(bottlenecks) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(patch_families)))
    ax.set_xticklabels(patch_families, rotation=45, ha="right")
    ax.set_yticks(range(len(bottlenecks)))
    ax.set_yticklabels(bottlenecks)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _search_ablation_curve(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("config_name") or "") == "baseline":
            continue
        key = (
            str(row.get("run_id") or ""),
            str(row.get("search_unit") or ""),
            str(row.get("patch_memory_enabled") or ""),
        )
        grouped[key].append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (run_id, search_unit, patch_memory_enabled), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(_safe_float(item.get("trial_id"), 0.0)))
        xs: List[int] = []
        ys: List[float] = []
        best = 0.0
        for index, item in enumerate(ordered, start=1):
            best = max(best, _safe_float(item.get("throughput_tokens_per_s")))
            xs.append(index)
            ys.append(best)
        label = f"{run_id}:{search_unit}:pm={'on' if _safe_bool(patch_memory_enabled) else 'off'}"
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label)
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Best-so-far Throughput")
    ax.set_title("Search Ablation Curves")
    ax.grid(alpha=0.2)
    if grouped:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _stateful_vs_coarse(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = "stateful" if _safe_float(row.get("layer_group_count")) > 0 else "coarse"
        grouped[key].append(_safe_float(row.get("throughput_gain_ratio")))
    labels = ["coarse", "stateful"]
    values = [_safe_float(sum(grouped.get(label, [])) / max(len(grouped.get(label, [])), 1)) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color=["#9aa0a6", "#1b998b"])
    ax.set_ylabel("Mean Throughput Gain Ratio")
    ax.set_title("Stateful vs Coarse Schedule")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _reload_interference_breakdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    shifted = [_safe_float(row.get("throughput_gain_ratio")) for row in rows if _safe_float(row.get("reload_shift_count")) > 0]
    stable = [_safe_float(row.get("throughput_gain_ratio")) for row in rows if _safe_float(row.get("reload_shift_count")) <= 0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        ["no_reload_shift", "reload_shift"],
        [
            _safe_float(sum(stable) / max(len(stable), 1)),
            _safe_float(sum(shifted) / max(len(shifted), 1)),
        ],
        color=["#577590", "#f3722c"],
    )
    ax.set_ylabel("Mean Throughput Gain Ratio")
    ax.set_title("Reload Interference Breakdown")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _budgeted_telemetry_cost(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("telemetry_level") or "summary")].append(_safe_float(row.get("throughput_gain_ratio")))
    labels = sorted(grouped) or ["summary"]
    values = [_safe_float(sum(grouped.get(label, [])) / max(len(grouped.get(label, [])), 1)) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#6d597a")
    ax.set_ylabel("Mean Throughput Gain Ratio")
    ax.set_title("Budgeted Telemetry Cost Profile")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _rewrite_gain_by_family(rows: List[Dict[str, Any]], rewrite_family: str, title: str, out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    active = [
        _safe_float(row.get("throughput_gain_ratio"))
        for row in rows
        if str(row.get("rewrite_family") or "") == str(rewrite_family)
    ]
    inactive = [
        _safe_float(row.get("throughput_gain_ratio"))
        for row in rows
        if str(row.get("rewrite_family") or "") != str(rewrite_family)
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        ["others", rewrite_family],
        [
            _safe_float(sum(inactive) / max(len(inactive), 1)),
            _safe_float(sum(active) / max(len(active), 1)),
        ],
        color=["#7f8c8d", "#2a9d8f"],
    )
    ax.set_ylabel("Mean Throughput Gain Ratio")
    ax.set_title(title)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_svg_image(path: Path):
    try:
        import cairosvg  # type: ignore
        from PIL import Image  # type: ignore

        png_bytes = cairosvg.svg2png(url=str(path))
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return None


def _case_study_compare(manifest: Dict[str, Any], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    cases = list(manifest.get("cases") or [])
    rows = max(len(cases), 1)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(4, rows * 4)))
    if rows == 1:
        axes = [axes]
    for row_axes, case in zip(axes, cases):
        baseline = dict(case.get("baseline") or {})
        candidate = dict(case.get("candidate") or {})
        metrics = dict(case.get("metrics") or {})
        for ax, payload, title in (
            (row_axes[0], baseline, f"{case.get('scenario_label')} baseline"),
            (row_axes[1], candidate, f"{case.get('scenario_label')} candidate"),
        ):
            visual_path = Path(str(payload.get("visual_path") or "")) if str(payload.get("visual_path") or "").strip() else None
            image = _load_svg_image(visual_path) if visual_path and visual_path.exists() else None
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(
                    0.02,
                    0.98,
                    f"{title}\n{payload.get('family')}\n{payload.get('visual_path') or 'no SVG available'}",
                    va="top",
                    ha="left",
                    fontsize=10,
                    family="monospace",
                )
            ax.set_title(title)
            ax.axis("off")
        metric_text = "\n".join(
            [
                f"patch_family: {candidate.get('patch_family') or 'n/a'}",
                f"patch_category: {candidate.get('patch_category') or 'n/a'}",
                f"step_gain_ratio: {_safe_float(metrics.get('step_gain_ratio')):.4f}",
                f"throughput_gain_ratio: {_safe_float(metrics.get('throughput_gain_ratio')):.4f}",
                f"bubble_ratio: {_safe_float(metrics.get('bubble_ratio')):.4f}",
                f"stage_skew_ratio: {_safe_float(metrics.get('stage_skew_ratio')):.4f}",
                f"memory_skew_ratio: {_safe_float(metrics.get('memory_skew_ratio')):.4f}",
                f"tail_ratio: {_safe_float(metrics.get('tail_ratio')):.4f}",
            ]
        )
        row_axes[2].text(0.02, 0.98, metric_text, va="top", ha="left", fontsize=10, family="monospace")
        row_axes[2].set_title(f"{case.get('scenario_label')} metrics")
        row_axes[2].axis("off")
    if len(cases) < rows:
        for row_axes in axes[len(cases):]:
            for ax in row_axes:
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_figures(analysis_dir: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_rows = _read_csv(analysis_dir / "patch_observations.csv")
    success_rows = _read_csv(analysis_dir / "bottleneck_patch_success.csv")
    gain_rows = _read_csv(analysis_dir / "bottleneck_patch_gain.csv")
    manifest_path = analysis_dir / "case_study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"cases": []}

    outputs = {
        "fig_patch_sparsity": out_dir / "fig_patch_sparsity.png",
        "fig_patch_count_hist": out_dir / "fig_patch_count_hist.png",
        "fig_bottleneck_patch_success_heatmap": out_dir / "fig_bottleneck_patch_success_heatmap.png",
        "fig_bottleneck_patch_gain_heatmap": out_dir / "fig_bottleneck_patch_gain_heatmap.png",
        "fig_search_ablation_curve": out_dir / "fig_search_ablation_curve.png",
        "fig_stateful_vs_coarse": out_dir / "fig_stateful_vs_coarse.png",
        "fig_reload_interference_breakdown": out_dir / "fig_reload_interference_breakdown.png",
        "fig_reload_shift_gain": out_dir / "fig_reload_shift_gain.png",
        "fig_adaptive_chunking_gain": out_dir / "fig_adaptive_chunking_gain.png",
        "fig_local_verticalization_gain": out_dir / "fig_local_verticalization_gain.png",
        "fig_budgeted_telemetry_cost": out_dir / "fig_budgeted_telemetry_cost.png",
        "fig_case_study_compare": out_dir / "fig_case_study_compare.png",
    }
    _scatter_patch_sparsity(patch_rows, outputs["fig_patch_sparsity"])
    _hist_high_gain_patch_count(patch_rows, outputs["fig_patch_count_hist"])
    _heatmap(success_rows, "success_rate", "Patch Success Rate by Bottleneck", outputs["fig_bottleneck_patch_success_heatmap"])
    _heatmap(gain_rows, "median_throughput_gain_ratio", "Median Throughput Gain by Bottleneck", outputs["fig_bottleneck_patch_gain_heatmap"])
    _search_ablation_curve(patch_rows, outputs["fig_search_ablation_curve"])
    _stateful_vs_coarse(patch_rows, outputs["fig_stateful_vs_coarse"])
    _reload_interference_breakdown(patch_rows, outputs["fig_reload_interference_breakdown"])
    _rewrite_gain_by_family(patch_rows, "reload_shift", "Reload Shift Gain", outputs["fig_reload_shift_gain"])
    _rewrite_gain_by_family(patch_rows, "adaptive_chunking", "Adaptive Chunking Gain", outputs["fig_adaptive_chunking_gain"])
    _rewrite_gain_by_family(patch_rows, "local_verticalization", "Local Verticalization Gain", outputs["fig_local_verticalization_gain"])
    _budgeted_telemetry_cost(patch_rows, outputs["fig_budgeted_telemetry_cost"])
    _case_study_compare(manifest, outputs["fig_case_study_compare"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper figures from patch observation analysis tables.")
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_figures(Path(args.analysis_dir), Path(args.out_dir))
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
