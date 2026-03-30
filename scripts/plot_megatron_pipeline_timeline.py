#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


ITERATION_RE = re.compile(
    r"""
    ^\s*\[
    (?P<timestamp>[^\]]+)
    \]\s+iteration\s+
    (?P<iteration>\d+)\s*/\s*(?P<total_iterations>\d+)
    \s+\|\s+consumed\s+samples:\s+(?P<consumed_samples>\d+)
    \s+\|\s+elapsed\s+time\s+per\s+iteration\s+\(ms\):\s+(?P<elapsed_ms>[\d.]+)
    \s+\|\s+throughput\s+per\s+GPU\s+\(TFLOP/s/GPU\):\s+(?P<throughput_tflops>[\d.]+)
    (?:\s+\|\s+MFU\s+\(%\):\s+(?P<mfu_percent>[\d.]+))?
    \s+\|\s+learning\s+rate:\s+(?P<learning_rate>[0-9.E+\-]+)
    \s+\|\s+global\s+batch\s+size:\s+(?P<global_batch_size>\d+)
    \s+\|\s+lm\s+loss:\s+(?P<lm_loss>[0-9.E+\-]+)
    """,
    re.VERBOSE,
)

TIMER_HEADER_RE = re.compile(r"INFO:megatron\.core\.timers:\(min, max\) time across ranks \(ms\):")
TIMER_LINE_RE = re.compile(r"^\s*(?P<name>[^.][^:]+?)\s*\.+:\s*\((?P<min_ms>[\d.]+),\s*(?P<max_ms>[\d.]+)\)\s*$")

PIPELINE_STAGE_ORDER = [
    "batch-generator",
    "forward-compute",
    "backward-compute",
    "embedding-grads-all-reduce",
    "all-grads-sync",
    "optimizer-copy-to-main-grad",
    "optimizer-inner-step",
    "optimizer-copy-main-to-model-params",
]

AGGREGATE_TIMER_ORDER = [
    "batch-generator",
    "forward-backward",
    "optimizer",
]

STAGE_COLORS = {
    "batch-generator": "#6c757d",
    "forward-compute": "#1f77b4",
    "backward-compute": "#ff7f0e",
    "embedding-grads-all-reduce": "#17becf",
    "all-grads-sync": "#2ca02c",
    "optimizer-copy-to-main-grad": "#9467bd",
    "optimizer-inner-step": "#d62728",
    "optimizer-copy-main-to-model-params": "#8c564b",
    "other": "#c7c7c7",
    "forward-backward": "#4c78a8",
    "optimizer": "#e45756",
}


@dataclass
class IterationRecord:
    timestamp: str
    iteration: int
    total_iterations: int
    consumed_samples: int
    elapsed_ms: float
    throughput_tflops: float
    mfu_percent: float | None
    learning_rate: float
    global_batch_size: int
    lm_loss: float
    timers_max_ms: dict[str, float] = field(default_factory=dict)


def _parse_float(text: str) -> float:
    return float(text.replace("E", "e"))


def parse_log(log_path: Path) -> list[IterationRecord]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[IterationRecord] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        iteration_match = ITERATION_RE.search(line)
        if not iteration_match:
            i += 1
            continue

        record = IterationRecord(
            timestamp=iteration_match.group("timestamp"),
            iteration=int(iteration_match.group("iteration")),
            total_iterations=int(iteration_match.group("total_iterations")),
            consumed_samples=int(iteration_match.group("consumed_samples")),
            elapsed_ms=float(iteration_match.group("elapsed_ms")),
            throughput_tflops=float(iteration_match.group("throughput_tflops")),
            mfu_percent=(
                float(iteration_match.group("mfu_percent"))
                if iteration_match.group("mfu_percent") is not None
                else None
            ),
            learning_rate=_parse_float(iteration_match.group("learning_rate")),
            global_batch_size=int(iteration_match.group("global_batch_size")),
            lm_loss=_parse_float(iteration_match.group("lm_loss")),
        )

        j = i + 1
        if j < len(lines) and TIMER_HEADER_RE.search(lines[j]):
            j += 1
            while j < len(lines):
                timer_match = TIMER_LINE_RE.match(lines[j])
                if not timer_match:
                    break
                timer_name = timer_match.group("name").strip()
                record.timers_max_ms[timer_name] = float(timer_match.group("max_ms"))
                j += 1

        records.append(record)
        i = max(j, i + 1)

    return records


def build_summary(records: list[IterationRecord]) -> dict:
    if not records:
        return {"iterations": 0}

    throughput_values = [r.throughput_tflops for r in records]
    elapsed_values = [r.elapsed_ms for r in records]

    stage_avgs = {}
    for stage in PIPELINE_STAGE_ORDER + ["forward-backward", "optimizer"]:
        values = [r.timers_max_ms.get(stage) for r in records if stage in r.timers_max_ms]
        if values:
            stage_avgs[stage] = mean(values)

    return {
        "iterations": len(records),
        "iteration_range": [records[0].iteration, records[-1].iteration],
        "avg_elapsed_ms": mean(elapsed_values),
        "avg_throughput_tflops_per_gpu": mean(throughput_values),
        "avg_mfu_percent": (
            mean([r.mfu_percent for r in records if r.mfu_percent is not None])
            if any(r.mfu_percent is not None for r in records)
            else None
        ),
        "avg_lm_loss": mean(r.lm_loss for r in records),
        "stage_avg_ms": stage_avgs,
    }


def get_timer_names(records: list[IterationRecord]) -> list[str]:
    extras = sorted(
        {
            timer_name
            for record in records
            for timer_name in record.timers_max_ms
            if timer_name not in PIPELINE_STAGE_ORDER and timer_name not in AGGREGATE_TIMER_ORDER
        }
    )
    return [*PIPELINE_STAGE_ORDER, *AGGREGATE_TIMER_ORDER, *extras]


def write_csv(records: list[IterationRecord], csv_path: Path) -> None:
    timer_names = get_timer_names(records)
    fieldnames = [
        "iteration",
        "timestamp",
        "elapsed_ms",
        "throughput_tflops_per_gpu",
        "mfu_percent",
        "learning_rate",
        "lm_loss",
        "global_batch_size",
        *timer_names,
        "other_detailed",
        "other_macro",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            detailed_total = sum(record.timers_max_ms.get(name, 0.0) for name in PIPELINE_STAGE_ORDER)
            macro_total = sum(record.timers_max_ms.get(name, 0.0) for name in AGGREGATE_TIMER_ORDER)
            row = {
                "iteration": record.iteration,
                "timestamp": record.timestamp,
                "elapsed_ms": record.elapsed_ms,
                "throughput_tflops_per_gpu": record.throughput_tflops,
                "mfu_percent": record.mfu_percent,
                "learning_rate": record.learning_rate,
                "lm_loss": record.lm_loss,
                "global_batch_size": record.global_batch_size,
                "other_detailed": max(record.elapsed_ms - detailed_total, 0.0),
                "other_macro": max(record.elapsed_ms - macro_total, 0.0),
            }
            for stage in timer_names:
                row[stage] = record.timers_max_ms.get(stage, 0.0)
            writer.writerow(row)


def render_plots(records: list[IterationRecord], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    iterations = [r.iteration for r in records]
    timer_names = get_timer_names(records)

    # Detailed pipeline-style Gantt chart.
    fig, ax = plt.subplots(figsize=(16, max(5, len(records) * 0.45)))
    y_positions = list(range(len(records)))
    y_labels = [f"iter {r.iteration}" for r in records]

    for idx, record in enumerate(records):
        left = 0.0
        for stage in PIPELINE_STAGE_ORDER:
            duration = record.timers_max_ms.get(stage, 0.0)
            if duration <= 0.0:
                continue
            ax.barh(
                idx,
                duration,
                left=left,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                height=0.7,
                label=stage if idx == 0 else None,
            )
            left += duration
        other = max(record.elapsed_ms - left, 0.0)
        if other > 0.0:
            ax.barh(
                idx,
                other,
                left=left,
                color=STAGE_COLORS["other"],
                edgecolor="white",
                height=0.7,
                label="other" if idx == 0 else None,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title("Megatron Iteration Pipeline Timeline (Detailed Sequentialized View)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    ax.legend(uniq_handles, uniq_labels, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_timeline_detailed.png", dpi=200)
    plt.close(fig)

    # Macro breakdown.
    fig, ax = plt.subplots(figsize=(16, max(5, len(records) * 0.35)))
    for idx, record in enumerate(records):
        left = 0.0
        for stage in AGGREGATE_TIMER_ORDER:
            duration = record.timers_max_ms.get(stage, 0.0)
            if duration <= 0.0:
                continue
            ax.barh(
                idx,
                duration,
                left=left,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                height=0.7,
                label=stage if idx == 0 else None,
            )
            left += duration
        other = max(record.elapsed_ms - left, 0.0)
        if other > 0.0:
            ax.barh(
                idx,
                other,
                left=left,
                color=STAGE_COLORS["other"],
                edgecolor="white",
                height=0.7,
                label="other" if idx == 0 else None,
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title("Megatron Iteration Pipeline Timeline (Macro View)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    ax.legend(uniq_handles, uniq_labels, loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_timeline_macro.png", dpi=200)
    plt.close(fig)

    # Stage trend lines.
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax_top, ax_bottom = axes
    for stage in ["forward-compute", "backward-compute", "optimizer-inner-step", "all-grads-sync"]:
        values = [r.timers_max_ms.get(stage, math.nan) for r in records]
        ax_top.plot(iterations, values, marker="o", linewidth=2, label=stage, color=STAGE_COLORS[stage])
    ax_top.set_ylabel("Time (ms)")
    ax_top.set_title("Critical Stage Timing Trends")
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend()

    ax_bottom.plot(iterations, [r.elapsed_ms for r in records], marker="o", linewidth=2, label="elapsed_ms", color="#4c78a8")
    ax_bottom.plot(iterations, [r.throughput_tflops for r in records], marker="s", linewidth=2, label="throughput_tflops_per_gpu", color="#f58518")
    if any(r.mfu_percent is not None for r in records):
        ax_bottom.plot(
            iterations,
            [r.mfu_percent if r.mfu_percent is not None else math.nan for r in records],
            marker="^",
            linewidth=2,
            label="mfu_percent",
            color="#54a24b",
        )
    ax_bottom.set_xlabel("Iteration")
    ax_bottom.set_ylabel("Value")
    ax_bottom.set_title("Iteration Throughput and End-to-End Runtime")
    ax_bottom.grid(True, linestyle="--", alpha=0.3)
    ax_bottom.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pipeline_stage_trends.png", dpi=200)
    plt.close(fig)

    # All-timer heatmap for a more detailed per-iteration view.
    heatmap = np.array(
        [[record.timers_max_ms.get(timer_name, 0.0) for record in records] for timer_name in timer_names],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.7), max(6, len(timer_names) * 0.45)))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels([str(r.iteration) for r in records], rotation=45, ha="right")
    ax.set_yticks(range(len(timer_names)))
    ax.set_yticklabels(timer_names)
    ax.set_xlabel("Iteration")
    ax.set_title("Megatron Timer Heatmap (Max Time Across Ranks, ms)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_dir / "timer_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Megatron stdout logs and render pipeline-style timing visualizations."
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to Megatron stdout/stderr text log containing iteration and timer lines.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write plots and summaries. Defaults to <log_dir>/pipeline_viz.",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else log_path.parent / "pipeline_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(log_path)
    if not records:
        raise SystemExit(f"No Megatron iteration/timer records found in: {log_path}")

    summary = build_summary(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(records, out_dir / "iteration_stage_breakdown.csv")
    render_plots(records, out_dir)

    print(f"Parsed {len(records)} iterations from {log_path}")
    print(f"Outputs written to {out_dir}")
    print(f"  - {(out_dir / 'pipeline_timeline_detailed.png')}")
    print(f"  - {(out_dir / 'pipeline_timeline_macro.png')}")
    print(f"  - {(out_dir / 'pipeline_stage_trends.png')}")
    print(f"  - {(out_dir / 'timer_heatmap.png')}")
    print(f"  - {(out_dir / 'iteration_stage_breakdown.csv')}")
    print(f"  - {(out_dir / 'summary.json')}")


if __name__ == "__main__":
    main()
