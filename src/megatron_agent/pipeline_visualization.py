from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_EVENT_COLORS = {
    "fwd": "#4C78A8",
    "bwd": "#F58518",
    "wgrad": "#E45756",
    "comm": "#B279A2",
    "offload": "#9C755F",
    "reload": "#72B7B2",
    "idle": "#B8B8B8",
}

DEFAULT_PHASE_COLORS = {
    "warmup": "#203864",
    "steady": "#EDF7E8",
    "cooldown": "#FCE7C7",
}


@dataclass(frozen=True)
class PipelinePanel:
    label: str
    summary: Dict[str, Any]
    phase_windows: List[Dict[str, Any]]
    lane_summaries: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    total_span_ms: float
    max_peak_reserved_gib: float
    source_path: str


def extract_pipeline_event_trace(payload: Mapping[str, Any]) -> Dict[str, Any]:
    direct = dict(payload)
    if str(direct.get("format") or "") == "pipeline_event_trace":
        return direct

    visualization = dict(direct.get("visualization_artifacts") or {})
    nested = dict(visualization.get("pipeline_event_trace") or {})
    if str(nested.get("format") or "") == "pipeline_event_trace":
        return nested

    evidence = dict(direct.get("evidence_record") or {})
    visualization = dict(evidence.get("visualization_artifacts") or {})
    nested = dict(visualization.get("pipeline_event_trace") or {})
    if str(nested.get("format") or "") == "pipeline_event_trace":
        return nested

    artifact = dict(direct.get("trial_artifact") or {})
    visualization = dict(artifact.get("visualization_artifacts") or {})
    nested = dict(visualization.get("pipeline_event_trace") or {})
    if str(nested.get("format") or "") == "pipeline_event_trace":
        return nested

    raise ValueError("payload does not contain a pipeline_event_trace artifact")


def load_pipeline_event_trace(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return extract_pipeline_event_trace(payload)


def prepare_pipeline_timeline_panels(
    traces: Sequence[Mapping[str, Any]],
    *,
    labels: Optional[Sequence[str]] = None,
    source_paths: Optional[Sequence[str | Path]] = None,
) -> List[PipelinePanel]:
    panels: List[PipelinePanel] = []
    label_list = list(labels or [])
    path_list = [str(item) for item in list(source_paths or [])]
    for index, raw_trace in enumerate(traces):
        trace = extract_pipeline_event_trace(raw_trace)
        summary = dict(trace.get("summary") or {})
        lane_summaries = sorted(
            [dict(item) for item in list(trace.get("lane_summaries") or [])],
            key=lambda item: int(item.get("stage_id") or 0),
        )
        events = sorted(
            [dict(item) for item in list(trace.get("events") or [])],
            key=lambda item: (
                float(item.get("start_ms") or 0.0),
                int(item.get("stage_id") or 0),
                int(item.get("lane_id") or 0),
            ),
        )
        total_span_ms = max(
            float(summary.get("projected_timeline_span_ms") or 0.0),
            max((float(item.get("end_ms") or 0.0) for item in events), default=0.0),
            1.0,
        )
        max_mem = max((float(item.get("peak_reserved_gib") or 0.0) for item in lane_summaries), default=0.0)
        default_label = str(summary.get("schedule_template") or f"trace_{index}")
        panel_label = str(label_list[index]).strip() if index < len(label_list) and str(label_list[index]).strip() else default_label
        panel_source = path_list[index] if index < len(path_list) else ""
        panels.append(
            PipelinePanel(
                label=panel_label,
                summary=summary,
                phase_windows=[dict(item) for item in list(trace.get("phase_windows") or [])],
                lane_summaries=lane_summaries,
                events=events,
                total_span_ms=total_span_ms,
                max_peak_reserved_gib=max_mem,
                source_path=panel_source,
            )
        )
    return panels


def _event_style(event: Mapping[str, Any]) -> tuple[str, float]:
    label = str(event.get("label") or "").strip().upper()
    op_kind = str(event.get("op_kind") or "idle").strip().lower()
    if label.startswith("W"):
        return DEFAULT_EVENT_COLORS["wgrad"], 0.94
    color = str(event.get("color") or "").strip()
    if color:
        if op_kind == "comm":
            return color, 0.75
        return color, 0.94
    opacity = 0.94
    if op_kind == "comm":
        opacity = 0.78
    if op_kind in {"idle", "bubble"}:
        opacity = 0.60
        op_kind = "idle"
    return DEFAULT_EVENT_COLORS.get(op_kind, "#7F7F7F"), opacity


def render_pipeline_timeline_figure(
    panels: Sequence[PipelinePanel],
    *,
    title: str = "Pipeline Schedule Timeline",
    annotate: bool = True,
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    panel_list = list(panels)
    if not panel_list:
        raise ValueError("at least one panel is required to render a pipeline timeline")

    figure_height = 1.5 + sum(max(2.2, 0.72 * max(len(panel.lane_summaries), 1)) for panel in panel_list)
    fig, axes = plt.subplots(len(panel_list), 1, figsize=(16, figure_height), squeeze=False)
    fig.patch.set_facecolor("white")

    legend_items = [
        ("fwd", DEFAULT_EVENT_COLORS["fwd"]),
        ("bwd", DEFAULT_EVENT_COLORS["bwd"]),
        ("wgrad", DEFAULT_EVENT_COLORS["wgrad"]),
        ("comm", DEFAULT_EVENT_COLORS["comm"]),
        ("offload/reload", DEFAULT_EVENT_COLORS["reload"]),
        ("bubble", DEFAULT_EVENT_COLORS["idle"]),
    ]

    for panel_index, panel in enumerate(panel_list):
        ax = axes[panel_index][0]
        ax.set_facecolor("#7B7B7B")
        lane_index = {
            int(item.get("stage_id") or idx): idx
            for idx, item in enumerate(panel.lane_summaries)
        }
        lane_count = max(len(panel.lane_summaries), 1)
        x_padding = panel.total_span_ms * 0.02
        right_label_x = panel.total_span_ms + x_padding * 1.35

        for phase in panel.phase_windows:
            phase_name = str(phase.get("name") or "").strip().lower()
            phase_color = DEFAULT_PHASE_COLORS.get(phase_name, "#D9D9D9")
            start_ms = float(phase.get("start_ms") or 0.0)
            width_ms = max(float(phase.get("end_ms") or 0.0) - start_ms, 0.0)
            if width_ms <= 0.0:
                continue
            alpha = 0.24 if phase_name != "warmup" else 0.34
            ax.axvspan(start_ms, start_ms + width_ms, color=phase_color, alpha=alpha, zorder=0)

        for idx, lane in enumerate(panel.lane_summaries):
            y = idx
            ax.add_patch(
                mpatches.Rectangle(
                    (0.0, y - 0.42),
                    panel.total_span_ms,
                    0.84,
                    facecolor="#7E7E7E",
                    edgecolor="#939393",
                    linewidth=0.6,
                    zorder=1,
                )
            )
            ax.text(
                -x_padding * 0.55,
                y,
                f"PP-{int(lane.get('stage_id') or idx)}",
                va="center",
                ha="right",
                fontsize=10,
                color="#222222",
                fontweight="bold",
            )
            bubble_events = [
                event for event in panel.events
                if int(event.get("stage_id") or -1) == int(lane.get("stage_id") or idx)
                and str(event.get("op_kind") or "").lower() in {"idle", "bubble"}
            ]
            bubble_ms = sum(float(event.get("duration_ms") or 0.0) for event in bubble_events)
            bubble_ratio = bubble_ms / max(panel.total_span_ms, 1.0)
            mem_text = f"bubble: {bubble_ms:.0f} ms ({bubble_ratio * 100:.0f}%)\nmem: {float(lane.get('peak_reserved_gib') or 0.0):.2f} GiB"
            ax.text(
                right_label_x,
                y,
                mem_text,
                va="center",
                ha="left",
                fontsize=8.5,
                color="#F4F4F4",
            )

        for event in panel.events:
            stage_id = int(event.get("stage_id") or 0)
            if stage_id not in lane_index:
                continue
            y = lane_index[stage_id]
            start_ms = float(event.get("start_ms") or 0.0)
            duration_ms = max(float(event.get("duration_ms") or 0.0), panel.total_span_ms * 0.0012)
            color, alpha = _event_style(event)
            ax.add_patch(
                mpatches.Rectangle(
                    (start_ms, y - 0.36),
                    duration_ms,
                    0.72,
                    facecolor=color,
                    edgecolor="#F7F7F7",
                    linewidth=0.4,
                    alpha=alpha,
                    zorder=3,
                )
            )
            label = str(event.get("label") or "").strip()
            if annotate and label and duration_ms >= (panel.total_span_ms * 0.012):
                ax.text(
                    start_ms + duration_ms / 2.0,
                    y,
                    label,
                    va="center",
                    ha="center",
                    fontsize=7.4,
                    color="#0F172A",
                    zorder=4,
                    clip_on=True,
                )

        pp_degree = int(panel.summary.get("pp_degree") or lane_count)
        step_ms = float(panel.summary.get("projected_timeline_span_ms") or panel.total_span_ms)
        subtitle = f"iteration 0 (iter_time: {step_ms:.0f} ms, max_mem: {panel.max_peak_reserved_gib:.2f} GiB, PP={pp_degree}, VPP={int(panel.summary.get('vpp_degree') or 1)})"
        if panel.source_path:
            subtitle = f"{subtitle}  [{Path(panel.source_path).name}]"

        ax.set_title(panel.label, fontsize=14, fontweight="bold", pad=14)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#555555")
        ax.set_xlim(-x_padding * 0.9, panel.total_span_ms + x_padding * 5.2)
        ax.set_ylim(lane_count - 0.3, -0.7)
        ax.set_yticks([])
        ax.grid(axis="x", linestyle="--", alpha=0.18, color="#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if panel_index == len(panel_list) - 1:
            ax.set_xlabel("time (ms)", fontsize=11)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    handles = [mpatches.Patch(color=color, label=name) for name, color in legend_items]
    fig.legend(handles=handles, loc="lower right", ncol=min(len(handles), 6), frameon=True)
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.97))
    return fig


def save_pipeline_timeline_figure(
    panels: Sequence[PipelinePanel],
    output_path: str | Path,
    *,
    title: str = "Pipeline Schedule Timeline",
    annotate: bool = True,
    dpi: int = 200,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = render_pipeline_timeline_figure(panels, title=title, annotate=annotate)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output


def load_panels_from_paths(
    trace_paths: Sequence[str | Path],
    *,
    labels: Optional[Sequence[str]] = None,
) -> List[PipelinePanel]:
    raw_payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in trace_paths]
    return prepare_pipeline_timeline_panels(raw_payloads, labels=labels, source_paths=trace_paths)
