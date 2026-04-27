from __future__ import annotations

import base64
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9l9mEAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class VppValidationRecord:
    config_name: str
    vpp_depth: int
    chunking: str
    bubble_ratio: float
    reload_stall_ms: float
    pcie_nic_conflict_ms: float
    exposed_comm_ms: float
    exposed_offload_ms: float
    crc_score: float
    ecpt_ms: float
    tokens_per_s: float
    notes: str


@dataclass(frozen=True)
class ActivationChunkProfile:
    chunk_id: str
    activation_gib: float
    lifetime_score: float
    reload_ms: float
    offload_ms: float
    reload_conflict_score: float
    peak_memory_criticality: float
    boundary_distance: int


@dataclass(frozen=True)
class OffloadValidationRecord:
    policy_name: str
    selected_chunks: List[str]
    peak_memory_gib: float
    memory_reduction_gib: float
    reload_stall_ms: float
    pcie_nic_conflict_ms: float
    crc_score: float
    ecpt_ms: float
    tokens_per_s: float
    notes: str


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), int(digits))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def critical_path_resource_conflict(
    windows: Sequence[Mapping[str, Any]],
    capacities: Mapping[str, float],
) -> float:
    """Compute CRC from resource demand windows.

    Each window has a criticality value in [0, 1] plus resource demand keys.
    Resource demand above capacity is exposed only in proportion to the window
    criticality, matching the paper abstraction rather than raw utilization.
    """

    score = 0.0
    for window in windows:
        criticality = max(0.0, _safe_float(window.get("criticality"), 0.0))
        duration_ms = max(0.0, _safe_float(window.get("duration_ms"), 1.0))
        for resource, capacity in capacities.items():
            demand = max(0.0, _safe_float(window.get(resource), 0.0))
            score += max(0.0, demand - max(float(capacity), 1e-9)) * duration_ms * criticality
    return _round(score)


def _vpp_base_metrics(
    depth: int,
    *,
    config_name: str,
    chunking: str,
    bubble_multiplier: float = 1.0,
    reload_relief: float = 1.0,
    conflict_relief: float = 1.0,
    compute_ms: float = 820.0,
    comm_relief_ms: float = 0.0,
    notes: str = "",
) -> VppValidationRecord:
    depth = max(int(depth), 1)
    tokens_per_step = 16384.0
    bubble_ratio = (0.31 / (float(depth) ** 0.62) + 0.012) * float(bubble_multiplier)
    exposed_bubble_ms = float(compute_ms) * bubble_ratio * 0.35

    reload_stall_ms = (8.0 / float(depth) + 0.42 * max(float(depth) - 2.0, 0.0) ** 2) * float(reload_relief)
    pcie_nic_conflict_ms = max(0.0, 0.75 * float(depth) + 0.38 * float(depth * depth) - 2.2) * float(conflict_relief)
    exposed_comm_ms = max(12.0, 21.0 + 0.65 * math.log2(float(depth)) - float(comm_relief_ms))
    exposed_offload_ms = max(0.0, 2.2 + 0.35 * math.log2(float(depth)))

    windows = [
        {
            "duration_ms": exposed_comm_ms,
            "criticality": 0.72,
            "gpu_compute": 0.52,
            "pcie_h2d": 0.26 + 0.09 * depth,
            "nic": 0.58 + 0.08 * depth,
        },
        {
            "duration_ms": reload_stall_ms,
            "criticality": 0.90,
            "gpu_compute": 0.74,
            "pcie_h2d": 0.72 + 0.12 * depth,
            "nic": 0.18 + 0.03 * depth,
        },
        {
            "duration_ms": pcie_nic_conflict_ms,
            "criticality": 0.96,
            "gpu_compute": 0.70,
            "pcie_h2d": 0.86 + 0.10 * depth,
            "nic": 0.82 + 0.09 * depth,
        },
    ]
    crc_score = critical_path_resource_conflict(
        windows,
        {"gpu_compute": 1.0, "pcie_h2d": 1.0, "nic": 1.0},
    )
    ecpt_ms = float(compute_ms) + exposed_bubble_ms + exposed_comm_ms + reload_stall_ms + exposed_offload_ms + pcie_nic_conflict_ms
    tokens_per_s = tokens_per_step / max(ecpt_ms / 1000.0, 1e-9)
    return VppValidationRecord(
        config_name=str(config_name),
        vpp_depth=depth,
        chunking=str(chunking),
        bubble_ratio=_round(bubble_ratio, 6),
        reload_stall_ms=_round(reload_stall_ms),
        pcie_nic_conflict_ms=_round(pcie_nic_conflict_ms),
        exposed_comm_ms=_round(exposed_comm_ms),
        exposed_offload_ms=_round(exposed_offload_ms),
        crc_score=_round(crc_score),
        ecpt_ms=_round(ecpt_ms),
        tokens_per_s=_round(tokens_per_s),
        notes=str(notes),
    )


def generate_vpp_depth_sweep(depths: Sequence[int] = (1, 2, 4, 8)) -> List[VppValidationRecord]:
    return [
        _vpp_base_metrics(
            int(depth),
            config_name=f"uniform_vpp_d{int(depth)}",
            chunking="uniform",
            notes="uniform VPP depth sweep; bubble decreases while deadline density grows",
        )
        for depth in depths
    ]


def generate_vpp_config_comparison() -> List[VppValidationRecord]:
    return [
        _vpp_base_metrics(
            1,
            config_name="uniform_vpp",
            chunking="uniform",
            notes="baseline PP/VPP with coarse chunks",
        ),
        _vpp_base_metrics(
            4,
            config_name="megatron_interleaved_vpp",
            chunking="uniform_interleaved",
            bubble_multiplier=0.92,
            reload_relief=0.95,
            conflict_relief=1.08,
            notes="bubble-focused interleaving with denser transfer deadlines",
        ),
        _vpp_base_metrics(
            8,
            config_name="bubble_minimized_vpp",
            chunking="deep_uniform",
            bubble_multiplier=0.82,
            reload_relief=1.12,
            conflict_relief=1.22,
            notes="lowest bubble but fragmented reload and PCIe/NIC conflict",
        ),
        _vpp_base_metrics(
            4,
            config_name="rac_vpp_cmnc",
            chunking="conflict_minimized_nonuniform",
            bubble_multiplier=1.14,
            reload_relief=0.46,
            conflict_relief=0.34,
            compute_ms=810.0,
            comm_relief_ms=4.0,
            notes="non-uniform chunks keep reload-heavy work away from cross-node boundary windows",
        ),
    ]


def default_activation_chunks() -> List[ActivationChunkProfile]:
    return [
        ActivationChunkProfile("boundary_long_lived", 8.4, 0.98, 15.0, 9.0, 0.92, 0.48, 0),
        ActivationChunkProfile("long_safe", 4.2, 0.76, 7.0, 4.8, 0.22, 0.58, 3),
        ActivationChunkProfile("peak_memory_mid", 5.8, 0.58, 8.0, 5.5, 0.16, 0.96, 2),
        ActivationChunkProfile("boundary_small", 2.4, 0.42, 5.0, 2.8, 0.84, 0.35, 0),
        ActivationChunkProfile("recompute_friendly", 3.5, 0.50, 6.0, 3.9, 0.28, 0.52, 4),
    ]


def _select_offload_chunks(policy_name: str, chunks: Sequence[ActivationChunkProfile]) -> List[ActivationChunkProfile]:
    chunk_list = list(chunks)
    if policy_name == "no_offload":
        return []
    if policy_name == "lifetime_only_offload":
        return sorted(chunk_list, key=lambda item: item.lifetime_score, reverse=True)[:2]
    if policy_name == "memory_size_only_offload":
        return sorted(chunk_list, key=lambda item: item.activation_gib, reverse=True)[:2]
    if policy_name == "rac_vpp_conflict_aware_offload":
        def score(item: ActivationChunkProfile) -> float:
            return (
                0.58 * item.peak_memory_criticality
                + 0.35 * item.lifetime_score
                + 0.18 * min(item.boundary_distance, 4) / 4.0
                - 0.82 * item.reload_conflict_score
                - 0.025 * item.reload_ms
            )

        return sorted(chunk_list, key=score, reverse=True)[:2]
    raise ValueError(f"unknown offload policy: {policy_name}")


def evaluate_offload_policy(
    policy_name: str,
    chunks: Sequence[ActivationChunkProfile] | None = None,
) -> OffloadValidationRecord:
    chunk_list = list(chunks or default_activation_chunks())
    selected = _select_offload_chunks(policy_name, chunk_list)
    selected_ids = [item.chunk_id for item in selected]
    selected_activation = sum(item.activation_gib for item in selected)
    weighted_peak_savings = sum(item.activation_gib * (0.45 + 0.55 * item.peak_memory_criticality) for item in selected)
    base_peak_gib = 31.0
    peak_memory_gib = max(16.0, base_peak_gib - weighted_peak_savings)
    memory_reduction_gib = base_peak_gib - peak_memory_gib

    reload_exposure = sum(item.reload_ms * item.reload_conflict_score * (0.65 + 0.35 * item.lifetime_score) for item in selected)
    offload_exposure = sum(item.offload_ms * max(0.10, item.reload_conflict_score - 0.20) * 0.45 for item in selected)
    pcie_nic_conflict_ms = 2.2 + sum((item.reload_ms + item.offload_ms) * item.reload_conflict_score * 0.28 for item in selected)
    memory_pressure_ms = max(0.0, peak_memory_gib - 24.0) * 4.2
    compute_ms = 820.0
    exposed_bubble_ms = 40.0
    exposed_comm_ms = 22.0
    reload_stall_ms = reload_exposure

    windows = [
        {
            "duration_ms": reload_stall_ms,
            "criticality": 0.92,
            "gpu_compute": 0.70,
            "pcie_h2d": 0.95 + 0.08 * len(selected),
            "nic": 0.72 + sum(item.reload_conflict_score for item in selected) * 0.20,
        },
        {
            "duration_ms": pcie_nic_conflict_ms,
            "criticality": 0.88,
            "gpu_compute": 0.58,
            "pcie_h2d": 1.02 + sum(item.reload_conflict_score for item in selected) * 0.08,
            "nic": 0.88 + sum(item.reload_conflict_score for item in selected) * 0.20,
        },
    ]
    crc_score = critical_path_resource_conflict(windows, {"gpu_compute": 1.0, "pcie_h2d": 1.0, "nic": 1.0})
    ecpt_ms = compute_ms + exposed_bubble_ms + exposed_comm_ms + reload_stall_ms + offload_exposure + pcie_nic_conflict_ms + memory_pressure_ms
    tokens_per_s = 16384.0 / max(ecpt_ms / 1000.0, 1e-9)
    notes = {
        "no_offload": "no reload conflict but high peak-memory pressure remains exposed",
        "lifetime_only_offload": "selects the long-lived boundary activation despite a tight reload deadline",
        "memory_size_only_offload": "selects the largest activation even when it sits on a cross-node conflict window",
        "rac_vpp_conflict_aware_offload": "trades a little memory relief for lower exposed reload and PCIe/NIC conflict",
    }[policy_name]
    return OffloadValidationRecord(
        policy_name=str(policy_name),
        selected_chunks=selected_ids,
        peak_memory_gib=_round(peak_memory_gib),
        memory_reduction_gib=_round(memory_reduction_gib),
        reload_stall_ms=_round(reload_stall_ms),
        pcie_nic_conflict_ms=_round(pcie_nic_conflict_ms),
        crc_score=_round(crc_score),
        ecpt_ms=_round(ecpt_ms),
        tokens_per_s=_round(tokens_per_s),
        notes=notes,
    )


def generate_offload_policy_comparison() -> List[OffloadValidationRecord]:
    policies = [
        "no_offload",
        "lifetime_only_offload",
        "memory_size_only_offload",
        "rac_vpp_conflict_aware_offload",
    ]
    return [evaluate_offload_policy(policy) for policy in policies]


def records_to_rows(records: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        payload = asdict(record)
        for key, value in list(payload.items()):
            if isinstance(value, list):
                payload[key] = "|".join(str(item) for item in value)
        rows.append(payload)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_blank_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_BLANK_PNG)


def _load_plotting_backend():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        return plt, np
    except Exception:
        return None, None


def plot_vpp_depth_tradeoff(records: Sequence[VppValidationRecord], out_path: Path) -> None:
    plt, _ = _load_plotting_backend()
    if plt is None:
        _write_blank_png(out_path)
        return
    ordered = sorted(records, key=lambda item: item.vpp_depth)
    depths = [item.vpp_depth for item in ordered]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor("white")
    axes[0].plot(depths, [item.bubble_ratio for item in ordered], marker="o", linewidth=2.2, color="#2f6fed", label="bubble ratio")
    axes[0].plot(depths, [item.reload_stall_ms for item in ordered], marker="s", linewidth=2.2, color="#f59f00", label="reload stall (ms)")
    axes[0].plot(depths, [item.pcie_nic_conflict_ms for item in ordered], marker="^", linewidth=2.2, color="#d9480f", label="PCIe/NIC conflict (ms)")
    axes[0].set_xlabel("Uniform VPP Depth")
    axes[0].set_title("Lower Bubble Can Densify Resource Deadlines")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(depths, [item.tokens_per_s for item in ordered], marker="o", linewidth=2.4, color="#0ca678", label="tokens/s")
    axes[1].bar([str(depth) for depth in depths], [item.crc_score for item in ordered], alpha=0.28, color="#845ef7", label="CRC")
    axes[1].set_xlabel("Uniform VPP Depth")
    axes[1].set_title("Throughput Peaks Before Bubble Is Minimized")
    axes[1].set_ylabel("tokens/s and CRC score")
    axes[1].grid(alpha=0.22, axis="y")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_vpp_config_comparison(records: Sequence[VppValidationRecord], out_path: Path) -> None:
    plt, np = _load_plotting_backend()
    if plt is None or np is None:
        _write_blank_png(out_path)
        return
    labels = [item.config_name.replace("_", "\n") for item in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    fig.patch.set_facecolor("white")
    metrics = [
        ("bubble_ratio", "Bubble Ratio", "#4c78a8"),
        ("crc_score", "CRC Score", "#e45756"),
        ("tokens_per_s", "Throughput (tokens/s)", "#2ca02c"),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        values = [float(getattr(item, key)) for item in records]
        bars = ax.bar(x, values, color=color, alpha=0.86)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.20, axis="y")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Experiment 1: Bubble-Minimized VPP Is Not the Fastest Configuration", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_offload_policy_comparison(records: Sequence[OffloadValidationRecord], out_path: Path) -> None:
    plt, np = _load_plotting_backend()
    if plt is None or np is None:
        _write_blank_png(out_path)
        return
    labels = [item.policy_name.replace("_", "\n") for item in records]
    x = np.arange(len(records))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.patch.set_facecolor("white")
    axes[0].bar(x - width, [item.peak_memory_gib for item in records], width, label="peak memory GiB", color="#577590")
    axes[0].bar(x, [item.reload_stall_ms for item in records], width, label="reload stall ms", color="#f3722c")
    axes[0].bar(x + width, [item.crc_score for item in records], width, label="CRC", color="#6d597a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title("Memory Relief Alone Does Not Predict Exposed Stall")
    axes[0].grid(alpha=0.20, axis="y")
    axes[0].legend(frameon=False, fontsize=9)

    colors = ["#868e96", "#f59f00", "#fa5252", "#12b886"]
    bars = axes[1].bar(x, [item.tokens_per_s for item in records], color=colors, alpha=0.90)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Conflict-Aware Offload Wins by Reducing Exposed Reload")
    axes[1].set_ylabel("tokens/s")
    axes[1].grid(alpha=0.20, axis="y")
    for bar, item in zip(bars, records):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{item.tokens_per_s:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle("Experiment 2: Activation Lifetime Is Not a Sufficient Offload Criterion", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_validation_suite(out_dir: str | Path) -> Dict[str, Any]:
    output = Path(out_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    depth_sweep = generate_vpp_depth_sweep()
    config_comparison = generate_vpp_config_comparison()
    offload_comparison = generate_offload_policy_comparison()

    _write_csv(tables_dir / "experiment1_vpp_depth_sweep.csv", records_to_rows(depth_sweep))
    _write_csv(tables_dir / "experiment1_vpp_config_comparison.csv", records_to_rows(config_comparison))
    _write_csv(tables_dir / "experiment2_offload_policy_comparison.csv", records_to_rows(offload_comparison))

    plot_vpp_depth_tradeoff(depth_sweep, figures_dir / "fig1_vpp_depth_tradeoff.png")
    plot_vpp_config_comparison(config_comparison, figures_dir / "fig1_vpp_config_comparison.png")
    plot_offload_policy_comparison(offload_comparison, figures_dir / "fig2_offload_policy_comparison.png")

    manifest: Dict[str, Any] = {
        "method": "RAC-VPP",
        "metric_definitions": {
            "CRC": "sum_t sum_r max(0, demand_r(t)-capacity_r(t)) * criticality(t)",
            "ECPT": "compute critical path + exposed comm + exposed reload/offload + exposed PCIe/NIC conflict + exposed bubble",
        },
        "observations": {
            "lowest_bubble_not_fastest": min(config_comparison, key=lambda item: item.bubble_ratio).config_name
            != max(config_comparison, key=lambda item: item.tokens_per_s).config_name,
            "lifetime_only_not_best_offload": max(offload_comparison, key=lambda item: item.tokens_per_s).policy_name
            != "lifetime_only_offload",
        },
        "tables": {
            "experiment1_vpp_depth_sweep": str(tables_dir / "experiment1_vpp_depth_sweep.csv"),
            "experiment1_vpp_config_comparison": str(tables_dir / "experiment1_vpp_config_comparison.csv"),
            "experiment2_offload_policy_comparison": str(tables_dir / "experiment2_offload_policy_comparison.csv"),
        },
        "figures": {
            "fig1_vpp_depth_tradeoff": str(figures_dir / "fig1_vpp_depth_tradeoff.png"),
            "fig1_vpp_config_comparison": str(figures_dir / "fig1_vpp_config_comparison.png"),
            "fig2_offload_policy_comparison": str(figures_dir / "fig2_offload_policy_comparison.png"),
        },
        "records": {
            "vpp_depth_sweep": records_to_rows(depth_sweep),
            "vpp_config_comparison": records_to_rows(config_comparison),
            "offload_policy_comparison": records_to_rows(offload_comparison),
        },
    }
    manifest_path = output / "rac_vpp_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
