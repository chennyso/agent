from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _normalize_forwarded_args(extra_args: Sequence[str]) -> List[str]:
    normalized = list(extra_args)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _variant_specs(work_root: Path) -> List[Dict[str, Any]]:
    return [
        {
            "name": "patch",
            "search_unit": "patch",
            "patch_memory_enabled": True,
            "workdir": work_root / "patch",
            "extra_args": [],
        },
        {
            "name": "whole_config",
            "search_unit": "whole_config",
            "patch_memory_enabled": True,
            "workdir": work_root / "whole_config",
            "extra_args": ["--search-unit", "whole_config"],
        },
        {
            "name": "patch_memory_off",
            "search_unit": "patch",
            "patch_memory_enabled": False,
            "workdir": work_root / "patch_memory_off",
            "extra_args": ["--disable-patch-memory"],
        },
    ]


def _agent_loop_command(
    *,
    python_executable: str,
    workdir: Path,
    forwarded_args: Sequence[str],
    variant_extra_args: Sequence[str],
    orchestrator_args: Sequence[str],
) -> List[str]:
    return [
        python_executable,
        "-m",
        "megatron_agent.agent_loop",
        *list(forwarded_args),
        *list(orchestrator_args),
        "--workdir",
        str(workdir),
        *list(variant_extra_args),
    ]


def _analysis_command(
    *,
    python_executable: str,
    run_dirs: Sequence[Path],
    analysis_dir: Path,
    case_study_topk: int,
) -> List[str]:
    return [
        python_executable,
        str(ROOT / "scripts" / "analyze_patch_observations.py"),
        "--runs",
        *[str(path) for path in run_dirs],
        "--out-dir",
        str(analysis_dir),
        "--case-study-topk",
        str(int(case_study_topk)),
    ]


def _plot_command(*, python_executable: str, analysis_dir: Path, figures_dir: Path) -> List[str]:
    return [
        python_executable,
        str(ROOT / "scripts" / "plot_patch_paper_figures.py"),
        "--analysis-dir",
        str(analysis_dir),
        "--out-dir",
        str(figures_dir),
    ]


def _run_command(command: Sequence[str], *, dry_run: bool) -> None:
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def _manifest_payload(
    *,
    work_root: Path,
    analysis_dir: Path,
    figures_dir: Path,
    dry_run: bool,
    analysis_only: bool,
    plots_only: bool,
    forwarded_agent_args: Sequence[str],
    variant_specs: Sequence[Dict[str, Any]],
    analyze_command: Sequence[str],
    plot_command: Sequence[str],
) -> Dict[str, Any]:
    return {
        "format": "patch_paper_ablation_manifest",
        "work_root": str(work_root),
        "analysis_dir": str(analysis_dir),
        "figures_dir": str(figures_dir),
        "dry_run": bool(dry_run),
        "analysis_only": bool(analysis_only),
        "plots_only": bool(plots_only),
        "forwarded_agent_args": list(forwarded_agent_args),
        "variants": [
            {
                "name": str(item.get("name") or ""),
                "search_unit": str(item.get("search_unit") or "patch"),
                "patch_memory_enabled": bool(item.get("patch_memory_enabled", True)),
                "workdir": str(item.get("workdir")),
                "command": list(item.get("command") or []),
            }
            for item in variant_specs
        ],
        "analysis_command": list(analyze_command),
        "plot_command": list(plot_command),
        "expected_outputs": {
            "analysis_tables": [
                str(analysis_dir / "patch_observations.csv"),
                str(analysis_dir / "bottleneck_patch_success.csv"),
                str(analysis_dir / "bottleneck_patch_gain.csv"),
                str(analysis_dir / "search_ablation.csv"),
                str(analysis_dir / "case_study_manifest.json"),
            ],
            "figure_files": [
                str(figures_dir / "fig_patch_sparsity.png"),
                str(figures_dir / "fig_patch_count_hist.png"),
                str(figures_dir / "fig_bottleneck_patch_success_heatmap.png"),
                str(figures_dir / "fig_bottleneck_patch_gain_heatmap.png"),
                str(figures_dir / "fig_search_ablation_curve.png"),
                str(figures_dir / "fig_stateful_vs_coarse.png"),
                str(figures_dir / "fig_reload_shift_gain.png"),
                str(figures_dir / "fig_adaptive_chunking_gain.png"),
                str(figures_dir / "fig_local_verticalization_gain.png"),
                str(figures_dir / "fig_budgeted_telemetry_cost.png"),
                str(figures_dir / "fig_case_study_compare.png"),
            ],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run the paper-priority patch-aware/whole-config ablations, then aggregate tables and figures."
    )
    parser.add_argument("--work-root", type=str, required=True)
    parser.add_argument("--analysis-dir", type=str, default=None)
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--case-study-topk", type=int, default=2)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--enable-hierarchical-orchestrator", action="store_true")
    parser.add_argument("--enable-reload-shift", action="store_true")
    parser.add_argument("--enable-adaptive-chunking", action="store_true")
    parser.add_argument("--enable-local-verticalization", action="store_true")
    parser.add_argument("--telemetry-budget", choices=["summary", "aggregated_grid", "full_debug"], default=None)
    parser.add_argument("--window-steps", type=int, default=None)
    args, extra_args = parser.parse_known_args(argv)
    return args, _normalize_forwarded_args(extra_args)


def main(argv: Sequence[str] | None = None) -> None:
    args, forwarded_agent_args = parse_args(argv)
    work_root = Path(args.work_root)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else work_root / "paper_analysis"
    figures_dir = Path(args.figures_dir) if args.figures_dir else analysis_dir / "figures"
    manifest_path = work_root / "paper_ablation_manifest.json"
    orchestrator_args: List[str] = []
    if bool(args.enable_hierarchical_orchestrator):
        orchestrator_args.append("--enable-hierarchical-orchestrator")
        orchestrator_args.append("--enable-stateful-schedule")
    if bool(args.enable_reload_shift):
        orchestrator_args.append("--enable-reload-shift")
    if bool(args.enable_adaptive_chunking):
        orchestrator_args.append("--enable-adaptive-chunking")
    if bool(args.enable_local_verticalization):
        orchestrator_args.append("--enable-local-verticalization")
    if args.telemetry_budget:
        orchestrator_args.extend(["--telemetry-budget", str(args.telemetry_budget)])
    if args.window_steps is not None:
        orchestrator_args.extend(["--window-steps", str(int(args.window_steps))])

    variants = _variant_specs(work_root)
    for item in variants:
        item["command"] = _agent_loop_command(
            python_executable=str(args.python),
            workdir=Path(item["workdir"]),
            forwarded_args=forwarded_agent_args,
            variant_extra_args=list(item.get("extra_args") or []),
            orchestrator_args=orchestrator_args,
        )

    analyze_command = _analysis_command(
        python_executable=str(args.python),
        run_dirs=[Path(item["workdir"]) for item in variants],
        analysis_dir=analysis_dir,
        case_study_topk=int(args.case_study_topk),
    )
    plot_command = _plot_command(
        python_executable=str(args.python),
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
    )

    manifest = _manifest_payload(
        work_root=work_root,
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        dry_run=bool(args.dry_run),
        analysis_only=bool(args.analysis_only),
        plots_only=bool(args.plots_only),
        forwarded_agent_args=forwarded_agent_args,
        variant_specs=variants,
        analyze_command=analyze_command,
        plot_command=plot_command,
    )
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.analysis_only and not args.plots_only:
        for item in variants:
            _run_command(item["command"], dry_run=bool(args.dry_run))

    if not args.plots_only:
        _run_command(analyze_command, dry_run=bool(args.dry_run))

    _run_command(plot_command, dry_run=bool(args.dry_run))

    print(f"manifest: {manifest_path}")
    for item in variants:
        print(f"variant[{item['name']}]: {' '.join(item['command'])}")
    print(f"analysis: {' '.join(analyze_command)}")
    print(f"figures: {' '.join(plot_command)}")


if __name__ == "__main__":
    main()
