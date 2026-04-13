from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from megatron_agent.config import (
    MegatronProgram,
    MegatronStrategy,
    default_backend_caps,
    default_dense_program,
    default_moe_smoke_program,
    validate_strategy,
)
from megatron_agent.metrics_parser import parse_megatron_logs
from megatron_agent.programs import CompiledProgram, compile_program
from megatron_agent.trace_reducer import (
    build_agent_observation,
    build_trial_artifact,
    classify_bottleneck,
    infer_program_patch_category,
    infer_program_patch_count,
    reduce_trial_trace,
)

DEFAULT_MEGATRON_ROOT = "/public/home/ssjxscy/agent/Megatron-LM"
DEFAULT_LAUNCHER_SCRIPT = "examples/qwen/train_qwen3_14b_rtx_8gpu.sh"
DEFAULT_TOKENIZER_MODEL = "/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
DEFAULT_DATA_PATH = "/public/home/ssjxscy/datasets/wikitext-103-raw-v1/data/processed_wikitext_text_document"

# Backward-compatible aliases for older imports.
DEFAULT_G5_MEGATRON_ROOT = DEFAULT_MEGATRON_ROOT
DEFAULT_G5_LAUNCHER_SCRIPT = DEFAULT_LAUNCHER_SCRIPT
DEFAULT_G5_TOKENIZER_MODEL = DEFAULT_TOKENIZER_MODEL
DEFAULT_G5_DATA_PATH = DEFAULT_DATA_PATH

ProgramLike = Union[MegatronProgram, MegatronStrategy]


@dataclass
class _GpuSnapshot:
    index: int
    memory_total_mib: int
    memory_free_mib: int
    memory_used_mib: int


def _load_args_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    args: List[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        args.extend(shlex.split(raw))
    return args


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(str(value).strip())
    except Exception:
        return default


def _parse_cuda_visible_devices(raw: Optional[str], nproc: int) -> List[int]:
    if raw:
        devices = [_safe_int(token) for token in str(raw).split(",")]
        parsed = [int(token) for token in devices if token is not None]
        return parsed[: max(int(nproc), 0)] if parsed else list(range(int(nproc)))
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value:
        devices = [_safe_int(token) for token in str(env_value).split(",")]
        parsed = [int(token) for token in devices if token is not None]
        return parsed[: max(int(nproc), 0)] if parsed else list(range(int(nproc)))
    return list(range(max(int(nproc), 0)))


def _resolve_megatron_entry(root: Optional[str], entry: str) -> Path:
    if os.path.isabs(entry):
        return Path(entry)
    if root:
        return Path(root) / entry
    env_root = os.environ.get("MEGATRON_LM_ROOT")
    if env_root:
        return Path(env_root) / entry
    return Path(entry)


def _resolve_launcher_script(root: Optional[str], launcher_script: Optional[str]) -> Optional[Path]:
    if not launcher_script:
        return None
    if os.path.isabs(launcher_script):
        return Path(launcher_script)
    if root:
        return Path(root) / launcher_script
    return Path(launcher_script)


def _default_run_root(args: argparse.Namespace) -> Path:
    return Path(args.run_root or "./runs_megatron").resolve()


def _trial_output_dirs(args: argparse.Namespace, trial_id: int) -> Dict[str, str]:
    run_root = _default_run_root(args)
    trial_dir = run_root / f"trial_{int(trial_id):03d}"
    return {
        "trial_dir": str(trial_dir),
        "analysis_dir": str(trial_dir / "analysis"),
        "checkpoint_path": str(trial_dir / "checkpoints"),
        "tensorboard_path": str(trial_dir / "tensorboard"),
        "torch_profile_path": str(trial_dir / "torch_profile"),
        "torchrun_log_dir": str(trial_dir / "torchrun_logs"),
        "chakra_path": str(trial_dir / "chakra"),
        "nsys_path": str(trial_dir / "nsys"),
        "data_cache_path": str(trial_dir / "cache"),
        "runtime_trace_dir": str(trial_dir / "runtime_schedule_traces"),
    }


def add_observability_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--observability-preset", type=str, choices=["none", "basic", "deep"], default="none")
    parser.add_argument("--profile-step-start", type=int, default=4)
    parser.add_argument("--profile-step-end", type=int, default=6)
    parser.add_argument("--profile-ranks", type=int, nargs="*", default=None)
    parser.add_argument("--enable-pytorch-profiler", action="store_true")
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-collect-callstack", action="store_true")
    parser.add_argument("--profile-collect-chakra", action="store_true")
    parser.add_argument("--enable-log-timers-to-tensorboard", action="store_true")
    parser.add_argument("--enable-log-memory-to-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log-interval", type=int, default=1)
    parser.add_argument("--enable-memory-history", action="store_true")
    parser.add_argument("--memory-snapshot-path", type=str, default="snapshot.pickle")
    parser.add_argument("--enable-straggler-log", action="store_true")
    parser.add_argument("--disable-straggler-on-startup", action="store_true")
    parser.add_argument("--straggler-minmax-count", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-exp-name", type=str, default="")
    parser.add_argument("--wandb-save-dir", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--enable-nsys", action="store_true")
    parser.add_argument("--nsys-output", type=str, default=None)
    parser.add_argument("--nsys-trace", type=str, default="cuda,nvtx")
    parser.add_argument("--stream-trial-logs", dest="stream_trial_logs", action="store_true")
    parser.add_argument("--no-stream-trial-logs", dest="stream_trial_logs", action="store_false")
    parser.set_defaults(stream_trial_logs=True)
    return parser


def _resolve_trial_relative_path(value: Optional[str], trial_dir: str, fallback: str) -> str:
    raw = str(value or fallback)
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str(Path(trial_dir) / path)


def _build_observability_config(
    args: argparse.Namespace,
    trial_id: int,
    output_dirs: Dict[str, str],
) -> Dict[str, Any]:
    preset = str(getattr(args, "observability_preset", "none") or "none").lower()
    enable_log_timers = bool(getattr(args, "enable_log_timers_to_tensorboard", False) or preset in {"basic", "deep"})
    enable_log_memory = bool(getattr(args, "enable_log_memory_to_tensorboard", False) or preset in {"basic", "deep"})
    enable_pytorch_profiler = bool(getattr(args, "enable_pytorch_profiler", False) or preset == "deep")
    enable_memory_history = bool(getattr(args, "enable_memory_history", False) or preset == "deep")
    enable_straggler = bool(getattr(args, "enable_straggler_log", False) or preset == "deep")
    enable_nsys = bool(getattr(args, "enable_nsys", False))
    profile_enabled = bool(getattr(args, "enable_profile", False) or enable_pytorch_profiler or enable_nsys)
    profile_step_start = max(int(getattr(args, "profile_step_start", 4) or 4), 0)
    profile_step_end = max(int(getattr(args, "profile_step_end", 6) or 6), profile_step_start + 1)
    raw_ranks = list(getattr(args, "profile_ranks", []) or [])
    profile_ranks = [int(rank) for rank in raw_ranks] if raw_ranks else ([0] if profile_enabled else [])

    wandb_project = str(getattr(args, "wandb_project", "") or "").strip()
    wandb_exp_name = str(getattr(args, "wandb_exp_name", "") or "").strip()
    wandb_save_dir = str(getattr(args, "wandb_save_dir", "") or "").strip()
    wandb_entity = str(getattr(args, "wandb_entity", "") or "").strip()
    if wandb_project and not wandb_exp_name:
        raise ValueError("wandb_project requires wandb_exp_name")
    if wandb_project and not wandb_save_dir:
        wandb_save_dir = str(Path(output_dirs["checkpoint_path"]) / "wandb")

    memory_snapshot_path = None
    if enable_memory_history:
        memory_snapshot_path = _resolve_trial_relative_path(
            getattr(args, "memory_snapshot_path", None),
            output_dirs["trial_dir"],
            "snapshot.pickle",
        )

    nsys_output = None
    if enable_nsys:
        nsys_output = _resolve_trial_relative_path(
            getattr(args, "nsys_output", None),
            output_dirs["trial_dir"],
            str(Path("nsys") / f"trial_{int(trial_id):03d}"),
        )

    return {
        "preset": preset,
        "profile_enabled": profile_enabled,
        "profile_step_start": profile_step_start,
        "profile_step_end": profile_step_end,
        "profile_ranks": profile_ranks,
        "enable_pytorch_profiler": enable_pytorch_profiler,
        "profile_record_shapes": bool(getattr(args, "profile_record_shapes", False) or preset == "deep"),
        "profile_collect_callstack": bool(getattr(args, "profile_collect_callstack", False)),
        "profile_collect_chakra": bool(getattr(args, "profile_collect_chakra", False)),
        "enable_log_timers_to_tensorboard": enable_log_timers,
        "enable_log_memory_to_tensorboard": enable_log_memory,
        "tensorboard_log_interval": max(int(getattr(args, "tensorboard_log_interval", 1) or 1), 1),
        "enable_memory_history": enable_memory_history,
        "memory_snapshot_path": memory_snapshot_path,
        "enable_straggler_log": enable_straggler,
        "disable_straggler_on_startup": bool(getattr(args, "disable_straggler_on_startup", False)),
        "straggler_minmax_count": max(int(getattr(args, "straggler_minmax_count", 8) or 1), 1),
        "wandb_project": wandb_project,
        "wandb_exp_name": wandb_exp_name,
        "wandb_save_dir": wandb_save_dir,
        "wandb_entity": wandb_entity,
        "enable_wandb": bool(wandb_project),
        "enable_nsys": enable_nsys,
        "nsys_output": nsys_output,
        "nsys_trace": str(getattr(args, "nsys_trace", "cuda,nvtx") or "cuda,nvtx"),
        "tensorboard_path": output_dirs["tensorboard_path"],
        "torch_profile_path": output_dirs["torch_profile_path"],
        "chakra_path": output_dirs["chakra_path"],
        "nsys_path": output_dirs["nsys_path"],
    }


def _build_nsys_command(observability: Dict[str, Any]) -> Optional[List[str]]:
    if not bool(observability.get("enable_nsys")):
        return None
    return [
        "nsys",
        "profile",
        "-s",
        "none",
        "-t",
        str(observability["nsys_trace"]),
        "-o",
        str(observability["nsys_output"]),
        "--force-overwrite",
        "true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end",
        "stop",
    ]


def _prepare_trial_artifact_dirs(output_dirs: Dict[str, str], observability: Dict[str, Any]) -> None:
    trial_dir = Path(output_dirs["trial_dir"])
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    for key in (
        "trial_dir",
        "analysis_dir",
        "checkpoint_path",
        "tensorboard_path",
        "data_cache_path",
        "torch_profile_path",
        "torchrun_log_dir",
        "chakra_path",
        "nsys_path",
        "runtime_trace_dir",
    ):
        path = output_dirs.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
    memory_snapshot_path = observability.get("memory_snapshot_path")
    if memory_snapshot_path:
        Path(str(memory_snapshot_path)).parent.mkdir(parents=True, exist_ok=True)
    nsys_output = observability.get("nsys_output")
    if nsys_output:
        Path(str(nsys_output)).parent.mkdir(parents=True, exist_ok=True)


def _trial_log_paths(output_dirs: Dict[str, str]) -> Dict[str, str]:
    trial_dir = Path(output_dirs["trial_dir"])
    return {
        "stdout_log": str(trial_dir / "stdout.log"),
        "stderr_log": str(trial_dir / "stderr.log"),
        "launch_plan_log": str(trial_dir / "launch_plan.json"),
    }


def _query_gpu_snapshots() -> List[_GpuSnapshot]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    snapshots: List[_GpuSnapshot] = []
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index = _safe_int(parts[0])
        total = _safe_int(parts[1])
        free = _safe_int(parts[2])
        used = _safe_int(parts[3])
        if None in {index, total, free, used}:
            continue
        snapshots.append(
            _GpuSnapshot(
                index=int(index),
                memory_total_mib=int(total),
                memory_free_mib=int(free),
                memory_used_mib=int(used),
            )
        )
    return snapshots


def _preflight_gpu_guard(args: argparse.Namespace) -> Optional[str]:
    if bool(getattr(args, "allow_busy_gpus", False)):
        return None
    snapshots = _query_gpu_snapshots()
    if not snapshots:
        return None
    selected = set(_parse_cuda_visible_devices(getattr(args, "cuda_visible_devices", None), int(args.nproc)))
    issues: List[str] = []
    min_free_mib = int(getattr(args, "min_free_gpu_memory_mib", 28672) or 0)
    busy_used_mib = int(getattr(args, "busy_gpu_memory_threshold_mib", 2048) or 0)
    for snap in snapshots:
        if snap.index not in selected:
            continue
        if snap.memory_free_mib < min_free_mib or snap.memory_used_mib > busy_used_mib:
            issues.append(
                f"gpu{snap.index}: free={snap.memory_free_mib}MiB used={snap.memory_used_mib}MiB total={snap.memory_total_mib}MiB"
            )
    if not issues:
        return None
    return (
        "GPU preflight rejected busy devices. "
        f"Selected GPUs={sorted(selected)} require free>={min_free_mib}MiB and used<={busy_used_mib}MiB. "
        "Observed: " + "; ".join(issues)
    )


def _consume_stream(
    pipe: Any,
    sink_path: str,
    console: Any,
    chunks: List[str],
) -> None:
    with Path(sink_path).open("w", encoding="utf-8") as sink:
        while True:
            line = pipe.readline()
            if not line:
                break
            chunks.append(line)
            sink.write(line)
            sink.flush()
            console.write(line)
            console.flush()
    pipe.close()


def _run_command_with_logging(
    cmd: List[str],
    cwd: Optional[str],
    env: Dict[str, str],
    stdout_log: str,
    stderr_log: str,
    stream_output: bool,
) -> subprocess.CompletedProcess[str]:
    if not stream_output:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
        )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=cwd,
        env=env,
    )
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    stdout_thread = threading.Thread(
        target=_consume_stream,
        args=(proc.stdout, stdout_log, sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_consume_stream,
        args=(proc.stderr, stderr_log, sys.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def _write_analysis_artifacts(output_dirs: Dict[str, str], metrics: Dict[str, Any]) -> Dict[str, str]:
    artifact_dir = Path(
        str(output_dirs.get("analysis_dir") or (Path(output_dirs["trial_dir"]) / "analysis"))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = dict(metrics.get("trial_artifact") or {})
    context_record = dict(metrics.get("context_record") or {})
    visualization = dict(artifact.get("visualization_artifacts") or {})
    perfetto_trace = dict(visualization.get("perfetto_trace") or {})
    pipeline_schedule_projection = dict(visualization.get("pipeline_schedule_projection") or {})
    pipeline_event_trace = dict(visualization.get("pipeline_event_trace") or {})
    pipeline_grid_trace = dict(visualization.get("pipeline_grid_trace") or {})
    pipeline_projection_svg = dict(visualization.get("pipeline_projection_svg") or {})
    compare_pipeline_svg = dict(visualization.get("compare_pipeline_svg") or {})
    search_space_blueprint = dict(artifact.get("search_space_blueprint") or {})
    bottleneck_breakdown = list(artifact.get("bottleneck_breakdown") or [])
    critical_operator_clusters = list(artifact.get("critical_operator_clusters") or [])
    trial_outcome = dict(metrics.get("trial_outcome") or {})
    trial_reflection = dict(metrics.get("trial_reflection") or {})
    failure_diagnosis = _build_failure_diagnosis(metrics)
    runtime_schedule_traces = list(metrics.get("runtime_schedule_traces") or [])

    outputs = {
        "trial_artifact_json": str(artifact_dir / "trial_artifact.json"),
        "context_record_json": str(artifact_dir / "context_record.json"),
        "perfetto_trace_json": str(artifact_dir / "perfetto_trace.json"),
        "pipeline_schedule_projection_json": str(artifact_dir / "pipeline_schedule_projection.json"),
        "pipeline_event_trace_json": str(artifact_dir / "pipeline_event_trace.json"),
        "pipeline_grid_trace_json": str(artifact_dir / "pipeline_grid_trace.json"),
        "pipeline_projection_svg": str(artifact_dir / "pipeline_projection.svg"),
        "compare_pipeline_svg": str(artifact_dir / "compare_pipeline.svg"),
        "search_space_blueprint_json": str(artifact_dir / "search_space_blueprint.json"),
        "bottleneck_breakdown_json": str(artifact_dir / "bottleneck_breakdown.json"),
        "critical_operator_clusters_json": str(artifact_dir / "critical_operator_clusters.json"),
        "trial_outcome_json": str(artifact_dir / "trial_outcome.json"),
        "trial_reflection_json": str(artifact_dir / "trial_reflection.json"),
        "failure_diagnosis_json": str(artifact_dir / "failure_diagnosis.json"),
        "runtime_schedule_traces_json": str(artifact_dir / "runtime_schedule_traces.json"),
    }
    Path(outputs["trial_artifact_json"]).write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(outputs["context_record_json"]).write_text(json.dumps(context_record, indent=2, ensure_ascii=False), encoding="utf-8")
    if perfetto_trace:
        Path(outputs["perfetto_trace_json"]).write_text(json.dumps(perfetto_trace, indent=2, ensure_ascii=False), encoding="utf-8")
    if pipeline_schedule_projection:
        Path(outputs["pipeline_schedule_projection_json"]).write_text(
            json.dumps(pipeline_schedule_projection, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if pipeline_event_trace:
        Path(outputs["pipeline_event_trace_json"]).write_text(
            json.dumps(pipeline_event_trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if pipeline_grid_trace:
        Path(outputs["pipeline_grid_trace_json"]).write_text(
            json.dumps(pipeline_grid_trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if str(pipeline_projection_svg.get("content") or "").strip():
        Path(outputs["pipeline_projection_svg"]).write_text(
            str(pipeline_projection_svg.get("content") or ""),
            encoding="utf-8",
        )
    if str(compare_pipeline_svg.get("content") or "").strip():
        Path(outputs["compare_pipeline_svg"]).write_text(
            str(compare_pipeline_svg.get("content") or ""),
            encoding="utf-8",
        )
    if search_space_blueprint:
        Path(outputs["search_space_blueprint_json"]).write_text(
            json.dumps(search_space_blueprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if bottleneck_breakdown:
        Path(outputs["bottleneck_breakdown_json"]).write_text(
            json.dumps(bottleneck_breakdown, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if critical_operator_clusters:
        Path(outputs["critical_operator_clusters_json"]).write_text(
            json.dumps(critical_operator_clusters, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if trial_outcome:
        Path(outputs["trial_outcome_json"]).write_text(
            json.dumps(trial_outcome, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if trial_reflection:
        Path(outputs["trial_reflection_json"]).write_text(
            json.dumps(trial_reflection, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if failure_diagnosis:
        Path(outputs["failure_diagnosis_json"]).write_text(
            json.dumps(failure_diagnosis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if runtime_schedule_traces:
        Path(outputs["runtime_schedule_traces_json"]).write_text(
            json.dumps(runtime_schedule_traces, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return outputs


def _load_runtime_schedule_traces(output_dirs: Dict[str, str]) -> List[Dict[str, Any]]:
    trace_dir = Path(str(output_dirs.get("runtime_trace_dir") or ""))
    if not trace_dir.exists():
        return []
    traces: List[Dict[str, Any]] = []
    for path in sorted(trace_dir.glob("schedule_runtime_trace_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("artifact_path", str(path))
            traces.append(payload)
    return traces


def _build_failure_diagnosis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    context_record = dict(metrics.get("context_record") or {})
    trial_artifact = dict(metrics.get("trial_artifact") or {})
    trial_outcome = dict(metrics.get("trial_outcome") or {})
    trial_reflection = dict(metrics.get("trial_reflection") or {})
    runtime = dict((context_record.get("runtime_evidence") or {}))
    optimization_hints = list(trial_artifact.get("optimization_hints") or context_record.get("optimization_hints") or [])
    failure_modes = list(trial_artifact.get("failure_modes") or context_record.get("failure_modes") or [])
    derived_bottlenecks = list(
        trial_artifact.get("derived_bottlenecks") or context_record.get("derived_bottlenecks") or []
    )

    recommended_actions: List[Dict[str, Any]] = []
    for item in optimization_hints[:5]:
        scope = str((item or {}).get("scope") or "").strip()
        rationale = str((item or {}).get("rationale") or "").strip()
        action = str((item or {}).get("action") or "").strip()
        if scope or rationale or action:
            recommended_actions.append(
                {
                    "source": "optimization_hint",
                    "scope": scope,
                    "action": action or "adjust_runtime_policy",
                    "rationale": rationale,
                }
            )
    next_action = str(trial_reflection.get("recommended_next_action") or "").strip()
    if next_action:
        recommended_actions.insert(
            0,
            {
                "source": "trial_reflection",
                "action": next_action,
                "summary": str(trial_reflection.get("summary") or "").strip(),
            },
        )

    return {
        "config_name": str(metrics.get("config_name") or ""),
        "program_kind": str(trial_artifact.get("program_kind") or ""),
        "returncode": int(metrics.get("returncode") or 0),
        "oom": bool(metrics.get("oom", False)),
        "error_msg": str(metrics.get("error_msg") or ""),
        "root_cause_excerpt": str(metrics.get("root_cause_excerpt") or ""),
        "stdout_tail": str(metrics.get("stdout_tail") or ""),
        "stderr_tail": str(metrics.get("stderr_tail") or ""),
        "failure_modes": failure_modes,
        "derived_bottlenecks": derived_bottlenecks,
        "bottleneck_signature": dict(metrics.get("bottleneck_signature") or {}),
        "optimization_hints": optimization_hints,
        "recommended_actions": recommended_actions,
        "trial_outcome": trial_outcome,
        "trial_reflection": trial_reflection,
        "runtime_focus": {
            "bubble_ratio": float(runtime.get("bubble_ratio") or 0.0),
            "peak_reserved_ratio": float(runtime.get("peak_reserved_ratio") or 0.0),
            "stage_tail_ratio": float(runtime.get("stage_tail_ratio") or 0.0),
            "tail_step_jitter_ratio": float(runtime.get("tail_step_jitter_ratio") or 0.0),
            "optimizer_exposed_ratio": float(runtime.get("optimizer_exposed_ratio") or 0.0),
            "comm_exposure_ratio": float(runtime.get("comm_exposure_ratio") or 0.0),
        },
    }


def write_analysis_artifacts_for_trial(
    args: argparse.Namespace,
    trial_id: int,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    output_dirs = _trial_output_dirs(args, trial_id)
    paths = _write_analysis_artifacts(output_dirs, metrics)
    metrics["analysis_artifact_paths"] = paths
    return paths


def _runtime_env_defaults() -> Dict[str, str]:
    env = {
        "CUDA_DEVICE_MAX_CONNECTIONS": str(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1")),
        "TORCH_NCCL_AVOID_RECORD_STREAMS": str(os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")),
        "PYTORCH_CUDA_ALLOC_CONF": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")),
        "NCCL_DEBUG": str(os.environ.get("NCCL_DEBUG", "WARN")),
        "PYTHONFAULTHANDLER": str(os.environ.get("PYTHONFAULTHANDLER", "1")),
        "TORCH_SHOW_CPP_STACKTRACES": str(os.environ.get("TORCH_SHOW_CPP_STACKTRACES", "1")),
    }
    cuda_home = _discover_cuda_home()
    if cuda_home:
        env.setdefault("CUDA_HOME", cuda_home)
        env.setdefault("CUDA_PATH", cuda_home)
        env.setdefault("CUDACXX", str(Path(cuda_home) / "bin" / "nvcc"))
    return env


def _discover_cuda_home() -> Optional[str]:
    candidates: List[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        raw = os.environ.get(key)
        if raw:
            candidates.append(Path(raw))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)

    for raw in (
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda-11.8",
        "/opt/cuda",
    ):
        candidates.append(Path(raw))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = str(candidate.resolve())
        except OSError:
            normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        nvcc_path = candidate / "bin" / "nvcc"
        if nvcc_path.exists():
            return str(candidate)
    return None


def _validate_cuda_toolchain() -> Dict[str, str]:
    runtime_env = _runtime_env_defaults()
    cuda_home = runtime_env.get("CUDA_HOME")
    cudacxx = runtime_env.get("CUDACXX")
    if not cuda_home or not cudacxx:
        raise RuntimeError(
            "CUDA toolchain not resolved for Megatron fused kernel compilation. "
            "Set CUDA_HOME/CUDA_PATH/CUDACXX so that nvcc is available."
        )
    if not Path(cudacxx).exists():
        raise RuntimeError(f"CUDACXX path does not exist: {cudacxx}")
    return runtime_env


def _resolve_transformer_impl(args: argparse.Namespace, program: MegatronProgram) -> str:
    raw = str(getattr(args, "transformer_impl", "auto") or "auto").strip().lower()
    if raw and raw != "auto":
        return raw
    if str(program.cluster.target) == "single_g5" and str(program.model.track) == "dense":
        return "transformer_engine"
    return "local"


def _resolve_backend_caps(args: argparse.Namespace, program: MegatronProgram) -> Dict[str, Any]:
    transformer_impl = _resolve_transformer_impl(args, program)
    return default_backend_caps(transformer_impl).to_dict()


def _sequence_parallel_requested(strategy: MegatronStrategy) -> bool:
    return bool(strategy.parallel.sp_enabled) and int(strategy.parallel.tp_degree) > 1


def _sequence_parallel_active(args: argparse.Namespace, program: MegatronProgram, strategy: MegatronStrategy) -> bool:
    backend_caps = _resolve_backend_caps(args, program)
    return _sequence_parallel_requested(strategy) and bool(backend_caps["supports_sequence_parallel"])


def _validate_runtime_stack(args: argparse.Namespace, program: MegatronProgram) -> Dict[str, str]:
    transformer_impl = _resolve_transformer_impl(args, program)
    if transformer_impl != "transformer_engine":
        return {"transformer_impl": transformer_impl}

    try:
        transformer_engine = importlib.import_module("transformer_engine")
        te_optim = importlib.import_module("transformer_engine.pytorch.optimizers")
        getattr(te_optim, "FusedAdam")
    except Exception as exc:
        raise RuntimeError(
            "transformer_engine path requires Transformer Engine with "
            "transformer_engine.pytorch.optimizers.FusedAdam available in the active environment."
        ) from exc

    details = {
        "transformer_impl": transformer_impl,
        "transformer_engine_version": str(getattr(transformer_engine, "__version__", "unknown")),
    }
    if str(program.cluster.target) == "single_g5" and str(program.model.track) == "dense":
        try:
            apex = importlib.import_module("apex")
            apex_optim = importlib.import_module("apex.optimizers")
            getattr(apex_optim, "FusedAdam")
        except Exception as exc:
            raise RuntimeError(
                "single_g5 dense high-performance path requires Apex with "
                "apex.optimizers.FusedAdam available in the active environment."
            ) from exc
        details["apex_path"] = str(getattr(apex, "__file__", ""))
    return details
def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _extract_error_excerpt_from_text(text: str, radius: int = 1200) -> Optional[str]:
    if not text or not text.strip():
        return None
    markers = [
        "Traceback (most recent call last):",
        "AssertionError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "KeyError",
        "ImportError",
        "ModuleNotFoundError",
        "FileNotFoundError",
        "PermissionError",
        "CUDA out of memory",
        "NCCL error",
        "NCCL WARN",
        "NCCL timeout",
        "Segmentation fault",
        "Signal 11",
        "Signal 15",
    ]
    for marker in markers:
        index = text.rfind(marker)
        if index >= 0:
            start = max(index - radius, 0)
            end = min(index + radius, len(text))
            return text[start:end].strip()
    return None


def _collect_torchrun_log_files(torchrun_log_dir: str) -> List[str]:
    root = Path(torchrun_log_dir)
    if not root.exists():
        return []
    files: List[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(str(path))
    return files


def _extract_failure_details(stdout_text: str, stderr_text: str, torchrun_log_dir: str) -> Dict[str, Any]:
    torchrun_log_files = _collect_torchrun_log_files(torchrun_log_dir)
    for file_path in torchrun_log_files:
        excerpt = _extract_error_excerpt_from_text(_safe_read_text(Path(file_path)))
        if excerpt:
            return {
                "root_cause_source": file_path,
                "root_cause_excerpt": excerpt[-4000:],
                "torchrun_log_files": torchrun_log_files,
            }
    for source_name, text in (("stderr", stderr_text), ("stdout", stdout_text)):
        excerpt = _extract_error_excerpt_from_text(text)
        if excerpt:
            return {
                "root_cause_source": source_name,
                "root_cause_excerpt": excerpt[-4000:],
                "torchrun_log_files": torchrun_log_files,
            }
    return {"torchrun_log_files": torchrun_log_files}


def _dense_shape_args(args: argparse.Namespace, program: MegatronProgram, strategy: MegatronStrategy) -> List[str]:
    backend_caps = _resolve_backend_caps(args, program)
    transformer_impl = backend_caps["transformer_impl"]
    shape_args = [
        "--num-layers",
        str(int(program.model.num_layers)),
        "--hidden-size",
        str(int(args.hidden_size)),
        "--ffn-hidden-size",
        str(int(args.ffn_hidden_size)),
        "--num-attention-heads",
        str(int(args.num_attention_heads)),
        "--kv-channels",
        str(int(args.kv_channels)),
        "--group-query-attention",
        "--num-query-groups",
        str(int(args.num_query_groups)),
        "--seq-length",
        str(int(strategy.seq_len)),
        "--max-position-embeddings",
        str(int(args.max_position_embeddings or strategy.seq_len)),
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--rotary-base",
        "1000000",
        "--use-rotary-position-embeddings",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--qk-layernorm",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--attention-softmax-in-fp32",
        "--no-masked-softmax-fusion",
        "--make-vocab-size-divisible-by",
        "128",
        "--vocab-size",
        str(int(args.vocab_size)),
    ]
    if transformer_impl != "transformer_engine":
        shape_args += ["--no-rope-fusion", "--no-persist-layer-norm"]
    return shape_args


def _moe_shape_args(args: argparse.Namespace, program: MegatronProgram, strategy: MegatronStrategy) -> List[str]:
    backend_caps = _resolve_backend_caps(args, program)
    transformer_impl = backend_caps["transformer_impl"]
    shape_args = [
        "--num-layers",
        str(int(program.model.num_layers)),
        "--hidden-size",
        str(int(args.moe_hidden_size)),
        "--ffn-hidden-size",
        str(int(args.moe_ffn_hidden_size)),
        "--num-attention-heads",
        str(int(args.moe_num_attention_heads)),
        "--kv-channels",
        str(int(args.moe_kv_channels)),
        "--num-query-groups",
        str(int(args.moe_num_query_groups)),
        "--seq-length",
        str(int(strategy.seq_len)),
        "--max-position-embeddings",
        str(int(args.moe_max_position_embeddings or strategy.seq_len)),
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--swiglu",
        "--disable-bias-linear",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--make-vocab-size-divisible-by",
        "128",
        "--vocab-size",
        str(int(args.moe_vocab_size)),
        "--num-experts",
        str(int(program.model.num_experts or 4)),
        "--moe-layer-freq",
        str(int(program.model.moe_layer_freq or 2)),
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-router-topk",
        str(int(program.metadata.get("moe_router_topk", 2) or 2)),
        "--moe-token-dispatcher-type",
        "alltoall",
        "--expert-model-parallel-size",
        str(int(program.parallel.ep_degree)),
        "--expert-tensor-parallel-size",
        str(int(program.parallel.expert_tp_degree)),
    ]
    if transformer_impl != "transformer_engine":
        shape_args += ["--no-rope-fusion", "--no-persist-layer-norm"]
    return shape_args


def _env_flag_enabled(env: Dict[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _env_csv_tokens(env: Dict[str, str], key: str) -> List[str]:
    raw = str(env.get(key) or "").strip()
    if not raw:
        return []
    tokens: List[str] = []
    for part in raw.replace(",", " ").split():
        token = str(part).strip()
        if token:
            tokens.append(token)
    return tokens


def _training_args(
    args: argparse.Namespace,
    program: MegatronProgram,
    strategy: MegatronStrategy,
    trial_id: int,
    launcher_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    output_dirs = _trial_output_dirs(args, trial_id)
    observability = _build_observability_config(args, trial_id=trial_id, output_dirs=output_dirs)
    backend_caps = _resolve_backend_caps(args, program)
    transformer_impl = backend_caps["transformer_impl"]
    sequence_parallel_active = _sequence_parallel_active(args, program, strategy)
    launcher_env = dict(launcher_env or {})
    enable_distributed_optimizer = _env_flag_enabled(launcher_env, "ENABLE_DISTRIBUTED_OPTIMIZER", True)
    enable_overlap_grad_reduce = _env_flag_enabled(
        launcher_env,
        "ENABLE_OVERLAP_GRAD_REDUCE",
        enable_distributed_optimizer,
    )
    enable_overlap_param_gather = _env_flag_enabled(
        launcher_env,
        "ENABLE_OVERLAP_PARAM_GATHER",
        enable_distributed_optimizer,
    )
    enable_overlap_param_gather_with_optimizer_step = _env_flag_enabled(
        launcher_env,
        "ENABLE_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP",
        False,
    )
    runtime_recompute_granularity = str(
        launcher_env.get("RECOMPUTE_GRANULARITY") or strategy.recompute_granularity or ""
    ).strip()
    enable_recompute_activations = _env_flag_enabled(
        launcher_env,
        "ENABLE_RECOMPUTE_ACTIVATIONS",
        bool(runtime_recompute_granularity),
    )
    runtime_recompute_modules = _env_csv_tokens(launcher_env, "RECOMPUTE_MODULES")
    if not runtime_recompute_modules and runtime_recompute_granularity == "selective":
        runtime_recompute_modules = ["core_attn"]
    enable_fine_grained_activation_offloading = _env_flag_enabled(
        launcher_env,
        "ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING",
        False,
    )
    runtime_offload_modules = _env_csv_tokens(launcher_env, "OFFLOAD_MODULES")
    common = [
        "--use-mcore-models",
        "--transformer-impl",
        transformer_impl,
        "--attention-backend",
        str(args.attention_backend),
        "--micro-batch-size",
        str(int(strategy.micro_batch_size)),
        "--global-batch-size",
        str(int(strategy.global_batch_size)),
        "--train-iters",
        str(int(args.train_iters)),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--lr-decay-style",
        str(args.lr_decay_style),
        "--lr-warmup-iters",
        str(int(args.lr_warmup_iters)),
        "--clip-grad",
        str(args.clip_grad),
        "--weight-decay",
        str(args.weight_decay),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-eps",
        str(args.adam_eps),
        "--calculate-per-token-loss",
        "--log-interval",
        str(int(args.log_interval)),
        "--timing-log-level",
        "2",
        "--log-throughput",
        "--ckpt-format",
        "torch_dist",
        "--distributed-timeout-minutes",
        str(int(args.distributed_timeout_minutes)),
        "--tensorboard-dir",
        output_dirs["tensorboard_path"],
        "--tensorboard-log-interval",
        str(int(observability["tensorboard_log_interval"])),
        "--data-cache-path",
        output_dirs["data_cache_path"],
    ]
    if enable_distributed_optimizer:
        common.append("--use-distributed-optimizer")
    if enable_overlap_grad_reduce:
        common.append("--overlap-grad-reduce")
    if enable_overlap_param_gather:
        common.append("--overlap-param-gather")
    if enable_overlap_param_gather_with_optimizer_step:
        common.append("--overlap-param-gather-with-optimizer-step")
    if strategy.use_bf16:
        common.append("--bf16")
    elif strategy.use_fp16:
        common.append("--fp16")
    if runtime_recompute_granularity:
        common += ["--recompute-granularity", runtime_recompute_granularity]
    if enable_recompute_activations:
        common.append("--recompute-activations")
    if runtime_recompute_modules:
        common += ["--recompute-modules", *runtime_recompute_modules]
    if enable_fine_grained_activation_offloading:
        common.append("--fine-grained-activation-offloading")
        if runtime_offload_modules:
            common += ["--offload-modules", *runtime_offload_modules]
    if not backend_caps["supports_sequence_parallel"]:
        common.append("--no-gradient-accumulation-fusion")
    if args.enable_tp_comm_overlap:
        if not sequence_parallel_active or not backend_caps["supports_tp_comm_overlap"]:
            raise ValueError("tp_comm_overlap requires tp_degree > 1 with sequence parallel enabled")
        common.append("--tp-comm-overlap")
    if sequence_parallel_active:
        common.append("--sequence-parallel")
    eval_iters = int(args.eval_iters)
    eval_interval = int(args.eval_interval)
    if eval_iters > 0:
        common += ["--eval-iters", str(eval_iters), "--eval-interval", str(max(eval_interval, 1))]
    else:
        # Megatron's current defaults are eval_iters=100 and eval_interval=None.
        # If we omit both flags when eval is disabled, dataset sizing can fail
        # before training starts. Pass an explicit no-eval pair instead.
        common += ["--eval-iters", "0", "--eval-interval", "1"]
    if int(args.save_interval) > 0:
        common += ["--save", output_dirs["checkpoint_path"], "--save-interval", str(int(args.save_interval))]
    if observability["enable_log_timers_to_tensorboard"]:
        common.append("--log-timers-to-tensorboard")
    if observability["enable_log_memory_to_tensorboard"]:
        common.append("--log-memory-to-tensorboard")
    if observability["profile_enabled"]:
        common += [
            "--profile",
            "--profile-step-start",
            str(int(observability["profile_step_start"])),
            "--profile-step-end",
            str(int(observability["profile_step_end"])),
        ]
        if observability["profile_ranks"]:
            common += ["--profile-ranks", *[str(rank) for rank in observability["profile_ranks"]]]
    if observability["enable_pytorch_profiler"]:
        common.append("--use-pytorch-profiler")
        if observability["profile_record_shapes"]:
            common.append("--pytorch-profiler-collect-shapes")
        if observability["profile_collect_callstack"]:
            common.append("--pytorch-profiler-collect-callstack")
        if observability["profile_collect_chakra"]:
            common.append("--pytorch-profiler-collect-chakra")
    if observability["enable_memory_history"]:
        common += ["--record-memory-history", "--memory-snapshot-path", str(observability["memory_snapshot_path"])]
    if observability["enable_straggler_log"]:
        common += ["--log-straggler", "--straggler-minmax-count", str(int(observability["straggler_minmax_count"]))]
        if observability["disable_straggler_on_startup"]:
            common.append("--disable-straggler-on-startup")
    if observability["enable_wandb"]:
        common += [
            "--wandb-project",
            str(observability["wandb_project"]),
            "--wandb-exp-name",
            str(observability["wandb_exp_name"]),
        ]
        if observability["wandb_save_dir"]:
            common += ["--wandb-save-dir", str(observability["wandb_save_dir"])]
        if observability["wandb_entity"]:
            common += ["--wandb-entity", str(observability["wandb_entity"])]
    return common


def _data_args(args: argparse.Namespace, program: MegatronProgram) -> List[str]:
    use_mock_data = bool(args.use_mock_data or program.model.track == "moe")
    if use_mock_data:
        return [
            "--mock-data",
            "--tokenizer-type",
            "NullTokenizer",
            "--split",
            str(args.data_split),
            "--no-create-attention-mask-in-dataloader",
            "--no-mmap-bin-files",
            "--num-workers",
            str(int(args.num_workers)),
        ]
    return [
        "--data-path",
        str(args.data_path),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(args.tokenizer_model),
        "--tokenizer-hf-include-special-tokens",
        "--split",
        str(args.data_split),
        "--no-create-attention-mask-in-dataloader",
        "--no-mmap-bin-files",
        "--num-workers",
        str(int(args.num_workers)),
    ]


def _build_megatron_cmd(
    args: argparse.Namespace,
    program: MegatronProgram,
    compiled: CompiledProgram,
    trial_id: int,
) -> List[str]:
    entry = _resolve_megatron_entry(args.megatron_root, args.megatron_entry)
    if not entry.exists():
        raise FileNotFoundError(f"Megatron entry not found: {entry}")
    output_dirs = _trial_output_dirs(args, trial_id)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc),
        "--log-dir",
        output_dirs["torchrun_log_dir"],
        "--redirects",
        "3",
    ]
    if int(args.nnodes) > 1:
        cmd += [
            "--nnodes",
            str(args.nnodes),
            "--node_rank",
            str(args.node_rank),
            "--master_addr",
            str(args.master_addr),
            "--master_port",
            str(args.master_port),
        ]

    strategy = compiled.strategy
    program = program.normalized()
    cmd.append(str(entry))
    cmd += [
        "--tensor-model-parallel-size",
        str(int(strategy.parallel.tp_degree)),
        "--pipeline-model-parallel-size",
        str(int(strategy.parallel.pp_degree)),
        "--context-parallel-size",
        str(int(strategy.parallel.cp_degree)),
        "--expert-model-parallel-size",
        str(int(strategy.parallel.ep_degree)),
        "--expert-tensor-parallel-size",
        str(int(strategy.parallel.expert_tp_degree)),
    ]

    layout = compiled.launcher_env.get("PIPELINE_LAYOUT")
    if layout:
        cmd += ["--pipeline-model-parallel-layout", layout]
    elif int(strategy.parallel.vpp_degree) > 1:
        total_virtual = int(strategy.parallel.pp_degree) * int(strategy.parallel.vpp_degree)
        if int(program.model.num_layers) % total_virtual != 0:
            raise ValueError(f"num_layers={program.model.num_layers} must be divisible by pp*vpp={total_virtual}")
        cmd += ["--num-layers-per-virtual-pipeline-stage", str(int(program.model.num_layers) // total_virtual)]

    if compiled.launcher_env.get("SCHEDULE_GROUP_SIZE"):
        cmd += [
            "--microbatch-group-size-per-virtual-pipeline-stage",
            str(compiled.launcher_env["SCHEDULE_GROUP_SIZE"]),
        ]

    shape_args = (
        _moe_shape_args(args, program, strategy) if program.model.track == "moe" else _dense_shape_args(args, program, strategy)
    )
    cmd += shape_args
    cmd += _training_args(args, program, strategy, trial_id, launcher_env=compiled.launcher_env)
    cmd += _data_args(args, program)
    cmd += _load_args_from_file(args.megatron_args_file)
    if args.megatron_args:
        cmd += shlex.split(args.megatron_args)
    if compiled.extra_args:
        cmd += [str(item) for item in compiled.extra_args if str(item).strip()]
    return cmd


def _build_launcher_env(
    args: argparse.Namespace,
    program: MegatronProgram,
    compiled: CompiledProgram,
    trial_id: int,
) -> Dict[str, str]:
    output_dirs = _trial_output_dirs(args, trial_id)
    env = os.environ.copy()
    env.update(_validate_cuda_toolchain())
    env.update(_launcher_env_overrides(args, program, compiled, trial_id, output_dirs=output_dirs))
    return env


def _launcher_env_overrides(
    args: argparse.Namespace,
    program: MegatronProgram,
    compiled: CompiledProgram,
    trial_id: int,
    output_dirs: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    output_dirs = output_dirs or _trial_output_dirs(args, trial_id)
    observability = _build_observability_config(args, trial_id=trial_id, output_dirs=output_dirs)
    transformer_impl = _resolve_transformer_impl(args, program)
    sequence_parallel_active = _sequence_parallel_active(args, program, compiled.strategy)
    env = dict(compiled.launcher_env)
    env.update(
        {
            "GPUS_PER_NODE": str(int(args.nproc)),
            "NUM_NODES": str(int(args.nnodes)),
            "NODE_RANK": str(int(args.node_rank)),
            "MASTER_ADDR": str(args.master_addr),
            "MASTER_PORT": str(args.master_port),
            "TRAIN_ITERS": str(int(args.train_iters)),
            "EVAL_ITERS": str(int(args.eval_iters)),
            "EVAL_INTERVAL": str(int(args.eval_interval)),
            "LOG_INTERVAL": str(int(args.log_interval)),
            "SAVE_INTERVAL": str(int(args.save_interval)),
            "CHECKPOINT_PATH": output_dirs["checkpoint_path"],
            "SCHEDULE_RUNTIME_TRACE_DIR": output_dirs["runtime_trace_dir"],
            "TENSORBOARD_LOGS_PATH": output_dirs["tensorboard_path"],
            "DATA_CACHE_PATH": output_dirs["data_cache_path"],
            "TRANSFORMER_IMPL": transformer_impl,
            "ENABLE_SP": "1" if sequence_parallel_active else "0",
            "ATTENTION_BACKEND": str(args.attention_backend),
            "ENABLE_TP_COMM_OVERLAP": "1" if bool(args.enable_tp_comm_overlap) else "0",
            "ENABLE_PROFILE": "1" if bool(observability["profile_enabled"]) else "0",
            "OBSERVABILITY_PRESET": str(observability["preset"]),
            "ENABLE_LOG_TIMERS_TO_TENSORBOARD": "1" if bool(observability["enable_log_timers_to_tensorboard"]) else "0",
            "ENABLE_LOG_MEMORY_TO_TENSORBOARD": "1" if bool(observability["enable_log_memory_to_tensorboard"]) else "0",
            "TENSORBOARD_LOG_INTERVAL": str(int(observability["tensorboard_log_interval"])),
            "ENABLE_PYTORCH_PROFILER": "1" if bool(observability["enable_pytorch_profiler"]) else "0",
            "PROFILE_RECORD_SHAPES": "1" if bool(observability["profile_record_shapes"]) else "0",
            "PROFILE_COLLECT_CALLSTACK": "1" if bool(observability["profile_collect_callstack"]) else "0",
            "PROFILE_COLLECT_CHAKRA": "1" if bool(observability["profile_collect_chakra"]) else "0",
            "PROFILE_STEP_START": str(int(observability["profile_step_start"])),
            "PROFILE_STEP_END": str(int(observability["profile_step_end"])),
            "PROFILE_RANKS": ",".join(str(rank) for rank in observability["profile_ranks"]),
            "ENABLE_MEMORY_HISTORY": "1" if bool(observability["enable_memory_history"]) else "0",
            "MEMORY_SNAPSHOT_PATH": str(observability["memory_snapshot_path"] or ""),
            "ENABLE_STRAGGLER_LOG": "1" if bool(observability["enable_straggler_log"]) else "0",
            "DISABLE_STRAGGLER_ON_STARTUP": "1" if bool(observability["disable_straggler_on_startup"]) else "0",
            "STRAGGLER_MINMAX_COUNT": str(int(observability["straggler_minmax_count"])),
            "WANDB_PROJECT": str(observability["wandb_project"]),
            "WANDB_EXP_NAME": str(observability["wandb_exp_name"]),
            "WANDB_SAVE_DIR": str(observability["wandb_save_dir"]),
            "WANDB_ENTITY": str(observability["wandb_entity"]),
            "ENABLE_NSYS": "1" if bool(observability["enable_nsys"]) else "0",
            "NSYS_OUTPUT": str(observability["nsys_output"] or ""),
            "NSYS_TRACE": str(observability["nsys_trace"]),
            "TOKENIZER_MODEL": "MOCK" if bool(args.use_mock_data or program.model.track == "moe") else str(args.tokenizer_model),
            "DATA_PATH": "MOCK" if bool(args.use_mock_data or program.model.track == "moe") else str(args.data_path),
            "MODEL_NAME": str(program.model.model_name),
        }
    )
    if str(env.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip() == "":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cuda_visible_devices = str(getattr(args, "cuda_visible_devices", "") or "").strip()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if args.preset:
        env["PRESET"] = str(args.preset)
    return env


def _parse_error(stderr: str, root_cause_excerpt: Optional[str] = None) -> str:
    combined = "\n".join(part for part in (root_cause_excerpt or "", stderr or "") if part)
    if "CUDA out of memory" in combined:
        return "CUDA OOM"
    if "NCCL" in combined and ("Watchdog" in combined or "timed out" in combined):
        return "NCCL timeout"
    if root_cause_excerpt:
        return f"Runtime Error: {root_cause_excerpt[-1200:]}"
    tail = stderr[-200:] if stderr else ""
    return f"Runtime Error: {tail}"


def _program_from_strategy(strategy: MegatronStrategy, target: str, model_track: str) -> MegatronProgram:
    base = default_moe_smoke_program(target) if model_track == "moe" else default_dense_program(target)
    base.parallel = strategy.parallel.normalized()
    base.batch_plan.micro_batch_size = int(strategy.micro_batch_size)
    base.batch_plan.global_batch_size = int(strategy.global_batch_size)
    base.batch_plan.target_tokens_per_step = int(strategy.global_batch_size) * int(strategy.seq_len)
    base.metadata.update(
        {
            "micro_batch_size": int(strategy.micro_batch_size),
            "global_batch_size": int(strategy.global_batch_size),
            "seq_len": int(strategy.seq_len),
            "use_bf16": bool(strategy.use_bf16),
            "use_fp16": bool(strategy.use_fp16),
            "recompute_granularity": strategy.recompute_granularity,
            "extra_args": list(strategy.extra_args or []),
        }
    )
    base.metadata["program_kind"] = "strategy_compat"
    return base


def _program_patch_family(program: MegatronProgram) -> str:
    normalized = program.normalized()
    applied_patch = normalized.applied_patch
    if applied_patch is not None and str(applied_patch.patch_family or "").strip():
        return str(applied_patch.patch_family or "").strip()
    return (
        str(normalized.metadata.get("patch_family") or "").strip()
        or str(normalized.metadata.get("program_kind") or "baseline").strip()
        or "baseline"
    )


def _update_trial_search_metadata(metrics: Dict[str, Any], program: MegatronProgram) -> None:
    metrics["patch_family"] = _program_patch_family(program)
    trace_summary = dict(metrics.get("trace_summary") or {})
    next_step = dict(trace_summary.get("next_step_hypotheses") or {})
    stop_signal = str(next_step.get("stop_signal") or "").strip()
    metrics["continue_search_hint"] = stop_signal or "continue_search"


def _search_unit_from_args(args: argparse.Namespace) -> str:
    token = str(getattr(args, "search_unit", "patch") or "patch").strip().lower()
    return token if token in {"patch", "whole_config"} else "patch"


def _patch_memory_enabled_from_args(args: argparse.Namespace) -> bool:
    return _search_unit_from_args(args) == "patch" and not bool(getattr(args, "disable_patch_memory", False))


def run_trial(
    args: argparse.Namespace,
    program_or_strategy: ProgramLike,
    trial_id: int,
) -> Dict[str, Any]:
    if isinstance(program_or_strategy, MegatronStrategy):
        strategy = validate_strategy(program_or_strategy)
        target = str(getattr(args, "run_target", None) or ("dual_g4_g5" if int(args.nnodes) > 1 else "single_g5"))
        model_track = str(getattr(args, "model_track", "dense") or "dense").lower()
        program = _program_from_strategy(strategy, target=target, model_track=model_track)
    else:
        program = program_or_strategy.normalized()
        strategy = None

    try:
        compiled = compile_program(program)
    except Exception as exc:
        metrics = {
            "trial_id": int(trial_id),
            "program": program.to_dict(),
            "program_hash": program.semantic_hash(),
            "candidate_rank": int(program.metadata.get("priority_rank") or 0),
            "returncode": 1,
            "error_msg": f"compile_program failed: {exc}",
            "oom": False,
            "patch_family": _program_patch_family(program),
            "patch_category": infer_program_patch_category(program),
            "patch_count": infer_program_patch_count(program),
            "search_unit": _search_unit_from_args(args),
            "patch_memory_enabled": _patch_memory_enabled_from_args(args),
            "continue_search_hint": "continue_search",
        }
        return metrics
    metrics: Dict[str, Any] = {
        "trial_id": int(trial_id),
        "program": program.to_dict(),
        "program_hash": program.semantic_hash(),
        "candidate_rank": int(program.metadata.get("priority_rank") or 0),
        "patch_family": _program_patch_family(program),
        "patch_category": infer_program_patch_category(program),
        "patch_count": infer_program_patch_count(program),
        "search_unit": _search_unit_from_args(args),
        "patch_memory_enabled": _patch_memory_enabled_from_args(args),
        "family": compiled.family.to_dict(),
        "legality": compiled.legality.to_dict(),
        "compiled": compiled.to_dict(),
        "trial_context": {
            "world_size_total": int(args.nproc) * int(args.nnodes),
            "nproc_per_node": int(args.nproc),
            "nnodes": int(args.nnodes),
            "runner_mode": None,
            "launcher_script": None,
            "compile_notes": list(compiled.compile_notes),
            "resolved_profile": dict(compiled.to_dict().get("resolved_profile") or {}),
            "resolved_backend_caps": _resolve_backend_caps(args, program),
        },
    }
    if not compiled.legality.is_valid:
        metrics["returncode"] = 1
        metrics["error_msg"] = "Program legality check failed"
        metrics["oom"] = False
        metrics["continue_search_hint"] = "continue_search"
        return metrics

    strategy = compiled.strategy if strategy is None else strategy
    launcher_script = _resolve_launcher_script(args.megatron_root, getattr(args, "launcher_script", None))
    runner_mode = "launcher_script" if launcher_script else "direct_entry"
    cwd = str(Path(args.megatron_root)) if args.megatron_root else None
    resolved_entry = _resolve_megatron_entry(args.megatron_root, args.megatron_entry)
    output_dirs = _trial_output_dirs(args, trial_id)
    log_paths = _trial_log_paths(output_dirs)
    try:
        runtime_stack = _validate_runtime_stack(args, program)
        resolved_backend_caps = _resolve_backend_caps(args, program)
        observability = _build_observability_config(args, trial_id=trial_id, output_dirs=output_dirs)
        launcher_env_overrides = _launcher_env_overrides(args, program, compiled, trial_id, output_dirs=output_dirs)
    except Exception as exc:
        metrics["returncode"] = 1
        metrics["error_msg"] = str(exc)
        metrics["oom"] = False
        metrics["continue_search_hint"] = "continue_search"
        return metrics
    metrics["trial_context"]["runner_mode"] = runner_mode
    metrics["trial_context"]["launcher_script"] = str(launcher_script) if launcher_script else None
    metrics["trial_context"]["megatron_entry"] = str(resolved_entry)
    metrics["trial_context"]["megatron_root"] = str(args.megatron_root)
    metrics["trial_context"]["observability"] = {
        "preset": observability["preset"],
        "enable_wandb": bool(observability["enable_wandb"]),
        "enable_pytorch_profiler": bool(observability["enable_pytorch_profiler"]),
        "enable_straggler_log": bool(observability["enable_straggler_log"]),
        "enable_memory_history": bool(observability["enable_memory_history"]),
        "enable_nsys": bool(observability["enable_nsys"]),
    }
    metrics["trial_context"]["runtime_stack"] = runtime_stack
    metrics["trial_context"]["resolved_backend_caps"] = resolved_backend_caps
    metrics["trial_context"]["resolved_paths"] = {
        "megatron_root": str(args.megatron_root),
        "megatron_entry": str(resolved_entry),
        "launcher_script": str(launcher_script) if launcher_script else None,
        "trial_dir": output_dirs["trial_dir"],
        "analysis_dir": output_dirs["analysis_dir"],
        "checkpoint_path": output_dirs["checkpoint_path"],
        "tensorboard_path": output_dirs["tensorboard_path"],
        "torch_profile_path": observability["torch_profile_path"],
        "torchrun_log_dir": output_dirs["torchrun_log_dir"],
        "runtime_trace_dir": output_dirs["runtime_trace_dir"],
        "chakra_path": observability["chakra_path"] if bool(observability["profile_collect_chakra"]) else None,
        "memory_snapshot_path": observability["memory_snapshot_path"],
        "nsys_output": observability["nsys_output"],
        "data_cache_path": output_dirs["data_cache_path"],
        "stdout_log": log_paths["stdout_log"],
        "stderr_log": log_paths["stderr_log"],
        "launch_plan_log": log_paths["launch_plan_log"],
    }
    metrics["strategy"] = strategy.to_dict()
    metrics["strategy_hash"] = strategy.semantic_hash()
    trial_evaluation_level = str(
        "dry_run" if bool(getattr(args, "dry_run", False)) else getattr(args, "trial_evaluation_level", "benchmark_trial")
    )
    metrics["trial_evaluation_level"] = trial_evaluation_level
    analysis_dir = Path(str((metrics["trial_context"]["resolved_paths"] or {}).get("trial_dir") or "")) / "analysis"

    try:
        megatron_cmd = _build_megatron_cmd(args, program, compiled, trial_id=trial_id)
        nsys_cmd = _build_nsys_command(observability)
        direct_command = list(megatron_cmd)
        executed_command = (list(nsys_cmd) + direct_command) if nsys_cmd else direct_command
        launcher_command = ["bash", str(launcher_script)] if launcher_script else direct_command
        if nsys_cmd:
            launcher_command = list(nsys_cmd) + launcher_command
        metrics["launch_plan"] = {
            "runner_mode": runner_mode,
            "megatron_command": megatron_cmd,
            "launcher_command": launcher_command,
            "executed_command": executed_command if not launcher_script else launcher_command,
            "launcher_env": launcher_env_overrides,
            "compile_notes": list(compiled.compile_notes),
            "schedule_detail": copy.deepcopy(compiled.schedule_detail),
            "partition_detail": copy.deepcopy(compiled.partition_detail),
            "overlap_detail": copy.deepcopy(compiled.overlap_detail),
            "memory_detail": copy.deepcopy(compiled.memory_detail),
            "config_resolution": copy.deepcopy(compiled.config_resolution),
            "artifact_links": {
                "program_file": str((metrics["trial_context"]["resolved_paths"] or {}).get("trial_dir") or ""),
                "launch_plan": log_paths["launch_plan_log"],
                "stdout_log": log_paths["stdout_log"],
                "stderr_log": log_paths["stderr_log"],
                "torchrun_log_dir": output_dirs["torchrun_log_dir"],
                "runtime_trace_dir": output_dirs["runtime_trace_dir"],
                "checkpoint_path": output_dirs["checkpoint_path"],
                "tensorboard_path": output_dirs["tensorboard_path"],
                "memory_snapshot_path": observability["memory_snapshot_path"],
                "nsys_output": observability["nsys_output"],
                "analysis_dir": output_dirs["analysis_dir"],
                "pipeline_schedule_projection": str(analysis_dir / "pipeline_schedule_projection.json"),
                "pipeline_event_trace": str(analysis_dir / "pipeline_event_trace.json"),
                "pipeline_grid_trace": str(analysis_dir / "pipeline_grid_trace.json"),
                "pipeline_projection_svg": str(analysis_dir / "pipeline_projection.svg"),
                "compare_pipeline_svg": str(analysis_dir / "compare_pipeline.svg"),
                "runtime_schedule_traces": str(analysis_dir / "runtime_schedule_traces.json"),
                "bottleneck_breakdown": str(analysis_dir / "bottleneck_breakdown.json"),
                "trial_reflection": str(analysis_dir / "trial_reflection.json"),
                "failure_diagnosis": str(analysis_dir / "failure_diagnosis.json"),
            },
            "applied_patch": copy.deepcopy(compiled.applied_patch),
            "trial_evaluation_level": trial_evaluation_level,
            "resolved_profile": dict(compiled.to_dict().get("resolved_profile") or {}),
            "resolved_backend_caps": resolved_backend_caps,
            "observability": {
                "preset": observability["preset"],
                "profile_enabled": bool(observability["profile_enabled"]),
                "profile_step_start": int(observability["profile_step_start"]),
                "profile_step_end": int(observability["profile_step_end"]),
                "profile_ranks": list(observability["profile_ranks"]),
                "enable_pytorch_profiler": bool(observability["enable_pytorch_profiler"]),
                "enable_memory_history": bool(observability["enable_memory_history"]),
                "enable_straggler_log": bool(observability["enable_straggler_log"]),
                "enable_nsys": bool(observability["enable_nsys"]),
                "tensorboard_path": observability["tensorboard_path"],
                "torch_profile_path": observability["torch_profile_path"],
                "chakra_path": observability["chakra_path"] if bool(observability["profile_collect_chakra"]) else None,
                "memory_snapshot_path": observability["memory_snapshot_path"],
                "nsys_output": observability["nsys_output"],
            },
        }
        Path(log_paths["launch_plan_log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(log_paths["launch_plan_log"]).write_text(
            json.dumps(metrics["launch_plan"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if bool(getattr(args, "dry_run", False)):
            metrics["dry_run"] = True
            metrics["returncode"] = 0
            metrics["oom"] = False
            metrics["continue_search_hint"] = "continue_search"
            return metrics
        preflight_error = _preflight_gpu_guard(args)
        if preflight_error:
            raise RuntimeError(preflight_error)
        _prepare_trial_artifact_dirs(output_dirs, observability)
        stream_output = bool(getattr(args, "stream_trial_logs", True))
        print(
            f"[megatron_agent] trial {int(trial_id):03d} launching ({runner_mode}) "
            f"stream_logs={'on' if stream_output else 'off'} "
            f"stdout={log_paths['stdout_log']}",
            flush=True,
        )
        if launcher_script:
            if not launcher_script.exists():
                raise FileNotFoundError(f"Launcher script not found: {launcher_script}")
            cmd = list(launcher_command)
            proc = _run_command_with_logging(
                cmd=cmd,
                cwd=cwd,
                env=_build_launcher_env(args, program, compiled, trial_id),
                stdout_log=log_paths["stdout_log"],
                stderr_log=log_paths["stderr_log"],
                stream_output=stream_output,
            )
        else:
            cmd = list(executed_command)
            proc = _run_command_with_logging(
                cmd=cmd,
                cwd=cwd,
                env=_build_launcher_env(args, program, compiled, trial_id),
                stdout_log=log_paths["stdout_log"],
                stderr_log=log_paths["stderr_log"],
                stream_output=stream_output,
            )
    except Exception as exc:
        metrics["returncode"] = 1
        metrics["error_msg"] = str(exc)
        metrics["oom"] = False
        metrics["continue_search_hint"] = "continue_search"
        return metrics

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    Path(log_paths["stdout_log"]).write_text(stdout_text, encoding="utf-8")
    Path(log_paths["stderr_log"]).write_text(stderr_text, encoding="utf-8")
    metrics["returncode"] = proc.returncode
    metrics["stdout_tail"] = stdout_text[-2000:]
    metrics["stderr_tail"] = stderr_text[-2000:]
    metrics["runtime_schedule_traces"] = _load_runtime_schedule_traces(output_dirs)
    print(
        f"[megatron_agent] trial {int(trial_id):03d} finished "
        f"returncode={int(proc.returncode)}",
        flush=True,
    )

    if proc.returncode != 0:
        failure_details = _extract_failure_details(stdout_text, stderr_text, output_dirs["torchrun_log_dir"])
        metrics.update(failure_details)
        metrics["error_msg"] = _parse_error(stderr_text, metrics.get("root_cause_excerpt"))
        metrics["oom"] = "CUDA OOM" in metrics["error_msg"]
        try:
            parsed = parse_megatron_logs(
                stdout=stdout_text,
                stderr=stderr_text,
                global_batch_size=int(strategy.global_batch_size),
                seq_len=int(strategy.seq_len),
            )
            metrics.update(parsed)
        except Exception:
            pass
        metrics["trace_summary"] = reduce_trial_trace(program, metrics=metrics)
        _update_trial_search_metadata(metrics, program)
        observation = build_agent_observation(program, trace_summary=metrics["trace_summary"])
        metrics["context_record"] = observation.to_dict()
        metrics["failure_modes"] = list(observation.failure_modes or [])
        metrics["bottleneck_signature"] = classify_bottleneck(program, metrics["trace_summary"])
        metrics["trial_artifact"] = build_trial_artifact(
            program,
            observation,
            bottleneck_signature=metrics["bottleneck_signature"],
            search_unit=_search_unit_from_args(args),
            patch_memory_enabled=_patch_memory_enabled_from_args(args),
        )
        metrics["patch_category"] = str(metrics["trial_artifact"].get("patch_category") or "")
        metrics["patch_count"] = int(metrics["trial_artifact"].get("patch_count") or 0)
        metrics["search_unit"] = str(metrics["trial_artifact"].get("search_unit") or _search_unit_from_args(args))
        metrics["patch_memory_enabled"] = bool(metrics["trial_artifact"].get("patch_memory_enabled"))
        metrics["analysis_artifact_paths"] = _write_analysis_artifacts(output_dirs, metrics)
        return metrics

    parsed = parse_megatron_logs(
        stdout=stdout_text,
        stderr=stderr_text,
        global_batch_size=int(strategy.global_batch_size),
        seq_len=int(strategy.seq_len),
    )
    metrics.update(parsed)
    metrics["trace_summary"] = reduce_trial_trace(program, metrics=metrics)
    _update_trial_search_metadata(metrics, program)
    observation = build_agent_observation(program, trace_summary=metrics["trace_summary"])
    metrics["context_record"] = observation.to_dict()
    metrics["failure_modes"] = list(observation.failure_modes or [])
    metrics["bottleneck_signature"] = classify_bottleneck(program, metrics["trace_summary"])
    metrics["trial_artifact"] = build_trial_artifact(
        program,
        observation,
        bottleneck_signature=metrics["bottleneck_signature"],
        search_unit=_search_unit_from_args(args),
        patch_memory_enabled=_patch_memory_enabled_from_args(args),
    )
    metrics["patch_category"] = str(metrics["trial_artifact"].get("patch_category") or "")
    metrics["patch_count"] = int(metrics["trial_artifact"].get("patch_count") or 0)
    metrics["search_unit"] = str(metrics["trial_artifact"].get("search_unit") or _search_unit_from_args(args))
    metrics["patch_memory_enabled"] = bool(metrics["trial_artifact"].get("patch_memory_enabled"))
    metrics["analysis_artifact_paths"] = _write_analysis_artifacts(output_dirs, metrics)
    metrics["oom"] = False
    return metrics


def _load_program_or_strategy(path: Path, default_model_track: str, default_target: str) -> ProgramLike:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cluster" in payload and "model" in payload:
        return MegatronProgram.from_dict(payload)
    strategy = MegatronStrategy.from_dict(payload)
    base = default_moe_smoke_program(default_target) if default_model_track == "moe" else default_dense_program(default_target)
    base.parallel = strategy.parallel.normalized()
    base.batch_plan.micro_batch_size = int(strategy.micro_batch_size)
    base.batch_plan.global_batch_size = int(strategy.global_batch_size)
    base.batch_plan.target_tokens_per_step = int(strategy.global_batch_size) * int(strategy.seq_len)
    base.metadata.update(
        {
            "micro_batch_size": int(strategy.micro_batch_size),
            "global_batch_size": int(strategy.global_batch_size),
            "seq_len": int(strategy.seq_len),
            "use_bf16": bool(strategy.use_bf16),
            "use_fp16": bool(strategy.use_fp16),
            "recompute_granularity": strategy.recompute_granularity,
            "extra_args": list(strategy.extra_args or []),
        }
    )
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Megatron program or low-level strategy trial.")
    parser.add_argument("--program-file", type=str, default=None)
    parser.add_argument("--strategy-file", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--megatron-root", type=str, default=DEFAULT_MEGATRON_ROOT)
    parser.add_argument("--launcher-script", type=str, default=DEFAULT_LAUNCHER_SCRIPT)
    parser.add_argument("--megatron-entry", type=str, default="pretrain_gpt.py")
    parser.add_argument("--megatron-args", type=str, default=None)
    parser.add_argument("--megatron-args-file", type=str, default=None)
    parser.add_argument("--model-track", type=str, choices=["dense", "moe"], default="dense")
    parser.add_argument("--run-target", type=str, choices=["single_g4", "single_g5", "dual_g4_g5", "dual_g5_g5"], default="single_g5")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--run-root", type=str, default="./runs_megatron")
    parser.add_argument("--tokenizer-model", type=str, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--use-mock-data", action="store_true")
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument("--enable-tp-comm-overlap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--trial-evaluation-level",
        type=str,
        choices=["dry_run", "smoke_trial", "benchmark_trial", "full_trial"],
        default="benchmark_trial",
    )
    add_observability_args(parser)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--min-free-gpu-memory-mib", type=int, default=28672)
    parser.add_argument("--busy-gpu-memory-threshold-mib", type=int, default=2048)
    parser.add_argument("--transformer-impl", type=str, default="auto")
    parser.add_argument("--attention-backend", type=str, default="auto")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_target in {"dual_g4_g5", "dual_g5_g5"} and int(args.nnodes) < 2:
        args.nnodes = 2
    if not args.program_file and not args.strategy_file:
        raise ValueError("either --program-file or --strategy-file is required")
    input_path = Path(args.program_file or args.strategy_file)
    item = _load_program_or_strategy(input_path, default_model_track=args.model_track, default_target=args.run_target)
    metrics = run_trial(args, item, trial_id=int(args.trial_id))
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if metrics.get("returncode") not in (0, None):
        sys.exit(int(metrics.get("returncode") or 1))


if __name__ == "__main__":
    main()
