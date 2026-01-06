from __future__ import annotations

import os
import math
import platform
from dataclasses import asdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
try:  # pragma: no cover
    from torch.profiler import record_function
except Exception:  # pragma: no cover
    from torch.autograd.profiler import record_function

from fsdp_agent.config import Fsdp2Strategy
from fsdp_agent.fsdp_apply import apply_strategy
from fsdp_agent.dataloaders import build_synthetic_loader
from fsdp_agent.dataset_stats import DatasetStats
from fsdp_agent.metrics_utils import score_strategy
from fsdp_agent.model_introspection import analyze_model_anatomy, extract_transformer_layers
from fsdp_agent.parallel_runtime import (
    apply_tp_sp,
    build_global_mesh,
    infer_tp_plan_id,
    summarize_parallel_spec,
)

try:  # pragma: no cover
    from torch.distributed.pipelining import (
        PipelineStage,
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
        ScheduleLoopedBFS,
        SplitPoint,
        pipeline,
    )
    from torch.distributed.pipelining.microbatch import TensorChunkSpec
except Exception:  # pragma: no cover
    PipelineStage = None
    Schedule1F1B = None
    ScheduleGPipe = None
    ScheduleInterleaved1F1B = None
    ScheduleLoopedBFS = None
    SplitPoint = None
    pipeline = None
    TensorChunkSpec = None

try:  # pragma: no cover
    from torch.distributed.tensor.experimental._context_parallel._attention import context_parallel
except Exception:  # pragma: no cover
    context_parallel = None

_MIN_STEP_TIME_MS = 1e-3
_CURRENT_STAGE = "init"
_LAST_STATIC_LAYER_STATS: Dict[str, Dict[str, float]] = {}
_LAST_MODEL_ANATOMY: Dict[str, Any] = {}


def _set_stage(stage: str) -> None:
    global _CURRENT_STAGE
    _CURRENT_STAGE = stage


def get_current_stage() -> str:
    return _CURRENT_STAGE


def _set_last_static_layer_stats(stats: Dict[str, Dict[str, float]]) -> None:
    global _LAST_STATIC_LAYER_STATS
    _LAST_STATIC_LAYER_STATS = stats


def get_last_static_layer_stats() -> Dict[str, Dict[str, float]]:
    return _LAST_STATIC_LAYER_STATS


def _set_last_model_anatomy(anatomy: Dict[str, Any]) -> None:
    global _LAST_MODEL_ANATOMY
    _LAST_MODEL_ANATOMY = anatomy


def get_last_model_anatomy() -> Dict[str, Any]:
    return _LAST_MODEL_ANATOMY


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _log_rank0(msg: str) -> None:
    if _is_rank0():
        print(msg, flush=True)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """加载任意 HF Causal LM。"""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要安装 transformers 才能加载模型") from exc

    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, config=cfg)
    return model


def build_optimizer(model: nn.Module, lr: float = 1e-4) -> Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _pipeline_loss_fn(output: Any, target: Optional[torch.Tensor]) -> torch.Tensor:
    if output is None:
        raise ValueError("pipeline output is None")
    if hasattr(output, "loss") and getattr(output, "loss") is not None:
        return output.loss
    logits = None
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    elif torch.is_tensor(output):
        logits = output
    if logits is None:
        raise ValueError("pipeline output has no logits for loss computation")
    if target is None:
        raise ValueError("pipeline loss requires target labels")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _build_pp_split_spec(
    layer_paths: Dict[str, str],
    pp_degree: int,
    pp_stages: Optional[List[List[int]]],
) -> Dict[str, Any]:
    if SplitPoint is None:
        raise RuntimeError("torch.distributed.pipelining is unavailable")
    layers: List[Tuple[int, str]] = []
    for name, path in (layer_paths or {}).items():
        if not name.startswith("layers."):
            continue
        try:
            idx = int(name.split(".")[1])
        except Exception:
            continue
        layers.append((idx, path))
    layers.sort(key=lambda x: x[0])
    if len(layers) < int(pp_degree):
        raise ValueError("pp_degree exceeds available transformer layers")
    split_spec: Dict[str, Any] = {}
    if pp_stages:
        for stage_idx, stage in enumerate(pp_stages):
            if stage_idx == 0:
                continue
            try:
                start = int(stage[0])
            except Exception:
                continue
            key = layer_paths.get(f"layers.{start}", f"layers.{start}")
            split_spec[key] = SplitPoint.BEGINNING
        return split_spec
    step = len(layers) / float(pp_degree)
    for stage in range(1, int(pp_degree)):
        split_idx = layers[int(stage * step)][0]
        key = layer_paths.get(f"layers.{split_idx}", f"layers.{split_idx}")
        split_spec[key] = SplitPoint.BEGINNING
    return split_spec


def _run_steps_pipeline(
    schedule,
    optimizer: Optimizer,
    dataloader,
    num_warmup: int,
    num_steps: int,
    profiler_ctx=None,
    progress_every: int = 0,
    dp_world_size: int = 1,
    pp_rank: int = 0,
    num_stages: int = 1,
    cp_context_factory=None,
) -> Tuple[List[float], List[float], List[int], List[float]]:
    it = iter(dataloader)
    losses: List[float] = []
    step_times_ms: List[float] = []
    effective_tokens_global: List[int] = []
    mem_allocated_mb: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_steps = num_warmup + num_steps
    progress_every = int(progress_every) if progress_every else 0
    for step in range(total_steps):
        if profiler_ctx:
            profiler_ctx.step()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            try:
                batch = next(it)
            except StopIteration as exc:
                raise RuntimeError(
                    "dataloader produced 0 batches; check SyntheticDataset length/global_batch_size configuration"
                ) from exc
        input_ids = batch["input_ids"] if pp_rank == 0 else None
        target = batch["labels"] if pp_rank == (num_stages - 1) else None
        if input_ids is not None:
            input_ids = input_ids.cuda()
        if target is not None:
            target = target.cuda()
        if input_ids is not None and "attention_mask" in batch:
            eff = int(batch["attention_mask"].sum().item())
        elif input_ids is not None:
            eff = int(input_ids.numel())
        else:
            eff = 0
        effective_tokens_global_step = eff * int(dp_world_size)
        optimizer.zero_grad(set_to_none=True)
        start_event.record()
        mb_losses: List[float] = []
        if cp_context_factory is not None:
            with cp_context_factory():
                schedule.step(input_ids, target=target, losses=mb_losses)
        else:
            schedule.step(input_ids, target=target, losses=mb_losses)
        optimizer.step()
        end_event.record()
        end_event.synchronize()
        iter_ms = start_event.elapsed_time(end_event)
        if step >= num_warmup:
            if mb_losses:
                loss_avg = sum(float(x) for x in mb_losses) / max(len(mb_losses), 1)
                losses.append(float(loss_avg))
            step_times_ms.append(iter_ms)
            if effective_tokens_global_step > 0:
                effective_tokens_global.append(int(effective_tokens_global_step))
            mem_allocated_mb.append(float(torch.cuda.memory_allocated()) / (1024 * 1024))
        if progress_every and _is_rank0():
            current = step + 1
            if current % progress_every == 0 or current == total_steps:
                phase = "warmup" if current <= num_warmup else "train"
                print(f"[train] step {current}/{total_steps} ({phase})", flush=True)
    return losses, step_times_ms, effective_tokens_global, mem_allocated_mb


def _collect_static_layer_stats(layers: nn.ModuleList, world_size: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    shard_size = max(int(world_size), 1)
    for idx, layer in enumerate(layers):
        numel = 0
        bytes_total = 0
        dim0_bytes = 0
        dim1_bytes = 0
        any_bytes = 0
        for p in layer.parameters(recurse=True):
            n = int(p.numel())
            numel += n
            try:
                bytes_val = n * int(p.element_size())
            except Exception:
                bytes_val = n * 2
            bytes_total += bytes_val
            if shard_size > 1 and getattr(p, "ndim", 0) >= 1:
                try:
                    shape = tuple(int(x) for x in p.shape)
                except Exception:
                    shape = ()
                if shape:
                    if shape[0] % shard_size == 0:
                        dim0_bytes += bytes_val
                    if len(shape) > 1 and shape[1] % shard_size == 0:
                        dim1_bytes += bytes_val
                    if any((d % shard_size) == 0 for d in shape):
                        any_bytes += bytes_val
        ratio = (float(bytes_total) if bytes_total > 0 else 1.0)
        stats[f"layers.{idx}"] = {
            "param_numel": float(numel),
            "param_bytes": float(bytes_total),
            "param_bytes_mb": float(bytes_total) / (1024.0 * 1024.0),
            "divisible_dim0_bytes": float(dim0_bytes),
            "divisible_dim1_bytes": float(dim1_bytes),
            "divisible_any_bytes": float(any_bytes),
            "divisible_dim0_ratio": float(dim0_bytes) / ratio,
            "divisible_dim1_ratio": float(dim1_bytes) / ratio,
            "divisible_any_ratio": float(any_bytes) / ratio,
        }
    return stats


def _infer_layer_type(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    low = str(name).lower()
    if "attn" in low or "attention" in low:
        return "attn"
    if "mlp" in low or "ffn" in low or "feed_forward" in low:
        return "mlp"
    if "moe" in low or "expert" in low:
        return "moe"
    if "norm" in low:
        return "norm"
    if "embed" in low:
        return "embed"
    return "other"


def _has_moe_experts(model: nn.Module) -> bool:
    for name, module in model.named_modules():
        low = f"{name}.{module.__class__.__name__}".lower()
        if "moe" in low or "expert" in low:
            return True
    return False


def _estimate_comm_split(total_comm_ms: float, fsdp_events: Dict[str, Any]) -> Dict[str, float]:
    if total_comm_ms <= 0:
        return {}
    counts = {
        "all_gather": int(fsdp_events.get("all_gather_calls", 0) or 0),
        "reduce_scatter": int(fsdp_events.get("reduce_scatter_calls", 0) or 0),
        "all_reduce": int(fsdp_events.get("all_reduce_calls", 0) or 0),
    }
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: total_comm_ms * (v / total) for k, v in counts.items()}


def _augment_layer_stats(
    layer_stats: Dict[str, Dict[str, float]],
    static_stats: Dict[str, Dict[str, float]],
    layer_paths: Dict[str, str],
    comm_split_ms: Dict[str, float],
    layer_comm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    if not layer_stats:
        return layer_stats
    total_param_bytes = 0.0
    total_time_ms = 0.0
    for st in static_stats.values():
        try:
            total_param_bytes += float(st.get("param_bytes") or 0.0)
        except Exception:
            continue
    for st in layer_stats.values():
        try:
            fwd = float(st.get("fwd_ms_p50") or 0.0)
            bwd = float(st.get("bwd_ms_p50") or 0.0)
            total_time_ms += (fwd + bwd)
        except Exception:
            continue
    out: Dict[str, Dict[str, float]] = {}
    for name, st in layer_stats.items():
        merged = dict(st)
        static = static_stats.get(name, {})
        param_bytes = float(static.get("param_bytes") or 0.0)
        merged["param_bytes"] = param_bytes
        merged["param_bytes_mb"] = param_bytes / (1024.0 * 1024.0) if param_bytes > 0 else 0.0
        layer_path = layer_paths.get(name)
        merged["layer_path"] = layer_path or ""
        merged["layer_type"] = _infer_layer_type(layer_path or name)
        comm_override = layer_comm_stats.get(name) if layer_comm_stats else None
        if comm_override:
            merged["all_gather_ms_est"] = float(comm_override.get("all_gather_ms") or 0.0)
            merged["reduce_scatter_ms_est"] = float(comm_override.get("reduce_scatter_ms") or 0.0)
            merged["all_reduce_ms_est"] = float(comm_override.get("all_reduce_ms") or 0.0)
            merged["comm_time_ms_est"] = float(comm_override.get("comm_time_ms") or 0.0)
            merged["comm_est_source"] = "profiler_stack"
        elif comm_split_ms and (total_time_ms > 0 or total_param_bytes > 0):
            if total_time_ms > 0:
                try:
                    weight = (float(st.get("fwd_ms_p50") or 0.0) + float(st.get("bwd_ms_p50") or 0.0)) / total_time_ms
                except Exception:
                    weight = 0.0
                comm_source = "global_time_weighted"
            else:
                weight = param_bytes / total_param_bytes
                comm_source = "global_param_weighted"
            merged["all_gather_ms_est"] = comm_split_ms.get("all_gather", 0.0) * weight
            merged["reduce_scatter_ms_est"] = comm_split_ms.get("reduce_scatter", 0.0) * weight
            merged["all_reduce_ms_est"] = comm_split_ms.get("all_reduce", 0.0) * weight
            merged["comm_time_ms_est"] = (
                merged["all_gather_ms_est"] + merged["reduce_scatter_ms_est"] + merged["all_reduce_ms_est"]
            )
            merged["comm_est_source"] = comm_source
        else:
            merged["all_gather_ms_est"] = 0.0
            merged["reduce_scatter_ms_est"] = 0.0
            merged["all_reduce_ms_est"] = 0.0
            merged["comm_time_ms_est"] = 0.0
            merged["comm_est_source"] = "none"
        out[name] = merged
    return out


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    batch: Dict[str, torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ctx = torch.cuda.amp.autocast() if scaler is not None else nullcontext()
    with ctx:
        out = model(batch["input_ids"].cuda(), labels=batch["labels"].cuda())
        loss = out.loss
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.item())


def run_steps(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader,
    num_warmup: int,
    num_steps: int,
    profiler_ctx=None,
    layer_probe=None,
    progress_every: int = 0,
    world_size: Optional[int] = None,
    cp_context_factory=None,
) -> Tuple[List[float], List[float], List[int], List[float]]:
    it = iter(dataloader)
    losses: List[float] = []
    step_times_ms: List[float] = []
    effective_tokens_global: List[int] = []
    mem_allocated_mb: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    world = int(world_size or (dist.get_world_size() if dist.is_initialized() else 1))
    total_steps = num_warmup + num_steps
    progress_every = int(progress_every) if progress_every else 0
    for step in range(total_steps):
        if profiler_ctx:
            profiler_ctx.step()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            try:
                batch = next(it)
            except StopIteration as exc:
                raise RuntimeError(
                    "dataloader produced 0 batches; check SyntheticDataset length/global_batch_size configuration"
                ) from exc
        input_ids = batch["input_ids"]
        if "attention_mask" in batch:
            eff = int(batch["attention_mask"].sum().item())
        else:
            eff = int(input_ids.numel())
        effective_tokens_global_step = eff * world
        start_event.record()
        if cp_context_factory is not None:
            with cp_context_factory():
                loss = train_step(model, optimizer, batch, scaler=None)
        else:
            loss = train_step(model, optimizer, batch, scaler=None)
        end_event.record()
        end_event.synchronize()
        iter_ms = start_event.elapsed_time(end_event)
        if step >= num_warmup:
            losses.append(loss)
            step_times_ms.append(iter_ms)
            effective_tokens_global.append(effective_tokens_global_step)
            mem_allocated_mb.append(float(torch.cuda.memory_allocated()) / (1024 * 1024))
            if layer_probe is not None:
                layer_probe.flush_step(record=True)
        else:
            if layer_probe is not None:
                layer_probe.flush_step(record=False)
        if progress_every and _is_rank0():
            current = step + 1
            if current % progress_every == 0 or current == total_steps:
                phase = "warmup" if current <= num_warmup else "train"
                print(f"[train] step {current}/{total_steps} ({phase})", flush=True)
    return losses, step_times_ms, effective_tokens_global, mem_allocated_mb


class _LayerProbe:
    """轻量 layer hooks：记录 forward/backward CUDA event 时间与显存变化（只用于诊断）。"""

    def __init__(self, layers: nn.ModuleList, *, record_ranges: bool = False):
        self._handles = []
        self._fwd_pending = {}
        self._bwd_pending = {}
        self._fwd_pairs = []
        self._bwd_pairs = []
        self._fwd_ms = {}
        self._bwd_ms = {}
        self._mem_delta_mb = {}
        self._fwd_mem_delta_mb = {}
        self._bwd_mem_delta_mb = {}
        self._record_ranges = record_ranges
        self._fwd_ranges = {}
        self._bwd_ranges = {}

        for idx, layer in enumerate(layers):
            name = f"layers.{idx}"

            def _fwd_pre(_, __, layer_name=name):
                if self._record_ranges:
                    ctx = record_function(f"layer::{layer_name}::fwd")
                    ctx.__enter__()
                    self._fwd_ranges[layer_name] = ctx
                s = torch.cuda.Event(enable_timing=True)
                s.record()
                mem0 = torch.cuda.memory_allocated()
                self._fwd_pending[layer_name] = (s, mem0)

            def _fwd_post(_, __, ___, layer_name=name):
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                mem1 = torch.cuda.memory_allocated()
                start, mem0 = self._fwd_pending.pop(layer_name, (None, None))
                if start is not None:
                    self._fwd_pairs.append((layer_name, start, e, mem0, mem1))
                if self._record_ranges:
                    ctx = self._fwd_ranges.pop(layer_name, None)
                    if ctx is not None:
                        ctx.__exit__(None, None, None)

            self._handles.append(layer.register_forward_pre_hook(_fwd_pre))
            self._handles.append(layer.register_forward_hook(_fwd_post))

            if hasattr(layer, "register_full_backward_pre_hook") and hasattr(layer, "register_full_backward_hook"):

                def _bwd_pre(_, __, layer_name=name):
                    if self._record_ranges:
                        ctx = record_function(f"layer::{layer_name}::bwd")
                        ctx.__enter__()
                        self._bwd_ranges[layer_name] = ctx
                    s = torch.cuda.Event(enable_timing=True)
                    s.record()
                    mem0 = torch.cuda.memory_allocated()
                    self._bwd_pending[layer_name] = (s, mem0)

                def _bwd_post(_, __, ___, layer_name=name):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    mem1 = torch.cuda.memory_allocated()
                    start, mem0 = self._bwd_pending.pop(layer_name, (None, None))
                    if start is not None:
                        self._bwd_pairs.append((layer_name, start, e, mem0, mem1))
                    if self._record_ranges:
                        ctx = self._bwd_ranges.pop(layer_name, None)
                        if ctx is not None:
                            ctx.__exit__(None, None, None)

                self._handles.append(layer.register_full_backward_pre_hook(_bwd_pre))
                self._handles.append(layer.register_full_backward_hook(_bwd_post))

    def flush_step(self, *, record: bool) -> None:
        # 依赖 run_steps 的 end_event.synchronize()，保证本 step 的 events 已完成
        if record:
            for layer_name, s, e, mem0, mem1 in self._fwd_pairs:
                self._fwd_ms.setdefault(layer_name, []).append(float(s.elapsed_time(e)))
                delta = float(mem1 - mem0) / (1024 * 1024)
                self._mem_delta_mb.setdefault(layer_name, []).append(delta)
                self._fwd_mem_delta_mb.setdefault(layer_name, []).append(delta)
            for layer_name, s, e, mem0, mem1 in self._bwd_pairs:
                self._bwd_ms.setdefault(layer_name, []).append(float(s.elapsed_time(e)))
                delta = float(mem1 - mem0) / (1024 * 1024)
                self._mem_delta_mb.setdefault(layer_name, []).append(delta)
                self._bwd_mem_delta_mb.setdefault(layer_name, []).append(delta)
        self._fwd_pairs.clear()
        self._bwd_pairs.clear()

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        def _p50(xs):
            if not xs:
                return 0.0
            ys = sorted(xs)
            return float(ys[len(ys) // 2])

        out = {}
        keys = set(self._fwd_ms.keys()) | set(self._bwd_ms.keys()) | set(self._mem_delta_mb.keys())
        for k in keys:
            out[k] = {
                "fwd_ms_p50": _p50(self._fwd_ms.get(k, [])),
                "bwd_ms_p50": _p50(self._bwd_ms.get(k, [])),
                "mem_delta_mb_p50": _p50(self._mem_delta_mb.get(k, [])),
                "fwd_mem_delta_mb_p50": _p50(self._fwd_mem_delta_mb.get(k, [])),
                "bwd_mem_delta_mb_p50": _p50(self._bwd_mem_delta_mb.get(k, [])),
            }
        return out


def _env_subset(prefixes: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        for p in prefixes:
            if k.startswith(p):
                out[k] = v
                break
    return out


def _model_feature_fingerprint(model: nn.Module) -> Dict[str, Any]:
    dropout_ps: List[float] = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            dropout_ps.append(float(m.p))
    cfg = getattr(model, "config", None)
    return {
        "dropout_p_max": max(dropout_ps) if dropout_ps else 0.0,
        "torch_compile_enabled": False,
        "graph_breaks": 0,
        "activation_checkpointing": False,
        "recompute": False,
        "flash_attn": bool(getattr(cfg, "attn_implementation", "") in {"flash_attention_2", "flash_attention"}),
        "attn_implementation": getattr(cfg, "attn_implementation", None),
    }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(q * (len(xs) - 1))
    return float(xs[idx])


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def _gather_rank_stats(step_time_ms: float, max_mem_bytes: int) -> Optional[List[Dict[str, Any]]]:
    if not dist.is_initialized():
        return None
    payload = {"rank": dist.get_rank(), "step_time_ms": float(step_time_ms), "max_mem_bytes": int(max_mem_bytes)}
    world = dist.get_world_size()
    gathered: Optional[List[Dict[str, Any]]] = [None for _ in range(world)] if dist.get_rank() == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    return gathered


def _safe_repr(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_repr(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_repr(v) for k, v in obj.items()}
    return repr(obj)


def _collect_execution_proof(model: nn.Module) -> Dict[str, Any]:
    proof: Dict[str, Any] = {}
    proof["wrap_plan"] = getattr(model, "_fsdp_agent_wrap_plan", None)
    try:
        from torch.distributed._composable.fsdp import fully_shard
    except Exception:
        return proof

    fsdp_modules: List[Dict[str, Any]] = []
    for name, mod in model.named_modules():
        try:
            state = fully_shard.state(mod)
        except Exception:
            continue
        pg = getattr(state, "_fsdp_param_group", None)
        if pg is None:
            continue
        entry = {
            "module": name,
            "post_forward_mesh_info": _safe_repr(getattr(pg, "post_forward_mesh_info", None) or getattr(pg, "_post_forward_mesh_info", None)),
            "mesh_info": _safe_repr(getattr(pg, "mesh_info", None) or getattr(pg, "_mesh_info", None)),
            "offload_policy": _safe_repr(getattr(pg, "offload_policy", None) or getattr(pg, "_offload_policy", None)),
            "mp_policy": _safe_repr(getattr(pg, "mp_policy", None) or getattr(pg, "_mp_policy", None)),
        }
        fsdp_modules.append(entry)
    proof["fsdp_modules"] = fsdp_modules
    return proof


def _estimate_max_unsharded_numel(wrap_plan: Optional[List[Dict[str, Any]]]) -> int:
    if not wrap_plan:
        return 0
    max_numel = 0
    for entry in wrap_plan:
        if entry.get("kind") not in {"layers", "named"}:
            continue
        layout = entry.get("layout") or {}
        raf = layout.get("reshard_after_forward")
        if raf is None:
            continue
        unsharded = False
        if isinstance(raf, bool):
            unsharded = not raf
        elif isinstance(raf, int):
            unsharded = True
        if not unsharded:
            continue
        numel = entry.get("param_numel")
        if isinstance(numel, (int, float)):
            max_numel = max(max_numel, int(numel))
    return max_numel


def _extract_profiler_metrics(prof) -> Dict[str, float]:
    events = prof.key_averages()
    # 兼容不同版本的 profiler 字段命名：优先 device_time_total/self_device_time_total（2.0+）
    def _get_cuda_time(evt):
        return (
            getattr(evt, "device_time_total", None)
            or getattr(evt, "self_device_time_total", None)
            or getattr(evt, "cuda_time_total", None)  # 旧字段
            or getattr(evt, "self_cuda_time_total", 0.0)  # 旧字段
            or 0.0
        )

    def _get_cpu_time(evt):
        return getattr(evt, "cpu_time_total", None) or getattr(evt, "self_cpu_time_total", 0.0) or 0.0

    # torch.profiler returns times in microseconds; convert to milliseconds.
    total_cuda_time = sum(_get_cuda_time(e) for e in events) / 1e3  # ms
    total_cpu_time = sum(_get_cpu_time(e) for e in events) / 1e3
    step_time_ms = total_cuda_time / max(prof.step_num, 1)

    comm_time_ms = 0.0
    all_gather_calls = 0
    reduce_scatter_calls = 0
    all_reduce_calls = 0
    for e in events:
        name = e.key
        if "all_gather" in name:
            comm_time_ms += _get_cuda_time(e) / 1e3
            all_gather_calls += 1
        elif "reduce_scatter" in name:
            comm_time_ms += _get_cuda_time(e) / 1e3
            reduce_scatter_calls += 1
        elif "all_reduce" in name:
            comm_time_ms += _get_cuda_time(e) / 1e3
            all_reduce_calls += 1
    compute_time_ms = max(total_cuda_time - comm_time_ms, 0.0)
    idle_ratio = None
    steps = int(getattr(prof, "step_num", 0) or 0)
    total_collective_calls = all_gather_calls + reduce_scatter_calls + all_reduce_calls
    calls_per_step = (total_collective_calls / steps) if steps > 0 else None
    calls_step_jitter = None
    if steps > 0:
        calls_step_jitter = abs(total_collective_calls - round(calls_per_step) * steps) / steps
    reshard_calls_est = reduce_scatter_calls
    reshard_calls_per_step = (reshard_calls_est / steps) if steps > 0 else None
    return {
        "step_time_ms": step_time_ms,
        "profiler_steps": steps,
        "total_cuda_time_ms": total_cuda_time,
        "total_cpu_time_ms": total_cpu_time,
        "comm_time_ms": comm_time_ms,
        "compute_time_ms": compute_time_ms,
        "idle_ratio": idle_ratio,
        "collective_calls_total": total_collective_calls,
        "collective_calls_per_step_est": calls_per_step,
        "collective_calls_step_jitter_est": calls_step_jitter,
        "reshard_calls_est": reshard_calls_est,
        "reshard_calls_per_step_est": reshard_calls_per_step,
        "fsdp_events": {
            "all_gather_calls": all_gather_calls,
            "reduce_scatter_calls": reduce_scatter_calls,
            "all_reduce_calls": all_reduce_calls,
        },
    }


def _extract_layer_comm_from_profiler(prof, layer_names: List[str]) -> Dict[str, Dict[str, float]]:
    if prof is None or not layer_names:
        return {}
    try:
        events = prof.key_averages(group_by_stack_n=5)
    except Exception:
        return {}

    def _get_cuda_time(evt) -> float:
        return (
            getattr(evt, "device_time_total", None)
            or getattr(evt, "self_device_time_total", None)
            or getattr(evt, "cuda_time_total", None)
            or getattr(evt, "self_cuda_time_total", 0.0)
            or 0.0
        )

    layer_comm: Dict[str, Dict[str, float]] = {}
    needle = {name: f"layer::{name}::" for name in layer_names}
    for evt in events:
        key = evt.key
        if "all_gather" in key:
            kind = "all_gather"
        elif "reduce_scatter" in key:
            kind = "reduce_scatter"
        elif "all_reduce" in key:
            kind = "all_reduce"
        else:
            continue
        stack = getattr(evt, "stack", None)
        if not stack:
            continue
        stack_text = "\n".join(stack)
        target = None
        for name, marker in needle.items():
            if marker in stack_text:
                target = name
                break
        if not target:
            continue
        stats = layer_comm.setdefault(target, {"all_gather_ms": 0.0, "reduce_scatter_ms": 0.0, "all_reduce_ms": 0.0})
        stats[f"{kind}_ms"] = stats.get(f"{kind}_ms", 0.0) + (_get_cuda_time(evt) / 1e3)
    for name, stats in layer_comm.items():
        stats["comm_time_ms"] = float(stats.get("all_gather_ms", 0.0) + stats.get("reduce_scatter_ms", 0.0) + stats.get("all_reduce_ms", 0.0))
    return layer_comm


def run_trial(
    strategy: Fsdp2Strategy,
    global_batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_warmup: int,
    num_steps: int,
    lr: float,
    trace_dir: str,
    trial_id: int,
    model_name: str,
    dataset_stats: DatasetStats,
    repeats: int = 1,
    mem_limit_gb: float = 70.0,
    profiling: str = "light",
    seed: Optional[int] = None,
) -> Dict:
    """
    执行单策略多次重复，取中位数吞吐，返回最佳一次的详细 metrics（含 trace_summary、MFU）。
    """
    _set_stage("init")
    if seed is not None:
        set_seeds(int(seed))
    all_runs = []
    world_size_total = dist.get_world_size() if dist.is_initialized() else 1
    parallel_spec = asdict(getattr(strategy, "parallel", None) or {})
    tp_degree = int(parallel_spec.get("tp_degree", 1) or 1)
    pp_degree = int(parallel_spec.get("pp_degree", 1) or 1)
    ep_degree = int(parallel_spec.get("ep_degree", 1) or 1)
    cp_degree = int(parallel_spec.get("cp_degree", 1) or 1)
    sp_enabled = bool(parallel_spec.get("sp_enabled", False))
    if sp_enabled and tp_degree <= 1:
        raise ValueError("sp_enabled requires tp_degree > 1")
    mesh = None
    dp_mesh = None
    tp_mesh = None
    pp_mesh = None
    ep_mesh = None
    cp_mesh = None
    dp_world_size = int(world_size_total)
    parallel_report_base = summarize_parallel_spec(parallel_spec) if parallel_spec else {}
    if tp_degree > 1 or pp_degree > 1 or ep_degree > 1 or cp_degree > 1 or sp_enabled:
        mesh, dp_world_size = build_global_mesh(
            world_size_total,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            ep_degree=ep_degree,
            cp_degree=cp_degree,
        )
        if mesh is not None:
            dp_mesh = mesh["dp"]
            pp_mesh = mesh["pp"]
            tp_mesh = mesh["tp"]
            ep_mesh = mesh["ep"]
            cp_mesh = mesh["cp"]
        if parallel_report_base:
            parallel_report_base = dict(parallel_report_base)
        else:
            parallel_report_base = {}
        parallel_report_base.update(
            {
                "world_size_total": int(world_size_total),
                "dp_world_size": int(dp_world_size),
            }
        )
    for r in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        _log_rank0(f"[trial] run {r + 1}/{repeats}: loading model")
        _set_stage("load_model")
        model = load_model(model_name=model_name)
        layers_static = extract_transformer_layers(model)
        layer_paths: Dict[str, str] = {}
        if layers_static is not None:
            name_by_id = {id(m): name for name, m in model.named_modules()}
            for idx, layer in enumerate(layers_static):
                name = name_by_id.get(id(layer))
                if name:
                    layer_paths[f"layers.{idx}"] = name
        static_layer_stats = (
            _collect_static_layer_stats(layers_static, world_size=dp_world_size) if layers_static is not None else {}
        )
        _set_last_static_layer_stats(static_layer_stats)
        anatomy = analyze_model_anatomy(model)
        _set_last_model_anatomy(anatomy)
        parallel_report = dict(parallel_report_base) if parallel_report_base else {}
        if tp_mesh is not None:
            plan_id = infer_tp_plan_id(model, model_name, parallel_spec.get("tp_plan", "auto"))
            tp_report = apply_tp_sp(model, tp_mesh, plan_id=plan_id, sp_enabled=sp_enabled)
            parallel_report["tp_plan_id"] = plan_id
            parallel_report["tp_sp_report"] = tp_report
        if ep_degree > 1 and not _has_moe_experts(model):
            raise ValueError("ep_degree>1 requested but model has no MoE experts")
        cp_context_factory = None
        if cp_degree > 1:
            if context_parallel is None or cp_mesh is None:
                raise RuntimeError("cp_degree>1 requires context_parallel and cp_mesh")
            cp_context_factory = lambda: context_parallel(cp_mesh)
        _log_rank0("[trial] applying strategy")
        _set_stage("apply_strategy")
        model = apply_strategy(model, strategy, world_size=dp_world_size, dp_mesh=dp_mesh)
        world = int(dp_world_size)
        per_rank_batch = int(math.ceil(global_batch_size / max(world, 1)))
        use_pipeline = bool(pp_degree > 1)
        schedule = None
        pp_rank = 0
        num_stages = 1
        if use_pipeline:
            if pipeline is None or PipelineStage is None:
                raise RuntimeError("pp_degree>1 requires torch.distributed.pipelining")
            if pp_mesh is None:
                raise RuntimeError("pp_degree>1 requires pp_mesh")
            if per_rank_batch < 1:
                raise ValueError("per_rank_batch must be >=1 for pipeline")
            split_spec = _build_pp_split_spec(layer_paths, pp_degree, parallel_spec.get("pp_stages"))
            device = torch.device("cuda")
            dummy_input = torch.zeros((per_rank_batch, seq_len), device=device, dtype=torch.long)
            pipe = pipeline(model, mb_args=(dummy_input,), split_spec=split_spec)
            pp_group = pp_mesh.get_group()
            pp_rank = dist.get_rank(pp_group)
            stage = pipe.build_stage(pp_rank, device, group=pp_group)
            num_stages = int(pp_degree)
            pp_microbatches = int(parallel_spec.get("pp_microbatches", 1) or 1)
            if pp_microbatches < 1:
                pp_microbatches = 1
            if pp_microbatches > per_rank_batch:
                raise ValueError("pp_microbatches cannot exceed per_rank_batch")
            schedule_name = str(parallel_spec.get("pp_schedule", "1f1b")).lower()
            chunk_spec = None
            if pp_microbatches > 1:
                if TensorChunkSpec is None:
                    raise RuntimeError("pp_microbatches>1 requires TensorChunkSpec")
                chunk_spec = TensorChunkSpec(0)
            if schedule_name in {"gpipe"}:
                schedule = ScheduleGPipe(
                    stage,
                    pp_microbatches,
                    loss_fn=_pipeline_loss_fn,
                    args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                    kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
                )
            elif schedule_name in {"interleaved1f1b", "interleaved"}:
                schedule = ScheduleInterleaved1F1B(
                    [stage],
                    pp_microbatches,
                    loss_fn=_pipeline_loss_fn,
                    args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                    kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
                )
            elif schedule_name in {"looped_bfs", "loopedbfs"}:
                schedule = ScheduleLoopedBFS([stage], pp_microbatches, loss_fn=_pipeline_loss_fn)
            else:
                schedule = Schedule1F1B(
                    stage,
                    pp_microbatches,
                    loss_fn=_pipeline_loss_fn,
                    args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                    kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
                )
            model_for_optimizer = getattr(stage, "submod", stage)
            _set_stage("build_optimizer")
            optimizer = build_optimizer(model_for_optimizer, lr=lr)
            parallel_report["pp_rank"] = int(pp_rank)
            parallel_report["pp_schedule"] = schedule_name
            parallel_report["pp_microbatches"] = int(pp_microbatches)
        else:
            _set_stage("build_optimizer")
            optimizer = build_optimizer(model, lr=lr)
        required_batches = num_warmup + num_steps + 1
        required_len = per_rank_batch * required_batches
        _set_stage("build_dataloader")
        dataloader = build_synthetic_loader(
            train_hyper={"global_batch_size": global_batch_size},
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=per_rank_batch,
            length=max(10_000, required_len),
            seed=seed,
        )
        _log_rank0("[trial] dataloader ready")

        prof = None
        record_ranges = False
        trace_path = None
        if profiling == "heavy":
            record_ranges = bool(os.environ.get("FSDP_AGENT_LAYER_COMM_STACK"))
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"trial_{trial_id}_run{r}")
            prof = profile(
                activities=activities,
                schedule=schedule(wait=0, warmup=1, active=num_steps),
                on_trace_ready=tensorboard_trace_handler(trace_path),
                record_shapes=False,
                profile_memory=True,
                with_stack=record_ranges,
            )

        layers = extract_transformer_layers(model) if not use_pipeline else None
        probe = _LayerProbe(layers, record_ranges=record_ranges) if layers is not None else None
        layer_stats_coverage = None
        layer_stats_incomplete = None
        try:
            progress_every = max(1, int(num_steps) // 5)
            _log_rank0(f"[trial] start steps (warmup={num_warmup}, steps={num_steps}, profile={profiling})")
            _set_stage("train_steps")
            if prof is None:
                if use_pipeline:
                    losses, step_times_ms, effective_tokens_global, mem_allocated_mb = _run_steps_pipeline(
                        schedule,
                        optimizer,
                        dataloader,
                        num_warmup=num_warmup,
                        num_steps=num_steps,
                        profiler_ctx=None,
                        progress_every=progress_every,
                        dp_world_size=dp_world_size,
                        pp_rank=pp_rank,
                        num_stages=num_stages,
                        cp_context_factory=cp_context_factory,
                    )
                else:
                    losses, step_times_ms, effective_tokens_global, mem_allocated_mb = run_steps(
                        model,
                        optimizer,
                        dataloader,
                        num_warmup=num_warmup,
                        num_steps=num_steps,
                        profiler_ctx=None,
                        layer_probe=probe,
                        progress_every=progress_every,
                        world_size=dp_world_size,
                        cp_context_factory=cp_context_factory,
                    )
            else:
                with prof:
                    if use_pipeline:
                        losses, step_times_ms, effective_tokens_global, mem_allocated_mb = _run_steps_pipeline(
                            schedule,
                            optimizer,
                            dataloader,
                            num_warmup=num_warmup,
                            num_steps=num_steps,
                            profiler_ctx=prof,
                            progress_every=progress_every,
                            dp_world_size=dp_world_size,
                            pp_rank=pp_rank,
                            num_stages=num_stages,
                            cp_context_factory=cp_context_factory,
                        )
                    else:
                        losses, step_times_ms, effective_tokens_global, mem_allocated_mb = run_steps(
                            model,
                            optimizer,
                            dataloader,
                            num_warmup=num_warmup,
                            num_steps=num_steps,
                            profiler_ctx=prof,
                            layer_probe=probe,
                            progress_every=progress_every,
                            world_size=dp_world_size,
                            cp_context_factory=cp_context_factory,
                        )
        finally:
            _set_stage("postprocess")
            layer_summary = probe.summary() if probe is not None else {}
            if layers is not None:
                total_layers = len(layers)
                if total_layers > 0:
                    layer_stats_coverage = len(layer_summary) / total_layers
                    layer_stats_incomplete = layer_stats_coverage < 0.8
            if probe is not None:
                probe.close()
        _log_rank0("[trial] steps done")

        torch.cuda.synchronize()
        mem_bytes = torch.cuda.max_memory_allocated()
        prof_metrics = _extract_profiler_metrics(prof) if prof is not None else {}
        comm_split = _estimate_comm_split(
            float(prof_metrics.get("comm_time_ms", 0.0) or 0.0),
            prof_metrics.get("fsdp_events") or {},
        )
        layer_comm_stats = _extract_layer_comm_from_profiler(prof, list(layer_paths.keys())) if record_ranges else {}
        prof_metrics["max_mem_bytes"] = mem_bytes
        prof_metrics["loss_mean"] = float(sum(losses) / max(len(losses), 1))
        prof_metrics["loss_std"] = 0.0
        prof_metrics["trace_dir"] = trace_dir
        prof_metrics["trace_path"] = trace_path
        prof_metrics["layer_stats"] = _augment_layer_stats(
            layer_summary,
            static_layer_stats,
            layer_paths,
            comm_split,
            layer_comm_stats=layer_comm_stats,
        )
        prof_metrics["layer_stats_coverage"] = layer_stats_coverage
        prof_metrics["layer_stats_incomplete"] = layer_stats_incomplete
        prof_metrics["layer_stats_static"] = static_layer_stats
        prof_metrics["layer_paths"] = layer_paths
        prof_metrics["model_anatomy"] = anatomy
        # 只用 CUDA event 的实测步时，避免 profiler 聚合带来的偏差（用 median 抵抗偶发抖动/0ms）
        sane = [t for t in step_times_ms if t >= _MIN_STEP_TIME_MS]
        if not sane:
            mean_step_ms = 0.0
        else:
            sane.sort()
            mean_step_ms = float(sane[len(sane) // 2])
        prof_metrics["step_time_ms"] = mean_step_ms
        prof_metrics["step_time_ms_p50"] = mean_step_ms
        prof_metrics["step_time_ms_p90"] = _percentile(sane, 0.9) if sane else 0.0
        _, step_time_std = _mean_std(sane)
        prof_metrics["step_time_ms_std"] = float(step_time_std)
        prof_metrics["step_time_ms_p90_p50"] = max(
            float(prof_metrics["step_time_ms_p90"] or 0.0) - float(mean_step_ms or 0.0),
            0.0,
        )

        mem_alloc_sane = [m for m in mem_allocated_mb if m >= 0.0]
        _, mem_std = _mean_std(mem_alloc_sane)
        mem_deltas = [mem_alloc_sane[i] - mem_alloc_sane[i - 1] for i in range(1, len(mem_alloc_sane))]
        abs_deltas = [abs(x) for x in mem_deltas]
        _, mem_abs_delta_std = _mean_std(abs_deltas)
        median_abs_delta = _percentile(abs_deltas, 0.5) if abs_deltas else 0.0
        spike_threshold_mb = max(16.0, 2.0 * median_abs_delta)
        spike_ratio = (sum(1 for x in abs_deltas if x > spike_threshold_mb) / len(abs_deltas)) if abs_deltas else 0.0
        prof_metrics["mem_allocated_std_mb"] = float(mem_std)
        prof_metrics["mem_allocated_delta_std_mb"] = float(mem_abs_delta_std)
        prof_metrics["alloc_free_spike_ratio"] = float(spike_ratio)

        eff_sane = [int(x) for x in effective_tokens_global if x > 0]
        eff_tokens_per_step = int(sorted(eff_sane)[len(eff_sane) // 2]) if eff_sane else 0
        prof_metrics["effective_tokens_per_step"] = eff_tokens_per_step
        prof_metrics["throughput_effective_tokens_per_s"] = (
            (eff_tokens_per_step / (mean_step_ms / 1000.0)) if mean_step_ms >= _MIN_STEP_TIME_MS and eff_tokens_per_step > 0 else 0.0
        )
        prof_metrics["throughput_tokens_per_s"] = _estimate_tokens_per_s(mean_step_ms, global_batch_size, seq_len)

        # 让 LLM 看到更稳健的“是否 CPU/等待占比高”的证据：用 profiler kernel 总时长 vs CUDA event wall time。
        gpu_busy_ratio_est = None
        host_overhead_ratio_est = None
        if profiling == "heavy":
            total_cuda = float(prof_metrics.get("total_cuda_time_ms", 0.0) or 0.0)
            steps = int(prof_metrics.get("profiler_steps", 0) or 0)
            wall = float(mean_step_ms or 0.0)
            if steps > 0 and wall >= _MIN_STEP_TIME_MS and total_cuda > 0:
                cuda_kernel_ms_per_step = total_cuda / steps
                busy = cuda_kernel_ms_per_step / wall
                # kernel sum 可能因为多 stream 重叠而 > wall；这里只关心“是否明显低于 wall”。
                gpu_busy_ratio_est = float(min(max(busy, 0.0), 1.0))
                host_overhead_ratio_est = float(max(1.0 - gpu_busy_ratio_est, 0.0))
                prof_metrics["cuda_kernel_ms_per_step_est"] = float(cuda_kernel_ms_per_step)
                if sane:
                    per_step_busy = [
                        float(min(max(cuda_kernel_ms_per_step / t, 0.0), 1.0)) if t >= _MIN_STEP_TIME_MS else 0.0
                        for t in sane
                    ]
                    busy_mean, busy_std = _mean_std(per_step_busy)
                    overlap_var = sum((x - busy_mean) ** 2 for x in per_step_busy) / max(len(per_step_busy), 1)
                    kernel_bubble = [max(1.0 - x, 0.0) for x in per_step_busy]
                    _, bubble_std = _mean_std(kernel_bubble)
                    prof_metrics["overlap_ratio_var"] = float(overlap_var)
                    prof_metrics["overlap_ratio_std"] = float(busy_std)
                    prof_metrics["kernel_bubble_ratio_std_est"] = float(bubble_std)
        prof_metrics["gpu_busy_ratio_est"] = gpu_busy_ratio_est
        prof_metrics["host_overhead_ratio_est"] = host_overhead_ratio_est
        prof_metrics.setdefault("overlap_ratio_var", None)
        prof_metrics.setdefault("overlap_ratio_std", None)
        prof_metrics.setdefault("kernel_bubble_ratio_std_est", None)

        # comm_ratio：仅 heavy 下可靠
        comm_ratio_valid = profiling == "heavy"
        if profiling == "heavy":
            total = float(prof_metrics.get("total_cuda_time_ms", 0.0) or 0.0)
            comm = float(prof_metrics.get("comm_time_ms", 0.0) or 0.0)
            prof_metrics["comm_ratio"] = (comm / total) if total > 0 else None
        else:
            prof_metrics.setdefault("comm_ratio", None)
        prof_metrics["comm_ratio_valid"] = comm_ratio_valid

        # MFU 暂不可信，置空，避免误导
        prof_metrics["mfu_percent"] = None
        prof_metrics["total_params"] = sum(p.numel() for p in model.parameters())

        prof_metrics["profiling"] = profiling
        prof_metrics["trial_context"] = {
            "requested_global_batch_size": int(global_batch_size),
            "effective_global_batch_size": int(per_rank_batch * dp_world_size),
            "per_rank_batch_size": int(per_rank_batch),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "grad_accum": 1,
            "trace_dir": trace_dir,
            "trace_path": trace_path,
            "microbatch_shape_bsh": [
                int(per_rank_batch),
                int(seq_len),
                int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0),
            ],
            "model_features": _model_feature_fingerprint(model),
            "dist_backend": dist.get_backend() if dist.is_initialized() else None,
            "nccl_env": _env_subset(["NCCL_", "TORCH_NCCL_"]),
            "torch": {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "platform": platform.platform(),
            },
        }
        prof_metrics["trial_context"]["dp_world_size"] = int(dp_world_size)
        prof_metrics["trial_context"]["world_size_total"] = int(world_size_total)
        if parallel_report:
            prof_metrics["parallel"] = parallel_report
            prof_metrics["trial_context"]["parallel"] = parallel_report
        execution_proof = _collect_execution_proof(model)
        prof_metrics["execution_proof"] = execution_proof
        prof_metrics["max_unsharded_numel_est"] = _estimate_max_unsharded_numel(execution_proof.get("wrap_plan"))
        if profiling != "heavy":
            prof_metrics.setdefault("total_cuda_time_ms", 0.0)
            prof_metrics.setdefault("total_cpu_time_ms", 0.0)
            prof_metrics.setdefault("comm_time_ms", 0.0)
            prof_metrics.setdefault("compute_time_ms", 0.0)
            prof_metrics.setdefault("idle_ratio", None)
            prof_metrics.setdefault("collective_calls_total", 0)
            prof_metrics.setdefault("collective_calls_per_step_est", None)
            prof_metrics.setdefault("collective_calls_step_jitter_est", None)
            prof_metrics.setdefault("reshard_calls_est", None)
            prof_metrics.setdefault("reshard_calls_per_step_est", None)
            prof_metrics.setdefault("overlap_ratio_var", None)
            prof_metrics.setdefault("overlap_ratio_std", None)
            prof_metrics.setdefault("kernel_bubble_ratio_std_est", None)
            prof_metrics.setdefault("fsdp_events", {"all_gather_calls": 0, "reduce_scatter_calls": 0, "all_reduce_calls": 0})

        prof_metrics["oom"] = False
        prof_metrics["trial_id"] = trial_id
        prof_metrics["run"] = r
        per_rank = _gather_rank_stats(mean_step_ms, int(mem_bytes))
        if per_rank is not None and (dist.get_rank() == 0):
            prof_metrics["per_rank"] = per_rank
        all_runs.append(prof_metrics)

    throughputs = sorted(x["throughput_effective_tokens_per_s"] for x in all_runs)
    median_tp = throughputs[len(throughputs) // 2]
    best = max(all_runs, key=lambda x: x["throughput_effective_tokens_per_s"])
    best["median_throughput_tokens_per_s"] = median_tp
    best["dataset_stats"] = dataset_stats.__dict__
    best["score"] = score_strategy(best, mem_limit_bytes=int(mem_limit_gb * 1024**3))

    _sanitize_metrics(best)

    # 结构化 trace summary，供 LLM 直接阅读
    mem_limit_bytes = int(mem_limit_gb * 1024**3)
    headroom_mb = (mem_limit_bytes - best.get("max_mem_bytes", 0)) // (1024 * 1024)
    best["oom_margin_gb"] = float(headroom_mb) / 1024.0
    best["memory_headroom_mb"] = int(headroom_mb)
    overlap_ratio = best.get("gpu_busy_ratio_est", None)
    peak_unsharded_groups = best.get("fsdp_events", {}).get("all_gather_calls", 0)
    best["trace_summary"] = {
        "all_gather_forward_late": (best.get("host_overhead_ratio_est") or 0.0) > 0.2,
        "overlap_ratio": overlap_ratio,
        "overlap_ratio_var": best.get("overlap_ratio_var", None),
        "overlap_ratio_std": best.get("overlap_ratio_std", None),
        "peak_unsharded_groups": peak_unsharded_groups,
        "max_unsharded_numel": best.get("max_unsharded_numel_est", 0),
        "memory_headroom_mb": headroom_mb,
        "mem_allocated_std_mb": best.get("mem_allocated_std_mb", None),
        "mem_allocated_delta_std_mb": best.get("mem_allocated_delta_std_mb", None),
        "alloc_free_spike_ratio": best.get("alloc_free_spike_ratio", None),
        "reshard_calls_per_step_est": best.get("reshard_calls_per_step_est", None),
        "collective_calls_per_step_est": best.get("collective_calls_per_step_est", None),
        "collective_calls_step_jitter_est": best.get("collective_calls_step_jitter_est", None),
        "mfu_percent": best.get("mfu_percent", 0.0),
        "tokens_per_step": global_batch_size * seq_len,
        "world_size": int(dp_world_size),
        "world_size_total": int(world_size_total),
        "gpu_busy_ratio_est": best.get("gpu_busy_ratio_est", None),
        "host_overhead_ratio_est": best.get("host_overhead_ratio_est", None),
        "comm_ratio": best.get("comm_ratio", None),
        "step_time_ms_std": best.get("step_time_ms_std", None),
        "step_time_ms_p90_p50": best.get("step_time_ms_p90_p50", None),
        "kernel_bubble_ratio_std_est": best.get("kernel_bubble_ratio_std_est", None),
    }
    return best


def _sanitize_metrics(metrics: Dict[str, Any]) -> None:
    warnings: List[str] = []
    mfu = metrics.get("mfu_percent")
    if isinstance(mfu, (int, float)) and (mfu < 0 or mfu > 100):
        metrics["mfu_percent"] = None
        warnings.append("mfu_out_of_range")
    total_cuda = float(metrics.get("total_cuda_time_ms", 0.0) or 0.0)
    comm = float(metrics.get("comm_time_ms", 0.0) or 0.0)
    if total_cuda > 0 and comm > total_cuda:
        metrics["comm_time_ms"] = total_cuda
        warnings.append("comm_time_clamped")
    if warnings:
        metrics["sanity_warnings"] = warnings


def _estimate_tokens_per_s(step_time_ms: float, global_batch: int, seq_len: int) -> float:
    if step_time_ms < _MIN_STEP_TIME_MS:
        return 0.0
    tokens = global_batch * seq_len
    return tokens / (step_time_ms / 1000.0)


def _get_gpu_peak_tflops(default_bf16: float = 312.0) -> float:
    """
    估算 GPU 峰值 BF16 TFLOPS。
    优先读取环境变量 FSDP_AGENT_GPU_PEAK_TFLOPS；否则默认 A800 312。
    """
    env_val = os.environ.get("FSDP_AGENT_GPU_PEAK_TFLOPS")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return default_bf16
