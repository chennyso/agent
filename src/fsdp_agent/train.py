from __future__ import annotations

import os
import math
import platform
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from fsdp_agent.config import Fsdp2Strategy
from fsdp_agent.fsdp_apply import apply_strategy
from fsdp_agent.dataloaders import build_synthetic_loader
from fsdp_agent.dataset_stats import DatasetStats
from fsdp_agent.metrics_utils import score_strategy
from fsdp_agent.model_introspection import extract_transformer_layers

_MIN_STEP_TIME_MS = 1e-3


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
) -> Tuple[List[float], List[float], List[int]]:
    it = iter(dataloader)
    losses: List[float] = []
    step_times_ms: List[float] = []
    effective_tokens_global: List[int] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    world = dist.get_world_size() if dist.is_initialized() else 1
    for step in range(num_warmup + num_steps):
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
        loss = train_step(model, optimizer, batch, scaler=None)
        end_event.record()
        end_event.synchronize()
        iter_ms = start_event.elapsed_time(end_event)
        if step >= num_warmup:
            losses.append(loss)
            step_times_ms.append(iter_ms)
            effective_tokens_global.append(effective_tokens_global_step)
            if layer_probe is not None:
                layer_probe.flush_step(record=True)
        else:
            if layer_probe is not None:
                layer_probe.flush_step(record=False)
    return losses, step_times_ms, effective_tokens_global


class _LayerProbe:
    """轻量 layer hooks：记录 forward/backward CUDA event 时间与显存变化（只用于诊断）。"""

    def __init__(self, layers: nn.ModuleList):
        self._handles = []
        self._fwd_pending = {}
        self._bwd_pending = {}
        self._fwd_pairs = []
        self._bwd_pairs = []
        self._fwd_ms = {}
        self._bwd_ms = {}
        self._mem_delta_mb = {}

        for idx, layer in enumerate(layers):
            name = f"layers.{idx}"

            def _fwd_pre(_, __, layer_name=name):
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

            self._handles.append(layer.register_forward_pre_hook(_fwd_pre))
            self._handles.append(layer.register_forward_hook(_fwd_post))

            if hasattr(layer, "register_full_backward_pre_hook") and hasattr(layer, "register_full_backward_hook"):

                def _bwd_pre(_, __, layer_name=name):
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

                self._handles.append(layer.register_full_backward_pre_hook(_bwd_pre))
                self._handles.append(layer.register_full_backward_hook(_bwd_post))

    def flush_step(self, *, record: bool) -> None:
        # 依赖 run_steps 的 end_event.synchronize()，保证本 step 的 events 已完成
        if record:
            for layer_name, s, e, mem0, mem1 in self._fwd_pairs:
                self._fwd_ms.setdefault(layer_name, []).append(float(s.elapsed_time(e)))
                self._mem_delta_mb.setdefault(layer_name, []).append(float(mem1 - mem0) / (1024 * 1024))
            for layer_name, s, e, mem0, mem1 in self._bwd_pairs:
                self._bwd_ms.setdefault(layer_name, []).append(float(s.elapsed_time(e)))
                self._mem_delta_mb.setdefault(layer_name, []).append(float(mem1 - mem0) / (1024 * 1024))
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
    idle_ratio = max(1.0 - (comm_time_ms + compute_time_ms) / max(total_cuda_time, 1e-6), 0.0)
    return {
        "step_time_ms": step_time_ms,
        "total_cuda_time_ms": total_cuda_time,
        "total_cpu_time_ms": total_cpu_time,
        "comm_time_ms": comm_time_ms,
        "compute_time_ms": compute_time_ms,
        "idle_ratio": idle_ratio,
        "fsdp_events": {
            "all_gather_calls": all_gather_calls,
            "reduce_scatter_calls": reduce_scatter_calls,
            "all_reduce_calls": all_reduce_calls,
        },
    }


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
) -> Dict:
    """
    执行单策略多次重复，取中位数吞吐，返回最佳一次的详细 metrics（含 trace_summary、MFU）。
    """
    set_seeds(0)
    all_runs = []
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for r in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        model = load_model(model_name=model_name)
        model = apply_strategy(model, strategy, world_size=dist.get_world_size() if dist.is_initialized() else 1)
        optimizer = build_optimizer(model, lr=lr)
        # Ensure synthetic loader won't be empty with drop_last=True and won't exhaust during warmup+steps.
        world = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_batch = int(math.ceil(global_batch_size / max(world, 1)))
        required_batches = num_warmup + num_steps + 1
        required_len = per_rank_batch * required_batches
        dataloader = build_synthetic_loader(
            train_hyper={"global_batch_size": global_batch_size},
            vocab_size=vocab_size,
            seq_len=seq_len,
            length=max(10_000, required_len),
        )

        prof = None
        if profiling == "heavy":
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"trial_{trial_id}_run{r}")
            prof = profile(
                activities=activities,
                schedule=schedule(wait=0, warmup=1, active=num_steps),
                on_trace_ready=tensorboard_trace_handler(trace_path),
                record_shapes=False,
                profile_memory=True,
            )

        layers = extract_transformer_layers(model)
        probe = _LayerProbe(layers) if layers is not None else None
        try:
            if prof is None:
                losses, step_times_ms, effective_tokens_global = run_steps(
                    model,
                    optimizer,
                    dataloader,
                    num_warmup=num_warmup,
                    num_steps=num_steps,
                    profiler_ctx=None,
                    layer_probe=probe,
                )
            else:
                with prof:
                    losses, step_times_ms, effective_tokens_global = run_steps(
                        model,
                        optimizer,
                        dataloader,
                        num_warmup=num_warmup,
                        num_steps=num_steps,
                        profiler_ctx=prof,
                        layer_probe=probe,
                    )
        finally:
            layer_summary = probe.summary() if probe is not None else {}
            if probe is not None:
                probe.close()

        torch.cuda.synchronize()
        mem_bytes = torch.cuda.max_memory_allocated()
        prof_metrics = _extract_profiler_metrics(prof) if prof is not None else {}
        prof_metrics["max_mem_bytes"] = mem_bytes
        prof_metrics["loss_mean"] = float(sum(losses) / max(len(losses), 1))
        prof_metrics["loss_std"] = 0.0
        prof_metrics["layer_stats"] = layer_summary
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

        eff_sane = [int(x) for x in effective_tokens_global if x > 0]
        eff_tokens_per_step = int(sorted(eff_sane)[len(eff_sane) // 2]) if eff_sane else 0
        prof_metrics["effective_tokens_per_step"] = eff_tokens_per_step
        prof_metrics["throughput_effective_tokens_per_s"] = (
            (eff_tokens_per_step / (mean_step_ms / 1000.0)) if mean_step_ms >= _MIN_STEP_TIME_MS and eff_tokens_per_step > 0 else 0.0
        )
        prof_metrics["throughput_tokens_per_s"] = _estimate_tokens_per_s(mean_step_ms, global_batch_size, seq_len)

        # MFU 暂不可信，置空，避免误导
        prof_metrics["mfu_percent"] = None
        prof_metrics["total_params"] = sum(p.numel() for p in model.parameters())

        prof_metrics["profiling"] = profiling
        prof_metrics["trial_context"] = {
            "requested_global_batch_size": int(global_batch_size),
            "effective_global_batch_size": int(per_rank_batch * world_size),
            "per_rank_batch_size": int(per_rank_batch),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "grad_accum": 1,
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
        prof_metrics["execution_proof"] = _collect_execution_proof(model)
        if profiling != "heavy":
            prof_metrics.setdefault("total_cuda_time_ms", 0.0)
            prof_metrics.setdefault("total_cpu_time_ms", 0.0)
            prof_metrics.setdefault("comm_time_ms", 0.0)
            prof_metrics.setdefault("compute_time_ms", 0.0)
            prof_metrics.setdefault("idle_ratio", 0.0)
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

    # 结构化 trace summary，供 LLM 直接阅读
    mem_limit_bytes = int(mem_limit_gb * 1024**3)
    headroom_mb = (mem_limit_bytes - best.get("max_mem_bytes", 0)) // (1024 * 1024)
    best["oom_margin_gb"] = float(headroom_mb) / 1024.0
    overlap_ratio = 0.0
    total = best.get("comm_time_ms", 0.0) + best.get("compute_time_ms", 0.0)
    if total > 0:
        overlap_ratio = 1.0 - (best.get("idle_ratio", 0.0))
    peak_unsharded_groups = best.get("fsdp_events", {}).get("all_gather_calls", 0)
    best["trace_summary"] = {
        "all_gather_forward_late": best.get("idle_ratio", 0.0) > 0.2,
        "overlap_ratio": overlap_ratio,
        "peak_unsharded_groups": peak_unsharded_groups,
        "memory_headroom_mb": headroom_mb,
        "mfu_percent": best.get("mfu_percent", 0.0),
        "tokens_per_step": global_batch_size * seq_len,
        "world_size": world_size,
    }
    return best


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
