from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from fsdp_agent.config import Fsdp2Strategy
from fsdp_agent.fsdp_apply import apply_fsdp2_strategy
from fsdp_agent.dataloaders import build_synthetic_loader
from fsdp_agent.dataset_stats import DatasetStats
from fsdp_agent.metrics_utils import score_strategy


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
) -> List[float]:
    it = iter(dataloader)
    losses: List[float] = []
    for step in range(num_warmup + num_steps):
        batch = next(it)
        ctx = profiler_ctx.step() if profiler_ctx else nullcontext()
        with ctx:
            loss = train_step(model, optimizer, batch, scaler=None)
        if step >= num_warmup:
            losses.append(loss)
    return losses


def _extract_profiler_metrics(prof) -> Dict[str, float]:
    events = prof.key_averages()
    total_cuda_time = sum(e.cuda_time_total for e in events) / 1e6  # ms
    total_cpu_time = sum(e.cpu_time_total for e in events) / 1e6
    step_time_ms = total_cuda_time / max(prof.step_num, 1)

    comm_time_ms = 0.0
    all_gather_calls = 0
    reduce_scatter_calls = 0
    all_reduce_calls = 0
    for e in events:
        name = e.key
        if "all_gather" in name:
            comm_time_ms += e.cuda_time_total / 1e6
            all_gather_calls += 1
        elif "reduce_scatter" in name:
            comm_time_ms += e.cuda_time_total / 1e6
            reduce_scatter_calls += 1
        elif "all_reduce" in name:
            comm_time_ms += e.cuda_time_total / 1e6
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
        model = apply_fsdp2_strategy(model, strategy)
        optimizer = build_optimizer(model, lr=lr)
        dataloader = build_synthetic_loader(
            train_hyper={"global_batch_size": global_batch_size},
            vocab_size=vocab_size,
            seq_len=seq_len,
        )

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"trial_{trial_id}_run{r}")

        prof = profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=3, active=num_steps),
            on_trace_ready=tensorboard_trace_handler(trace_path),
            record_shapes=False,
            profile_memory=True,
        )

        with prof:
            losses = run_steps(
                model,
                optimizer,
                dataloader,
                num_warmup=num_warmup,
                num_steps=num_steps,
                profiler_ctx=prof,
            )

        torch.cuda.synchronize()
        mem_bytes = torch.cuda.max_memory_allocated()
        prof_metrics = _extract_profiler_metrics(prof)
        prof_metrics["max_mem_bytes"] = mem_bytes
        prof_metrics["loss_mean"] = float(sum(losses) / max(len(losses), 1))
        prof_metrics["loss_std"] = 0.0
        prof_metrics["throughput_tokens_per_s"] = _estimate_tokens_per_s(
            prof_metrics["step_time_ms"], global_batch_size, seq_len
        )

        # MFU 估计（基于总参数量与 tokens_per_step）
        total_params = sum(p.numel() for p in model.parameters())
        step_time_s = max(prof_metrics["step_time_ms"] / 1000.0, 1e-6)
        tokens_per_step = global_batch_size * seq_len
        achieved_tflops = (6 * total_params * tokens_per_step / step_time_s) / 1e12
        peak_tflops = _get_gpu_peak_tflops()
        prof_metrics["mfu_percent"] = min(achieved_tflops / peak_tflops * 100.0, 999.0)
        prof_metrics["total_params"] = total_params

        prof_metrics["oom"] = False
        prof_metrics["trial_id"] = trial_id
        prof_metrics["run"] = r
        all_runs.append(prof_metrics)

    throughputs = sorted(x["throughput_tokens_per_s"] for x in all_runs)
    median_tp = throughputs[len(throughputs) // 2]
    best = max(all_runs, key=lambda x: x["throughput_tokens_per_s"])
    best["median_throughput_tokens_per_s"] = median_tp
    best["dataset_stats"] = dataset_stats.__dict__
    best["score"] = score_strategy(best, mem_limit_bytes=int(mem_limit_gb * 1024**3))

    # 结构化 trace summary，供 LLM 直接阅读
    mem_limit_bytes = int(mem_limit_gb * 1024**3)
    headroom_mb = (mem_limit_bytes - best.get("max_mem_bytes", 0)) // (1024 * 1024)
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
    if step_time_ms <= 0:
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
