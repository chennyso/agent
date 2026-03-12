from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import types

try:  # pragma: no cover
    import torch.profiler as torch_profiler
except Exception:  # pragma: no cover
    torch_profiler = None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: Optional[dist.ProcessGroup]):
        ctx.group = group
        if group is None or dist.get_world_size(group) == 1:
            return x
        y = x.contiguous()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def _vocab_partition(vocab_size: int, tp_rank: int, tp_world: int) -> Tuple[int, int, int]:
    part = int(math.ceil(int(vocab_size) / float(max(1, tp_world))))
    start = int(tp_rank) * int(part)
    end = min(int(start) + int(part), int(vocab_size))
    return start, end, part


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: Optional[dist.ProcessGroup],
        ignore_index: int,
    ):
        logits = vocab_parallel_logits.float()
        logits_max = torch.max(logits, dim=-1)[0]
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits -= logits_max.unsqueeze(dim=-1)

        target_mask = (target < int(vocab_start_index)) | (target >= int(vocab_end_index)) | (target == int(ignore_index))
        masked_target = target.clone() - int(vocab_start_index)
        masked_target[target_mask] = 0

        partition_vocab_size = logits.size(-1)
        logits_2d = logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target).clone().contiguous()
        predicted_logits[target_mask] = 0.0

        exp_logits = logits
        torch.exp(logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        ctx.partition_vocab_size = partition_vocab_size
        ctx.ignore_index = int(ignore_index)
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        partition_vocab_size = int(ctx.partition_vocab_size)

        grad_input = softmax
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size(0), device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).float()
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None, None, None, None


def _vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    group: Optional[dist.ProcessGroup],
    ignore_index: int = -100,
) -> torch.Tensor:
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits,
        target,
        int(vocab_start_index),
        int(vocab_end_index),
        group,
        int(ignore_index),
    )


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        tp_rank: int,
        tp_world: int,
        tp_group: Optional[dist.ProcessGroup],
    ) -> None:
        super().__init__()
        start, end, part = _vocab_partition(int(vocab_size), int(tp_rank), int(tp_world))
        self.vocab_size = int(vocab_size)
        self.vocab_start_index = int(start)
        self.vocab_end_index = int(end)
        self.partition_vocab_size = int(part)
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.empty((self.partition_vocab_size, int(hidden_size)), device="meta"))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= int(self.vocab_start_index)) & (input_ids < int(self.vocab_end_index))
        local_ids = (input_ids - int(self.vocab_start_index)).clamp(min=0)
        local_ids = torch.where(mask, local_ids, torch.zeros_like(local_ids))
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        if self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
            out = _AllReduceSum.apply(out, self.tp_group)
        return out


class VocabParallelLMHead(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        tp_rank: int,
        tp_world: int,
        tp_group: Optional[dist.ProcessGroup],
    ) -> None:
        super().__init__()
        start, end, part = _vocab_partition(int(vocab_size), int(tp_rank), int(tp_world))
        self.vocab_size = int(vocab_size)
        self.vocab_start_index = int(start)
        self.vocab_end_index = int(end)
        self.partition_vocab_size = int(part)
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.empty((self.partition_vocab_size, int(hidden_size)), device="meta"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def loss(self, hidden_states: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
        logits = self(hidden_states)
        loss = _vocab_parallel_cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            vocab_start_index=self.vocab_start_index,
            vocab_end_index=self.vocab_end_index,
            group=self.tp_group,
            ignore_index=int(ignore_index),
        )
        valid = (labels != int(ignore_index)).view(-1)
        denom = valid.sum().clamp(min=1).to(dtype=loss.dtype)
        return (loss.view(-1) * valid.to(dtype=loss.dtype)).sum() / denom

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

try:  # pragma: no cover
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module
except Exception:  # pragma: no cover
    ColwiseParallel = None  # type: ignore[assignment]
    RowwiseParallel = None  # type: ignore[assignment]
    SequenceParallel = None  # type: ignore[assignment]
    parallelize_module = None  # type: ignore[assignment]

try:  # pragma: no cover
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B
    from torch.distributed.pipelining.microbatch import TensorChunkSpec
except Exception:  # pragma: no cover
    PipelineStage = None
    Schedule1F1B = None
    ScheduleGPipe = None
    ScheduleInterleaved1F1B = None
    TensorChunkSpec = None

try:  # pragma: no cover
    import torch.distributed.pipelining.schedules as pipe_schedules
except Exception:  # pragma: no cover
    pipe_schedules = None  # type: ignore[assignment]


class SafeScheduleGPipe(ScheduleGPipe if ScheduleGPipe is not None else object):  # type: ignore[misc]
    """
    A safer GPipe schedule for mixed PP+TP(+FSDP2) runs.

    Difference from upstream ScheduleGPipe:
    - waits forward sends after each microbatch instead of delaying all waits
    - waits backward sends after each microbatch instead of delaying all waits

    This reduces overlap between PP p2p traffic and subsequent TP/FSDP collectives.
    """

    def _ensure_stage_runtime_infra(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
    ) -> None:
        stage = self._stage
        template_args = getattr(stage, "_agent_input_args_template", None)
        template_kwargs = getattr(stage, "_agent_input_kwargs_template", None)
        if template_args is None:
            raise RuntimeError("stage missing _agent_input_args_template; stage construction is incomplete")

        prepare_args = tuple(template_args)
        prepare_kwargs: Dict[str, Any] = {}
        if stage.is_first and arg_mbs and len(arg_mbs) > 0 and arg_mbs[0]:
            prepare_args = arg_mbs[0]
        if stage.is_first and kwarg_mbs and len(kwarg_mbs) > 0 and kwarg_mbs[0]:
            prepare_kwargs = kwarg_mbs[0]
        elif isinstance(template_kwargs, dict):
            prepare_kwargs = dict(template_kwargs)

        needs_fwd_init = any(chunk_id not in stage.args_recv_info for chunk_id in range(self._n_microbatches))
        if needs_fwd_init:
            fwd_params = list(inspect.signature(stage._prepare_forward_infra).parameters)
            if len(fwd_params) <= 1:
                stage._prepare_forward_infra(self._n_microbatches)
            elif len(fwd_params) == 2:
                stage._prepare_forward_infra(self._n_microbatches, prepare_args)
            else:
                stage._prepare_forward_infra(self._n_microbatches, prepare_args, prepare_kwargs)

        if self._has_backward:
            needs_bwd_init = any(chunk_id not in stage.grad_recv_info for chunk_id in range(self._n_microbatches))
            if needs_bwd_init:
                bwd_params = list(inspect.signature(stage._prepare_backward_infra).parameters)
                if len(bwd_params) <= 1:
                    stage._prepare_backward_infra(self._n_microbatches)
                elif len(bwd_params) == 2:
                    stage._prepare_backward_infra(self._n_microbatches, prepare_args)
                else:
                    stage._prepare_backward_infra(self._n_microbatches, prepare_args, prepare_kwargs)

    def _step_microbatches(  # type: ignore[override]
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
        *extra_args,
        **extra_kwargs,
    ):
        if pipe_schedules is None:
            raise RuntimeError("torch.distributed.pipelining.schedules unavailable")

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        self._ensure_stage_runtime_infra(arg_mbs, kwarg_mbs)

        for i in range(self._n_microbatches):
            with pipe_schedules.record_function(f"Forward {i}"):
                ops = [] if self._stage.is_first else self._stage.get_fwd_recv_ops(i)
                works = pipe_schedules._sorted_batch_p2p(ops, desc="fwd_recv")
                for work_idx, work in works.items():
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"fwd_recv chunk={i} wait_start idx={work_idx}",
                            flush=True,
                        )
                    work.wait()
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"fwd_recv chunk={i} wait_done idx={work_idx}",
                            flush=True,
                        )

                output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = pipe_schedules._sorted_batch_p2p(ops, desc="fwd_send")
                for work_idx, work in works.items():
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"fwd_send chunk={i} wait_start idx={work_idx}",
                            flush=True,
                        )
                    work.wait()
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"fwd_send chunk={i} wait_done idx={work_idx}",
                            flush=True,
                        )

            pipe_schedules.logger.debug(f"[{self._stage.stage_index}] Forwarded microbatch {i}")  # noqa: G004
            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        if not self._has_backward:
            return

        for i in range(self._n_microbatches):
            with pipe_schedules.record_function(f"Backward {i}"):
                ops = [] if self._stage.is_last else self._stage.get_bwd_recv_ops(i)
                works = pipe_schedules._sorted_batch_p2p(ops, desc="bwd_recv")
                for work_idx, work in works.items():
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"bwd_recv chunk={i} wait_start idx={work_idx}",
                            flush=True,
                        )
                    work.wait()
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"bwd_recv chunk={i} wait_done idx={work_idx}",
                            flush=True,
                        )

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(i, loss=loss)

                ops = self._stage.get_bwd_send_ops(i)
                works = pipe_schedules._sorted_batch_p2p(ops, desc="bwd_send")
                for work_idx, work in works.items():
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"bwd_send chunk={i} wait_start idx={work_idx}",
                            flush=True,
                        )
                    work.wait()
                    if getattr(self._stage, "_agent_runtime_debug_enabled", False):
                        print(
                            f"[p2p-wait][rank {dist.get_rank()}][stage {self._stage.stage_index}] "
                            f"bwd_send chunk={i} wait_done idx={work_idx}",
                            flush=True,
                        )

            pipe_schedules.logger.debug(f"[{self._stage.stage_index}] Backwarded microbatch {i}")  # noqa: G004

        self._update_losses(self._stage, losses)

try:  # pragma: no cover
    from safetensors import safe_open  # type: ignore
except Exception:  # pragma: no cover
    safe_open = None


def _infer_dtype(name: str) -> torch.dtype:
    n = str(name or "").lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_dist(timeout_seconds: int = 3600) -> Tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
        timeout=timedelta(seconds=int(timeout_seconds)),
    )
    return rank, world_size, local_rank


def _load_cfg(path: str) -> Dict[str, Any]:
    builtin_cfgs: Dict[str, Dict[str, Any]] = {
        "config_qwen3_2node_dense_pp4_gpipe_safe.json": {
            "model": {"path": "~/.cache/modelscope/hub/models/Qwen/Qwen3-32B", "dtype": "bf16", "seq_len": 512},
            "train": {
                "seed": 42,
                "steps": 20,
                "warmup_steps": 2,
                "global_batch_size": 8,
                "grad_accum_steps": 1,
                "lr": 0.0001,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "log_every": 1,
            },
            "parallel": {
                "pp": {
                    "degree": 4,
                    "vpp": 1,
                    "microbatches": 4,
                    "schedule": "gpipe",
                    "stages": [[0, 9], [10, 21], [22, 41], [42, 63]],
                },
                "tp": {"enabled": True, "degree": 2},
                "sp": {"enabled": False},
                "fsdp2": {
                    "enabled": False,
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "reshard_after_forward": True,
                    "reshard_after_forward_per_stage": [True, True, True, True],
                },
                "recompute": {"policy": "full", "per_stage": ["full", "full", "full", "full"]},
            },
            "runtime": {
                "cuda_alloc_conf": "expandable_segments:True",
                "dist_timeout_seconds": 3600,
                "debug_init_logs": True,
                "debug_train_logs": True,
                "debug_module_logs": True,
                "debug_p2p_logs": True,
                "serialize_pp_p2p": True,
                "pp_p2p_mode": "subgroup_group_peer",
            },
            "mfu": {"peak_tflops_total": 0.0, "flops_per_param_per_token": 6.0},
            "profile": {
                "enabled": False,
                "rank": 0,
                "trace_dir": "tb_traces",
                "wait": 1,
                "warmup": 1,
                "active": 3,
                "repeat": 1,
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
            },
        },
        "config_qwen3_2node_dense_pp4_safe.json": {
            "model": {"path": "~/.cache/modelscope/hub/models/Qwen/Qwen3-32B", "dtype": "bf16", "seq_len": 512},
            "train": {
                "seed": 42,
                "steps": 50,
                "warmup_steps": 5,
                "global_batch_size": 8,
                "grad_accum_steps": 1,
                "lr": 0.0001,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "log_every": 1,
            },
            "parallel": {
                "pp": {
                    "degree": 4,
                    "vpp": 1,
                    "microbatches": 4,
                    "schedule": "1f1b",
                    "stages": [[0, 9], [10, 21], [22, 41], [42, 63]],
                },
                "tp": {"enabled": True, "degree": 2},
                "sp": {"enabled": False},
                "fsdp2": {
                    "enabled": True,
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "reshard_after_forward": True,
                    "reshard_after_forward_per_stage": [True, True, True, True],
                },
                "recompute": {"policy": "full", "per_stage": ["full", "full", "full", "full"]},
            },
            "runtime": {
                "cuda_alloc_conf": "expandable_segments:True",
                "dist_timeout_seconds": 3600,
                "debug_init_logs": True,
                "debug_train_logs": True,
                "debug_module_logs": True,
                "debug_p2p_logs": True,
                "serialize_pp_p2p": True,
                "pp_p2p_mode": "subgroup_group_peer",
            },
            "mfu": {"peak_tflops_total": 0.0, "flops_per_param_per_token": 6.0},
            "profile": {
                "enabled": False,
                "rank": 0,
                "trace_dir": "tb_traces",
                "wait": 1,
                "warmup": 1,
                "active": 3,
                "repeat": 1,
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
            },
        },
        "config_qwen3_2node_dense_manual_pp.json": {
            "model": {"path": "~/.cache/modelscope/hub/models/Qwen/Qwen3-32B", "dtype": "bf16", "seq_len": 512},
            "train": {
                "seed": 42,
                "steps": 50,
                "warmup_steps": 5,
                "global_batch_size": 16,
                "grad_accum_steps": 1,
                "lr": 0.0001,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "log_every": 1,
            },
            "parallel": {
                "pp": {
                    "degree": 2,
                    "vpp": 1,
                    "microbatches": 2,
                    "schedule": "1f1b",
                    "stages": "auto",
                    "auto_mem_gb": [24, 32],
                    "auto_bias_stage0": -0.1,
                },
                "tp": {"enabled": True, "degree": 2},
                "sp": {"enabled": False},
                "fsdp2": {
                    "enabled": True,
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "reshard_after_forward": True,
                    "reshard_after_forward_per_stage": [True, False],
                },
                "recompute": {"policy": "full", "per_stage": ["full", "none"]},
            },
            "runtime": {
                "cuda_alloc_conf": "expandable_segments:True",
                "dist_timeout_seconds": 3600,
                "debug_init_logs": True,
                "debug_module_logs": True,
                "debug_p2p_logs": True,
            },
            "mfu": {"peak_tflops_total": 0.0, "flops_per_param_per_token": 6.0},
            "profile": {
                "enabled": False,
                "rank": 0,
                "trace_dir": "tb_traces",
                "wait": 1,
                "warmup": 1,
                "active": 3,
                "repeat": 1,
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
            },
        },
    }

    candidate = Path(path)
    candidates = [candidate]
    sibling = Path(__file__).resolve().parent / candidate.name
    if sibling != candidate:
        candidates.append(sibling)
    for cfg_path in candidates:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.loads(f.read())

    builtin = builtin_cfgs.get(candidate.name)
    if builtin is not None:
        print(f"[config] missing file {path}; using built-in preset {candidate.name}", flush=True)
        return copy.deepcopy(builtin)
    raise FileNotFoundError(f"config file not found: {path}")


def _extract_transformer_layers(hf_model: nn.Module) -> nn.ModuleList:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    raise RuntimeError("Cannot find transformer layers (model.layers / transformer.h)")


def _layer_param_bytes_from_meta(model: nn.Module) -> List[int]:
    layers = _extract_transformer_layers(model)
    out: List[int] = []
    for layer in layers:
        total = 0
        for p in layer.parameters(recurse=True):
            try:
                total += int(p.numel()) * int(p.element_size())
            except Exception:
                total += int(p.numel()) * 2
        out.append(int(total))
    return out


def _auto_pp_stages(
    *,
    model: nn.Module,
    pp_degree: int,
    mem_gb: List[float],
    bias_stage0: float,
) -> List[List[int]]:
    total_layers = len(_extract_transformer_layers(model))
    if int(pp_degree) != len(mem_gb):
        raise ValueError("parallel.pp.auto_mem_gb length must equal parallel.pp.degree")
    weights = _layer_param_bytes_from_meta(model)
    total_w = float(sum(weights)) if weights else 0.0
    mem = [float(x) for x in mem_gb]
    frac = [x / sum(mem) for x in mem]
    if frac and bias_stage0:
        frac[0] = min(0.9, max(0.1, float(frac[0]) + float(bias_stage0)))
        rest = 1.0 - frac[0]
        tail_sum = sum(frac[1:]) or 1.0
        for i in range(1, len(frac)):
            frac[i] = rest * (frac[i] / tail_sum)
    targets = [f * total_w for f in frac]
    stages: List[List[int]] = []
    start = 0
    cum = 0.0
    stage_idx = 0
    for i, w in enumerate(weights):
        remaining_layers = total_layers - i
        remaining_stages = int(pp_degree) - stage_idx
        if remaining_layers == remaining_stages:
            stages.append([start, i])
            start = i + 1
            stage_idx += 1
            continue
        cum += float(w)
        if stage_idx < int(pp_degree) - 1 and cum >= float(targets[stage_idx]) and i >= start:
            stages.append([start, i])
            start = i + 1
            stage_idx += 1
            cum = 0.0
    if start <= total_layers - 1:
        stages.append([start, total_layers - 1])
    if len(stages) != int(pp_degree):
        raise RuntimeError(f"auto PP split produced {len(stages)} stages, expected {pp_degree}")
    return stages


def _compute_virtual_slices(pp_stages: List[List[int]], vpp: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for start, end in pp_stages:
        start = int(start)
        end = int(end)
        n = end - start + 1
        if int(vpp) <= 1:
            out.append((start, end))
            continue
        cuts = [start]
        for k in range(1, int(vpp)):
            cut = start + int(round(k * n / float(vpp)))
            cut = max(start + 1, min(cut, end))
            cuts.append(cut)
        cuts.append(end + 1)
        for a, b in zip(cuts[:-1], cuts[1:]):
            out.append((a, b - 1))
    return out


def _call_with_supported_kwargs(fn, /, *args, **kwargs):
    try:
        sig = inspect.signature(fn)
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **supported)
    except Exception:
        return fn(*args, **kwargs)


class _StageBackbone(nn.Module):
    def __init__(
        self,
        *,
        embed_tokens: Optional[nn.Module],
        layers: nn.ModuleList,
        norm: Optional[nn.Module],
        rotary_emb: Optional[nn.Module],
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb


class DenseCausalLMStage(nn.Module):
    def __init__(
        self,
        *,
        backbone: _StageBackbone,
        lm_head: Optional[nn.Module],
        global_layer_start: int,
        is_first: bool,
        is_last: bool,
        seq_len: int,
        recompute: str,
        debug_module_logs: bool,
        debug_rank: int,
        stage_id: int,
    ) -> None:
        super().__init__()
        self.model = backbone
        self.lm_head = lm_head
        self.global_layer_start = int(global_layer_start)
        self.is_first = bool(is_first)
        self.is_last = bool(is_last)
        self.seq_len = int(seq_len)
        self.recompute = str(recompute or "none").lower()
        self.debug_module_logs = bool(debug_module_logs)
        self.debug_rank = int(debug_rank)
        self.stage_id = int(stage_id)
        self._forward_debug_calls = 0
        self._loss_debug_calls = 0

    def _debug_enabled(self) -> bool:
        return bool(self.debug_module_logs and self._forward_debug_calls == 0)

    def _log(self, msg: str) -> None:
        print(f"[module-debug][rank {self.debug_rank}][stage {self.stage_id}] {msg}", flush=True)

    @staticmethod
    def _describe_tensor(t: torch.Tensor) -> str:
        shape = tuple(int(x) for x in t.shape)
        return f"type={type(t).__name__} shape={shape} dtype={t.dtype} device={t.device}"

    def _run_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Any,
    ):
        out = _call_with_supported_kwargs(
            layer,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
            output_attentions=False,
        )
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def forward(self, *args):  # PipelineStage passes positional only
        debug_this_call = self._debug_enabled()
        if self.is_first:
            input_ids = args[0]
            if self.model.embed_tokens is None:
                raise RuntimeError("stage0 missing embed_tokens")
            if debug_this_call:
                self._log(f"before embed_tokens {self._describe_tensor(input_ids)}")
            hidden_states = self.model.embed_tokens(input_ids)
            hidden_states = hidden_states.contiguous()
            if debug_this_call:
                self._log(f"after embed_tokens {self._describe_tensor(hidden_states)}")
        else:
            hidden_states = args[0].contiguous()
            if debug_this_call:
                self._log(f"recv hidden_states {self._describe_tensor(hidden_states)}")

        bsz, seqlen = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        if seqlen != int(self.seq_len):
            raise RuntimeError(f"seq_len mismatch: got {seqlen}, expected {self.seq_len}")
        attention_mask = torch.ones((bsz, seqlen), device=hidden_states.device, dtype=torch.bool)
        position_ids = torch.arange(seqlen, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        position_embeddings = None
        if self.model.rotary_emb is not None:
            if debug_this_call:
                self._log("before rotary_emb")
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            if debug_this_call:
                if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
                    self._log(
                        "after rotary_emb "
                        f"cos={self._describe_tensor(position_embeddings[0])} "
                        f"sin={self._describe_tensor(position_embeddings[1])}"
                    )
                else:
                    self._log(f"after rotary_emb type={type(position_embeddings).__name__}")

        for layer_idx, layer in enumerate(self.model.layers):
            if debug_this_call:
                global_layer_idx = self.global_layer_start + int(layer_idx)
                self._log(f"before layer local={layer_idx} global={global_layer_idx}")
            if self.recompute in {"full", "checkpoint"}:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda hs: self._run_layer(layer, hs, attention_mask, position_ids, position_embeddings),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._run_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings)
            if debug_this_call:
                self._log(f"after layer local={layer_idx} global={global_layer_idx} {self._describe_tensor(hidden_states)}")

        if not self.is_last:
            hidden_states = hidden_states.contiguous()
            if debug_this_call:
                self._log(f"forward return hidden_states {self._describe_tensor(hidden_states)}")
                self._forward_debug_calls += 1
            return hidden_states

        if self.model.norm is not None:
            if debug_this_call:
                self._log("before final norm")
            hidden_states = self.model.norm(hidden_states)
            if debug_this_call:
                self._log(f"after final norm {self._describe_tensor(hidden_states)}")
        hidden_states = hidden_states.contiguous()
        if debug_this_call:
            self._log(f"forward return last-stage hidden_states {self._describe_tensor(hidden_states)}")
            self._forward_debug_calls += 1
        return hidden_states

    def compute_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("last stage missing lm_head")
        debug_this_call = bool(self.debug_module_logs and self._loss_debug_calls == 0)
        if debug_this_call:
            self._log(
                "before compute_loss "
                f"hidden_states={self._describe_tensor(hidden_states)} "
                f"labels={self._describe_tensor(labels)}"
            )

        labels = labels.long()
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if hasattr(self.lm_head, "loss"):
            out = self.lm_head.loss(shift_hidden, shift_labels, ignore_index=-100)  # type: ignore[call-arg]
            if debug_this_call:
                self._log(f"after compute_loss loss_shape={tuple(out.shape) if out.dim() else ()} dtype={out.dtype}")
                self._loss_debug_calls += 1
            return out

        logits = self.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        out = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        if debug_this_call:
            self._log(f"after compute_loss loss_shape={tuple(out.shape) if out.dim() else ()} dtype={out.dtype}")
            self._loss_debug_calls += 1
        return out


def _resolve_index(model_dir: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
    base = Path(model_dir)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = base / name
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") or {}
            return {str(k): str(v) for k, v in weight_map.items()}, [str(base / f) for f in sorted(set(weight_map.values()))]
    single = base / "model.safetensors"
    if single.exists():
        return None, [str(single)]
    raise FileNotFoundError(f"cannot find sharded index in {model_dir}")


def _load_tensors_safetensors(shard_path: str, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    if safe_open is None:
        raise RuntimeError("safetensors is required (pip install safetensors)")
    out: Dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[misc]
        for k in keys:
            if k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def _load_tensor0_slice_safetensors(shard_path: str, key: str, start: int, end: int) -> Optional[torch.Tensor]:
    if safe_open is None:
        raise RuntimeError("safetensors is required (pip install safetensors)")
    with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[misc]
        if key not in f.keys():
            return None
        if hasattr(f, "get_slice"):
            s = f.get_slice(key)  # type: ignore[attr-defined]
            return s[int(start) : int(end)]
        return f.get_tensor(key)[int(start) : int(end)]


def _load_state_dict_subset(model_dir: str, ckpt_keys: List[str]) -> Dict[str, torch.Tensor]:
    weight_map, shard_paths = _resolve_index(model_dir)
    needed = set(str(k) for k in ckpt_keys)
    if weight_map is None:
        return _load_tensors_safetensors(shard_paths[0], needed)
    by_file: Dict[str, List[str]] = {}
    for k in needed:
        fn = weight_map.get(k)
        if fn:
            by_file.setdefault(str(Path(model_dir) / fn), []).append(k)
    out: Dict[str, torch.Tensor] = {}
    for shard, keys in by_file.items():
        out.update(_load_tensors_safetensors(shard, keys))
    return out


def _translate_stage_key(stage_key: str, *, global_layer_start: int) -> str:
    if stage_key.startswith("model.layers."):
        parts = stage_key.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            local_idx = int(parts[2])
            parts[2] = str(int(global_layer_start) + local_idx)
            return ".".join(parts)
    return stage_key


def _materialize_and_load_stage(stage: DenseCausalLMStage, model_dir: str, dtype: torch.dtype) -> None:
    stage.to_empty(device="cpu")  # type: ignore[attr-defined]
    stage_keys = list(stage.state_dict().keys())
    local_to_ckpt = {k: _translate_stage_key(k, global_layer_start=stage.global_layer_start) for k in stage_keys}

    special_local: Dict[str, torch.Tensor] = {}
    weight_map, shard_paths = _resolve_index(model_dir)

    def _load_vocab_weight(ckpt_key: str, module: nn.Module) -> Optional[torch.Tensor]:
        if weight_map is None:
            shard_path = str(shard_paths[0])
        else:
            shard_rel = weight_map.get(ckpt_key)
            if not shard_rel:
                return None
            shard_path = str(Path(model_dir) / shard_rel)
        start = int(getattr(module, "vocab_start_index", 0))
        part = int(getattr(module, "partition_vocab_size", 0))
        t = _load_tensor0_slice_safetensors(shard_path, ckpt_key, start, start + part)
        if t is None:
            return None
        if t.size(0) < part:
            pad = torch.zeros((part - t.size(0), t.size(1)), dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        if torch.is_floating_point(t):
            t = t.to(dtype=dtype)
        return t

    # Avoid loading full vocab matrices; load only local vocab partition when present.
    if "model.embed_tokens.weight" in local_to_ckpt and stage.model.embed_tokens is not None and hasattr(stage.model.embed_tokens, "partition_vocab_size"):
        ck = local_to_ckpt["model.embed_tokens.weight"]
        t = _load_vocab_weight(ck, stage.model.embed_tokens)
        if t is None:
            raise RuntimeError(
                f"missing checkpoint tensor for embed_tokens: {ck}. "
                f"Make sure `{model_dir}` is a local HF directory with safetensors + index.json."
            )
        special_local["model.embed_tokens.weight"] = t
        local_to_ckpt.pop("model.embed_tokens.weight", None)
    if "lm_head.weight" in local_to_ckpt and stage.lm_head is not None and hasattr(stage.lm_head, "partition_vocab_size"):
        ck = local_to_ckpt["lm_head.weight"]
        t = _load_vocab_weight(ck, stage.lm_head)
        if t is None:
            raise RuntimeError(
                f"missing checkpoint tensor for lm_head: {ck}. "
                f"Make sure `{model_dir}` is a local HF directory with safetensors + index.json."
            )
        special_local["lm_head.weight"] = t
        local_to_ckpt.pop("lm_head.weight", None)

    ckpt = _load_state_dict_subset(model_dir, list(local_to_ckpt.values()))
    local_sd: Dict[str, torch.Tensor] = {}
    for lk, ck in local_to_ckpt.items():
        t = ckpt.get(ck)
        if t is None:
            continue
        if torch.is_floating_point(t):
            t = t.to(dtype=dtype)
        local_sd[lk] = t
    local_sd.update(special_local)
    stage.load_state_dict(local_sd, strict=False)


def _build_tp_plan(module: nn.Module, tp_degree: int, *, sp_enabled: bool) -> Dict[str, Any]:
    if ColwiseParallel is None or RowwiseParallel is None:
        raise RuntimeError("tensor parallel unavailable")
    mapping: Dict[str, Any] = {}
    col = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    row = ("o_proj", "down_proj")
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            if any(name.endswith(s) for s in col):
                if getattr(mod, "out_features", 0) % int(tp_degree) != 0:
                    continue
                mapping[name] = ColwiseParallel(use_local_output=True)
            elif any(name.endswith(s) for s in row):
                mapping[name] = RowwiseParallel()
    if sp_enabled and SequenceParallel is not None:
        for name, mod in module.named_modules():
            if isinstance(mod, nn.LayerNorm) or mod.__class__.__name__.lower() in {"rmsnorm"}:
                mapping.setdefault(name, SequenceParallel(sequence_dim=1, use_local_output=False))
    return mapping


def _apply_tp_sp(module: nn.Module, tp_mesh: DeviceMesh, tp_degree: int, *, sp_enabled: bool) -> None:
    if int(tp_degree) <= 1:
        return
    if parallelize_module is None:
        raise RuntimeError("parallelize_module unavailable")
    plan = _build_tp_plan(module, tp_degree, sp_enabled=sp_enabled)
    if plan:
        parallelize_module(module, device_mesh=tp_mesh, parallelize_plan=plan)


def _apply_fsdp2(module: DenseCausalLMStage, dp_mesh: DeviceMesh, *, param_dtype: torch.dtype, reduce_dtype: torch.dtype, reshard_after_forward: bool) -> None:
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    for layer in module.model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=bool(reshard_after_forward))


def _enable_stage_runtime_debug(stage: Any, *, rank: int, enabled: bool, p2p_mode: str = "original") -> None:
    p2p_mode = str(p2p_mode or "original").lower()
    if p2p_mode not in {"original", "default_group", "subgroup_group_peer"}:
        raise ValueError(f"unsupported p2p_mode: {p2p_mode}")
    if not enabled and p2p_mode == "original":
        return
    if getattr(stage, "_agent_runtime_debug_enabled", False):
        return
    stage._agent_runtime_debug_enabled = bool(enabled)
    stage._agent_runtime_debug_seen = set()

    def _should_log(key: str) -> bool:
        seen = stage._agent_runtime_debug_seen
        if key in seen:
            return False
        seen.add(key)
        return True

    def _log(msg: str) -> None:
        print(f"[p2p-debug][rank {rank}][stage {stage.stage_index}] {msg}", flush=True)

    def _tensor_meta(t: Any) -> str:
        if not isinstance(t, torch.Tensor):
            return f"type={type(t).__name__}"
        shape = tuple(int(x) for x in t.shape)
        return f"shape={shape} dtype={t.dtype} device={t.device}"

    def _log_ops(kind: str, chunk_id: int, ops: List[Any]) -> None:
        if not _should_log(f"{kind}_detail_{chunk_id}"):
            return
        for op_idx, op in enumerate(ops):
            peer = getattr(op, "peer", None)
            group_peer = getattr(op, "group_peer", None)
            tensor = getattr(op, "tensor", None)
            group = getattr(op, "group", None)
            group_id = None if group is None else id(group)
            group_ranks = None
            group_rank = None
            if group is not None and hasattr(dist, "get_process_group_ranks"):
                try:
                    group_ranks = dist.get_process_group_ranks(group)
                    group_rank = dist.get_rank(group)
                except Exception:
                    group_ranks = None
                    group_rank = None
            _log(
                f"{kind}[{op_idx}] chunk={chunk_id} peer={peer} group_peer={group_peer} "
                f"tensor={_tensor_meta(tensor)} "
                f"group_id={group_id} group_rank={group_rank} group_ranks={group_ranks}"
            )

    orig_get_fwd_recv_ops = stage.get_fwd_recv_ops
    orig_get_fwd_send_ops = stage.get_fwd_send_ops
    orig_get_bwd_recv_ops = getattr(stage, "get_bwd_recv_ops", None)
    orig_get_bwd_send_ops = getattr(stage, "get_bwd_send_ops", None)
    orig_forward_one_chunk = stage.forward_one_chunk

    def _rewrite_ops(ops: List[Any]) -> List[Any]:
        if p2p_mode == "original":
            return ops
        supports_group_peer = "group_peer" in inspect.signature(dist.P2POp).parameters
        rewritten: List[Any] = []
        for op in ops:
            op_fn = getattr(op, "op")
            tensor = getattr(op, "tensor")
            peer_global = int(getattr(op, "peer"))
            group = getattr(op, "group", None)
            tag = int(getattr(op, "tag", 0))
            if p2p_mode == "default_group" or group is None:
                rewritten.append(
                    dist.P2POp(
                        op_fn,
                        tensor,
                        peer_global,
                        group=None,
                        tag=tag,
                    )
                )
                continue

            if p2p_mode == "subgroup_group_peer" and supports_group_peer:
                group_ranks = dist.get_process_group_ranks(group)
                peer_in_group = int(group_ranks.index(peer_global))
                rewritten.append(
                    dist.P2POp(
                        op=op_fn,
                        tensor=tensor,
                        group=group,
                        group_peer=peer_in_group,
                        tag=tag,
                    )
                )
                continue

            rewritten.append(
                dist.P2POp(
                    op_fn,
                    tensor,
                    peer_global,
                    group=None,
                    tag=tag,
                )
            )
        return rewritten

    def get_fwd_recv_ops_wrapper(self, chunk_id: int):
        ops = _rewrite_ops(orig_get_fwd_recv_ops(chunk_id))
        if _should_log(f"recv_ops_{chunk_id}"):
            _log(f"get_fwd_recv_ops chunk={chunk_id} num_ops={len(ops)} is_first={self.is_first} is_last={self.is_last}")
        _log_ops("RECV", chunk_id, ops)
        return ops

    def get_fwd_send_ops_wrapper(self, chunk_id: int):
        ops = _rewrite_ops(orig_get_fwd_send_ops(chunk_id))
        if _should_log(f"send_ops_{chunk_id}"):
            _log(f"get_fwd_send_ops chunk={chunk_id} num_ops={len(ops)} is_first={self.is_first} is_last={self.is_last}")
        _log_ops("SEND", chunk_id, ops)
        return ops

    def get_bwd_recv_ops_wrapper(self, chunk_id: int):
        if orig_get_bwd_recv_ops is None:
            return []
        return _rewrite_ops(orig_get_bwd_recv_ops(chunk_id))

    def get_bwd_send_ops_wrapper(self, chunk_id: int):
        if orig_get_bwd_send_ops is None:
            return []
        return _rewrite_ops(orig_get_bwd_send_ops(chunk_id))

    def forward_one_chunk_wrapper(self, chunk_id: int, *args, **kwargs):
        if _should_log(f"forward_enter_{chunk_id}"):
            _log(f"forward_one_chunk enter chunk={chunk_id}")
        out = orig_forward_one_chunk(chunk_id, *args, **kwargs)
        if _should_log(f"forward_exit_{chunk_id}"):
            if torch.is_tensor(out):
                desc = f"shape={tuple(int(x) for x in out.shape)} dtype={out.dtype} device={out.device}"
            else:
                desc = f"type={type(out).__name__}"
            _log(f"forward_one_chunk exit chunk={chunk_id} out={desc}")
        return out

    stage.get_fwd_recv_ops = types.MethodType(get_fwd_recv_ops_wrapper, stage)
    stage.get_fwd_send_ops = types.MethodType(get_fwd_send_ops_wrapper, stage)
    if orig_get_bwd_recv_ops is not None:
        stage.get_bwd_recv_ops = types.MethodType(get_bwd_recv_ops_wrapper, stage)
    if orig_get_bwd_send_ops is not None:
        stage.get_bwd_send_ops = types.MethodType(get_bwd_send_ops_wrapper, stage)
    stage.forward_one_chunk = types.MethodType(forward_one_chunk_wrapper, stage)


def _make_synth_batch(vocab_size: int, seq_len: int, batch_size: int, *, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    ids = torch.randint(0, int(vocab_size), (batch_size, seq_len), generator=g, dtype=torch.long)
    return ids, ids.clone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cuda_alloc_conf = ((cfg.get("runtime") or {}).get("cuda_alloc_conf") or "").strip()
    if cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf)
    dist_timeout_seconds = int(((cfg.get("runtime") or {}).get("dist_timeout_seconds") or 3600))
    debug_init_logs = bool((cfg.get("runtime") or {}).get("debug_init_logs", True))
    debug_module_logs = bool((cfg.get("runtime") or {}).get("debug_module_logs", True))
    debug_p2p_logs = bool((cfg.get("runtime") or {}).get("debug_p2p_logs", True))
    runtime_cfg = cfg.get("runtime") or {}
    if "pp_p2p_mode" in runtime_cfg:
        pp_p2p_mode = str(runtime_cfg.get("pp_p2p_mode") or "subgroup_group_peer").strip()
    elif "pp_use_default_group_for_p2p" in runtime_cfg:
        pp_p2p_mode = "default_group" if bool(runtime_cfg.get("pp_use_default_group_for_p2p")) else "original"
    else:
        pp_p2p_mode = "subgroup_group_peer"

    rank, world_size, local_rank = _setup_dist(timeout_seconds=dist_timeout_seconds)
    try:
        train_cfg = cfg.get("train") or {}
        seed = int(train_cfg.get("seed") or 42)
        _seed_all(seed + rank)

        model_cfg = cfg.get("model") or {}
        model_dir = os.path.expandvars(os.path.expanduser(str(model_cfg.get("path") or "")))
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"model.path must exist: {model_dir}")
        dtype = _infer_dtype(model_cfg.get("dtype") or "bf16")
        seq_len = int(model_cfg.get("seq_len") or 1024)

        steps = int(train_cfg.get("steps") or 10)
        warmup_steps = int(train_cfg.get("warmup_steps") or 2)
        global_batch_size = int(train_cfg.get("global_batch_size") or 8)
        grad_accum_steps = int(train_cfg.get("grad_accum_steps") or 1)
        lr = float(train_cfg.get("lr") or 1e-4)
        wd = float(train_cfg.get("weight_decay") or 0.1)
        grad_clip = float(train_cfg.get("grad_clip") or 1.0)
        log_every = int(train_cfg.get("log_every") or 1)

        par = cfg.get("parallel") or {}
        profile_cfg = cfg.get("profile") or {}
        mfu_cfg = cfg.get("mfu") or {}
        pp_cfg = par.get("pp") or {}
        tp_cfg = par.get("tp") or {}
        sp_cfg = par.get("sp") or {}
        fsdp_cfg = par.get("fsdp2") or {}
        recompute_cfg = par.get("recompute") or {}

        pp_degree = int(pp_cfg.get("degree") or 2)
        vpp = int(pp_cfg.get("vpp") or 1)
        microbatches = int(pp_cfg.get("microbatches") or 4)
        schedule_name = str(pp_cfg.get("schedule") or "1f1b").lower()
        serialize_pp_p2p = bool((cfg.get("runtime") or {}).get("serialize_pp_p2p", False))
        pp_stages = pp_cfg.get("stages")
        pp_auto_mem_gb = pp_cfg.get("auto_mem_gb")
        pp_auto_bias0 = float(pp_cfg.get("auto_bias_stage0", -0.08) or -0.08)

        tp_enabled = bool(tp_cfg.get("enabled", True))
        tp_degree = int(tp_cfg.get("degree") or 2)
        if not tp_enabled:
            tp_degree = 1
        sp_enabled = bool(sp_cfg.get("enabled", False))

        fsdp_enabled = bool(fsdp_cfg.get("enabled", True))
        fsdp_enabled_per_stage = fsdp_cfg.get("enabled_per_stage")
        fsdp_param_dtype = _infer_dtype(fsdp_cfg.get("param_dtype") or "bf16")
        fsdp_reduce_dtype = _infer_dtype(fsdp_cfg.get("reduce_dtype") or "bf16")
        reshard_after_forward = bool(fsdp_cfg.get("reshard_after_forward", True))
        reshard_per_stage = fsdp_cfg.get("reshard_after_forward_per_stage")

        recompute = str(recompute_cfg.get("policy") or "full").lower()
        recompute_per_stage = recompute_cfg.get("per_stage")

        if PipelineStage is None or TensorChunkSpec is None:
            raise RuntimeError("torch.distributed.pipelining unavailable")

        if world_size % pp_degree != 0:
            raise ValueError("world_size must be divisible by pp_degree")
        ranks_per_stage = world_size // pp_degree
        if ranks_per_stage % tp_degree != 0:
            raise ValueError("ranks_per_stage must be divisible by tp_degree")
        dp_degree = ranks_per_stage // tp_degree

        any_fsdp_enabled = bool(fsdp_enabled)
        if isinstance(fsdp_enabled_per_stage, list):
            any_fsdp_enabled = any(bool(x) for x in fsdp_enabled_per_stage)

        if any_fsdp_enabled and dp_degree > 1 and pp_degree > 1:
            if reshard_after_forward or (
                isinstance(reshard_per_stage, list) and any(bool(x) for x in reshard_per_stage)
            ):
                if rank == 0:
                    print(
                        "[warn] forcing fsdp2 reshard_after_forward=False for PP+TP+FSDP2; "
                        "this matches PyTorch composability tests and avoids first-step all_gather deadlocks",
                        flush=True,
                    )
                reshard_after_forward = False
                if isinstance(reshard_per_stage, list):
                    reshard_per_stage = [False for _ in reshard_per_stage]
            if recompute != "none" or (
                isinstance(recompute_per_stage, list) and any(str(x).lower() not in {"none", "off", "false"} for x in recompute_per_stage)
            ):
                if rank == 0:
                    print(
                        "[warn] forcing recompute=none for PP+TP+FSDP2; "
                        "activation checkpointing inside FSDP2-wrapped pipeline stages is not aligned with the official composability path",
                        flush=True,
                    )
                recompute = "none"
                if isinstance(recompute_per_stage, list):
                    recompute_per_stage = ["none" for _ in recompute_per_stage]

        pp_rank = rank // ranks_per_stage
        local_in_stage = rank % ranks_per_stage
        dp_idx = local_in_stage // tp_degree
        tp_idx = local_in_stage % tp_degree

        def _r(pp: int, dp: int, tp: int) -> int:
            return int(pp) * int(ranks_per_stage) + int(dp) * int(tp_degree) + int(tp)

        root_mesh_tensor = torch.tensor(
            [[[_r(pp, dp, tp) for tp in range(tp_degree)] for dp in range(dp_degree)] for pp in range(pp_degree)],
            dtype=torch.int,
        )
        root_mesh = DeviceMesh("cuda", root_mesh_tensor, mesh_dim_names=("pp", "dp", "tp"))

        pp_mesh = root_mesh["pp"]
        pp_group = pp_mesh.get_group()

        stage_base = pp_rank * ranks_per_stage
        tp_group_ranks = [stage_base + dp_idx * tp_degree + t for t in range(tp_degree)]
        dp_group_ranks = [stage_base + d * tp_degree + tp_idx for d in range(dp_degree)]

        tp_mesh = root_mesh["tp"] if tp_degree > 1 else None
        dp_mesh = root_mesh["dp"] if dp_degree > 1 else None

        is_log_rank = bool(pp_rank == (pp_degree - 1) and dp_idx == 0 and tp_idx == 0)

        if rank == 0:
            print(f"[init] world={world_size} pp={pp_degree} dp={dp_degree} tp={tp_degree} vpp={vpp}", flush=True)
            print(f"[model] {model_dir}", flush=True)
            print(f"[init] dist_timeout_seconds={dist_timeout_seconds}", flush=True)

        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

        hf_cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        with torch.device("meta"):
            full_model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
        if getattr(full_model.config, "use_cache", None) is not None:
            full_model.config.use_cache = False
        if getattr(full_model.config, "return_dict", None) is not None:
            full_model.config.return_dict = False

        if isinstance(pp_stages, str) and pp_stages.strip().lower() == "auto":
            if not isinstance(pp_auto_mem_gb, list) or len(pp_auto_mem_gb) != pp_degree:
                raise ValueError("pp.stages='auto' requires parallel.pp.auto_mem_gb len==pp.degree")
            pp_stages = _auto_pp_stages(
                model=full_model, pp_degree=pp_degree, mem_gb=[float(x) for x in pp_auto_mem_gb], bias_stage0=pp_auto_bias0
            )
            if rank == 0:
                print(f"[pp] auto stages={pp_stages}", flush=True)
        if not isinstance(pp_stages, list) or len(pp_stages) != pp_degree:
            raise ValueError("parallel.pp.stages must be list len==pp.degree or 'auto'")

        if vpp > 1 and ScheduleInterleaved1F1B is None:
            if rank == 0:
                print("[warn] ScheduleInterleaved1F1B unavailable; forcing vpp=1", flush=True)
            vpp = 1

        virtual_slices = _compute_virtual_slices(pp_stages, vpp)
        num_virtual = pp_degree * vpp

        per_dp_batch = int(math.ceil(global_batch_size / float(dp_degree)))
        min_microbatches = 1
        if schedule_name in {"1f1b", "interleaved1f1b"}:
            min_microbatches = int(num_virtual)
        if microbatches < min_microbatches:
            if per_dp_batch >= min_microbatches:
                if rank == 0:
                    print(
                        f"[warn] microbatches={microbatches} too small for schedule={schedule_name}; "
                        f"auto-bumping to {min_microbatches}",
                        flush=True,
                    )
                microbatches = int(min_microbatches)
            else:
                if rank == 0 and schedule_name != "gpipe":
                    print(
                        f"[warn] microbatches={microbatches} and per_dp_batch={per_dp_batch} cannot satisfy "
                        f"schedule={schedule_name}; falling back to gpipe",
                        flush=True,
                    )
                schedule_name = "gpipe"
                min_microbatches = 1
        if microbatches > per_dp_batch:
            raise ValueError("pp.microbatches must be <= ceil(global_batch_size/dp_degree)")

        device = torch.device("cuda", local_rank)
        local_stage_ids = [dist.get_rank(pp_group) + k * pp_degree for k in range(vpp)]
        stages: List[Any] = []
        stage_modules: List[DenseCausalLMStage] = []

        layers_full = _extract_transformer_layers(full_model)
        hidden_size = int(getattr(hf_cfg, "hidden_size", getattr(hf_cfg, "n_embd", 0) or 0))
        vocab_size = int(getattr(hf_cfg, "vocab_size", 32000))
        model_params = int(sum(int(p.numel()) for p in full_model.parameters()))

        tp_group = tp_mesh.get_group() if tp_mesh is not None else None

        for stage_id in local_stage_ids:
            ls, le = virtual_slices[int(stage_id)]
            is_first = int(stage_id) == 0
            is_last = int(stage_id) == (num_virtual - 1)
            stage_t0 = time.time()
            if debug_init_logs:
                print(
                    f"[init][rank {rank}] build stage_id={stage_id} layers=[{ls},{le}] first={is_first} last={is_last}",
                    flush=True,
                )
            embed = VocabParallelEmbedding(vocab_size=vocab_size, hidden_size=hidden_size, tp_rank=tp_idx, tp_world=tp_degree, tp_group=tp_group) if is_first else None
            norm = getattr(full_model.model, "norm", None) if is_last else None
            lm_head = VocabParallelLMHead(vocab_size=vocab_size, hidden_size=hidden_size, tp_rank=tp_idx, tp_world=tp_degree, tp_group=tp_group) if is_last else None

            stage_recompute = recompute
            if isinstance(recompute_per_stage, list) and len(recompute_per_stage) == num_virtual:
                stage_recompute = str(recompute_per_stage[int(stage_id)]).lower()
            backbone = _StageBackbone(
                embed_tokens=embed,
                layers=nn.ModuleList([layers_full[i] for i in range(int(ls), int(le) + 1)]),
                norm=norm,
                rotary_emb=getattr(full_model.model, "rotary_emb", None),
            )
            stage_mod = DenseCausalLMStage(
                backbone=backbone,
                lm_head=lm_head,
                global_layer_start=int(ls),
                is_first=is_first,
                is_last=is_last,
                seq_len=seq_len,
                recompute=stage_recompute,
                debug_module_logs=debug_module_logs,
                debug_rank=rank,
                stage_id=int(stage_id),
            )
            if debug_init_logs:
                print(f"[init][rank {rank}] load weights stage_id={stage_id}", flush=True)
            _materialize_and_load_stage(stage_mod, model_dir=model_dir, dtype=dtype)
            stage_mod.to(device)

            if tp_degree > 1 and tp_mesh is not None:
                if debug_init_logs:
                    print(f"[init][rank {rank}] apply tp stage_id={stage_id}", flush=True)
                _apply_tp_sp(stage_mod, tp_mesh, tp_degree, sp_enabled=sp_enabled)
            stage_fsdp_enabled = bool(fsdp_enabled)
            if isinstance(fsdp_enabled_per_stage, list) and len(fsdp_enabled_per_stage) == num_virtual:
                stage_fsdp_enabled = bool(fsdp_enabled_per_stage[int(stage_id)])

            if stage_fsdp_enabled and dp_mesh is not None and dp_degree > 1:
                stage_reshard = bool(reshard_after_forward)
                if isinstance(reshard_per_stage, list) and len(reshard_per_stage) == num_virtual:
                    stage_reshard = bool(reshard_per_stage[int(stage_id)])
                if debug_init_logs:
                    print(
                        f"[init][rank {rank}] apply fsdp2 stage_id={stage_id} reshard_after_forward={stage_reshard}",
                        flush=True,
                    )
                _apply_fsdp2(stage_mod, dp_mesh, param_dtype=fsdp_param_dtype, reduce_dtype=fsdp_reduce_dtype, reshard_after_forward=stage_reshard)

            mb = max(1, int(math.ceil(per_dp_batch / float(microbatches))))
            dummy_ids = torch.zeros((mb, seq_len), device=device, dtype=torch.long)
            dummy_hs = torch.zeros((mb, seq_len, hidden_size), device=device, dtype=dtype)
            out_args = torch.zeros((mb, seq_len, hidden_size), device=device, dtype=dtype)
            stage = PipelineStage(
                stage_mod,
                int(stage_id),
                int(num_virtual),
                device,
                input_args=(dummy_ids,) if is_first else (dummy_hs,),
                output_args=out_args,
                group=pp_group,
            )
            stage._agent_input_args_template = (dummy_ids,) if is_first else (dummy_hs,)
            stage._agent_input_kwargs_template = {}
            _enable_stage_runtime_debug(
                stage,
                rank=rank,
                enabled=debug_p2p_logs,
                p2p_mode=pp_p2p_mode,
            )
            if debug_init_logs:
                print(
                    f"[init][rank {rank}] stage ready stage_id={stage_id} elapsed={time.time() - stage_t0:.1f}s",
                    flush=True,
                )
            stages.append(stage)
            stage_modules.append(stage_mod)

        chunk_spec = TensorChunkSpec(0) if microbatches > 1 else None
        loss_scale = 1.0 / float(max(1, grad_accum_steps))

        local_last_stage = next((m for m in stage_modules if m.is_last), None)

        def loss_fn(out: Any, target: Any = None) -> torch.Tensor:
            if local_last_stage is None:
                raise RuntimeError("loss_fn called on rank without last stage")
            if not torch.is_tensor(out):
                raise RuntimeError("loss_fn expected tensor hidden states")
            if target is None or not torch.is_tensor(target):
                raise RuntimeError("loss_fn expected target tensor")
            return local_last_stage.compute_loss(out, target) * float(loss_scale)

        if vpp > 1:
            if ScheduleInterleaved1F1B is None:
                raise RuntimeError("interleaved schedule unavailable; set vpp=1")
            sched = ScheduleInterleaved1F1B(
                stages,
                microbatches,
                loss_fn=loss_fn,
                args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
            )
        else:
            if schedule_name == "gpipe":
                if ScheduleGPipe is None:
                    raise RuntimeError("gpipe unavailable")
                use_safe_gpipe = bool(serialize_pp_p2p and pp_p2p_mode != "subgroup_group_peer")
                gpipe_cls = SafeScheduleGPipe if use_safe_gpipe else ScheduleGPipe
                if rank == 0 and use_safe_gpipe:
                    print("[warn] using SafeScheduleGPipe with per-microbatch P2P waits", flush=True)
                if rank == 0 and serialize_pp_p2p and pp_p2p_mode == "subgroup_group_peer":
                    print(
                        "[warn] disabling SafeScheduleGPipe for subgroup_group_peer mode; using upstream ScheduleGPipe",
                        flush=True,
                    )
                sched = gpipe_cls(
                    stages[0],
                    microbatches,
                    loss_fn=loss_fn,
                    args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                    kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
                )
            else:
                if Schedule1F1B is None:
                    raise RuntimeError("1f1b unavailable")
                sched = Schedule1F1B(
                    stages[0],
                    microbatches,
                    loss_fn=loss_fn,
                    args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                    kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
                )

        params: List[torch.nn.Parameter] = []
        seen: set[int] = set()
        for m in stage_modules:
            for p in m.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=wd)

        profile_enabled = bool(profile_cfg.get("enabled", False))
        profile_rank = int(profile_cfg.get("rank", -1))
        profile_dir = str(profile_cfg.get("trace_dir", "tb_traces"))
        profile_wait = int(profile_cfg.get("wait", 1))
        profile_warmup = int(profile_cfg.get("warmup", 1))
        profile_active = int(profile_cfg.get("active", 3))
        profile_repeat = int(profile_cfg.get("repeat", 1))
        debug_train_logs = bool((cfg.get("runtime") or {}).get("debug_train_logs", True))
        warmup_pp_group_collective = bool((cfg.get("runtime") or {}).get("warmup_pp_group_collective", True))

        eff_global_bsz = int(dp_degree) * int(per_dp_batch)
        flops_per_param_per_token = float(mfu_cfg.get("flops_per_param_per_token", 6.0))
        peak_tflops_total = float(mfu_cfg.get("peak_tflops_total", 0.0) or 0.0)

        def _train_loop_step(step: int) -> Tuple[float, List[float]]:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            optim.zero_grad(set_to_none=True)
            mb_losses: List[float] = []
            for ga in range(max(1, grad_accum_steps)):
                batch_seed = seed + step * 1000 + ga + dp_idx * 9973
                ids = None
                target = None
                pp_local_rank = dist.get_rank(pp_group)
                if pp_local_rank == 0:
                    ids_cpu, _ = _make_synth_batch(vocab_size, seq_len, per_dp_batch, seed=batch_seed)
                    ids = ids_cpu.to(device, non_blocking=True)
                if pp_local_rank == (pp_degree - 1):
                    _, target_cpu = _make_synth_batch(vocab_size, seq_len, per_dp_batch, seed=batch_seed)
                    target = target_cpu.to(device, non_blocking=True)
                if pp_local_rank == 0:
                    if debug_train_logs and step == 0 and ga == 0:
                        print(f"[train-debug][rank {rank}] before sched.step first-stage ids={tuple(ids.shape)}", flush=True)
                    sched.step(ids, losses=mb_losses, return_outputs=False)
                    if debug_train_logs and step == 0 and ga == 0:
                        print(f"[train-debug][rank {rank}] after sched.step first-stage", flush=True)
                elif pp_local_rank == (pp_degree - 1):
                    if debug_train_logs and step == 0 and ga == 0:
                        print(
                            f"[train-debug][rank {rank}] before sched.step last-stage target={tuple(target.shape)}",
                            flush=True,
                        )
                    sched.step(target=target, losses=mb_losses, return_outputs=False)
                    if debug_train_logs and step == 0 and ga == 0:
                        print(f"[train-debug][rank {rank}] after sched.step last-stage", flush=True)
                else:
                    if debug_train_logs and step == 0 and ga == 0:
                        print(f"[train-debug][rank {rank}] before sched.step middle-stage", flush=True)
                    sched.step(losses=mb_losses, return_outputs=False)
                    if debug_train_logs and step == 0 and ga == 0:
                        print(f"[train-debug][rank {rank}] after sched.step middle-stage", flush=True)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optim.step()
            end_ev.record()
            end_ev.synchronize()
            return float(start_ev.elapsed_time(end_ev)), mb_losses

        if debug_init_logs:
            print(f"[init][rank {rank}] entering pre-train barrier", flush=True)
        dist.barrier()
        if debug_init_logs:
            print(f"[init][rank {rank}] passed pre-train barrier", flush=True)
        if warmup_pp_group_collective:
            if debug_init_logs:
                print(f"[init][rank {rank}] warming up pp_group collective", flush=True)
            pp_warm = torch.zeros(1, device=device, dtype=torch.float32)
            dist.all_reduce(pp_warm, group=pp_group)
            torch.cuda.synchronize(device)
            if debug_init_logs:
                print(f"[init][rank {rank}] warmed up pp_group collective", flush=True)
        if is_log_rank:
            print(f"[stats] model_params={model_params:,}", flush=True)

        prof_ctx = None
        prof = None
        do_profile = bool(profile_enabled and torch_profiler is not None and (profile_rank < 0 or rank == profile_rank))
        if do_profile:
            os.makedirs(profile_dir, exist_ok=True)
            activities = [torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA]
            sched_prof = torch_profiler.schedule(wait=profile_wait, warmup=profile_warmup, active=profile_active, repeat=profile_repeat)
            handler = torch_profiler.tensorboard_trace_handler(profile_dir, worker_name=f"rank{rank}")
            prof_ctx = torch_profiler.profile(
                activities=activities,
                schedule=sched_prof,
                on_trace_ready=handler,
                record_shapes=bool(profile_cfg.get("record_shapes", True)),
                profile_memory=bool(profile_cfg.get("profile_memory", True)),
                with_stack=bool(profile_cfg.get("with_stack", False)),
            )
            prof = prof_ctx.__enter__()
            if rank == profile_rank or profile_rank < 0:
                print(f"[profile] enabled trace_dir={profile_dir} rank={rank}", flush=True)

        try:
            for step in range(warmup_steps + steps):
                step_ms, mb_losses = _train_loop_step(step)
                if prof is not None:
                    prof.step()
                if step >= warmup_steps and (step - warmup_steps) % max(1, log_every) == 0 and is_log_rank:
                    loss_val = float(sum(mb_losses) / max(1, len(mb_losses))) if mb_losses else float("nan")
                    tokens_global = eff_global_bsz * int(seq_len) * int(max(1, grad_accum_steps))
                    tps = float(tokens_global) / (step_ms / 1e3) if step_ms > 0 else 0.0
                    extra = ""
                    if peak_tflops_total > 0:
                        train_tflops = (tps * flops_per_param_per_token * float(model_params)) / 1e12
                        mfu = train_tflops / float(peak_tflops_total)
                        extra = f" tflops={train_tflops:,.1f} mfu={mfu*100:.1f}%"
                    mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
                    print(
                        f"[train] step={step-warmup_steps} loss={loss_val:.4f} step_ms={step_ms:.1f} tokens/s={tps:,.0f} mem_gb={mem_gb:.1f}{extra}",
                        flush=True,
                    )
        finally:
            if prof_ctx is not None:
                prof_ctx.__exit__(None, None, None)

        dist.barrier()
        if rank == 0:
            print("[done]", flush=True)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
