from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

try:  # pragma: no cover
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
except Exception:  # pragma: no cover
    ColwiseParallel = None  # type: ignore[assignment]
    RowwiseParallel = None  # type: ignore[assignment]
    parallelize_module = None  # type: ignore[assignment]

try:  # pragma: no cover
    from torch.distributed.pipelining import (
        PipelineStage,
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
        SplitPoint,
        pipeline,
    )
    from torch.distributed.pipelining.microbatch import TensorChunkSpec
except Exception:  # pragma: no cover
    PipelineStage = None
    Schedule1F1B = None
    ScheduleGPipe = None
    ScheduleInterleaved1F1B = None
    SplitPoint = None
    pipeline = None
    TensorChunkSpec = None


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


def _setup_dist() -> Tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )
    return rank, world_size, local_rank


def _get_transformer_layers(hf_model: nn.Module) -> nn.ModuleList:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    raise RuntimeError("Cannot find transformer layers (model.model.layers / transformer.h)")


def _extract_layer_paths(model: nn.Module) -> Dict[int, str]:
    layers = _get_transformer_layers(model)
    name_by_id = {id(m): n for n, m in model.named_modules()}
    paths: Dict[int, str] = {}
    for idx, layer in enumerate(layers):
        name = name_by_id.get(id(layer))
        if name:
            paths[idx] = name
    return paths


def _layer_param_bytes(model: nn.Module) -> List[int]:
    layers = _get_transformer_layers(model)
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
    bias_stage0: float = -0.05,
) -> List[List[int]]:
    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    if int(pp_degree) != len(mem_gb):
        raise ValueError("parallel.pp.auto_mem_gb length must equal parallel.pp.degree")
    if n_layers < int(pp_degree):
        raise ValueError("pp_degree exceeds available transformer layers")
    weights = _layer_param_bytes(model)
    total_w = float(sum(weights)) if weights else 0.0
    if total_w <= 0:
        raise RuntimeError("failed to estimate per-layer parameter bytes for auto PP splitting")

    mem = [float(x) for x in mem_gb]
    if any(x <= 0 for x in mem):
        raise ValueError("parallel.pp.auto_mem_gb entries must be > 0")
    frac = [x / sum(mem) for x in mem]
    if frac and bias_stage0:
        frac[0] = min(0.9, max(0.1, float(frac[0]) + float(bias_stage0)))
        rest = 1.0 - frac[0]
        if len(frac) > 1:
            tail_sum = sum(frac[1:]) or 1.0
            for i in range(1, len(frac)):
                frac[i] = rest * (frac[i] / tail_sum)

    # Greedy contiguous partition by cumulative weight.
    targets = [f * total_w for f in frac]
    stages: List[List[int]] = []
    start = 0
    cum = 0.0
    stage_idx = 0
    for i, w in enumerate(weights):
        remaining_layers = n_layers - i
        remaining_stages = int(pp_degree) - stage_idx
        # Ensure at least 1 layer per remaining stage.
        if remaining_layers == remaining_stages:
            if i >= start:
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
    if start <= n_layers - 1:
        stages.append([start, n_layers - 1])
    if len(stages) != int(pp_degree):
        raise RuntimeError(f"auto PP split produced {len(stages)} stages, expected {pp_degree}")
    return stages


def _compute_virtual_boundaries(
    *,
    total_layers: int,
    pp_degree: int,
    vpp: int,
    pp_stages: Optional[List[List[int]]],
) -> List[int]:
    num_virtual = int(pp_degree) * int(vpp)
    if num_virtual <= 1:
        return []
    boundaries: List[int] = []
    if pp_stages:
        if len(pp_stages) != int(pp_degree):
            raise ValueError("pp.stages must have length == pp.degree")
        for stage in pp_stages:
            if len(stage) != 2:
                raise ValueError("each pp.stages entry must be [start_layer, end_layer]")
        for stage_start, stage_end in pp_stages:
            stage_start = int(stage_start)
            stage_end = int(stage_end)
            if stage_start < 0 or stage_end < stage_start or stage_end >= total_layers:
                raise ValueError(f"invalid pp stage range [{stage_start}, {stage_end}] for total_layers={total_layers}")
            n = stage_end - stage_start + 1
            for k in range(1, int(vpp)):
                cut = stage_start + int(round(k * n / float(vpp)))
                cut = max(stage_start + 1, min(cut, stage_end))
                boundaries.append(cut)
        for i in range(1, int(pp_degree)):
            boundaries.append(int(pp_stages[i][0]))
        boundaries = sorted(set(boundaries))
        if len(boundaries) != num_virtual - 1:
            raise ValueError(
                f"computed {len(boundaries)} virtual boundaries but expected {num_virtual - 1}; "
                f"adjust pp.stages/vpp or set vpp=1"
            )
        return boundaries
    step = total_layers / float(num_virtual)
    for i in range(1, num_virtual):
        boundaries.append(int(round(i * step)))
    boundaries = sorted(set(max(1, min(int(x), total_layers - 1)) for x in boundaries))
    if len(boundaries) != num_virtual - 1:
        raise ValueError("failed to compute distinct virtual boundaries; set pp.stages explicitly")
    return boundaries


def _build_split_spec(layer_paths: Dict[int, str], boundaries: List[int]) -> Dict[str, Any]:
    if SplitPoint is None:
        raise RuntimeError("torch.distributed.pipelining is unavailable")
    split_spec: Dict[str, Any] = {}
    for start_idx in boundaries:
        key = layer_paths.get(int(start_idx))
        if not key:
            raise RuntimeError(f"missing module path for layer {start_idx}; cannot build split_spec")
        split_spec[key] = SplitPoint.BEGINNING
    return split_spec


def _apply_tp_qwen_like(model: nn.Module, tp_mesh, tp_degree: int) -> Dict[str, Any]:
    if int(tp_degree) <= 1:
        return {"tp_applied": False, "reason": "tp_degree<=1"}
    if parallelize_module is None or ColwiseParallel is None or RowwiseParallel is None:
        return {"tp_applied": False, "reason": "tensor_parallel_unavailable"}
    mapping: Dict[str, Any] = {}
    col = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    row = ("o_proj", "down_proj")
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name.endswith(s) for s in col):
            if getattr(mod, "out_features", 0) % int(tp_degree) != 0:
                continue
            mapping[name] = ColwiseParallel(use_local_output=True)
        elif any(name.endswith(s) for s in row):
            mapping[name] = RowwiseParallel()
    if not mapping:
        return {"tp_applied": False, "reason": "no_modules_matched"}
    parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=mapping)
    return {"tp_applied": True, "tp_mapping_size": len(mapping)}


def _apply_fsdp2_bottom_up(
    model: nn.Module,
    dp_mesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward: bool,
) -> None:
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    layers = _get_transformer_layers(model)
    for layer in layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=bool(reshard_after_forward))
    fully_shard(model, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=bool(reshard_after_forward))


def _apply_fsdp2_stage(
    stage_submod: nn.Module,
    dp_mesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward: bool,
) -> Dict[str, Any]:
    try:
        _apply_fsdp2_bottom_up(
            stage_submod,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward=reshard_after_forward,
        )
        return {"fsdp2_applied": True, "mode": "bottom_up_layers"}
    except Exception:
        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        fully_shard(stage_submod, mesh=dp_mesh, mp_policy=mp, reshard_after_forward=bool(reshard_after_forward))
        return {"fsdp2_applied": True, "mode": "root_only_fallback"}


def _loss_fn(output: Any, target: Optional[torch.Tensor]) -> torch.Tensor:
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
        raise ValueError("loss_fn requires target labels")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _make_synth_batch(vocab_size: int, seq_len: int, batch_size: int, *, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    input_ids = torch.randint(low=0, high=int(vocab_size), size=(batch_size, seq_len), generator=g, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML config requested but PyYAML is not installed; use JSON or install pyyaml") from exc
        return yaml.safe_load(raw)
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cuda_alloc_conf = ((cfg.get("runtime") or {}).get("cuda_alloc_conf") or "").strip()
    if cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf)

    rank, world_size, local_rank = _setup_dist()
    seed = int(((cfg.get("train") or {}).get("seed") or 42))
    _seed_all(seed + rank)

    model_cfg = cfg.get("model") or {}
    model_path = str(model_cfg.get("path"))
    dtype = _infer_dtype(model_cfg.get("dtype") or "bf16")
    seq_len = int(model_cfg.get("seq_len") or 1024)

    train_cfg = cfg.get("train") or {}
    steps = int(train_cfg.get("steps") or 10)
    warmup_steps = int(train_cfg.get("warmup_steps") or 0)
    global_batch_size = int(train_cfg.get("global_batch_size") or 8)
    lr = float(train_cfg.get("lr") or 1e-4)
    wd = float(train_cfg.get("weight_decay") or 0.1)
    grad_clip = float(train_cfg.get("grad_clip") or 1.0)
    log_every = int(train_cfg.get("log_every") or 1)
    grad_accum_steps = int(train_cfg.get("grad_accum_steps") or 1)
    if grad_accum_steps < 1:
        grad_accum_steps = 1
    mfu_cfg = train_cfg.get("mfu") or {}
    mfu_model_params_override = mfu_cfg.get("model_params")
    mfu_flops_per_param_token = float(mfu_cfg.get("flops_per_param_per_token") or 6.0)

    par = cfg.get("parallel") or {}
    pp_cfg = par.get("pp") or {}
    tp_cfg = par.get("tp") or {}
    fsdp_cfg = par.get("fsdp2") or {}

    pp_degree = int(pp_cfg.get("degree") or 1)
    vpp = int(pp_cfg.get("vpp") or 1)
    pp_microbatches = int(pp_cfg.get("microbatches") or 1)
    schedule_name = str(pp_cfg.get("schedule") or "1f1b").lower()
    pp_stages = pp_cfg.get("stages")  # optional [[start,end], ...] len==pp_degree
    pp_auto_mem_gb = pp_cfg.get("auto_mem_gb")
    pp_auto_bias0 = float(pp_cfg.get("auto_bias_stage0", -0.05) or -0.05)

    tp_enabled = bool(tp_cfg.get("enabled", True))
    tp_degree = int(tp_cfg.get("degree") or 1)
    if not tp_enabled:
        tp_degree = 1

    fsdp_enabled = bool(fsdp_cfg.get("enabled", True))
    fsdp_param_dtype = _infer_dtype(fsdp_cfg.get("param_dtype") or "bf16")
    fsdp_reduce_dtype = _infer_dtype(fsdp_cfg.get("reduce_dtype") or "bf16")
    reshard_after_forward = bool(fsdp_cfg.get("reshard_after_forward", True))

    if pp_degree < 1 or tp_degree < 1:
        raise ValueError("pp.degree and tp.degree must be >= 1")
    denom = int(pp_degree) * int(tp_degree)
    if world_size % denom != 0:
        raise ValueError(f"world_size={world_size} must be divisible by pp*tp={denom}")
    dp_degree = world_size // denom
    if dp_degree < 1:
        raise ValueError("computed dp_degree < 1")

    # Derive coords (assumes torchrun default contiguous rank assignment).
    tp_idx = rank % tp_degree
    dp_idx = (rank // tp_degree) % dp_degree
    pp_idx = rank // (dp_degree * tp_degree)
    is_log_rank = bool(pp_idx == (pp_degree - 1) and dp_idx == 0 and tp_idx == 0)

    if pp_idx < 0 or pp_idx >= pp_degree:
        raise RuntimeError(f"invalid pp_idx={pp_idx}; check rank mapping assumptions")

    def _r(pp: int, dp: int, tp: int) -> int:
        return int(pp) * int(dp_degree) * int(tp_degree) + int(dp) * int(tp_degree) + int(tp)

    # Build explicit stage-local TP/DP meshes + per-(dp,tp) pipeline group.
    # This avoids accidental cross-stage sharding (FSDP2 must shard only within a stage's DP group).
    stage_base = int(pp_idx) * int(dp_degree) * int(tp_degree)
    tp_group_ranks = [stage_base + int(dp_idx) * int(tp_degree) + t for t in range(int(tp_degree))]
    dp_group_ranks = [stage_base + d * int(tp_degree) + int(tp_idx) for d in range(int(dp_degree))]
    pp_group_ranks = [_r(p, int(dp_idx), int(tp_idx)) for p in range(int(pp_degree))]

    tp_mesh = DeviceMesh("cuda", tp_group_ranks, mesh_dim_names=("tp",)) if tp_degree > 1 else None
    dp_mesh = DeviceMesh("cuda", dp_group_ranks, mesh_dim_names=("dp",)) if dp_degree > 1 else None
    pp_group = dist.new_group(ranks=pp_group_ranks)

    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as exc:
        raise RuntimeError("transformers is required (pip install transformers>=4.42)") from exc

    if rank == 0:
        print(
            f"[init] world_size={world_size} pp={pp_degree} dp={dp_degree} tp={tp_degree} vpp={vpp} "
            f"(pp_idx={pp_idx} dp_idx={dp_idx} tp_idx={tp_idx})",
            flush=True,
        )

    runtime_cfg = cfg.get("runtime") or {}
    peak_tflops_per_gpu = runtime_cfg.get("peak_tflops_per_gpu")
    if peak_tflops_per_gpu is None:
        peak_tflops_per_gpu = []
    if not isinstance(peak_tflops_per_gpu, list):
        raise ValueError("runtime.peak_tflops_per_gpu must be a list (per-rank or per-node spec)")
    peak_flops_total = 0.0
    if peak_tflops_per_gpu:
        # If 2 entries and world_size is multiple of 16, interpret as [node0_gpu, node1_gpu] TFLOPs.
        if len(peak_tflops_per_gpu) == 2 and world_size % 16 == 0:
            per_node = int(world_size // 2)
            peak_flops_total = (float(peak_tflops_per_gpu[0]) * per_node + float(peak_tflops_per_gpu[1]) * per_node) * 1e12
        # If exactly world_size entries, interpret per-rank TFLOPs.
        elif len(peak_tflops_per_gpu) == world_size:
            peak_flops_total = sum(float(x) for x in peak_tflops_per_gpu) * 1e12
        else:
            if rank == 0:
                print(
                    f"[mfu] runtime.peak_tflops_per_gpu has len={len(peak_tflops_per_gpu)}; "
                    "supported: 2 (per-node) or world_size (per-rank). MFU will be reported as n/a.",
                    flush=True,
                )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.train()

    # Total params for MFU (same for all ranks).
    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    if mfu_model_params_override is not None:
        try:
            total_params = int(mfu_model_params_override)
        except Exception:
            pass

    if isinstance(pp_stages, str) and pp_stages.strip().lower() == "auto":
        if not isinstance(pp_auto_mem_gb, list) or not pp_auto_mem_gb:
            raise ValueError("pp.stages='auto' requires parallel.pp.auto_mem_gb=[...]")
        pp_stages = _auto_pp_stages(
            model=model,
            pp_degree=pp_degree,
            mem_gb=[float(x) for x in pp_auto_mem_gb],
            bias_stage0=pp_auto_bias0,
        )
        if rank == 0:
            print(f"[pp] auto stages={pp_stages}", flush=True)

    layer_paths = _extract_layer_paths(model)
    total_layers = len(_get_transformer_layers(model))
    boundaries = _compute_virtual_boundaries(
        total_layers=total_layers,
        pp_degree=pp_degree,
        vpp=vpp,
        pp_stages=pp_stages,
    )
    split_spec = _build_split_spec(layer_paths, boundaries)
    num_virtual_stages = pp_degree * vpp

    if pipeline is None or PipelineStage is None:
        raise RuntimeError("torch.distributed.pipelining is unavailable; upgrade torch>=2.4 with CUDA build")

    per_rank_batch = int(math.ceil(global_batch_size / float(dp_degree)))
    if per_rank_batch < 1:
        raise ValueError("global_batch_size too small for dp_degree")
    if pp_microbatches < 1:
        pp_microbatches = 1
    if pp_microbatches > per_rank_batch:
        raise ValueError("pp.microbatches cannot exceed per-rank batch size (global_batch_size/dp)")

    device = torch.device("cuda", local_rank)
    dummy_input = torch.zeros((per_rank_batch, seq_len), device=device, dtype=torch.long)
    pipe = pipeline(model, mb_args=(dummy_input,), split_spec=split_spec)

    pp_rank = dist.get_rank(pp_group)

    local_stage_ids = [pp_rank + k * pp_degree for k in range(vpp)]
    local_stages = [pipe.build_stage(sid, device, group=pp_group) for sid in local_stage_ids]

    # Apply TP/FSDP2 *after* pipeline split & stage materialization:
    # - avoids split_spec path mismatch if wrappers rename modules
    # - makes PP+VPP and FSDP2 co-exist by sharding only over the DP mesh within each stage
    tp_reports: List[Dict[str, Any]] = []
    fsdp_reports: List[Dict[str, Any]] = []
    for st in local_stages:
        submod = getattr(st, "submod", st)
        if tp_degree > 1 and tp_mesh is not None:
            tp_reports.append(_apply_tp_qwen_like(submod, tp_mesh, tp_degree))
        if fsdp_enabled and dp_mesh is not None and dp_degree > 1:
            fsdp_reports.append(
                _apply_fsdp2_stage(
                    submod,
                    dp_mesh,
                    param_dtype=fsdp_param_dtype,
                    reduce_dtype=fsdp_reduce_dtype,
                    reshard_after_forward=reshard_after_forward,
                )
            )
    if rank == 0:
        print(f"[tp] reports={tp_reports}", flush=True)
        print(f"[fsdp2] reports={fsdp_reports}", flush=True)

    chunk_spec = None
    if pp_microbatches > 1:
        if TensorChunkSpec is None:
            raise RuntimeError("pp.microbatches>1 requires TensorChunkSpec; upgrade torch or set microbatches=1")
        chunk_spec = TensorChunkSpec(0)

    loss_scale = 1.0 / float(grad_accum_steps)

    def _loss_fn_scaled(output: Any, target: Optional[torch.Tensor]) -> torch.Tensor:
        return _loss_fn(output, target) * loss_scale

    schedule = None
    if vpp > 1:
        if ScheduleInterleaved1F1B is None:
            raise RuntimeError("ScheduleInterleaved1F1B unavailable; set vpp=1 or upgrade torch")
        schedule = ScheduleInterleaved1F1B(
            local_stages,
            pp_microbatches,
            loss_fn=_loss_fn_scaled,
            args_chunk_spec=(chunk_spec,) if chunk_spec else None,
            kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
        )
    else:
        stage0 = local_stages[0]
        if schedule_name in {"gpipe"}:
            if ScheduleGPipe is None:
                raise RuntimeError("ScheduleGPipe unavailable")
            schedule = ScheduleGPipe(
                stage0,
                pp_microbatches,
                loss_fn=_loss_fn_scaled,
                args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
            )
        else:
            if Schedule1F1B is None:
                raise RuntimeError("Schedule1F1B unavailable")
            schedule = Schedule1F1B(
                stage0,
                pp_microbatches,
                loss_fn=_loss_fn_scaled,
                args_chunk_spec=(chunk_spec,) if chunk_spec else None,
                kwargs_chunk_spec={"target": chunk_spec} if chunk_spec else None,
            )

    # Optimizer over all local virtual stages' params.
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for st in local_stages:
        submod = getattr(st, "submod", st)
        for p in submod.parameters():
            pid = id(p)
            if pid not in seen:
                params.append(p)
                seen.add(pid)
    optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=wd)

    vocab_size = int(model_cfg.get("vocab_size_override") or getattr(getattr(model, "config", None), "vocab_size", 32000))
    batch = _make_synth_batch(vocab_size, seq_len, per_rank_batch, seed=seed + rank * 997)

    last_pp_rank = int((num_virtual_stages - 1) % pp_degree)
    dist.barrier()
    t0 = time.time()
    step0 = time.time()
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)

    prof_cfg = cfg.get("profile") or {}
    prof_enabled = bool(prof_cfg.get("enabled", False))
    trace_dir = str(prof_cfg.get("trace_dir") or "tb_traces")
    prof_wait = int(prof_cfg.get("wait", 1))
    prof_warmup = int(prof_cfg.get("warmup", 1))
    prof_active = int(prof_cfg.get("active", 5))
    prof_repeat = int(prof_cfg.get("repeat", 1))
    prof_record_shapes = bool(prof_cfg.get("record_shapes", False))
    prof_profile_memory = bool(prof_cfg.get("profile_memory", True))
    prof_with_stack = bool(prof_cfg.get("with_stack", False))
    prof_with_flops = bool(prof_cfg.get("with_flops", True))

    prof_ctx = None
    if prof_enabled:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"hybrid_pp{pp_degree}_tp{tp_degree}_dp{dp_degree}_rank{rank}")
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=prof_wait, warmup=prof_warmup, active=prof_active, repeat=prof_repeat),
            on_trace_ready=tensorboard_trace_handler(trace_path),
            record_shapes=prof_record_shapes,
            profile_memory=prof_profile_memory,
            with_stack=prof_with_stack,
            with_flops=prof_with_flops,
        )
        if is_log_rank:
            print(f"[profile] enabled trace_dir={trace_dir}", flush=True)

    try:
        if prof_ctx is not None:
            prof_ctx.__enter__()

        for step in range(warmup_steps + steps):
            if prof_ctx is not None:
                prof_ctx.step()
            step_start.record()

            optim.zero_grad(set_to_none=True)
            mb_losses: List[float] = []
            for _ga in range(int(grad_accum_steps)):
                input_ids = batch["input_ids"] if pp_rank == 0 else None
                target = batch["labels"] if pp_rank == last_pp_rank else None
                attention_mask = batch["attention_mask"] if pp_rank == 0 else None
                if input_ids is not None:
                    input_ids = input_ids.cuda(non_blocking=True)
                if target is not None:
                    target = target.cuda(non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda(non_blocking=True)

                if attention_mask is not None:
                    schedule.step(input_ids, target=target, attention_mask=attention_mask, losses=mb_losses)
                else:
                    schedule.step(input_ids, target=target, losses=mb_losses)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optim.step()

            step_end.record()
            step_end.synchronize()
            step_ms = float(step_start.elapsed_time(step_end))

            if step >= warmup_steps and (step - warmup_steps) % max(1, log_every) == 0 and is_log_rank:
                loss_val = float(sum(mb_losses) / max(1, len(mb_losses))) if mb_losses else float("nan")

                # Tokens/s: use attention_mask sum if present, else seq_len. Multiply by global_batch_size and grad_accum.
                tokens_per_sample = int(seq_len)
                tokens_global = int(global_batch_size) * tokens_per_sample * int(grad_accum_steps)
                tps = float(tokens_global) / (step_ms / 1e3) if step_ms > 0 else 0.0

                # MFU: approximate train FLOPs = flops_per_param_per_token * params * tokens.
                mfu = None
                achieved_tflops = None
                if peak_flops_total > 0:
                    flops = float(mfu_flops_per_param_token) * float(total_params) * float(tokens_global)
                    achieved = flops / (step_ms / 1e3)
                    achieved_tflops = achieved / 1e12
                    mfu = achieved / peak_flops_total

                elapsed = time.time() - t0
                if mfu is None:
                    print(
                        f"[train] step={step - warmup_steps} loss={loss_val:.4f} "
                        f"t={elapsed:.1f}s step_ms={step_ms:.1f} tokens/s={tps:,.0f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[train] step={step - warmup_steps} loss={loss_val:.4f} "
                        f"t={elapsed:.1f}s step_ms={step_ms:.1f} tokens/s={tps:,.0f} "
                        f"TFLOPs={achieved_tflops:.1f} MFU={mfu*100:.1f}%",
                        flush=True,
                    )
    finally:
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)

    dist.barrier()
    if rank == 0:
        print("[done]", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
