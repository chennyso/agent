from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import train_manual_pp as manual_pp
from train_manual_pp import (
    DenseCausalLMStage,
    VocabParallelEmbedding,
    VocabParallelLMHead,
    _StageBackbone,
    _extract_transformer_layers,
    _infer_dtype,
    _load_cfg,
    _make_synth_batch,
    _materialize_and_load_stage,
    _seed_all,
    _setup_dist,
)


def _tensor_meta(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "None"
    return f"shape={tuple(int(x) for x in tensor.shape)} dtype={tensor.dtype} device={tensor.device}"


def _log(rank: int, stage_idx: int, step: int, microbatch_idx: int, message: str) -> None:
    print(
        f"[handpp][rank {rank}][stage {stage_idx}][step {step}][mb {microbatch_idx}] {message}",
        flush=True,
    )


def _split_even_microbatches(tensor: torch.Tensor, microbatches: int) -> List[torch.Tensor]:
    batch = int(tensor.shape[0])
    if batch % int(microbatches) != 0:
        raise ValueError(
            f"per-dp batch {batch} must be divisible by microbatches {microbatches} for hand-rolled PP debug"
        )
    mb = batch // int(microbatches)
    return [tensor[i * mb : (i + 1) * mb].contiguous() for i in range(int(microbatches))]


def _msg_tag(kind: int, step: int, microbatch_idx: int) -> int:
    return int(kind) * 1_000_000 + int(step) * 1_000 + int(microbatch_idx)


def _recv_activation(
    *,
    rank: int,
    stage_idx: int,
    step: int,
    microbatch_idx: int,
    src_rank: int,
    group: dist.ProcessGroup,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.empty(shape, device=device, dtype=dtype)
    tag = _msg_tag(10, step, microbatch_idx)
    _log(rank, stage_idx, step, microbatch_idx, f"recv_act start src={src_rank} tag={tag} {_tensor_meta(tensor)}")
    dist.recv(tensor, src=src_rank, group=group, tag=tag)
    _log(rank, stage_idx, step, microbatch_idx, f"recv_act done src={src_rank} tag={tag} {_tensor_meta(tensor)}")
    return tensor


def _send_activation(
    *,
    rank: int,
    stage_idx: int,
    step: int,
    microbatch_idx: int,
    dst_rank: int,
    group: dist.ProcessGroup,
    tensor: torch.Tensor,
) -> None:
    tag = _msg_tag(10, step, microbatch_idx)
    send_tensor = tensor.detach().contiguous()
    _log(rank, stage_idx, step, microbatch_idx, f"send_act start dst={dst_rank} tag={tag} {_tensor_meta(send_tensor)}")
    dist.send(send_tensor, dst=dst_rank, group=group, tag=tag)
    _log(rank, stage_idx, step, microbatch_idx, f"send_act done dst={dst_rank} tag={tag}")


def _recv_grad(
    *,
    rank: int,
    stage_idx: int,
    step: int,
    microbatch_idx: int,
    src_rank: int,
    group: dist.ProcessGroup,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.empty(shape, device=device, dtype=dtype)
    tag = _msg_tag(20, step, microbatch_idx)
    _log(rank, stage_idx, step, microbatch_idx, f"recv_grad start src={src_rank} tag={tag} {_tensor_meta(tensor)}")
    dist.recv(tensor, src=src_rank, group=group, tag=tag)
    _log(rank, stage_idx, step, microbatch_idx, f"recv_grad done src={src_rank} tag={tag} {_tensor_meta(tensor)}")
    return tensor


def _send_grad(
    *,
    rank: int,
    stage_idx: int,
    step: int,
    microbatch_idx: int,
    dst_rank: int,
    group: dist.ProcessGroup,
    tensor: torch.Tensor,
) -> None:
    tag = _msg_tag(20, step, microbatch_idx)
    send_tensor = tensor.detach().contiguous()
    _log(rank, stage_idx, step, microbatch_idx, f"send_grad start dst={dst_rank} tag={tag} {_tensor_meta(send_tensor)}")
    dist.send(send_tensor, dst=dst_rank, group=group, tag=tag)
    _log(rank, stage_idx, step, microbatch_idx, f"send_grad done dst={dst_rank} tag={tag}")


def _build_line_ranks(world_size: int, pp_degree: int, ranks_per_stage: int, dp_idx: int) -> List[int]:
    return [pp_idx * int(ranks_per_stage) + int(dp_idx) for pp_idx in range(int(pp_degree))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    runtime_cfg = cfg.get("runtime") or {}
    cuda_alloc_conf = str(runtime_cfg.get("cuda_alloc_conf") or "").strip()
    if cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_alloc_conf)
    timeout_seconds = int(runtime_cfg.get("dist_timeout_seconds") or 3600)
    debug_init_logs = bool(runtime_cfg.get("debug_init_logs", True))
    debug_module_logs = bool(runtime_cfg.get("debug_module_logs", True))
    forward_debug_limit = int(runtime_cfg.get("debug_module_forward_limit", 8) or 8)
    loss_debug_limit = int(runtime_cfg.get("debug_module_loss_limit", 8) or 8)
    manual_pp._DEBUG_TP_LOSS = bool(runtime_cfg.get("debug_tp_loss_logs", True))

    rank, world_size, local_rank = _setup_dist(timeout_seconds=timeout_seconds)
    device = torch.device("cuda", local_rank)
    try:
        train_cfg = cfg.get("train") or {}
        par_cfg = cfg.get("parallel") or {}
        pp_cfg = par_cfg.get("pp") or {}
        tp_cfg = par_cfg.get("tp") or {}
        fsdp_cfg = par_cfg.get("fsdp2") or {}
        recompute_cfg = par_cfg.get("recompute") or {}
        model_cfg = cfg.get("model") or {}

        if bool(tp_cfg.get("enabled", False)) or int(tp_cfg.get("degree") or 1) != 1:
            raise ValueError("hand-rolled PP debug runner only supports TP=1")
        if bool(fsdp_cfg.get("enabled", False)):
            raise ValueError("hand-rolled PP debug runner only supports FSDP2 disabled")
        if str(recompute_cfg.get("policy") or "none").lower() not in {"none", "off", "false"}:
            raise ValueError("hand-rolled PP debug runner only supports recompute=none")
        if int(pp_cfg.get("vpp") or 1) != 1:
            raise ValueError("hand-rolled PP debug runner only supports vpp=1")

        pp_degree = int(pp_cfg.get("degree") or 1)
        if world_size % pp_degree != 0:
            raise ValueError("world_size must be divisible by pp.degree")
        ranks_per_stage = world_size // int(pp_degree)
        dp_degree = int(ranks_per_stage)
        pp_rank = rank // int(ranks_per_stage)
        dp_idx = rank % int(ranks_per_stage)
        line_ranks = _build_line_ranks(world_size, pp_degree, ranks_per_stage, dp_idx)
        line_groups: List[dist.ProcessGroup] = []
        line_group: Optional[dist.ProcessGroup] = None
        for line_dp_idx in range(dp_degree):
            ranks = _build_line_ranks(world_size, pp_degree, ranks_per_stage, line_dp_idx)
            group = dist.new_group(ranks=ranks)
            line_groups.append(group)
            if int(line_dp_idx) == int(dp_idx):
                line_group = group
        if line_group is None:
            raise RuntimeError("failed to create line_group")
        prev_rank = line_ranks[pp_rank - 1] if pp_rank > 0 else None
        next_rank = line_ranks[pp_rank + 1] if pp_rank < (pp_degree - 1) else None

        seed = int(train_cfg.get("seed") or 42)
        _seed_all(seed + rank)

        model_dir = os.path.expanduser(os.path.expandvars(str(model_cfg.get("path") or "")))
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"model.path must exist: {model_dir}")
        dtype = _infer_dtype(model_cfg.get("dtype") or "bf16")
        seq_len = int(model_cfg.get("seq_len") or 1024)
        steps = int(train_cfg.get("steps") or 10)
        global_batch_size = int(train_cfg.get("global_batch_size") or 8)
        grad_accum_steps = int(train_cfg.get("grad_accum_steps") or 1)
        microbatches = int(pp_cfg.get("microbatches") or 1)
        lr = float(train_cfg.get("lr") or 1e-4)
        wd = float(train_cfg.get("weight_decay") or 0.1)
        grad_clip = float(train_cfg.get("grad_clip") or 0.0)
        log_every = int(train_cfg.get("log_every") or 1)

        if debug_init_logs and rank == 0:
            print(
                f"[handpp-init] world={world_size} pp={pp_degree} dp={dp_degree} microbatches={microbatches}",
                flush=True,
            )

        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

        hf_cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        with torch.device("meta"):
            full_model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
        if getattr(full_model.config, "use_cache", None) is not None:
            full_model.config.use_cache = False
        if getattr(full_model.config, "return_dict", None) is not None:
            full_model.config.return_dict = False

        pp_stages = pp_cfg.get("stages")
        if not isinstance(pp_stages, list) or len(pp_stages) != pp_degree:
            raise ValueError("parallel.pp.stages must be a list with len==pp.degree")
        stage_start, stage_end = [int(x) for x in pp_stages[pp_rank]]
        is_first = pp_rank == 0
        is_last = pp_rank == (pp_degree - 1)

        layers_full = _extract_transformer_layers(full_model)
        hidden_size = int(getattr(hf_cfg, "hidden_size", getattr(hf_cfg, "n_embd", 0) or 0))
        vocab_size = int(getattr(hf_cfg, "vocab_size", 32000))

        if debug_init_logs:
            print(
                f"[handpp-init][rank {rank}] stage={pp_rank} line={line_ranks} layers=[{stage_start},{stage_end}] "
                f"first={is_first} last={is_last} prev={prev_rank} next={next_rank}",
                flush=True,
            )

        embed = VocabParallelEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tp_rank=0,
            tp_world=1,
            tp_group=None,
            dtype=dtype,
        ) if is_first else None
        norm = getattr(full_model.model, "norm", None) if is_last else None
        lm_head = VocabParallelLMHead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            tp_rank=0,
            tp_world=1,
            tp_group=None,
            dtype=dtype,
        ) if is_last else None
        backbone = _StageBackbone(
            embed_tokens=embed,
            layers=torch.nn.ModuleList([layers_full[i] for i in range(stage_start, stage_end + 1)]),
            norm=norm,
            rotary_emb=getattr(full_model.model, "rotary_emb", None),
        )
        stage_mod = DenseCausalLMStage(
            backbone=backbone,
            lm_head=lm_head,
            global_layer_start=stage_start,
            is_first=is_first,
            is_last=is_last,
            seq_len=seq_len,
            recompute="none",
            debug_module_logs=debug_module_logs,
            debug_rank=rank,
            stage_id=pp_rank,
            forward_debug_limit=forward_debug_limit,
            loss_debug_limit=loss_debug_limit,
        )
        _materialize_and_load_stage(stage_mod, model_dir=model_dir, dtype=dtype)
        stage_mod.to(device)
        params = [p for p in stage_mod.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=wd)

        per_dp_batch = int(math.ceil(global_batch_size / float(dp_degree)))
        if per_dp_batch % microbatches != 0:
            raise ValueError(
                f"ceil(global_batch_size / dp_degree)={per_dp_batch} must be divisible by microbatches={microbatches}"
            )
        mb_size = per_dp_batch // int(microbatches)
        hidden_shape = (mb_size, seq_len, hidden_size)
        loss_scale = 1.0 / float(max(1, grad_accum_steps) * max(1, microbatches))
        is_log_rank = bool(is_last and dp_idx == 0)

        dist.barrier()
        if debug_init_logs:
            print(f"[handpp-init][rank {rank}] passed pre-train barrier", flush=True)
        line_warm = torch.zeros(1, device=device, dtype=torch.float32)
        dist.all_reduce(line_warm, group=line_group)
        torch.cuda.synchronize(device)
        if debug_init_logs:
            print(f"[handpp-init][rank {rank}] warmed line_group={line_ranks}", flush=True)

        for step in range(steps):
            step_start = time.time()
            optim.zero_grad(set_to_none=True)
            step_losses: List[float] = []

            for ga in range(max(1, grad_accum_steps)):
                batch_seed = seed + step * 1000 + ga + dp_idx * 9973
                ids_cpu, labels_cpu = _make_synth_batch(vocab_size, seq_len, per_dp_batch, seed=batch_seed)
                ids_mbs = _split_even_microbatches(ids_cpu, microbatches)
                labels_mbs = _split_even_microbatches(labels_cpu, microbatches)

                cached_inputs: List[Optional[torch.Tensor]] = [None for _ in range(microbatches)]
                cached_outputs: List[torch.Tensor] = []
                cached_losses: List[Optional[torch.Tensor]] = [None for _ in range(microbatches)]

                for microbatch_idx in range(microbatches):
                    if is_first:
                        input_ids = ids_mbs[microbatch_idx].to(device, non_blocking=True)
                        _log(rank, pp_rank, step, microbatch_idx, f"forward start first-stage ids={_tensor_meta(input_ids)}")
                        stage_output = stage_mod(input_ids)
                    else:
                        recv_hidden = _recv_activation(
                            rank=rank,
                            stage_idx=pp_rank,
                            step=step,
                            microbatch_idx=microbatch_idx,
                            src_rank=int(prev_rank),
                            group=line_group,
                            shape=hidden_shape,
                            dtype=dtype,
                            device=device,
                        )
                        recv_hidden.requires_grad_(True)
                        cached_inputs[microbatch_idx] = recv_hidden
                        _log(rank, pp_rank, step, microbatch_idx, f"forward start recv_hidden={_tensor_meta(recv_hidden)}")
                        stage_output = stage_mod(recv_hidden)

                    cached_outputs.append(stage_output)
                    _log(rank, pp_rank, step, microbatch_idx, f"forward done out={_tensor_meta(stage_output)}")

                    if is_last:
                        target = labels_mbs[microbatch_idx].to(device, non_blocking=True)
                        _log(rank, pp_rank, step, microbatch_idx, f"loss start target={_tensor_meta(target)}")
                        loss = stage_mod.compute_loss(stage_output, target) * float(loss_scale)
                        cached_losses[microbatch_idx] = loss
                        step_losses.append(float(loss.detach().item()))
                        _log(rank, pp_rank, step, microbatch_idx, f"loss done value={float(loss.detach().item()):.6f}")
                    else:
                        _send_activation(
                            rank=rank,
                            stage_idx=pp_rank,
                            step=step,
                            microbatch_idx=microbatch_idx,
                            dst_rank=int(next_rank),
                            group=line_group,
                            tensor=stage_output,
                        )

                for microbatch_idx in reversed(range(microbatches)):
                    if is_last:
                        loss = cached_losses[microbatch_idx]
                        if loss is None:
                            raise RuntimeError("last stage missing cached loss")
                        _log(rank, pp_rank, step, microbatch_idx, "backward start loss.backward()")
                        loss.backward()
                        _log(rank, pp_rank, step, microbatch_idx, "backward done loss.backward()")
                        grad_to_prev = cached_inputs[microbatch_idx].grad if cached_inputs[microbatch_idx] is not None else None
                    else:
                        grad_out = _recv_grad(
                            rank=rank,
                            stage_idx=pp_rank,
                            step=step,
                            microbatch_idx=microbatch_idx,
                            src_rank=int(next_rank),
                            group=line_group,
                            shape=tuple(int(x) for x in cached_outputs[microbatch_idx].shape),
                            dtype=cached_outputs[microbatch_idx].dtype,
                            device=device,
                        )
                        _log(rank, pp_rank, step, microbatch_idx, f"backward start grad_out={_tensor_meta(grad_out)}")
                        torch.autograd.backward(cached_outputs[microbatch_idx], grad_tensors=grad_out)
                        _log(rank, pp_rank, step, microbatch_idx, "backward done local autograd")
                        grad_to_prev = cached_inputs[microbatch_idx].grad if cached_inputs[microbatch_idx] is not None else None

                    if not is_first:
                        if grad_to_prev is None:
                            raise RuntimeError(f"stage {pp_rank} microbatch {microbatch_idx} missing grad_to_prev")
                        _send_grad(
                            rank=rank,
                            stage_idx=pp_rank,
                            step=step,
                            microbatch_idx=microbatch_idx,
                            dst_rank=int(prev_rank),
                            group=line_group,
                            tensor=grad_to_prev,
                        )

                del cached_inputs
                del cached_outputs
                del cached_losses

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optim.step()
            torch.cuda.synchronize(device)

            step_time_s = time.time() - step_start
            if is_log_rank and step % max(1, log_every) == 0:
                loss_val = float(sum(step_losses) / max(1, len(step_losses))) if step_losses else float("nan")
                tokens = int(per_dp_batch) * int(seq_len) * int(dp_degree) * int(max(1, grad_accum_steps))
                tokens_per_s = float(tokens) / max(step_time_s, 1e-6)
                print(
                    f"[train] step={step} loss={loss_val:.6f} step_s={step_time_s:.3f} tokens/s={tokens_per_s:.1f}",
                    flush=True,
                )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
