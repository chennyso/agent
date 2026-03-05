# Hybrid parallel (PP + VPP + TP + FSDP2) example

This folder provides a **runnable baseline** for 2 nodes × 8 GPUs (16 total) training a HF `AutoModelForCausalLM`
with:

- **PP**: `pp.degree=2` (stage0 on ranks 0–7, stage1 on ranks 8–15, assuming default `torchrun` rank ordering)
- **VPP**: `pp.vpp>1` uses `ScheduleInterleaved1F1B`
- **TP**: `torch.distributed.tensor.parallel.parallelize_module` (Qwen/LLaMA-like linear suffix matching)
- **FSDP2**: `fully_shard` over the **DP** mesh inside each pipeline stage (`(pp, dp, tp)` mesh; shard over `dp`)

## Quick start

1) Edit `examples/hybrid_parallel/config_qwen3_2node_pp2_tp2_vpp2.json`:

- Set `model.path` to your local Qwen3-32B directory.
- For heterogenous VRAM, you can either:
  - Set `parallel.pp.stages` manually to balance nodes (give the 24GB node fewer layers), or
  - Set `parallel.pp.stages="auto"` with `parallel.pp.auto_mem_gb=[24,32]` and optionally tweak `auto_bias_stage0`.

2) Launch:

Node0:

```bash
export MASTER_ADDR=<node0_ip_or_hostname>
export MASTER_PORT=29500
bash examples/hybrid_parallel/launch_node0.sh examples/hybrid_parallel/config_qwen3_2node_pp2_tp2_vpp2.json
```

Node1:

```bash
export MASTER_ADDR=<node0_ip_or_hostname>
export MASTER_PORT=29500
bash examples/hybrid_parallel/launch_node1.sh examples/hybrid_parallel/config_qwen3_2node_pp2_tp2_vpp2.json
```

## Notes

- `pp.microbatches` must be `<= ceil(global_batch_size / dp_degree)` where `dp_degree = world_size / (pp_degree * tp_degree)`.
- This example uses **synthetic token data** (random IDs). Replace `_make_synth_batch()` with your real dataloader.
- Per-stage different TP degrees is **not** supported in this example (PyTorch mesh is uniform); adapt heterogeneity via `pp.stages` (layer split) + `pp.microbatches` (bubble) first.
- PP/VPP and FSDP2 do not conflict as long as FSDP2 sharding is only over the DP group (same `pp` and `tp` coords) and you apply FSDP2 to each stage submodule (not across stages).

## Metrics (tokens/s, MFU)

- The script prints `tokens/s` and an **MFU estimate** on the logging rank.
- Configure MFU peak math in `runtime.peak_tflops_per_gpu`:
  - `[]` => MFU prints as n/a
  - `[node0_tflops, node1_tflops]` => per-node GPU peak TFLOPs (assumes 2 nodes with equal GPU count)
  - `[... world_size entries ...]` => per-rank peak TFLOPs
- FLOPs model: `train_flops ~= flops_per_param_per_token * model_params * tokens_global`, default `6.0` (typical dense decoder training roughness).

## Profiler (TensorBoard)

- Enable `profile.enabled=true` in config to dump Torch Profiler traces under `profile.trace_dir` (default `tb_traces/`).
- View traces:
  - `bash examples/hybrid_parallel/tools/tensorboard.sh tb_traces 6006`

## Plugin / env check

- Run: `python examples/hybrid_parallel/tools/check_env.py`

## Throughput presets (2 nodes × 8 GPUs, 24GB + 32GB)

- Safer memory first: `examples/hybrid_parallel/config_qwen3_2node_oom_safe_tp2_dp4.json`
- Higher throughput (if it fits): `examples/hybrid_parallel/config_qwen3_2node_fast_tp4_dp2.json`

Both use:
- `pp.degree=2` + `pp.vpp=2` + `pp.stages="auto"` (bias stage0 lighter for 24GB)
- `train.grad_accum_steps` to increase effective batch without increasing activation peak per microbatch
