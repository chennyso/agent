# Hybrid parallel examples (PP + VPP + TP + FSDP2)

This folder contains **2-node (16 GPU)** training baselines for HF `AutoModelForCausalLM` with:

- **PP**: pipeline parallel (`pp.degree=2`)
- **VPP**: virtual pipeline stages (`pp.vpp>1` via interleaved 1F1B if available)
- **TP**: tensor parallel inside each stage
- **FSDP2**: `fully_shard` over the **DP** group inside each stage

## Which script to use

- Recommended: `examples/hybrid_parallel/train_manual_pp.py`
  - Manual stage construction (no `torch.export`), more robust on HF models.
- Experimental: `examples/hybrid_parallel/train_hybrid_qwen.py`
  - Uses `torch.distributed.pipelining.pipeline()` to auto-split graphs; may fail if the model cannot be fully captured.

## Quick start (manual PP)

1) Edit `examples/hybrid_parallel/config_qwen3_2node_dense_manual_pp.json`

- Set `model.path` to your local Qwen3-32B directory.
- For heterogenous VRAM (24GB + 32GB), keep `parallel.pp.stages="auto"` and set `parallel.pp.auto_mem_gb=[24,32]`.
  - You can also set `parallel.pp.stages=[[0, X], [X+1, 63]]` manually to force a split.

2) Launch

Node0 (g4 / 4090D / 24GB, NIC example `ens8f0`):

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens8f0
export GLOO_SOCKET_IFNAME=ens8f0
bash examples/hybrid_parallel/launch_manual_node0.sh examples/hybrid_parallel/config_qwen3_2node_dense_manual_pp.json
```

Node1 (g5 / 5090D / 32GB, NIC example `ens9f0`):

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
bash examples/hybrid_parallel/launch_manual_node1.sh examples/hybrid_parallel/config_qwen3_2node_dense_manual_pp.json
```

## Notes

- `pp.microbatches` must be `<= ceil(global_batch_size / dp_degree)` where `dp_degree = world_size / (pp_degree * tp_degree)`.
- `train_manual_pp.py` uses **synthetic token data** (random IDs). Replace `_make_synth_batch()` with your real dataloader.
- Per-stage different `tp.degree` is not supported (uniform mesh). Use:
  - `pp.stages` (layer split) to fit heterogenous VRAM
  - `recompute.per_stage` and `fsdp2.reshard_after_forward_per_stage` to trade memory vs throughput

## Metrics / profiler

- `train_manual_pp.py` prints `tokens/s` and `mem_gb` on the logging rank.
- Optional MFU estimate: set `mfu.peak_tflops_total` in config (aggregate peak TFLOPs for the whole job).
- Optional Torch Profiler traces: set `profile.enabled=true` and view with TensorBoard (`tensorboard --logdir tb_traces`).
