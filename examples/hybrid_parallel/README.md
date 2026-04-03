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

1) Recommended safe config: `examples/hybrid_parallel/config_qwen3_2node_dense_pp4_gpipe_safe.json`

- Set `model.path` to your local Qwen3-32B directory.
- This preset uses `pp=4`, `tp=2`, `vpp=1`, `schedule=gpipe` and a conservative manual split:
  - `[0, 9]`, `[10, 21]`, `[22, 41]`, `[42, 63]`
- The default `pp4_gpipe_safe` preset keeps `FSDP2` disabled on purpose.
  - Current stable path is `PP + TP (+ recompute)`.
  - `FSDP2 + TP + manual PP` remains experimental and may deadlock on Qwen3 in this setup.
- For the closest match to PyTorch's official composability pattern, use:
  - `examples/hybrid_parallel/config_qwen3_2node_dense_pp4_fsdp2_officialish.json`
  - It enforces `reshard_after_forward=false` and `recompute=none`.
- Use `config_qwen3_2node_dense_manual_pp.json` only if you want the older `pp=2` auto-split path.

2) Launch

Node0 (g4 / 4090D / 24GB, NIC example `ens8f0`):

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens8f0
export GLOO_SOCKET_IFNAME=ens8f0
bash examples/hybrid_parallel/launch_manual_node0.sh examples/hybrid_parallel/config_qwen3_2node_dense_pp4_gpipe_safe.json
```

Node1 (g5 / 5090D / 32GB, NIC example `ens9f0`):

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
bash examples/hybrid_parallel/launch_manual_node1.sh examples/hybrid_parallel/config_qwen3_2node_dense_pp4_gpipe_safe.json
```

## Notes

- `pp.microbatches` must be `<= ceil(global_batch_size / dp_degree)` where `dp_degree = world_size / (pp_degree * tp_degree)`.
- `Schedule1F1B` requires `microbatches >= num_stages`; the `pp4_gpipe_safe` preset avoids that constraint by using `GPipe`.
- If a known preset JSON is missing on a remote node, `train_manual_pp.py` now falls back to a built-in copy of:
  - `config_qwen3_2node_dense_pp4_gpipe_safe.json`
  - `config_qwen3_2node_dense_pp4_safe.json`
  - `config_qwen3_2node_dense_manual_pp.json`
- `train_manual_pp.py` uses **synthetic token data** (random IDs). Replace `_make_synth_batch()` with your real dataloader.
- Per-stage different `tp.degree` is not supported (uniform mesh). Use:
  - `pp.stages` (layer split) to fit heterogenous VRAM
  - `recompute.per_stage` and `fsdp2.reshard_after_forward_per_stage` to trade memory vs throughput

## Planner mode

- New config: `examples/hybrid_parallel/config_qwen3_2node_dense_planner_search.json`
- New capability in `train_manual_pp.py`:
  - `planner.enabled=true` lets the script generate candidate `PP/TP/VPP/microbatch/schedule` combinations
  - the planner emits a custom `parallel.pp.mesh`, so `stage -> rank/node` is no longer hard-coded
  - stage cuts are solved with a heterogeneity-aware objective instead of plain equal split
- Dry-run only the planner without launching training:

```bash
python examples/hybrid_parallel/train_manual_pp.py \
  --config examples/hybrid_parallel/config_qwen3_2node_dense_planner_search.json \
  --plan_only
```

## Hybrid policy schema

- New schema file: `examples/hybrid_parallel/hybrid_policy.py`
- New demo policy: `examples/hybrid_parallel/hybrid_policy_qwen3_2node_hetero_demo.json`
- New demo config: `examples/hybrid_parallel/config_qwen3_2node_dense_hybrid_policy_demo.json`

What it adds:

- one place to describe `PP / VPP / TP / CP / EP / FSDP2 / recompute`
- `stage_policies` for per-stage memory/runtime decisions
- `module_policies` for tail/embed/expert special handling
- `phase_policies` as placeholders for warmup vs steady-state policy changes

Current execution support:

- `train_manual_pp.py` and `train_handrolled_pp_debug.py` now read `hybrid_policy`
- the policy is merged into the existing manual runner config before launch
- unsupported manual-runner features (for example per-stage TP, asymmetric VPP, CP/EP runtime) are kept as metadata and emitted as warnings instead of being silently dropped

Export helpers:

```bash
python examples/hybrid_parallel/export_hybrid_policy.py \
  --config examples/hybrid_parallel/config_qwen3_2node_dense_hybrid_policy_demo.json \
  --format both \
  --total_layers 64
```

- `--format manual` prints the merged manual-runner config
- `--format torchtitan` prints TorchTitan-style override keys
- `--format both` prints both views

## TorchTitan execution path

- New TorchTitan experiment module: `torchtitan.experiments.hybrid_policy`
- New TorchTitan launch scripts:
  - `examples/hybrid_parallel/launch_torchtitan_hybrid_node0.sh`
  - `examples/hybrid_parallel/launch_torchtitan_hybrid_node1.sh`
- New TorchTitan configs:
  - `hybrid_policy:qwen3_hybrid_demo`
  - `hybrid_policy:qwen3_14b_single_g4_fsdp8`
  - `hybrid_policy:qwen3_14b_single_g4_fsdp8_conditioned`
  - `hybrid_policy:qwen3_32b_g4_g5_pp_only`
  - `hybrid_policy:qwen3_32b_g4_g5_pp_tp`
  - `hybrid_policy:qwen3_32b_g4_g5_pp_tp_fsdp2`

Example:

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens8f0
export GLOO_SOCKET_IFNAME=ens8f0
bash examples/hybrid_parallel/launch_torchtitan_hybrid_node0.sh qwen3_hybrid_demo
```

```bash
export MASTER_ADDR=192.168.10.241
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
bash examples/hybrid_parallel/launch_torchtitan_hybrid_node1.sh qwen3_hybrid_demo
```

Single-node 14B validation on `g4`:

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens8f0
export GLOO_SOCKET_IFNAME=ens8f0
bash examples/hybrid_parallel/launch_torchtitan_hybrid_node0.sh qwen3_14b_single_g4_fsdp8_conditioned
```

Single-node 14B validation on `5090D` / `g5`:

- New helper launcher:
  - `examples/hybrid_parallel/launch_torchtitan_qwen3_14b_single_5090d.sh`
- Supported presets:
  - `throughput`
    - maps to `qwen3_14b_single_5090d_tp4_fsdp2_throughput`
    - `PP=1, TP=4, FSDP2(shard=2)` throughput baseline
    - best when you want a shard-heavy control line and pipeline bubble is not the main problem
  - `vpp`
    - maps to `qwen3_14b_single_5090d_vpp_fsdp2`
    - `PP=2, VPP=2, TP=2, FSDP2(shard=2)` interleaved research preset
    - useful when you explicitly want to study `pipeline/VPP <-> FSDP2` interaction
  - `vpp_safe`
    - maps to `qwen3_14b_single_5090d_vpp_fsdp2_safe`
    - safer bring-up defaults for seq length and local batch
    - good first step before testing the more aggressive `vpp` preset
  - `vpp_budgeted`
    - maps to `qwen3_14b_single_5090d_vpp_fsdp2_budgeted`
    - keeps `PP=2, VPP=2, TP=2, FSDP2(shard=2)` but adds per-stage HBM budget, prefetch window, and watermark knobs
    - best preset for studying communication/materialization/offload coordination

Example:

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
bash examples/hybrid_parallel/launch_torchtitan_qwen3_14b_single_5090d.sh throughput
```

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
HYBRID_SEQ_LEN=1024 \
HYBRID_LOCAL_BATCH_SIZE=2 \
bash examples/hybrid_parallel/launch_torchtitan_qwen3_14b_single_5090d.sh vpp_safe
```

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ens9f0
export GLOO_SOCKET_IFNAME=ens9f0
HYBRID_SEQ_LEN=1024 \
HYBRID_GLOBAL_BATCH_SIZE=32 \
HYBRID_STAGE_HBM_BUDGET_GIB=28.5,30,30,28.5 \
HYBRID_FSDP_PREFETCH_WINDOW=1 \
HYBRID_FSDP_MATERIALIZATION_WATERMARK_GIB=29.0 \
bash examples/hybrid_parallel/launch_torchtitan_qwen3_14b_single_5090d.sh vpp_budgeted
```

Strategy notes for single-node `5090D`:

- If your main bottleneck is `all-gather` or parameter materialization pressure, start from `throughput`.
- If your main question is `pipeline/VPP` behavior under FSDP2, start from `vpp_safe`, then move to `vpp`.
- If your main question is communication vs prefetch/materialization coordination, use `vpp_budgeted`.
- For the current Qwen3-14B single-node runtime-optimization line, Megatron PP/VPP is still the stronger mainline for peak throughput; TorchTitan/FSDP2 is the better comparison axis for shard policy, prefetch, reshard, and budgeted-materialization studies.

Offline TorchTitan packaging on `g4` and sync to `g5`:

```bash
bash torchtitan/scripts/install_editable_local.sh
bash torchtitan/scripts/build_offline_wheelhouse.sh
bash examples/hybrid_parallel/push_torchtitan_bundle_to_g5.sh
```

Offline install on `g5`:

```bash
bash torchtitan/scripts/install_offline_from_wheelhouse.sh ~/agent/torchtitan/dist/wheelhouse
```

## Metrics / profiler

- `train_manual_pp.py` prints `tokens/s` and `mem_gb` on the logging rank.
- Optional MFU estimate: set `mfu.peak_tflops_total` in config (aggregate peak TFLOPs for the whole job).
- Optional Torch Profiler traces: set `profile.enabled=true` and view with TensorBoard (`tensorboard --logdir tb_traces`).
