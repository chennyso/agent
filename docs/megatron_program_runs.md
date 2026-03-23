# MegatronProgram Real Run Guide

This guide keeps the current `MegatronProgram + SearchSpaceSpec + check/compile/run` flow and focuses on manual, reproducible Linux cluster runs.

Sequence parallelism is modeled in the DSL as a TP-coupled toggle, not as an independent size axis. In practice this means exported programs may include a `candidate_sequence_parallel_toggle` variant only when `tp_degree > 1`.

## Preconditions

Run from the repo root and export `PYTHONPATH=src`:

```bash
cd /path/to/Agent
export PYTHONPATH=src
```

Set your Megatron paths explicitly when running on cluster:

```bash
export MEGATRON_ROOT=/public/home/ssjxscy/agent/Megatron-LM
export LAUNCHER_SCRIPT=examples/qwen/train_qwen3_14b_rtx_8gpu.sh
```

## 1. Export Programs Without Running Trials

Use `--export-only` to synthesize baseline and candidate programs, write program JSON files, and emit a manifest in `summary_megatron.json`.

### single_g4 dense

Use this target on a single `g4 / 4090D / 24GB` node.

```bash
python -m megatron_agent.agent_loop \
  --export-only \
  --workdir ./runs/single_g4_dense_export \
  --programs-dir ./runs/single_g4_dense_export/programs \
  --run-target single_g4 \
  --model-track dense \
  --candidate-limit 4
```

Expected outputs include at least a legal baseline and a synthesized candidate. The default `single_g4` dense baseline is intentionally more conservative than `single_g5`: `tp=2, pp=2`.

### single_g5 dense

Use this target on a single `g5 / 5090D / 32GB` node.

```bash
python -m megatron_agent.agent_loop \
  --export-only \
  --workdir ./runs/single_g5_dense_export \
  --programs-dir ./runs/single_g5_dense_export/programs \
  --run-target single_g5 \
  --model-track dense \
  --candidate-limit 4
```

Expected outputs:

- `./runs/single_g5_dense_export/summary_megatron.json`
- `./runs/single_g5_dense_export/programs/00_baseline.json`
- `./runs/single_g5_dense_export/programs/01_candidate_single_node_pp_split.json`
- `./runs/single_g5_dense_export/programs/02_candidate_sequence_parallel_toggle.json`

### dual_g4_g5 dense

```bash
python -m megatron_agent.agent_loop \
  --export-only \
  --workdir ./runs/dual_g4_g5_dense_export \
  --programs-dir ./runs/dual_g4_g5_dense_export/programs \
  --run-target dual_g4_g5 \
  --model-track dense \
  --nnodes 2 \
  --candidate-limit 4
```

### dual_g4_g5 moe

```bash
python -m megatron_agent.agent_loop \
  --export-only \
  --workdir ./runs/dual_g4_g5_moe_export \
  --programs-dir ./runs/dual_g4_g5_moe_export/programs \
  --run-target dual_g4_g5 \
  --model-track moe \
  --nnodes 2 \
  --candidate-limit 4
```

The recommended execution order is stored in `summary_megatron.json` under `recommended_execution_order`.

## Deep Observability

For detailed training diagnosis, the runner now supports:

- TensorBoard timers and memory logs
- PyTorch profiler traces exported as Chrome trace files
- CUDA memory history snapshots for OOM analysis
- Straggler detection logs
- Optional WandB logging
- Optional Nsight Systems wrapping

The easiest single-node `g5 / 5090D / Qwen3-14B` entrypoint is:

```bash
bash ./scripts/run_single_g5_qwen14b_deep_observability.sh
```

Useful environment overrides:

```bash
export MEGATRON_ROOT=/public/home/ssjxscy/agent/Megatron-LM
export WORKDIR=./runs/single_g5_qwen14b_deep
export TRAIN_ITERS=20
export ENABLE_NSYS=1
export WANDB_PROJECT=my-megatron-project
export WANDB_EXP_NAME=single_g5_qwen14b_deep
bash ./scripts/run_single_g5_qwen14b_deep_observability.sh
```

Expected artifacts:

- TensorBoard events: `trial_000/tensorboard`
- Chrome trace exports: `trial_000/torch_profile`
- Memory snapshots: `trial_000/checkpoints/snapshot.pickle` or the configured override
- Nsight output base: `trial_000/nsys/` when `ENABLE_NSYS=1`

## 2. Dry-Run Before Real Launch

Use `--dry-run` to verify resolved paths, launcher env overrides, runner mode, and the compiled Megatron command without starting training.

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/single_g5_dense_export/programs/00_baseline.json \
  --output ./runs/single_g5_dense_export/dry_run_baseline.json \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target single_g5 \
  --model-track dense \
  --nproc 8 \
  --nnodes 1 \
  --dry-run
```

Inspect the output JSON for:

- `trial_context.resolved_paths`
- `launch_plan.runner_mode`
- `launch_plan.launcher_env`
- `launch_plan.megatron_command`

## 3. Run single-node dense Baseline

### single_g4 / 4090D

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/single_g4_dense_export/programs/00_baseline.json \
  --output ./runs/single_g4_dense_export/baseline_metrics.json \
  --run-root ./runs/single_g4_dense_export/trials \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target single_g4 \
  --model-track dense \
  --nproc 8 \
  --nnodes 1
```

If your 4090 node is more stable on fp16, re-run with a program whose metadata selects `fp16`, or override at synthesis time by exporting a program generated with `--fp16`.

### single_g5 / 5090D

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/single_g5_dense_export/programs/00_baseline.json \
  --output ./runs/single_g5_dense_export/baseline_metrics.json \
  --run-root ./runs/single_g5_dense_export/trials \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target single_g5 \
  --model-track dense \
  --nproc 8 \
  --nnodes 1
```

## 4. Run single-node dense Family-Outside Candidate

### single_g4 / 4090D

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/single_g4_dense_export/programs/01_candidate_nonuniform_partition.json \
  --output ./runs/single_g4_dense_export/candidate_metrics.json \
  --run-root ./runs/single_g4_dense_export/trials \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target single_g4 \
  --model-track dense \
  --nproc 8 \
  --nnodes 1
```

The exact candidate filename depends on the exported manifest. Check `summary_megatron.json -> candidate_manifest`.

### single_g5 / 5090D

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/single_g5_dense_export/programs/01_candidate_single_node_pp_split.json \
  --output ./runs/single_g5_dense_export/candidate_single_node_pp_split_metrics.json \
  --run-root ./runs/single_g5_dense_export/trials \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target single_g5 \
  --model-track dense \
  --nproc 8 \
  --nnodes 1
```

## 5. Feed Baseline Summary Back Into Synthesis

If you already have a baseline metrics JSON, feed it back through `--runtime-summary` to rewrite the search space before exporting the next round of candidates.

```bash
python -m megatron_agent.agent_loop \
  --export-only \
  --workdir ./runs/single_g5_dense_round2 \
  --programs-dir ./runs/single_g5_dense_round2/programs \
  --run-target single_g5 \
  --model-track dense \
  --runtime-summary ./runs/single_g5_dense_export/baseline_metrics.json \
  --candidate-limit 4
```

## 6. Move To dual_g4_g5

Run in the same order after single-node validation:

1. `dual_g4_g5 dense` baseline
2. `dual_g4_g5 dense` synthesized candidates
3. `dual_g4_g5 moe` baseline
4. `dual_g4_g5 moe` synthesized candidates

For dual-node runs, keep `--nnodes 2` and set the standard multi-node torchrun fields:

```bash
python -m megatron_agent.trial_runner \
  --program-file ./runs/dual_g4_g5_dense_export/programs/00_baseline.json \
  --output ./runs/dual_g4_g5_dense_export/baseline_metrics_node0.json \
  --run-root ./runs/dual_g4_g5_dense_export/trials \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-target dual_g4_g5 \
  --model-track dense \
  --nproc 8 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <master-host> \
  --master-port 29500
```

Run the matching command on node 1 with `--node-rank 1`.

## Notes

- `cross_node_exposed_ratio` is still an optional external runtime-summary field. It is not auto-derived from current log parsing.
- `single_g4` is the single-node 4090D target, `single_g5` is the single-node 5090D target, and `dual_g4_g5` is the mixed 4090D + 5090D target.
- The launcher now honors `USE_BF16` / `USE_FP16` compiled from the program instead of always forcing bf16.
- `summary_megatron.json` now includes `candidate_manifest`, `compile_success_rate`, `family_outside_ratio`, `stage_load_variance`, `observed_comm_ratio`, and `baseline_vs_best`.
- The current workflow stays at program/control-plane level. It does not enable runtime PG rebuild, heterogeneous apipe execution, or submesh execution.
