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

For the `single_g5` Qwen3-14B high-performance path, the shared environment must provide both Transformer Engine and Apex. Install them on `g4` into the shared env, then verify them again on `g5`:

```bash
bash ./scripts/install_g5_perf_stack_on_g4.sh
```

Expected validation commands after install:

```bash
source /public/home/ssjxscy/envs/torchdist_ok/bin/activate
python -c "import transformer_engine; print(transformer_engine.__version__)"
python -c "import apex; print(apex.__file__)"
python -c "from transformer_engine.pytorch.optimizers import FusedAdam"
python -c "from apex.optimizers import FusedAdam"
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

## 7. Paper-Closure Ablation Workflow

This is the earliest complete workflow for the paper-priority loop. It does not attempt a full runtime rewrite. Instead, it uses the current patch-aware search, runtime-observed traces, structured artifacts, and the new multi-run aggregation scripts to answer four paper questions directly:

1. Are high-gain candidates usually sparse patch refinements?
2. Do different bottlenecks prefer different patch families?
3. Does `search_unit=patch` reach near-best results faster than `search_unit=whole_config`?
4. Does patch memory reduce repeated harmful trials?

### One-shot ablation driver

Use the new driver to run three aligned experiments under the same candidate space:

- `patch`
- `whole_config`
- `patch_memory_off`

The driver writes a manifest, launches the three runs, aggregates all `summary_megatron.json` files, and renders paper figures.

```bash
python ./scripts/run_patch_paper_ablation.py \
  --work-root ./runs/paper_patch_loop \
  --analysis-dir ./runs/paper_patch_loop/analysis \
  --figures-dir ./runs/paper_patch_loop/analysis/figures \
  --case-study-topk 2 \
  -- \
  --run-target single_g5 \
  --model-track dense \
  --candidate-limit 6 \
  --auto-tune-rounds 2 \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "$LAUNCHER_SCRIPT" \
  --run-root ./runs_megatron
```

The forwarded arguments after `--` are passed through to `python -m megatron_agent.agent_loop` for all three variants.

Expected run layout:

- `./runs/paper_patch_loop/patch`
- `./runs/paper_patch_loop/whole_config`
- `./runs/paper_patch_loop/patch_memory_off`
- `./runs/paper_patch_loop/paper_ablation_manifest.json`
- `./runs/paper_patch_loop/analysis/patch_observations.csv`
- `./runs/paper_patch_loop/analysis/search_ablation.csv`
- `./runs/paper_patch_loop/analysis/case_study_manifest.json`
- `./runs/paper_patch_loop/analysis/figures/fig_search_ablation_curve.png`

### Dry-run the paper loop

If you want the exact commands first, use `--dry-run`. The manifest is still written.

```bash
python ./scripts/run_patch_paper_ablation.py \
  --work-root ./runs/paper_patch_loop \
  --dry-run \
  -- \
  --run-target single_g5 \
  --model-track dense \
  --candidate-limit 6 \
  --auto-tune-rounds 2
```

### Re-analyze existing experiments only

If the three run directories already contain `summary_megatron.json`, you can skip training and regenerate tables plus figures:

```bash
python ./scripts/run_patch_paper_ablation.py \
  --work-root ./runs/paper_patch_loop \
  --analysis-only
```

If the analysis tables already exist and you only want to redraw figures:

```bash
python ./scripts/run_patch_paper_ablation.py \
  --work-root ./runs/paper_patch_loop \
  --plots-only
```

### Manual aggregation and figure rendering

If you prefer explicit step-by-step commands instead of the driver, run:

```bash
python ./scripts/analyze_patch_observations.py \
  --runs \
    ./runs/paper_patch_loop/patch \
    ./runs/paper_patch_loop/whole_config \
    ./runs/paper_patch_loop/patch_memory_off \
  --out-dir ./runs/paper_patch_loop/analysis \
  --case-study-topk 2
```

```bash
python ./scripts/plot_patch_paper_figures.py \
  --analysis-dir ./runs/paper_patch_loop/analysis \
  --out-dir ./runs/paper_patch_loop/analysis/figures
```

### Search controls used by the paper loop

The paper workflow relies on two new knobs in `agent_loop.py`:

- `--search-unit patch`
  Keeps `ProgramPatchSpec` as the search unit, enables patch-family priority, local refinement continuity, and patch memory recommend/avoid signals.
- `--search-unit whole_config`
  Uses the same candidate generator and verifier pipeline, but disables patch-family priority and treats proposals as whole configurations for the search ablation.
- `--disable-patch-memory`
  Turns off `recommend_patch_families`, `should_avoid_patch_family`, `record_trial_feedback`, and `policy_memory.json` updates for the patch-memory ablation.

### What to inspect in each run

Each `summary_megatron.json` now records:

- `search_unit`
- `patch_memory_enabled`
- `search_tree_history`
- `trial_reflections`
- `tested_trials[*].patch_family`
- `tested_trials[*].patch_category`
- `tested_trials[*].patch_count`

Each trial analysis directory should contain:

- `pipeline_schedule_projection.json`
- `pipeline_event_trace.json`
- `pipeline_grid_trace.json`
- `bottleneck_breakdown.json`
- `trial_reflection.json`
- `failure_diagnosis.json`
- `pipeline_projection.svg`
- `compare_pipeline.svg`

### Observation tables produced for the paper

`patch_observations.csv` is the flat trial table used for scatter plots and case filtering. Key columns:

- `search_unit`
- `patch_memory_enabled`
- `patch_family`
- `patch_category`
- `patch_count`
- `bottleneck_label`
- `step_gain_ratio`
- `throughput_gain_ratio`
- `bubble_ratio`
- `stage_skew_ratio`
- `memory_skew_ratio`
- `tail_ratio`
- `optimizer_exposed_ratio`

`bottleneck_patch_success.csv` answers which patch families work under which bottlenecks.

`bottleneck_patch_gain.csv` answers which patch families deliver the largest gains under which bottlenecks.

`search_ablation.csv` is the direct patch-aware versus whole-config versus patch-memory-off comparison.

`case_study_manifest.json` chooses top bubble-bound and memory-bound examples and points to their baseline/candidate artifacts.

### Figure files

The figure script writes:

- `fig_patch_sparsity.png`
- `fig_patch_count_hist.png`
- `fig_bottleneck_patch_success_heatmap.png`
- `fig_bottleneck_patch_gain_heatmap.png`
- `fig_search_ablation_curve.png`
- `fig_case_study_compare.png`

These are designed to be paper-ready intermediate assets, not just debugging plots.
