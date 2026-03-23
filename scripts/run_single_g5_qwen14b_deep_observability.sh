#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/public/home/ssjxscy/agent/Megatron-LM}"
WORKDIR="${WORKDIR:-./runs/single_g5_qwen14b_deep}"
EXPORT_DIR="${EXPORT_DIR:-${WORKDIR}/export}"
PROGRAMS_DIR="${PROGRAMS_DIR:-${EXPORT_DIR}/programs}"
RUN_ROOT="${RUN_ROOT:-${WORKDIR}/trials}"
OUTPUT_JSON="${OUTPUT_JSON:-${WORKDIR}/baseline_metrics.json}"
PROGRAM_FILE="${PROGRAM_FILE:-${PROGRAMS_DIR}/00_baseline.json}"

RUN_TARGET="${RUN_TARGET:-single_g5}"
MODEL_TRACK="${MODEL_TRACK:-dense}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-4}"
OBSERVABILITY_PRESET="${OBSERVABILITY_PRESET:-deep}"
PROFILE_STEP_START="${PROFILE_STEP_START:-4}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
ENABLE_NSYS="${ENABLE_NSYS:-0}"

TOKENIZER_MODEL="${TOKENIZER_MODEL:-/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen3-14B}"
DATA_PATH="${DATA_PATH:-/public/home/ssjxscy/datasets/wikitext-103-raw-v1/data/processed_wikitext_text_document}"

mkdir -p "$WORKDIR" "$EXPORT_DIR" "$PROGRAMS_DIR" "$RUN_ROOT"

PROFILE_RANK_ARGS=()
if [[ -n "$PROFILE_RANKS" ]]; then
  IFS=',' read -r -a PROFILE_RANK_ARRAY <<< "$PROFILE_RANKS"
  PROFILE_RANK_ARGS=(--profile-ranks "${PROFILE_RANK_ARRAY[@]}")
fi

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_EXP_NAME="${WANDB_EXP_NAME:-single_g5_qwen14b_deep_$(date +%Y%m%d_%H%M%S)}"
  WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-exp-name "$WANDB_EXP_NAME")
  if [[ -n "${WANDB_SAVE_DIR:-}" ]]; then
    WANDB_ARGS+=(--wandb-save-dir "$WANDB_SAVE_DIR")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

NSYS_ARGS=()
if [[ "$ENABLE_NSYS" == "1" ]]; then
  NSYS_ARGS+=(--enable-nsys)
fi

python -m megatron_agent.agent_loop \
  --export-only \
  --workdir "$EXPORT_DIR" \
  --programs-dir "$PROGRAMS_DIR" \
  --run-target "$RUN_TARGET" \
  --model-track "$MODEL_TRACK" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --train-iters "$TRAIN_ITERS" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --data-path "$DATA_PATH"

echo "Export finished."
echo "  summary: ${EXPORT_DIR}/summary_megatron.json"
echo "  baseline program: ${PROGRAM_FILE}"

if [[ ! -f "$PROGRAM_FILE" ]]; then
  echo "Error: exported baseline program not found: ${PROGRAM_FILE}"
  exit 1
fi

set +e
python -m megatron_agent.trial_runner \
  --program-file "$PROGRAM_FILE" \
  --output "$OUTPUT_JSON" \
  --run-root "$RUN_ROOT" \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "" \
  --run-target "$RUN_TARGET" \
  --model-track "$MODEL_TRACK" \
  --nproc 8 \
  --nnodes 1 \
  --train-iters "$TRAIN_ITERS" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --data-path "$DATA_PATH" \
  --observability-preset "$OBSERVABILITY_PRESET" \
  --profile-step-start "$PROFILE_STEP_START" \
  --profile-step-end "$PROFILE_STEP_END" \
  "${PROFILE_RANK_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${NSYS_ARGS[@]}"
TRIAL_RC=$?
set -e

if (( TRIAL_RC != 0 )); then
  echo "Trial runner failed with exit code ${TRIAL_RC}."
  if [[ -f "$OUTPUT_JSON" ]]; then
    echo "Failure summary from ${OUTPUT_JSON}:"
    python - <<'PY' "$OUTPUT_JSON"
import json, sys
path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
print(f"  returncode: {payload.get('returncode')}")
print(f"  error_msg: {payload.get('error_msg')}")
trial_context = payload.get("trial_context") or {}
resolved = trial_context.get("resolved_paths") or {}
launch_plan = payload.get("launch_plan") or {}
observability = launch_plan.get("observability") or {}
print(f"  megatron_entry: {trial_context.get('megatron_entry')}")
print(f"  trial_dir: {resolved.get('trial_dir')}")
print(f"  tensorboard: {resolved.get('tensorboard_path')}")
print(f"  torch_profile: {resolved.get('torch_profile_path')}")
print(f"  torchrun_log_dir: {resolved.get('torchrun_log_dir')}")
print(f"  memory_snapshot: {resolved.get('memory_snapshot_path')}")
print(f"  nsys_output: {resolved.get('nsys_output')}")
print(f"  stdout_log: {resolved.get('stdout_log')}")
print(f"  stderr_log: {resolved.get('stderr_log')}")
print(f"  launch_plan_log: {resolved.get('launch_plan_log')}")
print(f"  root_cause_source: {payload.get('root_cause_source')}")
print(f"  root_cause_excerpt: {payload.get('root_cause_excerpt')}")
print(f"  stderr_tail: {payload.get('stderr_tail')}")
print(f"  stdout_tail: {payload.get('stdout_tail')}")
print(f"  observability: {observability}")
PY
  else
    echo "No output json was produced."
  fi
  exit "$TRIAL_RC"
fi

echo "Run finished."
echo "  metrics: ${OUTPUT_JSON}"
echo "  tensorboard: ${RUN_ROOT}/trial_000/tensorboard"
echo "  torch_profile: ${RUN_ROOT}/trial_000/torch_profile"
echo "  checkpoint_dir: ${RUN_ROOT}/trial_000/checkpoints"
echo "  summary: ${EXPORT_DIR}/summary_megatron.json"
