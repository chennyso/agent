#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens8f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROGRAM_FILE="${PROGRAM_FILE:-${ROOT_DIR}/scripts/dual_g4_g5_qwen3_32b_tp2_pp8_local.json}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${ROOT_DIR}/Megatron-LM}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/dual_g4_g5_qwen3_32b_local}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/node0_metrics.json}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen3-32B}"
DATA_PATH="${DATA_PATH:-/public/home/ssjxscy/datasets/wikitext-103-raw-v1/data/processed_wikitext_text_document}"
MASTER_ADDR="${MASTER_ADDR:-192.168.10.228}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$RUN_ROOT" "$(dirname "$OUTPUT_JSON")"

DRY_RUN_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

python -m megatron_agent.trial_runner \
  --program-file "$PROGRAM_FILE" \
  --output "$OUTPUT_JSON" \
  --run-root "$RUN_ROOT" \
  --megatron-root "$MEGATRON_ROOT" \
  --launcher-script "" \
  --run-target dual_g4_g5 \
  --model-track dense \
  --nproc 8 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --transformer-impl local \
  --train-iters "$TRAIN_ITERS" \
  --eval-iters 0 \
  --eval-interval 0 \
  --num-layers 64 \
  --hidden-size 5120 \
  --ffn-hidden-size 25600 \
  --num-attention-heads 64 \
  --num-query-groups 8 \
  --kv-channels 128 \
  --max-position-embeddings 4096 \
  --vocab-size 151936 \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --data-path "$DATA_PATH" \
  "${DRY_RUN_ARGS[@]}"
