#!/usr/bin/env bash
set -euo pipefail

MODULE_PATH="${MODULE_PATH:-hybrid_policy}"
CONFIG_NAME="${1:-qwen3_hybrid_demo}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-torchtitan}"

export NODE_RANK="${NODE_RANK:-0}"
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${TORCHTITAN_DIR}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:?}:${MASTER_PORT:?}" \
  -m torchtitan.train --module "${MODULE_PATH}" --config "${CONFIG_NAME}" "${@:2}"
