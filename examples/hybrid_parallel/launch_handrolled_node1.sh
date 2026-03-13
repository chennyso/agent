#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-examples/hybrid_parallel/config_qwen3_2node_dense_pp4_handrolled_pp_only_debug.json}"

export NODE_RANK=1
export NNODES=2
export NPROC_PER_NODE=8

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:?}:${MASTER_PORT:?}" \
  examples/hybrid_parallel/train_handrolled_pp_debug.py --config "${CONFIG_PATH}"
