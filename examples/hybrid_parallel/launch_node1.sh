#!/usr/bin/env bash
set -euo pipefail

# Node1 (stage1) launch example.
# Usage:
#   export MASTER_ADDR=<node0_ip_or_hostname>
#   export MASTER_PORT=29500
#   bash examples/hybrid_parallel/launch_node1.sh /path/to/config.json

CONFIG_PATH="${1:-examples/hybrid_parallel/config_qwen3_2node_pp2_tp2_vpp2.json}"

export NODE_RANK=1
export NNODES=2
export NPROC_PER_NODE=8

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:?}:${MASTER_PORT:?}" \
  examples/hybrid_parallel/train_hybrid_qwen.py --config "${CONFIG_PATH}"

