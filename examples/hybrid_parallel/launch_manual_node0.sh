#!/usr/bin/env bash
set -euo pipefail

# Node0 (stage0) launch example for manual PP (no torch.export).
#
# Usage:
#   export MASTER_ADDR=<master_ip_or_hostname>
#   export MASTER_PORT=29500
#   export NCCL_SOCKET_IFNAME=<nic>   # e.g. ens8f0 on g4
#   export GLOO_SOCKET_IFNAME=<nic>
#   bash examples/hybrid_parallel/launch_manual_node0.sh /path/to/config.json

CONFIG_PATH="${1:-examples/hybrid_parallel/config_qwen3_2node_dense_pp4_gpipe_safe.json}"

export NODE_RANK=0
export NNODES=2
export NPROC_PER_NODE=8

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR:?}:${MASTER_PORT:?}" \
  examples/hybrid_parallel/train_manual_pp.py --config "${CONFIG_PATH}"
