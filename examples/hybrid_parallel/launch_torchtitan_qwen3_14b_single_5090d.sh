#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-throughput}"

case "${PRESET}" in
  throughput)
    CONFIG_NAME="qwen3_14b_single_5090d_tp4_fsdp2_throughput"
    ;;
  vpp)
    CONFIG_NAME="qwen3_14b_single_5090d_vpp_fsdp2"
    ;;
  vpp_safe)
    CONFIG_NAME="qwen3_14b_single_5090d_vpp_fsdp2_safe"
    ;;
  vpp_budgeted)
    CONFIG_NAME="qwen3_14b_single_5090d_vpp_fsdp2_budgeted"
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    echo "Supported presets: throughput, vpp, vpp_safe, vpp_budgeted" >&2
    exit 1
    ;;
esac

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

exec bash examples/hybrid_parallel/launch_torchtitan_hybrid_node0.sh "${CONFIG_NAME}" "${@:2}"
