#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../torchtitan"

export MASTER_ADDR="${MASTER_ADDR:-192.168.10.241}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens8f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

HF_ASSETS_PATH="${HF_ASSETS_PATH:-./assets/hf/Qwen3-32B}"
DUMP_FOLDER="${DUMP_FOLDER:-./outputs/qwen3_32b_tt_2node_hetero}"

EXTRA_ARGS=()
if [ -d "${HF_ASSETS_PATH}" ]; then
    EXTRA_ARGS+=(
        --hf_assets_path "${HF_ASSETS_PATH}"
        --checkpoint.initial_load_in_hf
        --checkpoint.initial_load_model_only
    )
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m torchtitan.train \
    --module qwen3 \
    --config qwen3_32b_2node_pp2_tp4_fsdp2_hetero \
    --dump_folder "${DUMP_FOLDER}" \
    --checkpoint.enable \
    "${EXTRA_ARGS[@]}" \
    "$@"
