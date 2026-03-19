#!/bin/bash

set -euo pipefail

# Single-node g5 (8x RTX 5090 32GB) mixed-parallel training presets for Qwen3-14B.
#
# Presets:
#   1) PRESET=throughput
#      - TP=4, PP=1, DP=2
#      - Best baseline to compare against TorchTitan FSDP2 throughput.
#   2) PRESET=vpp
#      - TP=2, PP=2, VPP=2, DP=2
#      - Research line to inspect PP/VPP bubble behavior on the same hardware.
#
# Usage examples:
#   bash Megatron-LM/examples/qwen/train_qwen3_14b_g5_8x5090.sh
#   PRESET=vpp bash Megatron-LM/examples/qwen/train_qwen3_14b_g5_8x5090.sh
#   DATA_PATH=/path/to/indexed_dataset TOKENIZER_MODEL=/path/to/Qwen3-14B \
#     bash Megatron-LM/examples/qwen/train_qwen3_14b_g5_8x5090.sh

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}

PRESET=${PRESET:-throughput}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6000}
PRETRAIN_SCRIPT_PATH=${PRETRAIN_SCRIPT_PATH:-pretrain_gpt.py}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/qwen3_14b_${PRESET}}
TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-tensorboard_logs/qwen3_14b_${PRESET}}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-${PWD}/benchmark_cache_qwen3_14b_${PRESET}}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen3-14B}
DATA_PATH=${DATA_PATH:-MOCK}
DATA_SPLIT=${DATA_SPLIT:-99,1,0}
NUM_WORKERS=${NUM_WORKERS:-1}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-0}

TRAIN_ITERS=${TRAIN_ITERS:-20}
EVAL_ITERS=${EVAL_ITERS:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}

mkdir -p "$CHECKPOINT_PATH"
mkdir -p "$TENSORBOARD_LOGS_PATH"
mkdir -p "$DATA_CACHE_PATH"

case "$PRESET" in
  throughput)
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-1}
    CP_SIZE=${CP_SIZE:-1}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
    SEQ_LENGTH=${SEQ_LENGTH:-1024}
    MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-40960}
    EXTRA_PARALLEL_ARGS=(
      --tensor-model-parallel-size "$TP_SIZE"
      --pipeline-model-parallel-size "$PP_SIZE"
      --context-parallel-size "$CP_SIZE"
      --sequence-parallel
      --use-distributed-optimizer
      --overlap-grad-reduce
      --overlap-param-gather
      --tp-comm-overlap
    )
    PRESET_NOTE="TP4 + DP2 mixed parallel baseline"
    ;;
  vpp)
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-2}
    CP_SIZE=${CP_SIZE:-1}
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
    SEQ_LENGTH=${SEQ_LENGTH:-1024}
    MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-40960}
    NUM_LAYERS_PER_VIRTUAL_STAGE=${NUM_LAYERS_PER_VIRTUAL_STAGE:-10}
    EXTRA_PARALLEL_ARGS=(
      --tensor-model-parallel-size "$TP_SIZE"
      --pipeline-model-parallel-size "$PP_SIZE"
      --context-parallel-size "$CP_SIZE"
      --num-layers-per-virtual-pipeline-stage "$NUM_LAYERS_PER_VIRTUAL_STAGE"
      --sequence-parallel
      --use-distributed-optimizer
      --overlap-grad-reduce
      --overlap-param-gather
      --tp-comm-overlap
    )
    if [[ -n "${PIPELINE_LAYOUT:-}" ]]; then
      EXTRA_PARALLEL_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_LAYOUT")
    fi
    PRESET_NOTE="TP2 + PP2 + VPP2 + DP2 mixed parallel research line"
    ;;
  *)
    echo "Unsupported PRESET=$PRESET. Choose throughput or vpp."
    exit 1
    ;;
esac

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
PRODUCT=$((TP_SIZE * PP_SIZE * CP_SIZE))
if (( WORLD_SIZE % PRODUCT != 0 )); then
  echo "Invalid parallel product: WORLD_SIZE=${WORLD_SIZE}, TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}"
  exit 1
fi
DP_SIZE=$((WORLD_SIZE / PRODUCT))

echo "Launching Qwen3-14B on g5 with preset: ${PRESET_NOTE}"
echo "World size=${WORLD_SIZE}, DP=${DP_SIZE}, TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NUM_NODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
  --use-mcore-models
  --transformer-impl transformer_engine
  --attention-backend flash
  --num-layers 40
  --hidden-size 5120
  --ffn-hidden-size 17408
  --num-attention-heads 40
  --kv-channels 128
  --group-query-attention
  --num-query-groups 8
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --use-rotary-position-embeddings
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --qk-layernorm
  --swiglu
  --untie-embeddings-and-output-weights
  --disable-bias-linear
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --no-masked-softmax-fusion
  --make-vocab-size-divisible-by 128
  --vocab-size 151936
)

TRAINING_ARGS=(
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --train-iters "$TRAIN_ITERS"
  --lr 1.0e-4
  --min-lr 1.0e-5
  --lr-decay-style cosine
  --lr-warmup-iters 5
  --clip-grad 1.0
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1.0e-8
  --bf16
  --grad-reduce-in-bf16
  --recompute-granularity selective
  --recompute-activations
  --recompute-modules core_attn
  --calculate-per-token-loss
)

DATA_ARGS=()
if [[ "$DATA_PATH" == "MOCK" ]]; then
  DATA_ARGS+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --data-cache-path "$DATA_CACHE_PATH"
    --split "$DATA_SPLIT"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers "$NUM_WORKERS"
  )
else
  DATA_ARGS+=(
    --data-path "$DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --tokenizer-hf-include-special-tokens
    --data-cache-path "$DATA_CACHE_PATH"
    --split "$DATA_SPLIT"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers "$NUM_WORKERS"
  )
fi

LOGGING_ARGS=(
  --log-interval "$LOG_INTERVAL"
  --timing-log-level 2
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-throughput
  --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
  --ckpt-format torch_dist
  --distributed-timeout-minutes 60
  --save "$CHECKPOINT_PATH"
)

if (( LOAD_CHECKPOINT > 0 )); then
  LOGGING_ARGS+=(--load "$CHECKPOINT_PATH")
fi

if (( EVAL_ITERS > 0 && EVAL_INTERVAL > 0 )); then
  LOGGING_ARGS+=(--eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL")
fi
if (( SAVE_INTERVAL > 0 )); then
  LOGGING_ARGS+=(--save-interval "$SAVE_INTERVAL")
fi
if (( ENABLE_PROFILE > 0 )); then
  LOGGING_ARGS+=(--profile --profile-step-start 10 --profile-step-end 12 --profile-ranks 0)
fi

if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
  echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
  echo "Run this script from the Megatron-LM repository root, or set PRETRAIN_SCRIPT_PATH."
  exit 1
fi

set -x
torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$PRETRAIN_SCRIPT_PATH" \
  "${MODEL_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${EXTRA_PARALLEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOGGING_ARGS[@]}"
