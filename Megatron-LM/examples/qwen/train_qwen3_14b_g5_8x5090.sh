#!/bin/bash

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

RUN_TARGET=${RUN_TARGET:-single_g5}
MODEL_TRACK=${MODEL_TRACK:-dense}
PRESET=${PRESET:-auto}

if [[ "$PRESET" == "safe_single_node" ]]; then
  MODEL_TRACK="dense"
  if [[ "$RUN_TARGET" == "dual_g4_g5" ]]; then
    RUN_TARGET="single_g5"
  fi
elif [[ "$PRESET" == "vpp_research" ]]; then
  MODEL_TRACK="dense"
  if [[ "$RUN_TARGET" == "dual_g4_g5" ]]; then
    RUN_TARGET="single_g5"
  fi
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
if [[ "$RUN_TARGET" == "dual_g4_g5" ]]; then
  NUM_NODES=${NUM_NODES:-2}
fi
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6000}
PRETRAIN_SCRIPT_PATH=${PRETRAIN_SCRIPT_PATH:-pretrain_gpt.py}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/${MODEL_TRACK}_${RUN_TARGET}}
TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-tensorboard_logs/${MODEL_TRACK}_${RUN_TARGET}}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-${PWD}/benchmark_cache_${MODEL_TRACK}_${RUN_TARGET}}

TOKENIZER_MODEL=${TOKENIZER_MODEL:-/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen3-14B}
DATA_PATH=${DATA_PATH:-/public/home/ssjxscy/datasets/wikitext-103-raw-v1/data/processed_wikitext_text_document}
DATA_SPLIT=${DATA_SPLIT:-99,1,0}
NUM_WORKERS=${NUM_WORKERS:-1}

TRAIN_ITERS=${TRAIN_ITERS:-10}
EVAL_ITERS=${EVAL_ITERS:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
OBSERVABILITY_PRESET=${OBSERVABILITY_PRESET:-none}
ENABLE_LOG_TIMERS_TO_TENSORBOARD=${ENABLE_LOG_TIMERS_TO_TENSORBOARD:-0}
ENABLE_LOG_MEMORY_TO_TENSORBOARD=${ENABLE_LOG_MEMORY_TO_TENSORBOARD:-0}
TENSORBOARD_LOG_INTERVAL=${TENSORBOARD_LOG_INTERVAL:-1}
ENABLE_PYTORCH_PROFILER=${ENABLE_PYTORCH_PROFILER:-0}
PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-0}
PROFILE_COLLECT_CALLSTACK=${PROFILE_COLLECT_CALLSTACK:-0}
PROFILE_COLLECT_CHAKRA=${PROFILE_COLLECT_CHAKRA:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-4}
PROFILE_STEP_END=${PROFILE_STEP_END:-6}
PROFILE_RANKS=${PROFILE_RANKS:-0}
ENABLE_MEMORY_HISTORY=${ENABLE_MEMORY_HISTORY:-0}
MEMORY_SNAPSHOT_PATH=${MEMORY_SNAPSHOT_PATH:-snapshot.pickle}
ENABLE_STRAGGLER_LOG=${ENABLE_STRAGGLER_LOG:-0}
DISABLE_STRAGGLER_ON_STARTUP=${DISABLE_STRAGGLER_ON_STARTUP:-0}
STRAGGLER_MINMAX_COUNT=${STRAGGLER_MINMAX_COUNT:-8}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
ENABLE_NSYS=${ENABLE_NSYS:-0}
NSYS_OUTPUT=${NSYS_OUTPUT:-}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}

TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-local}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
ENABLE_TP_COMM_OVERLAP=${ENABLE_TP_COMM_OVERLAP:-0}
ENABLE_SP=${ENABLE_SP:-1}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-0}
USE_BF16=${USE_BF16:-1}
USE_FP16=${USE_FP16:-0}

TP_SIZE=${TP_SIZE:-}
PP_SIZE=${PP_SIZE:-}
VPP_SIZE=${VPP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-}
EXPERT_TP_SIZE=${EXPERT_TP_SIZE:-}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
SEQ_LENGTH=${SEQ_LENGTH:-}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-}
NUM_LAYERS=${NUM_LAYERS:-}
PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-}
SCHEDULE_GROUP_SIZE=${SCHEDULE_GROUP_SIZE:-}
NUM_LAYERS_PER_VIRTUAL_STAGE=${NUM_LAYERS_PER_VIRTUAL_STAGE:-}

MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-4}
MOE_LAYER_FREQ=${MOE_LAYER_FREQ:-2}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-2}
ENABLE_PLANE_MAP=${ENABLE_PLANE_MAP:-0}
ATTENTION_TP_SIZE=${ATTENTION_TP_SIZE:-$TP_SIZE}
ATTENTION_CP_SIZE=${ATTENTION_CP_SIZE:-$CP_SIZE}
MOE_EP_SIZE=${MOE_EP_SIZE:-$EP_SIZE}
MOE_EXPERT_TP_SIZE=${MOE_EXPERT_TP_SIZE:-$EXPERT_TP_SIZE}

mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH" "$DATA_CACHE_PATH"

if [[ "$OBSERVABILITY_PRESET" == "basic" ]]; then
  ENABLE_LOG_TIMERS_TO_TENSORBOARD=1
  ENABLE_LOG_MEMORY_TO_TENSORBOARD=1
elif [[ "$OBSERVABILITY_PRESET" == "deep" ]]; then
  ENABLE_LOG_TIMERS_TO_TENSORBOARD=1
  ENABLE_LOG_MEMORY_TO_TENSORBOARD=1
  ENABLE_PROFILE=1
  ENABLE_PYTORCH_PROFILER=1
  PROFILE_RECORD_SHAPES=1
  ENABLE_MEMORY_HISTORY=1
  ENABLE_STRAGGLER_LOG=1
fi

if [[ -z "$WANDB_SAVE_DIR" ]] && [[ -n "$WANDB_PROJECT" ]]; then
  WANDB_SAVE_DIR="${CHECKPOINT_PATH}/wandb"
fi

if [[ ! "$MEMORY_SNAPSHOT_PATH" = /* ]]; then
  MEMORY_SNAPSHOT_PATH="${CHECKPOINT_PATH}/${MEMORY_SNAPSHOT_PATH}"
fi

if [[ -z "$NSYS_OUTPUT" ]]; then
  NSYS_OUTPUT="${CHECKPOINT_PATH}/nsys/profile"
fi

mkdir -p "$(dirname "$MEMORY_SNAPSHOT_PATH")" "$(dirname "$NSYS_OUTPUT")"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Error: ${label} not found at ${path}"
    exit 1
  fi
}

append_if_enabled() {
  local flag="$1"
  local enabled="$2"
  if [[ "$enabled" == "1" ]]; then
    EXTRA_PARALLEL_ARGS+=("$flag")
  fi
}

if [[ "$MODEL_TRACK" == "dense" ]]; then
  if [[ "$RUN_TARGET" == "dual_g4_g5" ]]; then
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-2}
  elif [[ "$RUN_TARGET" == "single_g4" ]]; then
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-2}
  else
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-1}
  fi
  NUM_LAYERS=${NUM_LAYERS:-40}
  SEQ_LENGTH=${SEQ_LENGTH:-1024}
  MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-40960}
  EP_SIZE=1
  EXPERT_TP_SIZE=1
elif [[ "$MODEL_TRACK" == "moe" ]]; then
  if [[ "$RUN_TARGET" == "dual_g4_g5" ]]; then
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-2}
  elif [[ "$RUN_TARGET" == "single_g4" ]]; then
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-2}
  else
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-1}
  fi
  EP_SIZE=${EP_SIZE:-2}
  EXPERT_TP_SIZE=${EXPERT_TP_SIZE:-1}
  NUM_LAYERS=${NUM_LAYERS:-8}
  SEQ_LENGTH=${SEQ_LENGTH:-512}
  MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}
fi

if [[ "$USE_BF16" == "1" ]] && [[ "$USE_FP16" == "1" ]]; then
  echo "Invalid precision preset: USE_BF16 and USE_FP16 cannot both be enabled"
  exit 1
fi
if [[ "$USE_BF16" != "1" ]] && [[ "$USE_FP16" != "1" ]]; then
  USE_BF16=1
fi
PRECISION_NAME=$([[ "$USE_BF16" == "1" ]] && echo bf16 || echo fp16)

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
PRODUCT=$((TP_SIZE * PP_SIZE * CP_SIZE * EP_SIZE * EXPERT_TP_SIZE))
if (( PRODUCT <= 0 || WORLD_SIZE % PRODUCT != 0 )); then
  echo "Invalid parallel product: WORLD_SIZE=${WORLD_SIZE}, TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}, EP=${EP_SIZE}, ETP=${EXPERT_TP_SIZE}"
  exit 1
fi

if (( VPP_SIZE > 1 )) && [[ -z "$PIPELINE_LAYOUT" ]]; then
  if (( PP_SIZE <= 1 )); then
    echo "Invalid preset: VPP requires PP > 1"
    exit 1
  fi
  TOTAL_VIRTUAL=$((PP_SIZE * VPP_SIZE))
  if (( NUM_LAYERS % TOTAL_VIRTUAL != 0 )); then
    echo "Invalid preset: NUM_LAYERS=${NUM_LAYERS} must be divisible by PP*VPP=${TOTAL_VIRTUAL}"
    exit 1
  fi
  if [[ -z "$NUM_LAYERS_PER_VIRTUAL_STAGE" ]]; then
    NUM_LAYERS_PER_VIRTUAL_STAGE=$((NUM_LAYERS / TOTAL_VIRTUAL))
  fi
fi

if [[ "$RUN_TARGET" == "single_g4" ]] && (( NUM_NODES != 1 )); then
  echo "single_g4 target requires NUM_NODES=1"
  exit 1
fi
if [[ "$RUN_TARGET" == "single_g5" ]] && (( NUM_NODES != 1 )); then
  echo "single_g5 target requires NUM_NODES=1"
  exit 1
fi
if [[ "$RUN_TARGET" == "dual_g4_g5" ]] && (( NUM_NODES < 2 )); then
  echo "dual_g4_g5 target requires NUM_NODES>=2"
  exit 1
fi

if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
  echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
  echo "Run this script from the Megatron-LM repository root, or set PRETRAIN_SCRIPT_PATH."
  exit 1
fi

USE_MOCK_DATA=0
if [[ "$DATA_PATH" == "MOCK" ]] || [[ "$TOKENIZER_MODEL" == "MOCK" ]] || [[ "$MODEL_TRACK" == "moe" ]]; then
  USE_MOCK_DATA=1
fi
if (( USE_MOCK_DATA == 0 )); then
  require_path "$TOKENIZER_MODEL" "tokenizer model"
  require_path "${DATA_PATH}.bin" "indexed dataset bin"
  require_path "${DATA_PATH}.idx" "indexed dataset idx"
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NUM_NODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

COMMON_MODEL_ARGS=(
  --use-mcore-models
  --transformer-impl "$TRANSFORMER_IMPL"
  --attention-backend "$ATTENTION_BACKEND"
  --num-layers "$NUM_LAYERS"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
  --position-embedding-type rope
  --no-rope-fusion
  --rotary-percent 1.0
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --disable-bias-linear
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --make-vocab-size-divisible-by 128
)

if [[ "$MODEL_TRACK" == "dense" ]]; then
  MODEL_ARGS=(
    --hidden-size 5120
    --ffn-hidden-size 17408
    --num-attention-heads 40
    --kv-channels 128
    --group-query-attention
    --num-query-groups 8
    --rotary-base 1000000
    --use-rotary-position-embeddings
    --qk-layernorm
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --vocab-size 151936
  )
else
  MODEL_ARGS=(
    --hidden-size 1024
    --ffn-hidden-size 4096
    --num-attention-heads 16
    --kv-channels 64
    --num-query-groups 4
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 32768
    --num-experts "$MOE_NUM_EXPERTS"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk "$MOE_ROUTER_TOPK"
    --moe-token-dispatcher-type alltoall
  )
fi

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
  --calculate-per-token-loss
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
)
if [[ "$USE_BF16" == "1" ]]; then
  TRAINING_ARGS+=(--bf16)
elif [[ "$USE_FP16" == "1" ]]; then
  TRAINING_ARGS+=(--fp16)
fi

if [[ "$MODEL_TRACK" == "dense" ]]; then
  TRAINING_ARGS+=(--recompute-granularity selective --recompute-activations --recompute-modules core_attn)
fi

EXTRA_PARALLEL_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --pipeline-model-parallel-size "$PP_SIZE"
  --context-parallel-size "$CP_SIZE"
  --expert-model-parallel-size "$EP_SIZE"
  --expert-tensor-parallel-size "$EXPERT_TP_SIZE"
)
if [[ -n "$PIPELINE_LAYOUT" ]]; then
  EXTRA_PARALLEL_ARGS+=(--pipeline-model-parallel-layout "$PIPELINE_LAYOUT")
elif (( PP_SIZE > 1 && VPP_SIZE > 1 )); then
  EXTRA_PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage "$NUM_LAYERS_PER_VIRTUAL_STAGE")
fi
if [[ -n "$SCHEDULE_GROUP_SIZE" ]]; then
  EXTRA_PARALLEL_ARGS+=(--microbatch-group-size-per-virtual-pipeline-stage "$SCHEDULE_GROUP_SIZE")
fi
append_if_enabled --sequence-parallel "$ENABLE_SP"
append_if_enabled --tp-comm-overlap "$ENABLE_TP_COMM_OVERLAP"

DATA_ARGS=()
if (( USE_MOCK_DATA > 0 )); then
  DATA_ARGS+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --split "$DATA_SPLIT"
    --data-cache-path "$DATA_CACHE_PATH"
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
    --split "$DATA_SPLIT"
    --data-cache-path "$DATA_CACHE_PATH"
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers "$NUM_WORKERS"
  )
fi

LOGGING_ARGS=(
  --log-interval "$LOG_INTERVAL"
  --timing-log-level 2
  --log-throughput
  --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
  --tensorboard-log-interval "$TENSORBOARD_LOG_INTERVAL"
  --ckpt-format torch_dist
  --distributed-timeout-minutes 60
)
if (( LOAD_CHECKPOINT > 0 )); then
  LOGGING_ARGS+=(--load "$CHECKPOINT_PATH")
fi
if (( EVAL_ITERS > 0 && EVAL_INTERVAL > 0 )); then
  LOGGING_ARGS+=(--eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL")
fi
if (( SAVE_INTERVAL > 0 )); then
  LOGGING_ARGS+=(--save "$CHECKPOINT_PATH" --save-interval "$SAVE_INTERVAL")
fi
if (( ENABLE_PROFILE > 0 )); then
  LOGGING_ARGS+=(--profile --profile-step-start "$PROFILE_STEP_START" --profile-step-end "$PROFILE_STEP_END")
  if [[ -n "$PROFILE_RANKS" ]]; then
    IFS=',' read -r -a PROFILE_RANK_ARRAY <<< "$PROFILE_RANKS"
    LOGGING_ARGS+=(--profile-ranks "${PROFILE_RANK_ARRAY[@]}")
  fi
fi
if (( ENABLE_LOG_TIMERS_TO_TENSORBOARD > 0 )); then
  LOGGING_ARGS+=(--log-timers-to-tensorboard)
fi
if (( ENABLE_LOG_MEMORY_TO_TENSORBOARD > 0 )); then
  LOGGING_ARGS+=(--log-memory-to-tensorboard)
fi
if (( ENABLE_PYTORCH_PROFILER > 0 )); then
  LOGGING_ARGS+=(--use-pytorch-profiler)
fi
if (( PROFILE_RECORD_SHAPES > 0 )); then
  LOGGING_ARGS+=(--pytorch-profiler-collect-shapes)
fi
if (( PROFILE_COLLECT_CALLSTACK > 0 )); then
  LOGGING_ARGS+=(--pytorch-profiler-collect-callstack)
fi
if (( PROFILE_COLLECT_CHAKRA > 0 )); then
  LOGGING_ARGS+=(--pytorch-profiler-collect-chakra)
fi
if (( ENABLE_MEMORY_HISTORY > 0 )); then
  LOGGING_ARGS+=(--record-memory-history --memory-snapshot-path "$MEMORY_SNAPSHOT_PATH")
fi
if (( ENABLE_STRAGGLER_LOG > 0 )); then
  LOGGING_ARGS+=(--log-straggler --straggler-minmax-count "$STRAGGLER_MINMAX_COUNT")
fi
if (( DISABLE_STRAGGLER_ON_STARTUP > 0 )); then
  LOGGING_ARGS+=(--disable-straggler-on-startup)
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  LOGGING_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_EXP_NAME" ]]; then
  LOGGING_ARGS+=(--wandb-exp-name "$WANDB_EXP_NAME")
fi
if [[ -n "$WANDB_SAVE_DIR" ]]; then
  LOGGING_ARGS+=(--wandb-save-dir "$WANDB_SAVE_DIR")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  LOGGING_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

echo "Launching Megatron program"
echo "  RUN_TARGET=${RUN_TARGET}"
echo "  MODEL_TRACK=${MODEL_TRACK}"
echo "  PRECISION=${PRECISION_NAME}"
echo "  WORLD_SIZE=${WORLD_SIZE} TP=${TP_SIZE} PP=${PP_SIZE} VPP=${VPP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${EXPERT_TP_SIZE}"
echo "  PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-<none>}"
echo "  SCHEDULE_GROUP_SIZE=${SCHEDULE_GROUP_SIZE:-<none>}"
echo "  ENABLE_PLANE_MAP=${ENABLE_PLANE_MAP} ATTN_TP=${ATTENTION_TP_SIZE} ATTN_CP=${ATTENTION_CP_SIZE} MOE_EP=${MOE_EP_SIZE} MOE_ETP=${MOE_EXPERT_TP_SIZE}"
echo "  OBSERVABILITY_PRESET=${OBSERVABILITY_PRESET} PROFILE=${ENABLE_PROFILE} PYTORCH_PROFILER=${ENABLE_PYTORCH_PROFILER} STRAGGLER=${ENABLE_STRAGGLER_LOG} MEMORY_HISTORY=${ENABLE_MEMORY_HISTORY} NSYS=${ENABLE_NSYS}"
echo "  TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH}"
echo "  TORCH_PROFILE_PATH=$(dirname "$TENSORBOARD_LOGS_PATH")/torch_profile"
echo "  MEMORY_SNAPSHOT_PATH=${MEMORY_SNAPSHOT_PATH}"
echo "  NSYS_OUTPUT=${NSYS_OUTPUT}"

set -x
TORCHRUN_CMD=(
  torchrun "${DISTRIBUTED_ARGS[@]}"
  "$PRETRAIN_SCRIPT_PATH"
  "${COMMON_MODEL_ARGS[@]}"
  "${MODEL_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${EXTRA_PARALLEL_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${LOGGING_ARGS[@]}"
)

if (( ENABLE_NSYS > 0 )); then
  nsys profile -s none -t "$NSYS_TRACE" -o "$NSYS_OUTPUT" --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end stop "${TORCHRUN_CMD[@]}"
else
  "${TORCHRUN_CMD[@]}"
fi
