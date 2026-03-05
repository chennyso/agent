#!/usr/bin/env bash
set -euo pipefail

TRACE_DIR="${1:-tb_traces}"
PORT="${2:-6006}"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "tensorboard not found in PATH" >&2
  exit 1
fi

tensorboard --logdir "${TRACE_DIR}" --port "${PORT}" --bind_all

