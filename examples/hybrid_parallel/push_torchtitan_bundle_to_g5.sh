#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE_DIR="${1:-torchtitan/dist/wheelhouse}"
REMOTE_HOST="${REMOTE_HOST:-g5}"
REMOTE_DIR="${REMOTE_DIR:-~/agent/torchtitan/dist/wheelhouse}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found; falling back to scp"
  scp -r "${WHEELHOUSE_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"
  exit 0
fi

rsync -av "${WHEELHOUSE_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
