#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/dist/wheelhouse}"

mkdir -p "${OUT_DIR}"

echo "[offline] downloading Python wheels into ${OUT_DIR}"
python -m pip download -d "${OUT_DIR}" -r "${ROOT_DIR}/requirements-offline.txt"

echo "[offline] building local torchtitan wheel"
python -m pip wheel -w "${OUT_DIR}" "${ROOT_DIR}"

echo "[offline] wheelhouse ready at ${OUT_DIR}"
