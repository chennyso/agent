#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE_DIR="${1:?usage: install_offline_from_wheelhouse.sh <wheelhouse_dir> [package_name]}"
PACKAGE_NAME="${2:-torchtitan}"

python -m pip install --no-index --find-links "${WHEELHOUSE_DIR}" "${PACKAGE_NAME}"
