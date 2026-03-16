#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if env | grep -Eiq '^(http|https|all)_proxy='; then
  echo "[install] proxy environment detected; if pip reports 'Missing dependencies for SOCKS support',"
  echo "[install] either unset the proxy variables for this shell or preinstall PySocks."
fi

python -m pip install -e "${ROOT_DIR}"
