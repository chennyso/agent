#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if env | grep -Eiq '^(http|https|all)_proxy='; then
  echo "[install] proxy environment detected; if pip reports 'Missing dependencies for SOCKS support',"
  echo "[install] either unset the proxy variables for this shell or preinstall PySocks."
fi

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("setuptools", "wheel") if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write(
        "[install] missing local build prerequisites: "
        + ", ".join(missing)
        + "\n"
    )
    sys.stderr.write(
        "[install] install them first, or use the offline wheelhouse flow.\n"
    )
    raise SystemExit(1)
PY

python -m pip install --no-build-isolation -e "${ROOT_DIR}"
