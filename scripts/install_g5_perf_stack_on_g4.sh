#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-${ROOT_DIR}/Megatron-LM}"
ENV_PREFIX="${ENV_PREFIX:-/public/home/ssjxscy/envs/torchdist_ok}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
APEX_REPO_DIR="${APEX_REPO_DIR:-/public/home/ssjxscy/src/apex}"
APEX_REF="${APEX_REF:-}"
MAX_JOBS="${MAX_JOBS:-4}"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Error: shared env not found: ${ENV_PREFIX}"
  exit 1
fi

if [[ ! -d "${MEGATRON_ROOT}" ]]; then
  echo "Error: Megatron-LM root not found: ${MEGATRON_ROOT}"
  exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Error: nvcc not found under CUDA_HOME=${CUDA_HOME}"
  exit 1
fi

source "${ENV_PREFIX}/bin/activate"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found in PATH after activating ${ENV_PREFIX}. Install uv in the shared environment before running this script."
  exit 1
fi

export CUDA_HOME
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS

echo "Using shared env: ${ENV_PREFIX}"
echo "Using Megatron root: ${MEGATRON_ROOT}"
echo "Using CUDA_HOME: ${CUDA_HOME}"

readarray -t TE_SOURCE < <(
  python - <<'PY' "${MEGATRON_ROOT}/pyproject.toml"
import sys
import tomllib
from pathlib import Path

payload = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = payload["tool"]["uv"]["sources"]["transformer-engine"]
print(source["git"])
print(source["rev"])
PY
)

TE_GIT="${TE_SOURCE[0]}"
TE_REV="${TE_SOURCE[1]}"

cd "${MEGATRON_ROOT}"
uv pip install --group build
uv pip install --no-build-isolation "git+${TE_GIT}@${TE_REV}"

if [[ ! -d "${APEX_REPO_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA/apex.git "${APEX_REPO_DIR}"
else
  git -C "${APEX_REPO_DIR}" fetch --tags --prune origin
fi

if [[ -n "${APEX_REF}" ]]; then
  git -C "${APEX_REPO_DIR}" checkout "${APEX_REF}"
else
  if git -C "${APEX_REPO_DIR}" rev-parse --verify origin/main >/dev/null 2>&1; then
    git -C "${APEX_REPO_DIR}" checkout main
    git -C "${APEX_REPO_DIR}" pull --ff-only origin main
  else
    git -C "${APEX_REPO_DIR}" checkout master
    git -C "${APEX_REPO_DIR}" pull --ff-only origin master
  fi
fi

cd "${APEX_REPO_DIR}"
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .

python - <<'PY'
import transformer_engine
import apex
from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam  # noqa: F401
from apex.optimizers import FusedAdam as ApexFusedAdam  # noqa: F401

print("Transformer Engine:", getattr(transformer_engine, "__version__", "unknown"))
print("Apex:", getattr(apex, "__file__", ""))
PY
