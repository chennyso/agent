from __future__ import annotations

import os
import shutil
import sys


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def main() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"[fail] torch import: {exc}", flush=True)
        sys.exit(1)

    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {getattr(torch.version, 'cuda', None)}")
    print(f"nccl_available: {torch.distributed.is_nccl_available() if hasattr(torch, 'distributed') else False}")
    if torch.cuda.is_available():
        print(f"device_count: {torch.cuda.device_count()}")
        print(f"device0: {torch.cuda.get_device_name(0)}")

    pipelining_ok = True
    tensor_parallel_ok = True
    try:
        import torch.distributed.pipelining  # noqa: F401
    except Exception:
        pipelining_ok = False
    try:
        import torch.distributed.tensor.parallel  # noqa: F401
    except Exception:
        tensor_parallel_ok = False
    print(f"torch.distributed.pipelining: {'ok' if pipelining_ok else 'missing'}")
    print(f"torch.distributed.tensor.parallel: {'ok' if tensor_parallel_ok else 'missing'}")

    try:
        import transformers  # noqa: F401

        import transformers as _t

        print(f"transformers: {_t.__version__}")
    except Exception as exc:
        print(f"transformers: missing ({exc})")

    print(f"tensorboard: {'ok' if _cmd_exists('tensorboard') else 'missing'}")
    print(f"nsys: {'ok' if _cmd_exists('nsys') else 'missing'}")
    print(f"nvidia-smi: {'ok' if _cmd_exists('nvidia-smi') else 'missing'}")

    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')}")


if __name__ == '__main__':
    main()

