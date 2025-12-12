from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class HardwareInfo:
    """硬件/拓扑摘要，供 Agent 提示词和 mesh 选择使用。"""

    num_nodes: int = 1
    gpus_per_node: int = 4
    gpu_name: str = "Unknown"
    memory_gb: float = 0.0
    interconnect: str = "NVLink"  # 可选 NVLink/PCIe/InfiniBand
    mesh_shape: Optional[List[int]] = None
    notes: str = ""


def detect_hardware() -> HardwareInfo:
    """尽力探测本机 GPU 信息；失败时返回占位。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return HardwareInfo(gpu_name="No CUDA", gpus_per_node=0, memory_gb=0.0, notes="CUDA unavailable")
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return HardwareInfo(
            num_nodes=1,
            gpus_per_node=n,
            gpu_name=name,
            memory_gb=mem,
            interconnect="NVLink",
        )
    except Exception:
        return HardwareInfo(notes="hardware detection failed")


def load_hardware_info(path: str) -> HardwareInfo:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return HardwareInfo(**payload)


def write_hardware_info(info: HardwareInfo, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2, ensure_ascii=False)
