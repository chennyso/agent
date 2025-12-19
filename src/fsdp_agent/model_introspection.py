from __future__ import annotations

from typing import Optional

import torch.nn as nn


def extract_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """尽量在常见 HF 结构中定位 Transformer block 列表。"""
    candidate_paths = [
        ("model", "layers"),  # Llama/Mistral/Qwen2/... (CausalLM.model.layers)
        ("model", "model", "layers"),  # wrapper
        ("model", "decoder", "layers"),  # OPT 等 decoder-only
        ("model", "model", "decoder", "layers"),
        ("transformer", "h"),  # GPT-2 family
        ("transformer", "blocks"),  # MPT
        ("gpt_neox", "layers"),  # GPT-NeoX
        ("layers",),  # 已经是 backbone
    ]
    for path in candidate_paths:
        cursor: nn.Module | None = model
        for attr in path:
            cursor = getattr(cursor, attr, None)
            if cursor is None:
                break
        if isinstance(cursor, nn.ModuleList) and len(cursor) > 0:
            return cursor
    return None

