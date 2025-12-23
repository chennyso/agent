from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def analyze_model_anatomy(model: nn.Module, *, topk: int = 6) -> Dict[str, Any]:
    name_by_id = {id(m): name for name, m in model.named_modules()}
    param_modules: List[Tuple[int, str, nn.Module]] = []
    for name, mod in model.named_modules():
        params = list(mod.parameters(recurse=False))
        if not params:
            continue
        total_bytes = 0
        for p in params:
            try:
                total_bytes += int(p.numel()) * int(p.element_size())
            except Exception:
                total_bytes += int(p.numel()) * 2
        param_modules.append((total_bytes, name, mod))
    param_modules.sort(reverse=True, key=lambda x: x[0])

    embedding_names = [name for _, name, mod in param_modules if isinstance(mod, nn.Embedding) or "embed" in name.lower()]
    lm_head_names = [name for _, name, _ in param_modules if "lm_head" in name.lower() or name.lower().endswith("lm_head")]

    large_linear_names: List[str] = []
    for bytes_total, name, mod in param_modules:
        if isinstance(mod, nn.Linear):
            large_linear_names.append(name)
        if len(large_linear_names) >= max(int(topk), 1):
            break

    comm_hotspots = sorted(set(embedding_names + lm_head_names + large_linear_names))

    narrow_linear: List[str] = []
    for _, name, mod in param_modules:
        if not isinstance(mod, nn.Linear):
            continue
        weight = getattr(mod, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        try:
            m, n = int(weight.shape[0]), int(weight.shape[1])
        except Exception:
            continue
        small = min(m, n)
        big = max(m, n)
        if small <= 128 or (small / max(big, 1)) <= 0.25:
            narrow_linear.append(name)
    narrow_linear = sorted(set(narrow_linear))

    layers = extract_transformer_layers(model)
    block_names: List[str] = []
    if layers is not None:
        for layer in layers:
            name = name_by_id.get(id(layer))
            if name:
                block_names.append(name)

    def _regex_from_names(names: List[str]) -> List[str]:
        out = []
        for name in names:
            parts = []
            for part in str(name).split("."):
                if part.isdigit():
                    parts.append(r"\d+")
                else:
                    parts.append(part)
            out.append(".".join(parts))
        return sorted(set(out))

    def _override_keys(names: List[str]) -> List[str]:
        keys: set[str] = set()
        for name in names:
            parts = str(name).split(".")
            if not parts:
                continue
            if len(parts) >= 2:
                keys.add(".".join(parts[-2:]))
            else:
                keys.add(parts[-1])
        return sorted(keys)

    anatomy = {
        "comm_hotspots": {
            "paths": comm_hotspots,
            "regex": _regex_from_names(comm_hotspots),
            "named_override_keys": _override_keys(comm_hotspots),
        },
        "latency_hotspots": {
            "paths": narrow_linear,
            "regex": _regex_from_names(narrow_linear),
            "named_override_keys": _override_keys(narrow_linear),
        },
        "stable_blocks": {
            "paths": block_names,
            "regex": _regex_from_names(block_names),
        },
        "notes": {
            "comm_hotspots_topk": int(topk),
            "narrow_linear_threshold": "min_dim<=128 or min/max<=0.25",
        },
    }
    return anatomy
