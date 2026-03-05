from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

_SEMANTIC_HASH_SALT = "megatron_strategy_v1"


@dataclass
class MegatronParallelSpec:
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    cp_degree: int = 1
    sp_enabled: bool = False


@dataclass
class MegatronStrategy:
    parallel: MegatronParallelSpec = field(default_factory=MegatronParallelSpec)
    micro_batch_size: int = 1
    global_batch_size: int = 8
    seq_len: int = 2048
    use_bf16: bool = True
    use_fp16: bool = False
    recompute_granularity: Optional[str] = None
    extra_args: Optional[List[str]] = None
    schema_version: int = 1

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "parallel": asdict(self.parallel),
            "micro_batch_size": int(self.micro_batch_size),
            "global_batch_size": int(self.global_batch_size),
            "seq_len": int(self.seq_len),
            "use_bf16": bool(self.use_bf16),
            "use_fp16": bool(self.use_fp16),
            "recompute_granularity": self.recompute_granularity,
            "extra_args": list(self.extra_args or []),
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "MegatronStrategy":
        parallel_raw = payload.get("parallel") or {}
        parallel = MegatronParallelSpec(
            tp_degree=int(parallel_raw.get("tp_degree", 1) or 1),
            pp_degree=int(parallel_raw.get("pp_degree", 1) or 1),
            ep_degree=int(parallel_raw.get("ep_degree", 1) or 1),
            cp_degree=int(parallel_raw.get("cp_degree", 1) or 1),
            sp_enabled=bool(parallel_raw.get("sp_enabled", False)),
        )
        return cls(
            parallel=parallel,
            micro_batch_size=int(payload.get("micro_batch_size", 1) or 1),
            global_batch_size=int(payload.get("global_batch_size", 8) or 1),
            seq_len=int(payload.get("seq_len", 2048) or 1),
            use_bf16=bool(payload.get("use_bf16", True)),
            use_fp16=bool(payload.get("use_fp16", False)),
            recompute_granularity=payload.get("recompute_granularity"),
            extra_args=list(payload.get("extra_args") or []),
            schema_version=int(payload.get("schema_version", 1) or 1),
        )

    def normalized(self) -> "MegatronStrategy":
        norm = copy.deepcopy(self)
        if norm.use_bf16:
            norm.use_fp16 = False
        norm.parallel.tp_degree = max(int(norm.parallel.tp_degree), 1)
        norm.parallel.pp_degree = max(int(norm.parallel.pp_degree), 1)
        norm.parallel.ep_degree = max(int(norm.parallel.ep_degree), 1)
        norm.parallel.cp_degree = max(int(norm.parallel.cp_degree), 1)
        norm.micro_batch_size = max(int(norm.micro_batch_size), 1)
        norm.global_batch_size = max(int(norm.global_batch_size), 1)
        norm.seq_len = max(int(norm.seq_len), 1)
        if norm.extra_args is not None:
            norm.extra_args = [str(x) for x in norm.extra_args if str(x).strip()]
        return norm

    def semantic_hash(self) -> str:
        norm = self.normalized()
        payload = norm.to_dict()
        payload["_hash_salt"] = _SEMANTIC_HASH_SALT
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def validate_strategy(strategy: MegatronStrategy) -> MegatronStrategy:
    s = strategy.normalized()
    if s.parallel.sp_enabled and s.parallel.tp_degree <= 1:
        raise ValueError("sequence parallel requires tp_degree > 1")
    if s.parallel.tp_degree * s.parallel.pp_degree * s.parallel.ep_degree * s.parallel.cp_degree <= 0:
        raise ValueError("parallel degrees must be positive")
    return s
