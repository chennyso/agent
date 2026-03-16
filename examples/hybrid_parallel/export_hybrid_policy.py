from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_policy import apply_hybrid_policy_to_config, load_hybrid_policy
from train_manual_pp import _load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--format", type=str, default="torchtitan", choices=["manual", "torchtitan", "both"])
    ap.add_argument("--total_layers", type=int, default=0)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    policy = load_hybrid_policy(cfg, config_path=args.config)
    if policy is None:
        raise ValueError("config has no hybrid_policy section")

    total_layers = int(args.total_layers) or None
    payload = {}
    if args.format in {"manual", "both"}:
        manual_cfg, manual_warnings = apply_hybrid_policy_to_config(cfg, config_path=args.config)
        payload["manual"] = {
            "config": manual_cfg,
            "warnings": manual_warnings,
        }
    if args.format in {"torchtitan", "both"}:
        overrides, warnings = policy.export_torchtitan(total_layers=total_layers)
        payload["torchtitan"] = {
            "overrides": overrides,
            "warnings": warnings,
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
