#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os

from torchtitan.experiments.hybrid_policy.config_registry import qwen3_hybrid_demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=False, default="")
    args = ap.parse_args()

    if args.policy:
        os.environ["HYBRID_POLICY_PATH"] = args.policy

    cfg = qwen3_hybrid_demo()
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
