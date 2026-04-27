from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megatron_agent.rac_vpp_validation import run_validation_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RAC-VPP validation tables and figures for the first two paper observations."
    )
    parser.add_argument("--out-dir", type=str, default="./runs/rac_vpp_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_validation_suite(Path(args.out_dir))
    print(json.dumps({"manifest": manifest["manifest_path"], "figures": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
