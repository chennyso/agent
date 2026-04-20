#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megatron_agent.pipeline_visualization import load_panels_from_paths, save_pipeline_timeline_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render paper-style PP/VPP schedule timelines from pipeline_event_trace artifacts."
    )
    parser.add_argument(
        "--trace",
        dest="traces",
        action="append",
        required=True,
        help="Path to a pipeline_event_trace.json file or a trial/context artifact containing visualization_artifacts.pipeline_event_trace.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Optional panel label. Repeat to match --trace order.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output image path. Use .png or .svg.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pipeline Schedule Timeline",
        help="Figure title.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable tile labels like F0:0 / B0:0 inside blocks.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster DPI when writing PNG.",
    )
    args = parser.parse_args()

    panels = load_panels_from_paths(args.traces, labels=args.labels)
    output = save_pipeline_timeline_figure(
        panels,
        args.out,
        title=args.title,
        annotate=not bool(args.no_annotate),
        dpi=max(int(args.dpi), 72),
    )
    print(f"Rendered {len(panels)} pipeline panel(s) to {output}")


if __name__ == "__main__":
    main()
