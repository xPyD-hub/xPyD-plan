"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Recommend optimal Prefill:Decode node ratio for xPyD proxy",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to benchmark results (JSON) from xpyd-bench",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file for offline estimation (without running benchmark)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total number of available nodes/GPUs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for recommendation (JSON)",
    )

    args = parser.parse_args(argv)

    if not args.data and not args.dataset:
        parser.error("Either --data or --dataset is required")

    from xpyd_plan import __version__

    # TODO: implement planner logic
    print(f"xpyd-plan v{__version__}")
    print(f"  Data:    {args.data}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Budget:  {args.budget} nodes")
    print()
    print("Planner not yet implemented.")
