"""CLI cold-start command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.cold_start import ColdStartDetector, ColdStartSeverity


def _cmd_cold_start(args: argparse.Namespace) -> None:
    """Handle the 'cold-start' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    detector = ColdStartDetector()
    report = detector.detect(
        data,
        warmup_window=args.warmup_window,
        threshold=args.threshold,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Cold Start Detection")
    table.add_column("Metric", justify="left")
    table.add_column("Warmup P50", justify="right")
    table.add_column("Warmup P95", justify="right")
    table.add_column("Steady P50", justify="right")
    table.add_column("Steady P95", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Stabilizes At", justify="right")
    table.add_column("Severity", justify="center")

    for m in report.metrics:
        style = {
            ColdStartSeverity.NONE: "green",
            ColdStartSeverity.MILD: "yellow",
            ColdStartSeverity.MODERATE: "bold yellow",
            ColdStartSeverity.SEVERE: "bold red",
        }[m.severity]

        table.add_row(
            m.metric,
            f"{m.warmup_p50:.2f}",
            f"{m.warmup_p95:.2f}",
            f"{m.steady_p50:.2f}",
            f"{m.steady_p95:.2f}",
            f"[{style}]{m.ratio:.2f}×[/{style}]",
            f"req #{m.stabilization_index}",
            f"[{style}]{m.severity.value}[/{style}]",
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_cold_start_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the cold-start subcommand parser."""
    parser = subparsers.add_parser(
        "cold-start",
        help="Detect cold start effects in benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--warmup-window",
        type=int,
        default=10,
        help="Number of initial requests to consider as warmup (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Detection threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_cold_start)
