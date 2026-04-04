"""CLI decompose command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.decomposer import BottleneckType, LatencyDecomposer


def _cmd_decompose(args: argparse.Namespace) -> None:
    """Handle the 'decompose' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    threshold = getattr(args, "bottleneck_threshold", 0.5)
    decomposer = LatencyDecomposer(data, bottleneck_threshold=threshold)
    report = decomposer.analyze()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Phase stats table
    table = Table(title="Latency Decomposition")
    table.add_column("Phase", justify="left")
    table.add_column("Mean Fraction", justify="right")
    table.add_column("P50 Fraction", justify="right")
    table.add_column("P95 Fraction", justify="right")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")

    for ps in report.phase_stats:
        table.add_row(
            ps.phase.capitalize(),
            f"{ps.mean_fraction:.1%}",
            f"{ps.p50_fraction:.1%}",
            f"{ps.p95_fraction:.1%}",
            f"{ps.mean_ms:.1f}",
            f"{ps.p50_ms:.1f}",
            f"{ps.p95_ms:.1f}",
        )

    console.print(table)

    # Bottleneck
    style = {
        BottleneckType.PREFILL_BOUND: "bold yellow",
        BottleneckType.DECODE_BOUND: "bold cyan",
        BottleneckType.OVERHEAD_BOUND: "bold red",
        BottleneckType.BALANCED: "green",
    }[report.bottleneck]

    console.print(
        f"\n[bold]Bottleneck:[/bold] [{style}]{report.bottleneck.value}[/{style}]"
    )
    console.print(f"[dim]{report.recommendation}[/dim]")


def add_decompose_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the decompose subcommand parser."""
    parser = subparsers.add_parser(
        "decompose",
        help="Decompose latency into prefill, decode, and overhead phases",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--bottleneck-threshold",
        type=float,
        default=0.5,
        help="Fraction threshold for bottleneck classification (default: 0.5)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_decompose)
