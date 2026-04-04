"""CLI throughput command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.throughput import ThroughputAnalyzer


def _cmd_throughput(args: argparse.Namespace) -> None:
    """Handle the 'throughput' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = ThroughputAnalyzer(bucket_size=args.bucket_size)
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output — summary stats
    table = Table(title="Throughput Percentile Analysis")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    s = report.stats
    table.add_row("Duration", f"{report.duration_seconds:.1f}s")
    table.add_row("Total Requests", str(report.total_requests))
    table.add_row("Buckets", f"{report.total_buckets} × {report.bucket_size}s")
    table.add_row("Min RPS", f"{s.min_rps:.1f}")
    table.add_row("P5 RPS (sustainable)", f"{s.p5_rps:.1f}")
    table.add_row("Mean RPS", f"{s.mean_rps:.1f}")
    table.add_row("P50 RPS", f"{s.p50_rps:.1f}")
    table.add_row("P95 RPS", f"{s.p95_rps:.1f}")
    table.add_row("P99 RPS", f"{s.p99_rps:.1f}")
    table.add_row("Max RPS", f"{s.max_rps:.1f}")
    table.add_row("CV", f"{s.cv:.3f}")

    stab_style = {
        "STABLE": "green",
        "VARIABLE": "yellow",
        "UNSTABLE": "red",
    }[s.stability.value]
    table.add_row("Stability", f"[{stab_style}]{s.stability.value}[/{stab_style}]")
    table.add_row("Bottleneck Buckets", str(len(report.bottleneck_buckets)))

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_throughput_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the throughput subcommand parser."""
    parser = subparsers.add_parser(
        "throughput",
        help="Analyze per-second throughput distribution",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--bucket-size",
        type=float,
        default=1.0,
        help="Time bucket size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_throughput)
