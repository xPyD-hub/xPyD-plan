"""CLI arrival-pattern command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.arrival_pattern import ArrivalPatternAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto


def _cmd_arrival_pattern(args: argparse.Namespace) -> None:
    """Handle the 'arrival-pattern' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    burst_threshold = getattr(args, "burst_threshold", 0.1)

    analyzer = ArrivalPatternAnalyzer(burst_threshold=burst_threshold)
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    console.print(f"\n📊 Arrival Pattern: [bold]{report.pattern.value.upper()}[/bold]")
    console.print(f"   Confidence: {report.confidence:.1%}")
    console.print(f"   Requests: {report.request_count}, Duration: {report.duration_s:.1f}s")
    console.print(f"   Measured QPS: {report.measured_qps:.1f}\n")

    # Inter-arrival stats
    table = Table(title="Inter-Arrival Time Statistics")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    ia = report.inter_arrival
    table.add_row("Count", str(ia.count))
    table.add_row("Mean (ms)", f"{ia.mean_ms:.1f}")
    table.add_row("Std (ms)", f"{ia.std_ms:.1f}")
    table.add_row("CV", f"{ia.cv:.3f}")
    table.add_row("Min (ms)", f"{ia.min_ms:.1f}")
    table.add_row("P50 (ms)", f"{ia.p50_ms:.1f}")
    table.add_row("P95 (ms)", f"{ia.p95_ms:.1f}")
    table.add_row("P99 (ms)", f"{ia.p99_ms:.1f}")
    table.add_row("Max (ms)", f"{ia.max_ms:.1f}")

    console.print(table)

    if report.poisson_fit_p_value is not None:
        console.print(f"\nPoisson fit p-value: {report.poisson_fit_p_value:.4f}")

    if report.burst_info:
        console.print(f"\n🔥 Bursts: {report.burst_info.burst_count}")
        console.print(f"   Avg burst size: {report.burst_info.avg_burst_size:.1f}")
        console.print(f"   Burst fraction: {report.burst_info.burst_fraction:.1%}")


def add_arrival_pattern_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the arrival-pattern subcommand parser."""
    parser = subparsers.add_parser(
        "arrival-pattern",
        help="Analyze request arrival patterns (Poisson, bursty, periodic, uniform)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--burst-threshold",
        type=float,
        default=0.1,
        help="Burst detection threshold as fraction of mean IAT (default: 0.1)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_arrival_pattern)
