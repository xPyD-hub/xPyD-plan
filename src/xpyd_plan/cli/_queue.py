"""CLI queue command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.queue_analysis import QueueAnalyzer


def _cmd_queue(args: argparse.Namespace) -> None:
    """Handle the 'queue' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = QueueAnalyzer()
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        result = report.model_dump()
        # Exclude verbose profile points from JSON by default
        if not getattr(args, "include_points", False):
            result["concurrency"]["points"] = []
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output — queue stats
    table = Table(title="Request Queuing Time Analysis")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Total Requests", str(report.total_requests))
    table.add_row("Duration", f"{report.duration_seconds:.1f}s")

    q = report.queue_stats
    table.add_row("Min Queue Delay", f"{q.min_queue_ms:.1f} ms")
    table.add_row("Mean Queue Delay", f"{q.mean_queue_ms:.1f} ms")
    table.add_row("P50 Queue Delay", f"{q.p50_queue_ms:.1f} ms")
    table.add_row("P95 Queue Delay", f"{q.p95_queue_ms:.1f} ms")
    table.add_row("P99 Queue Delay", f"{q.p99_queue_ms:.1f} ms")
    table.add_row("Max Queue Delay", f"{q.max_queue_ms:.1f} ms")
    table.add_row("Queue Ratio", f"{q.queue_ratio:.1%}")

    c = report.concurrency
    table.add_row("Peak Concurrency", str(c.peak_concurrency))
    table.add_row("Mean Concurrency", f"{c.mean_concurrency:.1f}")
    table.add_row("P95 Concurrency", f"{c.p95_concurrency:.1f}")

    cong_style = {
        "LOW": "green",
        "MODERATE": "yellow",
        "HIGH": "red",
        "CRITICAL": "bold red",
    }[report.congestion_level.value]
    table.add_row(
        "Congestion",
        f"[{cong_style}]{report.congestion_level.value}[/{cong_style}]",
    )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the queue subcommand parser."""
    parser = subparsers.add_parser(
        "queue",
        help="Analyze request queuing times and concurrency",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--include-points",
        action="store_true",
        help="Include concurrency profile points in JSON output",
    )
    parser.set_defaults(func=_cmd_queue)
