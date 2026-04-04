"""CLI batch-analysis command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.batch_analysis import BatchAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto


def _cmd_batch_analysis(args: argparse.Namespace) -> None:
    """Handle the 'batch-analysis' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = BatchAnalyzer(window_ms=args.window_ms)
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    summary = Table(title="Batch Size Impact Analysis")
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="right")

    summary.add_row("Total Requests", str(report.total_requests))
    summary.add_row("Total Batches", str(report.total_batches))
    summary.add_row("Window", f"{report.window_ms:.0f} ms")
    summary.add_row("Min Batch Size", str(report.min_batch_size))
    summary.add_row("Max Batch Size", str(report.max_batch_size))
    summary.add_row("Mean Batch Size", f"{report.mean_batch_size:.1f}")
    summary.add_row("Optimal Batch Size", f"[bold green]{report.optimal_batch_size}[/bold green]")

    console.print(summary)
    console.print()

    # Bucket detail table
    if report.buckets:
        detail = Table(title="Per-Batch-Size Statistics")
        detail.add_column("Size", justify="right")
        detail.add_column("Count", justify="right")
        detail.add_column("P50 TTFT", justify="right")
        detail.add_column("P95 TTFT", justify="right")
        detail.add_column("P50 Total", justify="right")
        detail.add_column("P95 Total", justify="right")
        detail.add_column("Throughput", justify="right")
        detail.add_column("Efficiency", justify="center")

        eff_styles = {
            "OPTIMAL": "bold green",
            "GOOD": "green",
            "ACCEPTABLE": "yellow",
            "POOR": "red",
        }

        for b in report.buckets:
            style = eff_styles.get(b.efficiency.value, "")
            detail.add_row(
                str(b.batch_size),
                str(b.count),
                f"{b.p50_ttft_ms:.1f} ms",
                f"{b.p95_ttft_ms:.1f} ms",
                f"{b.p50_total_latency_ms:.1f} ms",
                f"{b.p95_total_latency_ms:.1f} ms",
                f"{b.throughput_rps:.1f} rps",
                f"[{style}]{b.efficiency.value}[/{style}]",
            )

        console.print(detail)
        console.print()

    console.print(f"[bold]{report.recommendation}[/bold]")


def add_batch_analysis_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the batch-analysis subcommand parser."""
    parser = subparsers.add_parser(
        "batch-analysis",
        help="Analyze batch size impact on latency and throughput",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=100.0,
        help="Temporal grouping window in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_batch_analysis)
