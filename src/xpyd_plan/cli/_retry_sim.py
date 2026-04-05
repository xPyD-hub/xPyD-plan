"""CLI retry-sim command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.retry_sim import BackoffType, RetryConfig, RetrySimulator


def _cmd_retry_sim(args: argparse.Namespace) -> None:
    """Handle the 'retry-sim' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    config = RetryConfig(
        max_retries=args.max_retries,
        retry_threshold_ttft_ms=args.retry_threshold_ttft,
        retry_threshold_tpot_ms=args.retry_threshold_tpot,
        retry_threshold_total_ms=args.retry_threshold_total,
        backoff_ms=args.backoff_ms,
        backoff_type=BackoffType(args.backoff_type),
    )
    simulator = RetrySimulator(config)
    report = simulator.simulate(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    table = Table(title="Retry Simulation Results")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    la = report.load_amplification
    table.add_row("Original Requests", str(la.original_requests))
    table.add_row("Total With Retries", str(la.total_requests_with_retries))
    table.add_row("Retry Attempts", str(la.total_retry_attempts))
    table.add_row("Retry Rate", f"{la.retry_rate:.1%}")
    table.add_row("Amplification Factor", f"{la.amplification_factor:.2f}x")
    table.add_row("Original Goodput", f"{report.original_goodput:.1%}")
    table.add_row("Effective Goodput", f"{report.effective_goodput:.1%}")

    console.print(table)

    # Latency impact table
    if report.latency_impact:
        lat_table = Table(title="Latency Impact")
        lat_table.add_column("Metric")
        lat_table.add_column("Orig P50", justify="right")
        lat_table.add_column("Orig P95", justify="right")
        lat_table.add_column("Orig P99", justify="right")
        lat_table.add_column("Post P50", justify="right")
        lat_table.add_column("Post P95", justify="right")
        lat_table.add_column("Post P99", justify="right")

        for impact in report.latency_impact:
            lat_table.add_row(
                impact.metric,
                f"{impact.original_p50:.1f}",
                f"{impact.original_p95:.1f}",
                f"{impact.original_p99:.1f}",
                f"{impact.post_retry_p50:.1f}",
                f"{impact.post_retry_p95:.1f}",
                f"{impact.post_retry_p99:.1f}",
            )

        console.print()
        console.print(lat_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_retry_sim_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the retry-sim subcommand parser."""
    parser = subparsers.add_parser(
        "retry-sim",
        help="Simulate request retry impact on latency and load",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per request (default: 3)",
    )
    parser.add_argument(
        "--retry-threshold-ttft",
        type=float,
        default=None,
        help="TTFT threshold triggering retry (ms)",
    )
    parser.add_argument(
        "--retry-threshold-tpot",
        type=float,
        default=None,
        help="TPOT threshold triggering retry (ms)",
    )
    parser.add_argument(
        "--retry-threshold-total",
        type=float,
        default=None,
        help="Total latency threshold triggering retry (ms)",
    )
    parser.add_argument(
        "--backoff-ms",
        type=float,
        default=100.0,
        help="Base backoff delay in ms (default: 100.0)",
    )
    parser.add_argument(
        "--backoff-type",
        choices=["constant", "exponential"],
        default="constant",
        help="Backoff strategy (default: constant)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_retry_sim)
