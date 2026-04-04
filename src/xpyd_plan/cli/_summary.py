"""CLI summary command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.summary import SummaryGenerator


def _cmd_summary(args: argparse.Namespace) -> None:
    """Handle the 'summary' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    generator = SummaryGenerator(data)
    report = generator.generate()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Overview table
    overview = Table(title="Benchmark Summary")
    overview.add_column("Property", justify="left")
    overview.add_column("Value", justify="right")

    overview.add_row("Requests", str(report.request_count))
    overview.add_row("Duration", f"{report.duration_seconds:.1f}s")
    overview.add_row("Measured QPS", f"{report.measured_qps:.1f}")
    overview.add_row("P:D Ratio", report.pd_ratio)
    overview.add_row("Prefill Instances", str(report.num_prefill_instances))
    overview.add_row("Decode Instances", str(report.num_decode_instances))

    console.print(overview)

    # Token stats table
    token_table = Table(title="Token Distribution")
    token_table.add_column("Metric", justify="left")
    token_table.add_column("Min", justify="right")
    token_table.add_column("Mean", justify="right")
    token_table.add_column("P50", justify="right")
    token_table.add_column("P95", justify="right")
    token_table.add_column("Max", justify="right")

    for name, stats in [("Prompt Tokens", report.prompt_tokens),
                         ("Output Tokens", report.output_tokens)]:
        token_table.add_row(
            name,
            str(stats.min),
            f"{stats.mean:.1f}",
            f"{stats.p50:.1f}",
            f"{stats.p95:.1f}",
            str(stats.max),
        )

    console.print(token_table)

    # Latency table
    lat_table = Table(title="Latency Overview")
    lat_table.add_column("Metric", justify="left")
    lat_table.add_column("Min", justify="right")
    lat_table.add_column("Mean", justify="right")
    lat_table.add_column("P50", justify="right")
    lat_table.add_column("P95", justify="right")
    lat_table.add_column("P99", justify="right")
    lat_table.add_column("Max", justify="right")

    for name, lat in [("TTFT", report.ttft),
                       ("TPOT", report.tpot),
                       ("Total Latency", report.total_latency)]:
        lat_table.add_row(
            name,
            f"{lat.min_ms:.1f}",
            f"{lat.mean_ms:.1f}",
            f"{lat.p50_ms:.1f}",
            f"{lat.p95_ms:.1f}",
            f"{lat.p99_ms:.1f}",
            f"{lat.max_ms:.1f}",
        )

    console.print(lat_table)


def add_summary_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the summary subcommand parser."""
    parser = subparsers.add_parser(
        "summary",
        help="Quick overview of a benchmark file (requests, tokens, latency stats)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_summary)
