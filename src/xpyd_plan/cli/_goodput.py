"""CLI goodput command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.goodput import GoodputAnalyzer


def _cmd_goodput(args: argparse.Namespace) -> None:
    """Handle the 'goodput' subcommand."""
    console = Console()

    sla_ttft = getattr(args, "sla_ttft", None)
    sla_tpot = getattr(args, "sla_tpot", None)
    sla_total = getattr(args, "sla_total", None)

    data = load_benchmark_auto(args.benchmark)
    analyzer = GoodputAnalyzer(
        sla_ttft_ms=sla_ttft,
        sla_tpot_ms=sla_tpot,
        sla_total_latency_ms=sla_total,
        window_size=args.window_size,
    )
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    table = Table(title="Goodput Analysis")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Total Requests", str(report.total_requests))
    table.add_row("Passing Requests", str(report.passing_requests))
    table.add_row("Failing Requests", str(report.failing_requests))

    ratio_style = {
        "EXCELLENT": "green",
        "GOOD": "cyan",
        "FAIR": "yellow",
        "POOR": "red",
    }[report.grade.value]
    table.add_row(
        "Goodput Ratio",
        f"[{ratio_style}]{report.goodput_ratio:.1%}[/{ratio_style}]",
    )
    table.add_row("Grade", f"[{ratio_style}]{report.grade.value}[/{ratio_style}]")
    table.add_row("Raw QPS", f"{report.raw_qps:.1f}")
    table.add_row("Goodput QPS", f"{report.goodput_qps:.1f}")
    table.add_row("Worst Window Goodput", f"{report.worst_window_goodput:.1%}")

    fb = report.failure_breakdown
    table.add_row("TTFT Failures", str(fb.ttft_failures))
    table.add_row("TPOT Failures", str(fb.tpot_failures))
    table.add_row("Total Latency Failures", str(fb.total_latency_failures))
    table.add_row("Multi-metric Failures", str(fb.multi_metric_failures))

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_goodput_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the goodput subcommand parser."""
    parser = subparsers.add_parser(
        "goodput",
        help="Analyze effective throughput of SLA-compliant requests",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="TTFT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold in ms",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Time window size in seconds for windowed tracking (default: 5.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_goodput)
