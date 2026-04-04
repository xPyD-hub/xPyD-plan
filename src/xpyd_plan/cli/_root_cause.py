"""CLI root-cause command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.root_cause import FactorSignificance, RootCauseAnalyzer


def _cmd_root_cause(args: argparse.Namespace) -> None:
    """Handle the 'root-cause' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = RootCauseAnalyzer()
    report = analyzer.analyze(data, args.sla_metric, args.sla_threshold)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print("[bold]Root Cause Analysis[/bold]")
    console.print(
        f"Total: {report.total_requests} | "
        f"Pass: {report.passing_requests} | "
        f"Fail: {report.failing_requests} | "
        f"Failure rate: {report.failure_rate:.1%}"
    )
    console.print(
        f"SLA: {report.sla_metric} ≤ {report.sla_threshold_ms}ms"
    )
    console.print()

    if not report.factors:
        console.print(f"[bold]{report.recommendation}[/bold]")
        return

    # Factor table
    table = Table(title="Contributing Factors")
    table.add_column("Factor", justify="left")
    table.add_column("Significance", justify="center")
    table.add_column("p-value", justify="right")
    table.add_column("Effect Size", justify="right")
    table.add_column("Pass Mean", justify="right")
    table.add_column("Fail Mean", justify="right")

    significance_style = {
        FactorSignificance.HIGH: "bold red",
        FactorSignificance.MEDIUM: "dark_orange",
        FactorSignificance.LOW: "yellow",
        FactorSignificance.NONE: "green",
    }

    for f in report.factors:
        style = significance_style[f.significance]
        table.add_row(
            f.factor,
            f"[{style}]{f.significance.value}[/{style}]",
            f"{f.p_value:.4f}",
            f"{f.effect_size:+.3f}",
            f"{f.pass_mean:.1f}",
            f"{f.fail_mean:.1f}",
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_root_cause_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the root-cause subcommand parser."""
    parser = subparsers.add_parser(
        "root-cause",
        help="Analyze root causes of SLA violations in benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--sla-metric",
        choices=["ttft", "tpot", "total_latency"],
        default="ttft",
        help="SLA metric to check (default: ttft)",
    )
    parser.add_argument(
        "--sla-threshold",
        type=float,
        default=100.0,
        help="SLA threshold in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_root_cause)
