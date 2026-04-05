"""CLI error-budget command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.error_budget import ErrorBudgetAnalyzer


def _cmd_error_budget(args: argparse.Namespace) -> None:
    """Handle the 'error-budget' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = ErrorBudgetAnalyzer(
        slo_target=args.slo_target,
        sla_ttft_ms=getattr(args, "sla_ttft", None),
        sla_tpot_ms=getattr(args, "sla_tpot", None),
        sla_total_latency_ms=getattr(args, "sla_total", None),
        window_size=args.window_size,
        warning_burn_rate=args.warning_burn_rate,
        critical_burn_rate=args.critical_burn_rate,
    )
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    table = Table(title="SLO Error Budget Burn Rate")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    level_styles = {
        "SAFE": "green",
        "WARNING": "yellow",
        "CRITICAL": "red",
        "EXHAUSTED": "bold red",
    }
    style = level_styles[report.overall_level.value]

    table.add_row("SLO Target", f"{report.config.slo_target:.3%}")
    table.add_row("Error Budget", f"{report.budget_status.total_budget:.3%}")
    table.add_row("Total Requests", str(report.total_requests))
    table.add_row("Total Failures", str(report.total_failures))
    table.add_row("Overall Error Rate", f"{report.overall_error_rate:.4%}")
    table.add_row(
        "Overall Burn Rate",
        f"[{style}]{report.overall_burn_rate:.1f}x[/{style}]",
    )
    table.add_row(
        "Level",
        f"[{style}]{report.overall_level.value}[/{style}]",
    )
    table.add_row(
        "Budget Consumed",
        f"{report.budget_status.consumed:.1%}",
    )
    table.add_row(
        "Budget Remaining",
        f"{report.budget_status.remaining:.1%}",
    )
    table.add_row(
        "Worst Window Burn Rate",
        f"{report.worst_window_burn_rate:.1f}x",
    )
    table.add_row(
        "Safe Windows",
        f"{report.safe_window_fraction:.0%}",
    )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the error-budget subcommand."""
    parser = subparsers.add_parser(
        "error-budget",
        help="Analyze SLO error budget burn rate",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--slo-target",
        type=float,
        default=0.999,
        help="SLO target (default: 0.999 for 99.9%%)",
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
        default=10.0,
        help="Time window size in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--warning-burn-rate",
        type=float,
        default=2.0,
        help="Burn rate threshold for WARNING (default: 2.0)",
    )
    parser.add_argument(
        "--critical-burn-rate",
        type=float,
        default=10.0,
        help="Burn rate threshold for CRITICAL (default: 10.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_error_budget)
