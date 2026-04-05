"""CLI health-check command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.health_check import HealthChecker, HealthStatus


def _cmd_health_check(args: argparse.Namespace) -> None:
    """Handle the 'health-check' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    checker = HealthChecker(
        convergence_steps=getattr(args, "convergence_steps", 10),
        convergence_threshold=getattr(args, "convergence_threshold", 0.05),
        load_profile_window=getattr(args, "load_profile_window", 5.0),
        iqr_multiplier=getattr(args, "iqr_multiplier", 1.5),
    )
    report = checker.check(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Status color/emoji
    status_style = {
        HealthStatus.PASS: ("[green]PASS[/green]", "✅"),
        HealthStatus.WARN: ("[yellow]WARN[/yellow]", "⚠️"),
        HealthStatus.FAIL: ("[red]FAIL[/red]", "❌"),
    }

    styled, emoji = status_style[report.overall]
    console.print(f"\n{emoji} Overall: {styled}  ({report.request_count} requests)\n")

    table = Table(title="Health Checks")
    table.add_column("Check", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Detail", justify="left")

    for check in report.checks:
        s, e = status_style[check.status]
        table.add_row(check.name, f"{e} {s}", check.detail)

    console.print(table)


def add_health_check_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the health-check subcommand parser."""
    parser = subparsers.add_parser(
        "health-check",
        help="Composite readiness check: validation, convergence, load profile, outlier impact",
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
    parser.add_argument(
        "--convergence-steps",
        type=int,
        default=10,
        help="Number of convergence windows (default: 10)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.05,
        help="CV threshold for convergence stability (default: 0.05)",
    )
    parser.add_argument(
        "--load-profile-window",
        type=float,
        default=5.0,
        help="Window size in seconds for load profile (default: 5.0)",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.set_defaults(func=_cmd_health_check)
