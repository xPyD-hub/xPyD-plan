"""CLI sla-headroom command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.sla_headroom import SLAHeadroomCalculator


def _cmd_sla_headroom(args: argparse.Namespace) -> None:
    """Handle the 'sla-headroom' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    calc = SLAHeadroomCalculator()
    report = calc.calculate(
        data,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
        percentile=args.percentile,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not report.metrics:
        console.print("[yellow]No SLA thresholds configured.[/yellow]")
        return

    table = Table(title=f"SLA Headroom Analysis (P{args.percentile})")
    table.add_column("Metric", justify="left")
    table.add_column("SLA (ms)", justify="right")
    table.add_column("Actual (ms)", justify="right")
    table.add_column("Headroom (ms)", justify="right")
    table.add_column("Headroom (%)", justify="right")
    table.add_column("Status", justify="center")

    for m in report.metrics:
        status_style = {
            "critical": "[bold red]CRITICAL[/bold red]",
            "tight": "[yellow]TIGHT[/yellow]",
            "adequate": "[green]ADEQUATE[/green]",
            "comfortable": "[bold green]COMFORTABLE[/bold green]",
        }
        table.add_row(
            m.metric,
            f"{m.sla_threshold_ms:.1f}",
            f"{m.actual_ms:.1f}",
            f"{m.headroom_ms:.1f}",
            f"{m.headroom_pct:.1f}%",
            status_style.get(m.safety_level.value, m.safety_level.value),
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_sla_headroom_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the sla-headroom subcommand parser."""
    parser = subparsers.add_parser(
        "sla-headroom",
        help="Calculate SLA headroom margins for benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
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
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile to evaluate (default: 95.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_sla_headroom)
