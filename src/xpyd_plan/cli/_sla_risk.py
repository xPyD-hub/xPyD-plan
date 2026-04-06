"""CLI sla-risk command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.sla_risk import SLARiskScorer


def _cmd_sla_risk(args: argparse.Namespace) -> None:
    """Handle the 'sla-risk' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    scorer = SLARiskScorer()
    report = scorer.assess(
        data,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Risk score summary
    level_style = {
        "low": "[bold green]LOW[/bold green]",
        "moderate": "[yellow]MODERATE[/yellow]",
        "high": "[bold yellow]HIGH[/bold yellow]",
        "critical": "[bold red]CRITICAL[/bold red]",
    }
    level_val = report.risk_score.risk_level.value
    styled = level_style.get(level_val, level_val)
    console.print(
        f"\nSLA Risk Score: [bold]{report.risk_score.total_score:.1f}"
        f"[/bold] / 100 — {styled}"
    )
    console.print()

    # Factor table
    table = Table(title="Risk Factor Breakdown")
    table.add_column("Factor", justify="left")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right")
    table.add_column("Detail", justify="left")

    for f in report.factors:
        table.add_row(
            f.name,
            f"{f.score:.1f}",
            f"{f.weight:.0%}",
            f"{f.weighted_score:.1f}",
            f.detail,
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the sla-risk subcommand."""
    parser = subparsers.add_parser(
        "sla-risk",
        help="Composite SLA risk score combining headroom, tail, jitter, convergence, burn rate",
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
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
