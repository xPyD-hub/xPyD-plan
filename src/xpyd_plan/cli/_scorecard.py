"""CLI scorecard command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import SLAConfig
from xpyd_plan.scorecard import ScorecardCalculator, ScoreGrade


def _cmd_scorecard(args: argparse.Namespace) -> None:
    """Handle the 'scorecard' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    sla = SLAConfig(
        ttft_ms=getattr(args, "sla_ttft", None),
        tpot_ms=getattr(args, "sla_tpot", None),
        max_latency_ms=getattr(args, "sla_total", None),
    )

    analyzer = BenchmarkAnalyzer()
    analyzer._data = data
    total_instances = data.metadata.total_instances
    analysis = analyzer.find_optimal_ratio(total_instances, sla)

    calc = ScorecardCalculator(
        sla_weight=args.sla_weight,
        utilization_weight=args.util_weight,
        waste_weight=args.waste_weight,
    )
    report = calc.score(analysis)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    table = Table(title="Efficiency Scorecard")
    table.add_column("Ratio", justify="left")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")
    table.add_column("SLA", justify="center")
    table.add_column("SLA Score", justify="right")
    table.add_column("Util Score", justify="right")
    table.add_column("Waste Score", justify="right")

    for sc in report.scorecards:
        grade_style = {
            ScoreGrade.A: "bold green",
            ScoreGrade.B: "green",
            ScoreGrade.C: "yellow",
            ScoreGrade.D: "red",
            ScoreGrade.F: "bold red",
        }[sc.grade]

        sla_icon = "[green]✓[/green]" if sc.sla_passed else "[red]✗[/red]"

        table.add_row(
            sc.ratio,
            f"[{grade_style}]{sc.composite_score:.1f}[/{grade_style}]",
            f"[{grade_style}]{sc.grade.value}[/{grade_style}]",
            sla_icon,
            f"{sc.dimensions[0].score:.0f}",
            f"{sc.dimensions[1].score:.0f}",
            f"{sc.dimensions[2].score:.0f}",
        )

    console.print(table)
    console.print(f"\n[bold]{report.summary}[/bold]")


def add_scorecard_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the scorecard subcommand parser."""
    parser = subparsers.add_parser(
        "scorecard",
        help="Compute efficiency scorecards for P:D configurations",
    )
    parser.add_argument(
        "--benchmark", required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--sla-ttft", type=float, default=None,
        help="TTFT SLA limit in ms",
    )
    parser.add_argument(
        "--sla-tpot", type=float, default=None,
        help="TPOT SLA limit in ms",
    )
    parser.add_argument(
        "--sla-total", type=float, default=None,
        help="Total latency SLA limit in ms",
    )
    parser.add_argument(
        "--sla-weight", type=float, default=0.4,
        help="Weight for SLA compliance (default: 0.4)",
    )
    parser.add_argument(
        "--util-weight", type=float, default=0.3,
        help="Weight for utilization (default: 0.3)",
    )
    parser.add_argument(
        "--waste-weight", type=float, default=0.3,
        help="Weight for resource waste (default: 0.3)",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_scorecard)
