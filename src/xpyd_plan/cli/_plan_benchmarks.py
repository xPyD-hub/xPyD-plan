"""CLI plan-benchmarks command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.plan_generator import BenchmarkPlanGenerator, RatioPriority


def _cmd_plan_benchmarks(args: argparse.Namespace) -> None:
    """Handle the 'plan-benchmarks' subcommand."""
    console = Console()

    generator = BenchmarkPlanGenerator(
        total_instances=args.total_instances,
        max_runs=getattr(args, "max_runs", None),
    )
    plan = generator.generate()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(plan.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    table = Table(title=f"Benchmark Plan — {plan.total_instances} instances")
    table.add_column("#", justify="right")
    table.add_column("Ratio", justify="left")
    table.add_column("Priority", justify="center")
    table.add_column("Reason", justify="left")

    priority_style = {
        RatioPriority.CRITICAL: "bold red",
        RatioPriority.HIGH: "bold yellow",
        RatioPriority.MEDIUM: "green",
        RatioPriority.LOW: "dim",
    }

    for i, r in enumerate(plan.ratios, 1):
        style = priority_style[r.priority]
        table.add_row(
            str(i),
            r.ratio_str,
            f"[{style}]{r.priority.value}[/{style}]",
            r.reason,
        )

    console.print(table)
    console.print(f"\n[bold]{len(plan.ratios)} configurations recommended[/bold]")


def add_plan_benchmarks_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the plan-benchmarks subcommand parser."""
    parser = subparsers.add_parser(
        "plan-benchmarks",
        help="Recommend which P:D ratios to benchmark",
    )
    parser.add_argument(
        "--total-instances", type=int, required=True,
        help="Total number of instances available",
    )
    parser.add_argument(
        "--max-runs", type=int, default=None,
        help="Maximum number of benchmark runs to recommend",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_plan_benchmarks)
