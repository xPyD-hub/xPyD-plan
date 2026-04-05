"""CLI variance command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.variance_decomp import ComponentSignificance, VarianceDecomposer


def _cmd_variance(args: argparse.Namespace) -> None:
    """Handle the 'variance' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    decomposer = VarianceDecomposer(n_temporal_bins=getattr(args, "temporal_bins", 10))
    report = decomposer.decompose(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    for metric_decomp in report.metrics:
        table = Table(title=f"Variance Decomposition — {metric_decomp.metric}")
        table.add_column("Component", justify="left")
        table.add_column("Sum of Squares", justify="right")
        table.add_column("Contribution %", justify="right")
        table.add_column("Significance", justify="center")

        for comp in metric_decomp.components:
            style = {
                ComponentSignificance.DOMINANT: "bold red",
                ComponentSignificance.SIGNIFICANT: "yellow",
                ComponentSignificance.MINOR: "dim",
                ComponentSignificance.NEGLIGIBLE: "dim",
            }[comp.significance]

            table.add_row(
                comp.name,
                f"{comp.sum_of_squares:,.2f}",
                f"[{style}]{comp.contribution_pct:.1f}%[/{style}]",
                f"[{style}]{comp.significance.value}[/{style}]",
            )

        console.print(table)
        if metric_decomp.dominant_factor:
            console.print(
                f"  Dominant factor: [bold]{metric_decomp.dominant_factor}[/bold]  "
                f"R² = {metric_decomp.r_squared:.4f}"
            )
        else:
            console.print(f"  No dominant factor  R² = {metric_decomp.r_squared:.4f}")
        console.print()

    console.print(f"[bold]{report.recommendation}[/bold]")


def add_variance_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the variance subcommand parser."""
    parser = subparsers.add_parser(
        "variance",
        help="Decompose latency variance into prompt, output, temporal, and residual components",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=10,
        help="Number of temporal bins (default: 10)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_variance)
