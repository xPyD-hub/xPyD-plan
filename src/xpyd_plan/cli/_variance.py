"""CLI variance command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.variance_decomp import VarianceDecomposer


def _cmd_variance(args: argparse.Namespace) -> None:
    """Handle the 'variance' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    decomposer = VarianceDecomposer(temporal_bins=getattr(args, "temporal_bins", 10))
    report = decomposer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    for decomp in report.decompositions:
        table = Table(title=f"Variance Decomposition — {decomp.metric} (R²={decomp.r_squared:.3f})")
        table.add_column("Factor", style="cyan")
        table.add_column("Sum of Squares", justify="right")
        table.add_column("Contribution %", justify="right")
        table.add_column("Significance", style="bold")

        for comp in decomp.components:
            sig_style = {
                "HIGH": "[red]HIGH[/red]",
                "MODERATE": "[yellow]MODERATE[/yellow]",
                "LOW": "[dim]LOW[/dim]",
                "NEGLIGIBLE": "[dim]NEGLIGIBLE[/dim]",
            }.get(comp.significance.value, comp.significance.value)

            marker = " ★" if comp.factor == decomp.dominant_factor else ""
            table.add_row(
                f"{comp.factor}{marker}",
                f"{comp.sum_of_squares:,.2f}",
                f"{comp.contribution_pct:.1f}%",
                sig_style,
            )
        console.print(table)
        console.print()

    if report.recommendations:
        console.print("[bold]Recommendations:[/bold]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")
