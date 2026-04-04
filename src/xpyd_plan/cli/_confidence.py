"""CLI confidence command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.confidence import analyze_confidence


def _cmd_confidence(args: argparse.Namespace) -> None:
    """Handle the 'confidence' subcommand."""
    console = Console()

    benchmarks = [load_benchmark_auto(p) for p in args.benchmark]
    if not benchmarks:
        console.print("[red]Error:[/red] No benchmark files provided.")
        sys.exit(1)

    # Use the first benchmark for analysis
    data = benchmarks[0]

    report = analyze_confidence(
        data=data,
        percentile=args.percentile,
        confidence_level=args.confidence_level,
        iterations=args.iterations,
        seed=args.seed,
    )

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(json.dumps(report.model_dump(), indent=2))
        return

    # Table output
    console.print(
        f"\n[bold]Bootstrap Confidence Intervals[/bold] "
        f"(P{report.percentile:.0f}, {report.confidence_level:.0%} CI, "
        f"{report.iterations} iterations, n={report.sample_size})\n"
    )

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Point Estimate", justify="right")
    table.add_column("CI Lower", justify="right")
    table.add_column("CI Upper", justify="right")
    table.add_column("CI Width", justify="right")
    table.add_column("Relative Width", justify="right")
    table.add_column("Adequacy", justify="center")

    for mc in report.metrics:
        ci = mc.intervals[0]
        adequacy_style = {
            "sufficient": "[green]SUFFICIENT[/green]",
            "marginal": "[yellow]MARGINAL[/yellow]",
            "insufficient": "[red]INSUFFICIENT[/red]",
        }
        table.add_row(
            ci.metric,
            f"{ci.point_estimate:.2f} ms",
            f"{ci.ci_lower:.2f} ms",
            f"{ci.ci_upper:.2f} ms",
            f"{ci.ci_width:.2f} ms",
            f"{ci.relative_ci_width:.1%}",
            adequacy_style.get(mc.adequacy.value, mc.adequacy.value),
        )

    console.print(table)

    if report.warnings:
        console.print()
        for w in report.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

    if not report.adequate:
        console.print(
            "\n[yellow]Some metrics have insufficient sample sizes. "
            "Consider collecting more benchmark data.[/yellow]"
        )
