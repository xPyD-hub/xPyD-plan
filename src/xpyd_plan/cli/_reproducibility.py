"""CLI reproducibility command."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.reproducibility import ReproducibilityAnalyzer


def _cmd_reproducibility(args: argparse.Namespace) -> None:
    """Handle the 'reproducibility' subcommand."""
    console = Console()

    datasets = [load_benchmark_auto(f) for f in args.benchmark]
    analyzer = ReproducibilityAnalyzer(cv_threshold=args.cv_threshold)
    report = analyzer.analyze(datasets)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        console.print(json.dumps(report.model_dump(), indent=2))
        return

    # Summary
    console.print(
        f"\n[bold]Benchmark Reproducibility Analysis[/bold]"
        f" ({report.num_runs} runs, {report.total_requests} total requests)\n"
    )

    grade_style = {
        "EXCELLENT": "bright_green",
        "GOOD": "green",
        "FAIR": "yellow",
        "POOR": "red",
    }.get(report.grade.value, "")
    console.print(
        f"  Score: [bold]{report.composite_score}[/bold] / 100"
        f"  Grade: [{grade_style}]{report.grade.value}[/{grade_style}]"
    )
    console.print(f"  Recommended minimum runs: {report.recommended_min_runs}\n")

    if report.unreliable_metrics:
        console.print(
            f"  [yellow]⚠ Unreliable metrics (CV > {args.cv_threshold:.0%}):"
            f" {', '.join(report.unreliable_metrics)}[/yellow]\n"
        )

    # Metrics table
    table = Table(title="Metric Reproducibility")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Reliable", justify="center")

    for m in report.metrics:
        reliable_str = "[green]✓[/green]" if m.reliable else "[red]✗[/red]"
        table.add_row(
            m.metric_name,
            f"{m.mean:.2f}",
            f"{m.std:.2f}",
            f"{m.cv:.4f}",
            reliable_str,
        )
    console.print(table)

    # KS test summary
    if report.pair_tests:
        console.print("\n[bold]Distribution Consistency (KS Tests)[/bold]")
        consistent = sum(1 for t in report.pair_tests if t.distributions_consistent)
        total = len(report.pair_tests)
        console.print(
            f"  {consistent}/{total} pairs consistent (p > 0.05)\n"
        )


def add_reproducibility_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the reproducibility subcommand parser."""
    parser = subparsers.add_parser(
        "reproducibility",
        help="Analyze benchmark reproducibility across repeated runs",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        help="Paths to benchmark JSON files (2+ required)",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.10,
        help="CV threshold for reliability flagging (default: 0.10)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
