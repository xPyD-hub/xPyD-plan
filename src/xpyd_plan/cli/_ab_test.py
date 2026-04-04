"""CLI ab-test command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_ab_test(args: argparse.Namespace) -> None:
    """Execute the ab-test subcommand."""
    from xpyd_plan.ab_test import ABTestAnalyzer, ABTestConfig

    config = ABTestConfig(alpha=args.alpha)
    if args.metric:
        config.metrics = args.metric

    analyzer = ABTestAnalyzer(config=config)
    report = analyzer.analyze(args.control, args.treatment)

    if args.output_format == "json":
        print(report.model_dump_json(indent=2))
        return

    console = Console()

    console.print("\n[bold]A/B Test Report[/bold]")
    console.print(f"Control:   {report.control_file}")
    console.print(f"Treatment: {report.treatment_file}")
    console.print(f"Alpha:     {report.alpha}")
    console.print()

    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Control Mean", justify="right")
    table.add_column("Treatment Mean", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Rel Diff", justify="right")
    table.add_column("p-value (t)", justify="right")
    table.add_column("Significant", justify="center")
    table.add_column("Effect Size", justify="right")
    table.add_column("Winner", justify="center")

    for r in report.results:
        sig_style = "green" if r.welch_t.significant else "dim"
        table.add_row(
            r.metric,
            f"{r.control_mean:.2f}",
            f"{r.treatment_mean:.2f}",
            f"{r.mean_difference:+.2f}",
            f"{r.relative_difference:+.1%}",
            f"{r.welch_t.p_value:.4f}",
            f"[{sig_style}]{'Yes' if r.welch_t.significant else 'No'}[/{sig_style}]",
            f"{r.effect_size.cohens_d:.2f} ({r.effect_size.magnitude.value})",
            r.winner or "-",
        )

    console.print(table)
    console.print()

    # CI table
    ci_table = Table(title="Confidence Intervals (Mean Difference)")
    ci_table.add_column("Metric", style="cyan")
    ci_table.add_column("Lower", justify="right")
    ci_table.add_column("Upper", justify="right")
    ci_table.add_column("Level", justify="right")

    for r in report.results:
        ci_table.add_row(
            r.metric,
            f"{r.ci.lower:.2f}",
            f"{r.ci.upper:.2f}",
            f"{r.ci.confidence_level:.0%}",
        )

    console.print(ci_table)
    console.print()

    # Power warnings
    for r in report.results:
        if not r.power_warning.adequate:
            console.print(f"[yellow]⚠ {r.metric}: {r.power_warning.reason}[/yellow]")

    console.print(f"\n[bold]Summary:[/bold] {report.summary}\n")
