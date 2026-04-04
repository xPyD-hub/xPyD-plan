"""CLI compare command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_compare(args: argparse.Namespace) -> None:
    """Execute the compare subcommand."""

    from xpyd_plan.comparator import compare_benchmarks

    console = Console()
    result = compare_benchmarks(args.baseline, args.current, threshold=args.threshold)

    if args.output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
        return

    # Table output
    console.print("\n[bold]📊 Benchmark Comparison[/bold]")
    console.print(f"   Threshold: {result.threshold * 100:.0f}%")
    console.print(f"   Baseline QPS: {result.baseline_qps:.1f}")
    console.print(f"   Current QPS:  {result.current_qps:.1f}")

    qps = result.qps_delta
    qps_color = "red" if qps.is_regression else "green" if qps.relative_delta > 0 else "white"
    console.print(
        f"   QPS change: [{qps_color}]{qps.relative_delta:+.1%}[/{qps_color}]"
        f"{'  ⚠️  REGRESSION' if qps.is_regression else ''}"
    )

    table = Table(title="\nLatency Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Status", justify="center")

    for delta in result.latency_deltas:
        color = "red" if delta.is_regression else "green" if delta.relative_delta < 0 else "white"
        status = "⚠️  REGRESSED" if delta.is_regression else "✅"
        table.add_row(
            delta.metric,
            f"{delta.baseline:.1f}",
            f"{delta.current:.1f}",
            f"[{color}]{delta.relative_delta:+.1%}[/{color}]",
            status,
        )

    console.print(table)

    if result.has_regression:
        console.print(
            f"\n[bold red]⚠️  {result.regression_count} regression(s) detected![/bold red]"
        )
    else:
        console.print("\n[bold green]✅ No regressions detected.[/bold green]")
