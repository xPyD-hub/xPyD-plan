"""CLI interpolate command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _cmd_interpolate(args: argparse.Namespace) -> None:
    """Run performance interpolation across P:D ratios."""
    import json as json_mod

    from xpyd_plan.interpolator import (
        InterpolationMethod,
        interpolate_performance,
    )

    console = Console()
    paths = args.benchmark
    if not paths or len(paths) < 2:
        console.print("[red]Need at least 2 benchmark files for interpolation.[/red]")
        sys.exit(1)

    from xpyd_plan.analyzer import BenchmarkAnalyzer
    analyzer = BenchmarkAnalyzer()
    datasets = analyzer.load_multi_data(paths)

    method = InterpolationMethod(args.method)

    target_ratios = None
    if args.ratios:
        target_ratios = []
        for r in args.ratios:
            parts = r.replace("P", "").replace("D", "").replace("p", "").replace("d", "").split(":")
            if len(parts) != 2:
                console.print(f"[red]Invalid ratio format '{r}'. Use 'P:D' e.g. '2:6'.[/red]")
                sys.exit(1)
            target_ratios.append((int(parts[0]), int(parts[1])))

    result = interpolate_performance(datasets, target_ratios=target_ratios, method=method)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(json_mod.dumps(result.model_dump(), indent=2))
        return

    # Table output
    table = Table(title="Performance Interpolation Results")
    table.add_column("P:D Ratio", style="cyan")
    table.add_column("TTFT P95 (ms)", justify="right")
    table.add_column("TPOT P95 (ms)", justify="right")
    table.add_column("Total Lat P95 (ms)", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("Confidence", justify="center")
    table.add_column("Measured", justify="center")

    for pred in result.predictions:
        conf_style = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(pred.confidence.value, "white")
        table.add_row(
            f"{pred.num_prefill}P:{pred.num_decode}D",
            f"{pred.ttft_p95_ms:.1f}",
            f"{pred.tpot_p95_ms:.1f}",
            f"{pred.total_latency_p95_ms:.1f}",
            f"{pred.throughput_qps:.1f}",
            f"[{conf_style}]{pred.confidence.value.upper()}[/{conf_style}]",
            "✓" if pred.is_measured else "",
        )

    console.print(table)

    if result.best_predicted:
        bp = result.best_predicted
        console.print(
            f"\n[bold green]Best predicted:[/bold green] "
            f"{bp.num_prefill}P:{bp.num_decode}D "
            f"(QPS={bp.throughput_qps:.1f}, confidence={bp.confidence.value})"
        )

    if result.notes:
        for note in result.notes:
            console.print(f"[dim]Note: {note}[/dim]")
