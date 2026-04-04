"""CLI validate command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _cmd_validate(args: argparse.Namespace) -> None:
    """Handle the 'validate' subcommand."""

    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.validator import DataValidator, OutlierMethod

    console = Console()

    if not args.benchmark:
        console.print("[red]Error: --benchmark is required.[/red]")
        sys.exit(1)

    data = load_benchmark_auto(args.benchmark)
    method = OutlierMethod(args.outlier_method)
    validator = DataValidator(method=method)
    result = validator.validate(data, filter_outliers=True)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(result.model_dump_json(indent=2, exclude={"filtered_data"}))
        return

    # Table output
    console.print("\n[bold]📊 Benchmark Data Validation[/bold]")
    console.print(f"   File: {args.benchmark}")
    console.print(f"   Method: {result.method.value}")
    console.print(f"   Total requests: {result.total_requests}")
    console.print(f"   Outliers: {result.outlier_count}")
    console.print()

    # Quality scores
    q = result.quality
    quality_table = Table(title="Data Quality Scores")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Score", justify="right")
    quality_table.add_row("Completeness", f"{q.completeness:.4f}")
    quality_table.add_row("Consistency", f"{q.consistency:.4f}")
    quality_table.add_row("Outlier Ratio", f"{q.outlier_ratio:.4f}")
    quality_table.add_row("Overall", f"[bold]{q.overall:.4f}[/bold]")
    console.print(quality_table)

    # Outlier details
    if result.outliers:
        console.print()
        outlier_table = Table(title=f"Outliers ({result.outlier_count} unique requests)")
        outlier_table.add_column("Request ID", style="cyan")
        outlier_table.add_column("Index", justify="right")
        outlier_table.add_column("Metric")
        outlier_table.add_column("Value", justify="right")
        outlier_table.add_column("Reason")
        for o in result.outliers[:50]:  # Cap display at 50
            outlier_table.add_row(o.request_id, str(o.index), o.metric, f"{o.value:.2f}", o.reason)
        console.print(outlier_table)
        if len(result.outliers) > 50:
            console.print(f"[dim]   ... and {len(result.outliers) - 50} more[/dim]")
