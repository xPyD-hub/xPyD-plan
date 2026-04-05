"""CLI handler for the cross-validate subcommand."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.cross_validation import CrossValidator, InterpolationMethod


def _cmd_cross_validate(args: argparse.Namespace) -> None:
    """Execute cross-validate subcommand."""
    console = Console()

    if len(args.benchmark) < 3:
        console.print("[red]Error: at least 3 benchmark files required[/red]")
        sys.exit(1)

    data = [load_benchmark_auto(p) for p in args.benchmark]
    method = InterpolationMethod(args.method)
    validator = CrossValidator(data, method=method)
    report = validator.validate()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        print(json.dumps(report.model_dump(), indent=2, default=str))
        return

    # Summary table
    summary = Table(title="Cross-Validation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Method", report.method.value)
    summary.add_row("Folds", str(report.num_folds))
    summary.add_row("Mean TTFT Error", f"{report.mean_ttft_error_pct:.2f}%")
    summary.add_row("Mean TPOT Error", f"{report.mean_tpot_error_pct:.2f}%")
    summary.add_row("Mean Total Latency Error", f"{report.mean_total_error_pct:.2f}%")
    summary.add_row("Mean QPS Error", f"{report.mean_qps_error_pct:.2f}%")
    summary.add_row("Overall Mean Error", f"{report.overall_mean_error_pct:.2f}%")
    summary.add_row("Worst Fold Error", f"{report.worst_fold_error_pct:.2f}%")
    summary.add_row("Accuracy", report.accuracy.value.upper())
    console.print(summary)

    # Fold details table
    if report.fold_errors:
        detail = Table(title="Per-Fold Errors")
        detail.add_column("P:D", style="cyan")
        detail.add_column("TTFT Err%", justify="right")
        detail.add_column("TPOT Err%", justify="right")
        detail.add_column("Total Err%", justify="right")
        detail.add_column("QPS Err%", justify="right")
        for e in report.fold_errors:
            detail.add_row(
                f"{e.num_prefill}:{e.num_decode}",
                f"{e.ttft_error_pct:.1f}",
                f"{e.tpot_error_pct:.1f}",
                f"{e.total_error_pct:.1f}",
                f"{e.qps_error_pct:.1f}",
            )
        console.print(detail)

    # Recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for r in report.recommendations:
            console.print(f"  • {r}")


def add_cross_validate_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add cross-validate subcommand parser."""
    parser = subparsers.add_parser(
        "cross-validate",
        help="Leave-one-out cross-validation for interpolation model accuracy",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 3)",
    )
    parser.add_argument(
        "--method",
        choices=["linear", "spline"],
        default="linear",
        help="Interpolation method (default: linear)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
