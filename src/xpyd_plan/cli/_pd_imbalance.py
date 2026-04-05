"""CLI pd-imbalance command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.pd_imbalance import PDImbalanceDetector


def _cmd_pd_imbalance(args: argparse.Namespace) -> None:
    """Handle the 'pd-imbalance' subcommand."""
    console = Console()

    datasets = [load_benchmark_auto(path) for path in args.benchmark]
    detector = PDImbalanceDetector(datasets)
    report = detector.analyze()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    # Summary
    console.print(f"\n[bold]P:D Imbalance Analysis[/bold] ({report.num_benchmarks} benchmarks)\n")
    console.print(f"  Classification: [bold]{report.classification.value}[/bold]")
    console.print(f"  Severity:       {report.level.value}")
    if report.sensitivity_ratio is not None:
        console.print(f"  Sensitivity Ratio: {report.sensitivity_ratio:.2f}")
    console.print(f"\n  Recommendation: {report.recommendation}\n")

    # Sensitivity table
    table = Table(title="Metric Sensitivities")
    table.add_column("Metric")
    table.add_column("Instance Type")
    table.add_column("Sensitivity (ms/instance)", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Data Points", justify="right")

    for sens in [report.ttft_sensitivity, report.tpot_sensitivity]:
        if sens:
            r2 = f"{sens.r_squared:.3f}" if sens.r_squared is not None else "N/A"
            table.add_row(
                sens.metric,
                sens.instance_type,
                f"{sens.sensitivity_ms_per_instance:.2f}",
                r2,
                str(sens.data_points),
            )

    console.print(table)


def add_pd_imbalance_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the pd-imbalance subcommand."""
    parser = subparsers.add_parser(
        "pd-imbalance",
        help="Detect prefill-decode imbalance across P:D ratio benchmarks",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        help="Benchmark JSON files (different P:D ratios)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_pd_imbalance)
