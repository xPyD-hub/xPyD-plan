"""CLI scaling command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.scaling import ScalingAnalyzer


def _cmd_scaling(args: argparse.Namespace) -> None:
    """Handle the 'scaling' subcommand."""
    console = Console()

    if len(args.benchmark) < 2:
        console.print("[red]Error: need at least 2 benchmark files for scaling analysis[/red]")
        sys.exit(1)

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    analyzer = ScalingAnalyzer(knee_threshold=args.knee_threshold)
    report = analyzer.analyze(benchmarks)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Throughput Scaling Analysis")
    table.add_column("Instances", justify="right")
    table.add_column("P:D", justify="center")
    table.add_column("QPS", justify="right")
    table.add_column("Per-Instance QPS", justify="right")
    table.add_column("Efficiency", justify="right")
    table.add_column("Note", justify="left")

    for point in report.curve.points:
        note = ""
        knee = report.curve.knee_point
        if knee and point.total_instances == knee.total_instances:
            note = "⚠ knee"
        if point.total_instances == report.curve.optimal_point.total_instances:
            note = "★ optimal" if not note else "⚠ knee"

        eff_style = "green" if point.scaling_efficiency >= args.knee_threshold else "red"

        table.add_row(
            str(point.total_instances),
            f"{point.num_prefill}:{point.num_decode}",
            f"{point.measured_qps:.1f}",
            f"{point.per_instance_qps:.2f}",
            f"[{eff_style}]{point.scaling_efficiency:.0%}[/{eff_style}]",
            note,
        )

    console.print(table)
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_scaling_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the scaling subcommand parser."""
    parser = subparsers.add_parser(
        "scaling",
        help="Analyze throughput scaling across different instance counts",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 2, different instance counts)",
    )
    parser.add_argument(
        "--knee-threshold",
        type=float,
        default=0.8,
        help="Scaling efficiency threshold for knee detection (default: 0.8)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_scaling)
