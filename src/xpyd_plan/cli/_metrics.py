"""CLI metrics command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.metrics_export import MetricsExporter


def _cmd_metrics(args: argparse.Namespace) -> None:
    """Handle the 'metrics' subcommand."""
    console = Console()

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    exporter = MetricsExporter()
    report = exporter.export(benchmarks)

    output_format = getattr(args, "output_format", "text")

    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif output_format == "table":
        table = Table(title="Exported Metrics")
        table.add_column("Name", style="cyan")
        table.add_column("Labels", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Type", style="green")

        for m in report.metrics:
            label_str = ", ".join(f"{k}={v}" for k, v in sorted(m.labels.items()))
            table.add_row(m.name, label_str, f"{m.value:.4f}", m.metric_type)

        console.print(table)
    else:
        # Default: OpenMetrics text format
        sys.stdout.write(report.text)

    # Write to file if requested
    output_path = getattr(args, "output", None)
    if output_path:
        Path(output_path).write_text(report.text)
        console.print(f"[green]Metrics written to {output_path}[/green]", stderr=True)


def add_metrics_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the metrics subcommand parser."""
    parser = subparsers.add_parser(
        "metrics",
        help="Export benchmark analysis as Prometheus/OpenMetrics format",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write OpenMetrics text to file",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "table"],
        default="text",
        help="Output format (default: text = OpenMetrics)",
    )
    parser.set_defaults(func=_cmd_metrics)
