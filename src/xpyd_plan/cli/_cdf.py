"""CLI cdf command."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.cdf import generate_cdf


def _cmd_cdf(args: argparse.Namespace) -> None:
    """Handle the 'cdf' subcommand."""
    console = Console()

    sla = getattr(args, "sla_threshold", None)
    labels_raw = getattr(args, "labels", None)
    labels = labels_raw.split(",") if labels_raw else None
    output_format = getattr(args, "output_format", "table")

    report = generate_cdf(
        benchmarks=args.benchmark,
        metric=args.metric,
        n_points=args.points,
        sla_threshold_ms=sla,
        labels=labels,
    )

    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if output_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["source", "metric", "latency_ms", "cumulative_probability"])
        for curve in report.curves:
            for pt in curve.points:
                writer.writerow(
                    [curve.source, curve.metric, pt.latency_ms, pt.cumulative_probability]
                )
        return

    # Table output
    for curve in report.curves:
        table = Table(title=f"CDF: {curve.source} — {curve.metric}")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Cumulative Probability", justify="right")
        for pt in curve.points:
            table.add_row(f"{pt.latency_ms:.2f}", f"{pt.cumulative_probability:.4f}")
        console.print(table)

        if curve.sla_marker:
            console.print(
                f"  SLA threshold {curve.sla_marker.threshold_ms:.1f} ms → "
                f"{curve.sla_marker.percentile_at_threshold:.1f}% of requests pass"
            )
        console.print()


def add_cdf_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the cdf subcommand parser."""
    parser = subparsers.add_parser(
        "cdf",
        help="Export latency CDF data for visualization",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON file(s)",
    )
    parser.add_argument(
        "--metric",
        default="total_latency",
        choices=["ttft", "tpot", "total_latency"],
        help="Latency metric (default: total_latency)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=100,
        help="Number of CDF data points per curve (default: 100)",
    )
    parser.add_argument(
        "--sla-threshold",
        type=float,
        default=None,
        help="SLA threshold (ms) to mark on CDF",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels for each benchmark file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
