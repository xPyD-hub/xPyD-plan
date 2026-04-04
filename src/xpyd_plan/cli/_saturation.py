"""CLI saturation command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.saturation import SaturationDetector


def _cmd_saturation(args: argparse.Namespace) -> None:
    """Handle the 'saturation' subcommand."""
    console = Console()

    if len(args.benchmark) < 2:
        console.print("[red]Error: need at least 2 benchmark files for saturation analysis[/red]")
        sys.exit(1)

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    detector = SaturationDetector(increase_threshold=args.increase_threshold)
    report = detector.analyze(benchmarks)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table: QPS points
    table = Table(title="Saturation Analysis — QPS vs Latency")
    table.add_column("QPS", justify="right")
    table.add_column("P:D", justify="center")
    table.add_column("Requests", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("Total P95", justify="right")
    table.add_column("TTFT P99", justify="right")
    table.add_column("TPOT P99", justify="right")
    table.add_column("Total P99", justify="right")

    for point in report.points:
        table.add_row(
            f"{point.measured_qps:.1f}",
            f"{point.num_prefill}:{point.num_decode}",
            str(point.request_count),
            f"{point.ttft_p95_ms:.1f}",
            f"{point.tpot_p95_ms:.1f}",
            f"{point.total_latency_p95_ms:.1f}",
            f"{point.ttft_p99_ms:.1f}",
            f"{point.tpot_p99_ms:.1f}",
            f"{point.total_latency_p99_ms:.1f}",
        )

    console.print(table)

    # Saturation thresholds
    if report.thresholds:
        console.print()
        thresh_table = Table(title="Detected Saturation Thresholds")
        thresh_table.add_column("Metric", justify="left")
        thresh_table.add_column("Safe QPS", justify="right")
        thresh_table.add_column("Saturated QPS", justify="right")
        thresh_table.add_column("Latency (safe)", justify="right")
        thresh_table.add_column("Latency (saturated)", justify="right")
        thresh_table.add_column("Increase", justify="right")

        for t in report.thresholds:
            thresh_table.add_row(
                t.metric,
                f"{t.safe_qps:.1f}",
                f"{t.saturated_qps:.1f}",
                f"{t.latency_at_safe:.1f} ms",
                f"{t.latency_at_saturated:.1f} ms",
                f"[red]+{t.increase_pct:.1f}%[/red]",
            )

        console.print(thresh_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_saturation_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the saturation subcommand parser."""
    parser = subparsers.add_parser(
        "saturation",
        help="Detect QPS saturation point from benchmarks at different load levels",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 2, different QPS levels)",
    )
    parser.add_argument(
        "--increase-threshold",
        type=float,
        default=0.5,
        help="Relative latency increase threshold for saturation detection (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_saturation)
