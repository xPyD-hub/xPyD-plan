"""CLI warmup-filter command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.warmup import WarmupFilter


def _cmd_warmup_filter(args: argparse.Namespace) -> None:
    """Handle the 'warmup-filter' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    warmup_seconds = getattr(args, "warmup_seconds", None)
    warmup_factor = getattr(args, "warmup_factor", 2.0)
    window_size = getattr(args, "window_size", 10.0)

    filt = WarmupFilter(
        warmup_seconds=warmup_seconds,
        warmup_factor=warmup_factor,
        window_size_s=window_size,
    )
    filtered_data, report = filt.filter(data)

    output_format = getattr(args, "output_format", "table")
    output_file = getattr(args, "output", None)

    # Write filtered data if output requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(filtered_data.model_dump(), f, indent=2)
        console.print(f"Filtered benchmark written to {output_file}")

    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    if not report.warmup_detected:
        console.print("\n✅ No warm-up period detected.\n")
        return

    console.print(f"\n🔥 Warm-up detected: {report.warmup_window.duration_s}s")
    console.print(
        f"   Excluded {report.warmup_window.requests_excluded} requests "
        f"({report.original_request_count} → {report.filtered_request_count})\n"
    )

    if report.latency_comparison:
        table = Table(title="Latency Comparison (total_latency_ms)")
        table.add_column("Percentile", justify="center")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")

        lc = report.latency_comparison
        table.add_row("P50", f"{lc.before_p50_ms:.1f}", f"{lc.after_p50_ms:.1f}")
        table.add_row("P95", f"{lc.before_p95_ms:.1f}", f"{lc.after_p95_ms:.1f}")
        table.add_row("P99", f"{lc.before_p99_ms:.1f}", f"{lc.after_p99_ms:.1f}")

        console.print(table)

    console.print(f"\nQPS: {report.original_qps:.1f} → {report.adjusted_qps:.1f}")


def add_warmup_filter_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the warmup-filter subcommand parser."""
    parser = subparsers.add_parser(
        "warmup-filter",
        help="Detect and remove warm-up requests from benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=None,
        help="Fixed warm-up duration to exclude (seconds). Auto-detect if not set.",
    )
    parser.add_argument(
        "--warmup-factor",
        type=float,
        default=2.0,
        help="P95 multiplier threshold for auto warm-up detection (default: 2.0)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="Window size in seconds for warm-up detection (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for filtered benchmark JSON",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_warmup_filter)
