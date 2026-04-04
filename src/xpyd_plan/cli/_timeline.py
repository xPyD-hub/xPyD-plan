"""CLI timeline command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.timeline import LatencyTrendDirection, TimelineAnalyzer


def _cmd_timeline(args: argparse.Namespace) -> None:
    """Handle the 'timeline' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = TimelineAnalyzer(
        window_size_s=args.window_size,
        warmup_factor=getattr(args, "warmup_factor", 2.0),
    )
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Windows table
    table = Table(title="Request Timeline Analysis")
    table.add_column("#", justify="right")
    table.add_column("Time Range", justify="center")
    table.add_column("Requests", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("Total P95", justify="right")

    for w in report.windows:
        if w.request_count == 0:
            continue
        table.add_row(
            str(w.window_index),
            f"{w.start_time:.1f}–{w.end_time:.1f}",
            str(w.request_count),
            f"{w.ttft_p95_ms:.1f}ms",
            f"{w.tpot_p95_ms:.1f}ms",
            f"{w.total_p95_ms:.1f}ms",
        )

    console.print(table)
    console.print()

    # Warmup
    if report.warmup.detected:
        console.print(
            f"[bold yellow]⚠ Warmup detected:[/bold yellow] "
            f"{report.warmup.warmup_windows} window(s), "
            f"{report.warmup.warmup_duration_s}s, "
            f"peak {report.warmup.warmup_peak_p95_ms:.1f}ms vs "
            f"steady-state {report.warmup.steady_state_p95_ms:.1f}ms"
        )
    else:
        console.print("[green]✓ No warmup period detected[/green]")

    # Trend
    style_map = {
        LatencyTrendDirection.DEGRADING: "bold red",
        LatencyTrendDirection.IMPROVING: "bold green",
        LatencyTrendDirection.STABLE: "green",
    }
    style = style_map[report.trend.direction]
    console.print(
        f"[{style}]Trend: {report.trend.direction.value} "
        f"(slope: {report.trend.slope_ms_per_s:+.3f} ms/s, "
        f"R²={report.trend.r_squared:.3f})[/{style}]"
    )
    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_timeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the timeline subcommand parser."""
    parser = subparsers.add_parser(
        "timeline",
        help="Analyze latency patterns over time (warmup, trends)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=10.0,
        help="Time window size in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--warmup-factor",
        type=float,
        default=2.0,
        help="Warmup detection threshold multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_timeline)
