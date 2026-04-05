"""CLI concurrency-util command."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.concurrency_util import ConcurrencyUtilizationAnalyzer


def _cmd_concurrency_util(args: argparse.Namespace) -> None:
    """Handle the 'concurrency-util' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = ConcurrencyUtilizationAnalyzer(window_size=args.window_size)
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        console.print(json.dumps(report.model_dump(), indent=2))
        return

    # Summary
    n = report.total_requests
    console.print(
        f"\n[bold]Concurrency Utilization Analysis[/bold]"
        f" ({n} requests)\n"
    )
    dur = report.duration_seconds
    inst = report.total_instances
    console.print(f"  Duration: {dur:.1f}s | Instances: {inst}")
    peak = report.peak_concurrent
    avg_u = report.avg_utilization_pct
    console.print(f"  Peak concurrent: {peak} | Avg util: {avg_u:.1f}%")
    iw = report.idle_windows
    hw = report.high_windows
    console.print(f"  Idle windows: {iw} | High-util windows: {hw}\n")

    # Windows table
    table = Table(title="Time Windows")
    table.add_column("Window", style="cyan")
    table.add_column("Peak", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("Util%", justify="right")
    table.add_column("Level", style="bold")
    table.add_column("Started", justify="right")
    table.add_column("Completed", justify="right")

    for w in report.windows:
        level_style = {
            "idle": "red",
            "low": "yellow",
            "moderate": "green",
            "high": "bright_green",
        }.get(w.level.value, "")
        table.add_row(
            f"{w.window_start:.1f}–{w.window_end:.1f}",
            str(w.concurrent_requests),
            f"{w.avg_concurrent:.1f}",
            f"{w.utilization_pct:.1f}%",
            f"[{level_style}]{w.level.value.upper()}[/{level_style}]",
            str(w.requests_started),
            str(w.requests_completed),
        )
    console.print(table)

    # Recommendation
    rec = report.recommendation
    if rec:
        console.print("\n[bold]Right-Sizing Recommendation[/bold]")
        console.print(f"  Current instances: {rec.current_instances}")
        avg_c = rec.avg_concurrent
        p95_c = rec.p95_concurrent
        console.print(
            f"  Peak: {rec.peak_concurrent} | "
            f"Avg: {avg_c:.1f} | P95: {p95_c}"
        )
        r_min = rec.recommended_min
        r_tgt = rec.recommended_target
        console.print(
            f"  Recommended min: {r_min} | Target: {r_tgt}"
        )
        if rec.over_provisioned:
            console.print("  [yellow]⚠ Over-provisioned — consider reducing instances[/yellow]")
        if rec.under_provisioned:
            console.print("  [red]⚠ Under-provisioned — consider adding instances[/red]")


def add_concurrency_util_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the concurrency-util subcommand parser."""
    parser = subparsers.add_parser(
        "concurrency-util",
        help="Analyze time-windowed concurrency and instance utilization",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=1.0,
        help="Window size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
