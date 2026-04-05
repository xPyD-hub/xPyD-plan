"""CLI spike command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.spike import SpikeDetector, SpikeSeverity


def _cmd_spike(args: argparse.Namespace) -> None:
    """Handle the 'spike' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    detector = SpikeDetector()
    report = detector.detect(
        data,
        window_size=args.window_size,
        threshold=args.threshold,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    summary_table = Table(title="Spike Summary")
    summary_table.add_column("Metric", justify="left")
    summary_table.add_column("Spikes", justify="right")
    summary_table.add_column("Affected Reqs", justify="right")
    summary_table.add_column("Affected %", justify="right")
    summary_table.add_column("Worst Magnitude", justify="right")
    summary_table.add_column("Worst Severity", justify="center")

    for s in report.summaries:
        style = {
            SpikeSeverity.MINOR: "yellow",
            SpikeSeverity.MODERATE: "bold yellow",
            SpikeSeverity.SEVERE: "bold red",
        }[s.worst_severity]
        severity_str = f"[{style}]{s.worst_severity.value}[/{style}]" if s.spike_count > 0 else "-"
        mag_str = f"[{style}]{s.worst_magnitude:.1f}×[/{style}]" if s.spike_count > 0 else "-"

        summary_table.add_row(
            s.metric,
            str(s.spike_count),
            str(s.total_affected_requests),
            f"{s.affected_fraction:.1%}",
            mag_str,
            severity_str,
        )

    console.print(summary_table)

    # Events table if any
    if report.events:
        console.print()
        event_table = Table(title=f"Spike Events ({len(report.events)} total)")
        event_table.add_column("#", justify="right")
        event_table.add_column("Metric", justify="left")
        event_table.add_column("Requests", justify="right")
        event_table.add_column("Index Range", justify="right")
        event_table.add_column("Peak", justify="right")
        event_table.add_column("Baseline", justify="right")
        event_table.add_column("Magnitude", justify="right")
        event_table.add_column("Severity", justify="center")

        for i, e in enumerate(report.events, 1):
            style = {
                SpikeSeverity.MINOR: "yellow",
                SpikeSeverity.MODERATE: "bold yellow",
                SpikeSeverity.SEVERE: "bold red",
            }[e.severity]
            event_table.add_row(
                str(i),
                e.metric,
                str(e.request_count),
                f"{e.start_index}-{e.end_index}",
                f"{e.peak_value:.2f}",
                f"{e.baseline_value:.2f}",
                f"[{style}]{e.magnitude:.1f}×[/{style}]",
                f"[{style}]{e.severity.value}[/{style}]",
            )

        console.print(event_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_spike_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the spike subcommand parser."""
    parser = subparsers.add_parser(
        "spike",
        help="Detect latency spikes in benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Rolling window size for baseline computation (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Spike threshold multiplier above baseline (default: 3.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_spike)
