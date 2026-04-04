"""CLI load-profile command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.load_profile import LoadProfileClassifier, ProfileType


def _cmd_load_profile(args: argparse.Namespace) -> None:
    """Handle the 'load-profile' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    classifier = LoadProfileClassifier(data)
    report = classifier.classify(window_size=args.window_size)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    profile = report.profile
    style_map = {
        ProfileType.STEADY_STATE: "bold green",
        ProfileType.RAMP_UP: "bold yellow",
        ProfileType.RAMP_DOWN: "bold yellow",
        ProfileType.BURST: "bold red",
        ProfileType.CYCLIC: "bold cyan",
        ProfileType.UNKNOWN: "bold dim",
    }
    style = style_map.get(profile.profile_type, "bold")

    console.print(f"Total requests: {report.total_requests}")
    console.print(f"Duration: {report.duration_seconds:.1f}s")
    console.print(f"Overall rate: {report.overall_rate_rps:.2f} rps")
    console.print(
        f"Profile: [{style}]{profile.profile_type.value}[/{style}] "
        f"(confidence: {profile.confidence:.0%})"
    )
    console.print(f"Description: {profile.description}")
    console.print()

    # Windows table
    table = Table(title="Rate by Time Window")
    table.add_column("Window", justify="left")
    table.add_column("Requests", justify="right")
    table.add_column("Rate (rps)", justify="right")

    for w in report.windows:
        table.add_row(
            f"{w.start_time:.1f}–{w.end_time:.1f}",
            str(w.request_count),
            f"{w.rate_rps:.2f}",
        )

    console.print(table)


def add_load_profile_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the load-profile subcommand parser."""
    parser = subparsers.add_parser(
        "load-profile",
        help="Classify the load pattern of a benchmark (steady-state, ramp, burst, cyclic)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Time window size in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_load_profile)
