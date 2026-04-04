"""CLI threshold-advisor command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.threshold_advisor import ThresholdAdvisor


def _cmd_threshold_advisor(args: argparse.Namespace) -> None:
    """Handle the 'threshold-advisor' subcommand."""
    console = Console()

    analyzer = BenchmarkAnalyzer(args.benchmark)
    data = analyzer.data

    pass_rates = [float(x) for x in args.pass_rates.split(",")]
    advisor = ThresholdAdvisor(data, pass_rates=pass_rates)
    report = advisor.advise()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Suggestions table
    table = Table(title="SLA Threshold Recommendations")
    table.add_column("Metric", justify="left")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Threshold (ms)", justify="right")

    for s in report.suggestions:
        table.add_row(
            s.metric,
            f"{s.pass_rate:.0%}",
            f"{s.threshold_ms:.2f}",
        )

    console.print(table)

    # Sweet spots
    if report.sweet_spots:
        console.print()
        st = Table(title="Sweet Spots (high gain per ms relaxation)")
        st.add_column("Metric", justify="left")
        st.add_column("Threshold (ms)", justify="right")
        st.add_column("Pass Rate Range", justify="center")
        st.add_column("Gain", justify="right")

        for spot in report.sweet_spots:
            st.add_row(
                spot.metric,
                f"{spot.threshold_ms:.2f}",
                f"{spot.pass_rate_below:.1%} → {spot.pass_rate_above:.1%}",
                f"+{spot.gain:.1%}",
            )
        console.print(st)


def add_threshold_advisor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the threshold-advisor subcommand parser."""
    parser = subparsers.add_parser(
        "threshold-advisor",
        help="Recommend optimal SLA thresholds from benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--pass-rates",
        type=str,
        default="0.90,0.95,0.99",
        help="Comma-separated target pass rates (default: 0.90,0.95,0.99)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_threshold_advisor)
