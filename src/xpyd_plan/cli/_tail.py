"""CLI tail command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.tail import TailAnalyzer, TailClassification


def _cmd_tail(args: argparse.Namespace) -> None:
    """Handle the 'tail' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = TailAnalyzer(data)
    report = analyzer.analyze()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Metrics table
    table = Table(title="Tail Latency Analysis")
    table.add_column("Field", justify="left")
    table.add_column("P50", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right")
    table.add_column("P99.9", justify="right")
    table.add_column("P99.99", justify="right")
    table.add_column("P99/P50", justify="right")
    table.add_column("Classification", justify="center")

    for m in report.metrics:
        style = {
            TailClassification.LIGHT: "green",
            TailClassification.MODERATE: "yellow",
            TailClassification.HEAVY: "bold red",
            TailClassification.EXTREME: "bold magenta",
        }[m.classification]

        table.add_row(
            m.field,
            f"{m.p50:.1f}",
            f"{m.p90:.1f}",
            f"{m.p95:.1f}",
            f"{m.p99:.1f}",
            f"{m.p999:.1f}",
            f"{m.p9999:.1f}",
            f"[{style}]{m.tail_ratio_p99:.2f}x[/{style}]",
            f"[{style}]{m.classification.value}[/{style}]",
        )

    console.print(table)

    # Long-tail profiles
    if report.long_tail_profiles:
        console.print()
        ptable = Table(title="Long-Tail Request Profiles (P99+)")
        ptable.add_column("Field", justify="left")
        ptable.add_column("Tail Count", justify="right")
        ptable.add_column("Avg Prompt Tokens", justify="right")
        ptable.add_column("Avg Output Tokens", justify="right")
        ptable.add_column("Prompt Ratio vs All", justify="right")
        ptable.add_column("Output Ratio vs All", justify="right")

        for p in report.long_tail_profiles:
            ptable.add_row(
                p.field,
                str(p.tail_count),
                f"{p.avg_prompt_tokens:.0f}",
                f"{p.avg_output_tokens:.0f}",
                f"{p.prompt_token_ratio:.2f}x",
                f"{p.output_token_ratio:.2f}x",
            )

        console.print(ptable)

    if report.worst_tail:
        console.print(f"\n[bold]Heaviest tail:[/bold] {report.worst_tail}")


def add_tail_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the tail subcommand parser."""
    parser = subparsers.add_parser(
        "tail",
        help="Analyze tail latency behavior (extended percentiles, tail classification)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_tail)
