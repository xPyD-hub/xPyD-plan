"""CLI dedup command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.dedup import DeduplicationAnalyzer


def _cmd_dedup(args: argparse.Namespace) -> None:
    """Handle the 'dedup' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = DeduplicationAnalyzer()
    report = analyzer.analyze(data, tolerance_ms=args.tolerance)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print("\n[bold]Deduplication Analysis[/bold]")
    console.print(f"Total requests: {report.total_requests}")
    console.print(f"Unique requests: {report.unique_requests}")
    console.print(f"Duplicates: {report.duplicate_count}")
    console.print(f"Duplication rate: {report.duplication_rate:.2%}")
    console.print(f"Duplicate groups: {report.group_count}")
    if report.largest_group_size > 0:
        console.print(f"Largest group: {report.largest_group_size} requests")
    console.print(f"\n[italic]{report.recommendation}[/italic]\n")

    if report.groups:
        table = Table(title="Duplicate Groups")
        table.add_column("Group", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("Prompt Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Exact", justify="center")

        for g in report.groups[:20]:  # limit display
            table.add_row(
                str(g.group_id),
                str(g.count),
                str(g.representative_prompt_tokens),
                str(g.representative_output_tokens),
                "✓" if g.is_exact else "~",
            )

        if len(report.groups) > 20:
            table.add_row("...", f"+{len(report.groups) - 20} more", "", "", "")

        console.print(table)

    # Optionally write deduplicated output
    output = getattr(args, "output", None)
    if output:
        deduped = analyzer.deduplicate(data, tolerance_ms=args.tolerance)
        with open(output, "w") as f:
            json.dump(deduped.model_dump(), f, indent=2)
        console.print(f"\nDeduplicated data written to {output}")


def add_dedup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the dedup subcommand parser."""
    parser = subparsers.add_parser(
        "dedup",
        help="Detect and remove duplicate requests in benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Latency tolerance in ms for near-duplicate detection (default: 0.0 = exact only)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write deduplicated benchmark JSON (optional)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_dedup)
