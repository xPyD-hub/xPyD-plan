"""CLI discover command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.discovery import BenchmarkDiscovery, ValidationStatus


def _cmd_discover(args: argparse.Namespace) -> None:
    """Handle the 'discover' subcommand."""
    console = Console()

    discoverer = BenchmarkDiscovery()
    report = discoverer.discover(
        root_dir=args.dir,
        pattern=args.pattern,
        max_depth=args.max_depth,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print(f"\n[bold]Benchmark Discovery: {report.root_dir}[/bold]")
    console.print(
        f"Pattern: {report.pattern} | "
        f"Scanned: {report.total_files_scanned} | "
        f"Valid: {report.valid_count} | "
        f"Invalid: {report.invalid_count}"
    )

    if not report.benchmarks:
        console.print("[yellow]No JSON files found.[/yellow]")
        return

    # Benchmarks table
    table = Table(title="Discovered Benchmarks")
    table.add_column("Path", justify="left", max_width=60)
    table.add_column("Status", justify="center")
    table.add_column("Config", justify="center")
    table.add_column("Requests", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("Size", justify="right")

    for b in report.benchmarks:
        status_style = "green" if b.status == ValidationStatus.VALID else "red"
        table.add_row(
            b.path,
            f"[{status_style}]{b.status.value}[/{status_style}]",
            b.config_key or "-",
            str(b.num_requests) if b.num_requests is not None else "-",
            f"{b.measured_qps:.1f}" if b.measured_qps is not None else "-",
            f"{b.file_size_bytes:,}",
        )

    console.print(table)

    # Groups table
    if report.groups:
        console.print()
        gtable = Table(title="Configuration Groups")
        gtable.add_column("Config", justify="center")
        gtable.add_column("Prefill", justify="right")
        gtable.add_column("Decode", justify="right")
        gtable.add_column("Total", justify="right")
        gtable.add_column("Files", justify="right")

        for g in report.groups:
            gtable.add_row(
                g.config_key,
                str(g.num_prefill_instances),
                str(g.num_decode_instances),
                str(g.total_instances),
                str(g.count),
            )

        console.print(gtable)


def add_discover_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the discover subcommand parser."""
    parser = subparsers.add_parser(
        "discover",
        help="Discover benchmark files in a directory tree",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Root directory to scan",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.json",
        help="Glob pattern for matching files (default: **/*.json)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory depth to scan (default: unlimited)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_discover)
