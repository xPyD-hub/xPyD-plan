"""CLI sqlite-export command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.sqlite_export import SQLiteExportConfig, SQLiteExporter


def _cmd_sqlite_export(args: argparse.Namespace) -> None:
    """Handle the 'sqlite-export' subcommand."""
    console = Console()

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    config = SQLiteExportConfig(
        append=args.append,
        benchmark_id=args.benchmark_id,
        include_summary=not args.no_summary,
        create_indexes=not args.no_indexes,
    )

    exporter = SQLiteExporter(config)
    result = exporter.export(
        benchmarks,
        args.output,
        source_tags=args.source_tags,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(result.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    t = Table(title="SQLite Export Result")
    t.add_column("Field", justify="left")
    t.add_column("Value", justify="right")
    t.add_row("Output Path", result.output_path)
    t.add_row("Total Requests", str(result.total_requests))
    t.add_row("Total Benchmarks", str(result.total_benchmarks))
    t.add_row("Benchmark IDs", ", ".join(result.benchmark_ids))
    t.add_row("Tables", ", ".join(result.tables_created))
    t.add_row("Indexes Created", str(result.indexes_created))
    t.add_row("File Size", f"{result.file_size_bytes:,} bytes")
    t.add_row("Appended", "Yes" if result.appended else "No")
    console.print(t)


def add_sqlite_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the sqlite-export subcommand parser."""
    parser = subparsers.add_parser(
        "sqlite-export",
        help="Export benchmark data to SQLite database",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files to export",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output SQLite file path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing database instead of overwriting",
    )
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Custom benchmark ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--source-tags",
        nargs="+",
        default=None,
        help="Labels for each benchmark file (used as benchmark_id)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        default=False,
        help="Skip creating analysis_summary table",
    )
    parser.add_argument(
        "--no-indexes",
        action="store_true",
        default=False,
        help="Skip creating indexes",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format for export result (default: table)",
    )
    parser.set_defaults(func=_cmd_sqlite_export)
