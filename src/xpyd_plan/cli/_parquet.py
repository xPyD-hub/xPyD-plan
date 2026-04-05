"""CLI parquet command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.parquet_export import ExportMode, ParquetConfig, ParquetExporter


def _cmd_parquet(args: argparse.Namespace) -> None:
    """Handle the 'parquet' subcommand."""
    console = Console()

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    config = ParquetConfig(
        mode=ExportMode(args.mode),
        enrich=args.enrich,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )

    exporter = ParquetExporter(config)
    try:
        result = exporter.export(
            benchmarks,
            args.output,
            source_tags=args.source_tags,
        )
    except ImportError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(result.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    t = Table(title="Parquet Export Result")
    t.add_column("Field", justify="left")
    t.add_column("Value", justify="right")
    t.add_row("Output Path", result.output_path)
    t.add_row("Mode", result.mode.value)
    t.add_row("Total Requests", str(result.total_requests))
    t.add_row("Total Benchmarks", str(result.total_benchmarks))
    t.add_row("Columns", str(len(result.columns)))
    t.add_row("File Size", f"{result.file_size_bytes:,} bytes")
    t.add_row("Enriched", "Yes" if result.enriched else "No")
    console.print(t)

    col_table = Table(title="Exported Columns")
    col_table.add_column("#", justify="right")
    col_table.add_column("Column", justify="left")
    for i, col in enumerate(result.columns, 1):
        col_table.add_row(str(i), col)
    console.print(col_table)


def add_parquet_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the parquet subcommand parser."""
    parser = subparsers.add_parser(
        "parquet",
        help="Export benchmark data to Apache Parquet format",
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
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--mode",
        choices=["requests", "summary", "both"],
        default="requests",
        help="Export mode (default: requests)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Add SLA compliance and workload category columns",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="TTFT SLA threshold in ms (for enrichment)",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold in ms (for enrichment)",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold in ms (for enrichment)",
    )
    parser.add_argument(
        "--source-tags",
        nargs="+",
        default=None,
        help="Labels for each benchmark file (for multi-file exports)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format for export result (default: table)",
    )
    parser.set_defaults(func=_cmd_parquet)
