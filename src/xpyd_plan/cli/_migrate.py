"""CLI migrate command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.schema_migrate import (
    LATEST_VERSION,
    MigrationResult,
    SchemaMigrator,
    SchemaVersion,
)


def _cmd_migrate(args: argparse.Namespace) -> None:
    """Handle the 'migrate' subcommand."""
    console = Console()
    migrator = SchemaMigrator()

    target = SchemaVersion.parse(args.target_version) if args.target_version else None
    dry_run = getattr(args, "dry_run", False)
    output_path = getattr(args, "output", None)
    output_format = getattr(args, "output_format", "table")

    results: list[MigrationResult] = []
    for bench_path in args.benchmark:
        try:
            result = migrator.migrate_file(
                bench_path,
                target_version=target,
                dry_run=dry_run,
                output_path=output_path,
            )
            results.append(result)
        except (ValueError, FileNotFoundError) as e:
            console.print(f"[red]Error processing {bench_path}: {e}[/red]")
            sys.exit(1)

    if output_format == "json":
        out = [
            {
                "file": str(bench_path),
                "source_version": r.source_version,
                "target_version": r.target_version,
                "migrated": r.migrated,
                "changes": r.changes,
            }
            for bench_path, r in zip(args.benchmark, results)
        ]
        console.print_json(json.dumps(out))
        return

    # Table output
    table = Table(title="Schema Migration Results")
    table.add_column("File", style="cyan")
    table.add_column("From", justify="center")
    table.add_column("To", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Changes")

    for bench_path, result in zip(args.benchmark, results):
        if result.migrated:
            status = "[green]migrated[/green]"
        elif result.changes:
            status = "[yellow]dry-run[/yellow]"
        else:
            status = "[dim]up-to-date[/dim]"

        changes_str = "; ".join(result.changes) if result.changes else "—"
        table.add_row(
            str(bench_path),
            result.source_version,
            result.target_version,
            status,
            changes_str,
        )

    console.print(table)


def add_migrate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'migrate' subcommand parser."""
    p = subparsers.add_parser(
        "migrate",
        help="Migrate benchmark files to a newer schema version",
    )
    p.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON file(s) to migrate",
    )
    p.add_argument(
        "--target-version",
        default=None,
        help=f"Target schema version (default: {LATEST_VERSION})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output file path (default: overwrite in place; only for single file)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_cmd_migrate)
