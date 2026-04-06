"""CLI catalog command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.catalog import CatalogQuery, DatasetCatalog


def _cmd_catalog(args: argparse.Namespace) -> None:
    """Handle the 'catalog' subcommand."""
    console = Console()
    db_path = getattr(args, "db_path", "~/.xpyd-plan/catalog.db")
    output_format = getattr(args, "output_format", "table")
    catalog = DatasetCatalog(db_path=db_path)

    try:
        sub = args.catalog_action

        if sub == "add":
            try:
                entry = catalog.add(args.file, notes=getattr(args, "notes", ""))
            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
            if output_format == "json":
                json.dump(entry.model_dump(), sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                console.print(
                    f"[green]Added[/green] {entry.file_path} "
                    f"(id={entry.id}, {entry.request_count} requests, "
                    f"P:D={entry.pd_ratio or 'N/A'}, QPS={entry.measured_qps:.1f})"
                )

        elif sub == "list":
            report = catalog.list_all()
            if output_format == "json":
                json.dump(report.model_dump(), sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                title = f"Catalog ({report.total_count} entries)"
                _print_entries_table(console, report.entries, title)

        elif sub == "search":
            query = CatalogQuery(
                gpu_type=getattr(args, "gpu_type", None),
                model_name=getattr(args, "model_name", None),
                min_qps=getattr(args, "min_qps", None),
                max_qps=getattr(args, "max_qps", None),
                pd_ratio=getattr(args, "pd_ratio", None),
                min_instances=getattr(args, "min_instances", None),
                max_instances=getattr(args, "max_instances", None),
            )
            report = catalog.search(query)
            if output_format == "json":
                json.dump(report.model_dump(), sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                title = f"Search Results ({report.total_count})"
                _print_entries_table(console, report.entries, title)

        elif sub == "show":
            entry = catalog.get(args.id)
            if not entry:
                console.print(f"[red]Entry {args.id} not found[/red]")
                sys.exit(1)
            if output_format == "json":
                json.dump(entry.model_dump(), sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                console.print(f"\n[bold]Entry {entry.id}[/bold]")
                console.print(f"  File:       {entry.file_path}")
                console.print(f"  Hash:       {entry.file_hash[:16]}...")
                console.print(f"  GPU:        {entry.gpu_type or 'N/A'}")
                console.print(f"  Model:      {entry.model_name or 'N/A'}")
                console.print(f"  P:D Ratio:  {entry.pd_ratio or 'N/A'}")
                p = entry.prefill_instances
                d = entry.decode_instances
                console.print(
                    f"  Instances:  {entry.total_instances}"
                    f" (P={p}, D={d})"
                )
                console.print(f"  QPS:        {entry.measured_qps:.1f}")
                console.print(f"  Requests:   {entry.request_count}")
                console.print(f"  Added:      {entry.date_added}")
                if entry.notes:
                    console.print(f"  Notes:      {entry.notes}")

        elif sub == "remove":
            removed = catalog.remove(args.id)
            if removed:
                console.print(f"[green]Removed[/green] entry {args.id}")
            else:
                console.print(f"[red]Entry {args.id} not found[/red]")
                sys.exit(1)

        else:
            console.print(f"[red]Unknown catalog action: {sub}[/red]")
            sys.exit(1)

    finally:
        catalog.close()


def _print_entries_table(console: Console, entries: list, title: str) -> None:
    """Print entries as a Rich table."""
    table = Table(title=title)
    table.add_column("ID", justify="right")
    table.add_column("P:D", justify="center")
    table.add_column("QPS", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Instances", justify="right")
    table.add_column("GPU", justify="left")
    table.add_column("File", justify="left", max_width=40)

    for e in entries:
        table.add_row(
            str(e.id),
            e.pd_ratio or "N/A",
            f"{e.measured_qps:.1f}",
            str(e.request_count),
            str(e.total_instances),
            e.gpu_type or "N/A",
            e.file_path.split("/")[-1] if "/" in e.file_path else e.file_path,
        )

    console.print(table)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register catalog subcommand."""
    parser = subparsers.add_parser("catalog", help="Manage benchmark dataset catalog")
    parser.add_argument(
        "--db-path",
        default="~/.xpyd-plan/catalog.db",
        help="Catalog database path",
    )
    parser.add_argument("--output-format", choices=["table", "json"], default="table")

    sub = parser.add_subparsers(dest="catalog_action")

    # add
    add_p = sub.add_parser("add", help="Add a benchmark file to catalog")
    add_p.add_argument("file", help="Path to benchmark JSON file")
    add_p.add_argument("--notes", default="", help="Optional notes")

    # list
    sub.add_parser("list", help="List all catalog entries")

    # search
    search_p = sub.add_parser("search", help="Search catalog")
    search_p.add_argument("--gpu-type", help="Filter by GPU type")
    search_p.add_argument("--model-name", help="Filter by model name")
    search_p.add_argument("--min-qps", type=float, help="Minimum QPS")
    search_p.add_argument("--max-qps", type=float, help="Maximum QPS")
    search_p.add_argument("--pd-ratio", help="Filter by P:D ratio (e.g. 2:6)")
    search_p.add_argument("--min-instances", type=int, help="Minimum total instances")
    search_p.add_argument("--max-instances", type=int, help="Maximum total instances")

    # show
    show_p = sub.add_parser("show", help="Show details for a catalog entry")
    show_p.add_argument("id", type=int, help="Entry ID")

    # remove
    rm_p = sub.add_parser("remove", help="Remove a catalog entry")
    rm_p.add_argument("id", type=int, help="Entry ID")
