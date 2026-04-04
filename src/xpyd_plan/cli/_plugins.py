"""CLI subcommand: plugins — list installed xPyD-plan plugins."""

from __future__ import annotations

import argparse
import json
from typing import Any

from rich.console import Console
from rich.table import Table

from xpyd_plan.plugin import PluginListReport, list_plugins


def add_plugins_subcommand(subparsers: Any) -> None:
    """Register the ``plugins`` subcommand."""
    parser = subparsers.add_parser(
        "plugins",
        help="List installed xPyD-plan plugins",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_run_plugins)


def _run_plugins(args: argparse.Namespace) -> None:
    report = list_plugins()

    if args.output_format == "json":
        print(json.dumps(report.model_dump(), indent=2))
    else:
        _print_table(report)


def _print_table(report: PluginListReport) -> None:
    console = Console()

    if report.total == 0:
        console.print("[dim]No plugins installed.[/dim]")
        console.print(
            "\n[dim]Install plugins via pip. "
            "They register using the 'xpyd_plan.plugins' entry point group.[/dim]"
        )
        return

    table = Table(title="Installed Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Type")
    table.add_column("Description")
    table.add_column("Author")
    table.add_column("Status")

    for p in report.plugins:
        status = "[green]✓ loaded[/green]" if p.loaded else f"[red]✗ {p.error}[/red]"
        table.add_row(
            p.name,
            p.version,
            p.plugin_type.value,
            p.description,
            p.author,
            status,
        )

    console.print(table)
    console.print(
        f"\n[dim]Total: {report.total} | Loaded: {report.loaded} | "
        f"Failed: {report.failed}[/dim]"
    )
