"""CLI annotate command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _parse_tags(tag_list: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE tag arguments into a dict."""
    tags: dict[str, str] = {}
    for item in tag_list:
        if "=" not in item:
            Console().print(f"[red]Invalid tag format: {item} (expected KEY=VALUE)[/red]")
            sys.exit(1)
        key, value = item.split("=", 1)
        tags[key.strip()] = value.strip()
    return tags


def _cmd_annotate(args: argparse.Namespace) -> None:
    """Handle annotate subcommand."""
    from xpyd_plan.annotation import AnnotationManager

    manager = AnnotationManager()
    console = Console()

    if args.annotate_action == "add":
        tags = _parse_tags(args.tag)
        result = manager.add_tags(args.benchmark, tags)
        console.print(f"[green]Added {len(tags)} tag(s) to {args.benchmark}[/green]")
        for k, v in sorted(result.tags.items()):
            console.print(f"  {k} = {v}")

    elif args.annotate_action == "remove":
        result = manager.remove_tags(args.benchmark, args.key)
        console.print(f"[green]Removed tag(s) from {args.benchmark}[/green]")
        if result.tags:
            for k, v in sorted(result.tags.items()):
                console.print(f"  {k} = {v}")
        else:
            console.print("  (no tags remaining)")

    elif args.annotate_action == "clear":
        manager.clear_tags(args.benchmark)
        console.print(f"[green]Cleared all tags from {args.benchmark}[/green]")

    elif args.annotate_action == "list":
        result = manager.get_tags(args.benchmark)
        if getattr(args, "output_format", "table") == "json":
            import json as json_mod

            console.print(json_mod.dumps({"tags": result.tags}, indent=2))
        elif result.tags:
            table = Table(title=f"Tags: {args.benchmark}")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in sorted(result.tags.items()):
                table.add_row(k, v)
            console.print(table)
        else:
            console.print(f"No tags on {args.benchmark}")

    elif args.annotate_action == "filter":
        tags = _parse_tags(args.tag)
        result = manager.filter_by_tags(args.dir, tags)
        if getattr(args, "output_format", "table") == "json":
            import json as json_mod

            console.print(
                json_mod.dumps(
                    {
                        "query_tags": result.query_tags,
                        "matched": [
                            {"path": m.benchmark_path, "tags": m.tags}
                            for m in result.matched
                        ],
                        "total_scanned": result.total_scanned,
                    },
                    indent=2,
                )
            )
        else:
            console.print(
                f"Scanned {result.total_scanned} file(s), "
                f"{len(result.matched)} matched"
            )
            if result.matched:
                table = Table(title="Matching Benchmarks")
                table.add_column("File", style="cyan")
                table.add_column("Tags", style="green")
                for m in result.matched:
                    tag_str = ", ".join(f"{k}={v}" for k, v in sorted(m.tags.items()))
                    table.add_row(m.benchmark_path, tag_str)
                console.print(table)

    else:
        console.print("[red]Usage: xpyd-plan annotate {add|list|remove|filter|clear}[/red]")
        sys.exit(1)
