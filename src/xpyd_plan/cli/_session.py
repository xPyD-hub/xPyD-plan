"""CLI session command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _cmd_session(args: argparse.Namespace) -> None:
    """Handle the 'session' subcommand."""
    from xpyd_plan.session import SessionManager

    console = Console()
    db_path = args.db or "xpyd-plan-sessions.db"
    mgr = SessionManager(db_path=db_path)

    try:
        if args.session_action == "create":
            if not args.name:
                console.print("[red]Error: --name is required for 'create'[/red]")
                sys.exit(1)
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
            session = mgr.create(
                name=args.name,
                description=args.description or "",
                tags=tags,
            )
            if args.output_format == "json":
                print(session.model_dump_json(indent=2))
            else:
                console.print(
                    f"[green]✅ Created session #{session.id}:[/green] {session.name}"
                )

        elif args.session_action == "add":
            if not args.name or not args.benchmark:
                console.print("[red]Error: --name and --benchmark required for 'add'[/red]")
                sys.exit(1)
            entry = mgr.add(session_name=args.name, benchmark_path=args.benchmark)
            if args.output_format == "json":
                print(entry.model_dump_json(indent=2))
            else:
                console.print(
                    f"[green]✅ Added to '{args.name}':[/green] "
                    f"{entry.benchmark_path} ({entry.num_requests} requests, "
                    f"QPS={entry.measured_qps:.1f})"
                )

        elif args.session_action == "list":
            sessions = mgr.list_sessions()
            if args.output_format == "json":
                import json

                print(json.dumps([s.model_dump() for s in sessions], indent=2))
                return

            if not sessions:
                console.print("[dim]No sessions found.[/dim]")
                return

            table = Table(title="Benchmark Sessions")
            table.add_column("ID", justify="right")
            table.add_column("Name", style="cyan")
            table.add_column("Description")
            table.add_column("Tags", style="green")
            table.add_column("Files", justify="right")
            table.add_column("Total Requests", justify="right")

            for s in sessions:
                total_reqs = sum(e.num_requests for e in s.entries)
                table.add_row(
                    str(s.id),
                    s.name,
                    s.description or "-",
                    ", ".join(s.tags) if s.tags else "-",
                    str(len(s.entries)),
                    str(total_reqs),
                )
            console.print(table)

        elif args.session_action == "show":
            if not args.name:
                console.print("[red]Error: --name is required for 'show'[/red]")
                sys.exit(1)
            session = mgr.show(args.name)
            if args.output_format == "json":
                print(session.model_dump_json(indent=2))
                return

            console.print(f"\n[bold]📋 Session: {session.name}[/bold]")
            if session.description:
                console.print(f"  Description: {session.description}")
            if session.tags:
                console.print(f"  Tags: {', '.join(session.tags)}")
            console.print(f"  Files: {len(session.entries)}")

            if session.entries:
                table = Table(title="Benchmark Files")
                table.add_column("ID", justify="right")
                table.add_column("Path", style="cyan")
                table.add_column("Requests", justify="right")
                table.add_column("QPS", justify="right")
                table.add_column("Config", style="green")

                for e in session.entries:
                    table.add_row(
                        str(e.id),
                        e.benchmark_path,
                        str(e.num_requests),
                        f"{e.measured_qps:.1f}",
                        f"{e.num_prefill}P:{e.num_decode}D",
                    )
                console.print(table)

        elif args.session_action == "delete":
            if not args.name:
                console.print("[red]Error: --name is required for 'delete'[/red]")
                sys.exit(1)
            mgr.delete(args.name)
            if args.output_format == "json":
                import json

                print(json.dumps({"deleted": args.name}))
            else:
                console.print(f"[green]✅ Deleted session '{args.name}'[/green]")

        elif args.session_action == "remove":
            if not args.name or not args.benchmark:
                console.print(
                    "[red]Error: --name and --benchmark required for 'remove'[/red]"
                )
                sys.exit(1)
            mgr.remove(session_name=args.name, benchmark_path=args.benchmark)
            if args.output_format == "json":
                import json

                print(json.dumps({"removed": args.benchmark, "session": args.name}))
            else:
                console.print(
                    f"[green]✅ Removed '{args.benchmark}' from '{args.name}'[/green]"
                )

        else:
            console.print(
                "[red]Error: specify 'create', 'add', 'list',"
                " 'show', 'delete', or 'remove'[/red]"
            )
            sys.exit(1)
    finally:
        mgr.close()
