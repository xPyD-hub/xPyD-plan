"""CLI trend command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _cmd_trend(args: argparse.Namespace) -> None:
    """Handle the 'trend' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.trend import TrendTracker

    console = Console()
    db_path = args.db or "xpyd-plan-trend.db"
    tracker = TrendTracker(db_path=db_path)

    try:
        if args.trend_action == "add":
            data = load_benchmark_auto(args.benchmark)
            entry = tracker.add(data, label=args.label)
            if args.output_format == "json":
                print(entry.model_dump_json(indent=2))
            else:
                console.print(
                    f"[green]✅ Added entry #{entry.id}:[/green] "
                    f"{entry.label} (QPS={entry.measured_qps:.1f}, "
                    f"{entry.num_requests} requests)"
                )

        elif args.trend_action == "show":
            entries = tracker.list_entries(limit=args.limit)
            if args.output_format == "json":
                import json

                print(json.dumps([e.model_dump() for e in entries], indent=2))
                return

            if not entries:
                console.print("[dim]No trend entries found.[/dim]")
                return

            table = Table(title="Trend History")
            table.add_column("ID", justify="right")
            table.add_column("Label", style="cyan")
            table.add_column("QPS", justify="right")
            table.add_column("Config", style="green")
            table.add_column("TTFT P95", justify="right")
            table.add_column("TPOT P95", justify="right")
            table.add_column("Latency P95", justify="right")
            table.add_column("Requests", justify="right")

            for e in entries:
                table.add_row(
                    str(e.id),
                    e.label,
                    f"{e.measured_qps:.1f}",
                    f"{e.num_prefill}P:{e.num_decode}D",
                    f"{e.ttft_p95_ms:.1f}",
                    f"{e.tpot_p95_ms:.1f}",
                    f"{e.total_latency_p95_ms:.1f}",
                    str(e.num_requests),
                )
            console.print(table)

        elif args.trend_action == "check":
            report = tracker.check(
                lookback=args.lookback,
                threshold=args.threshold,
            )
            if args.output_format == "json":
                print(report.model_dump_json(indent=2))
                return

            console.print(f"\n[bold]📈 Trend Analysis ({report.lookback_count} entries)[/bold]")

            if report.lookback_count < 2:
                console.print("[dim]Not enough data for trend analysis (need >= 2 entries).[/dim]")
                return

            degrading = [a for a in report.alerts if a.is_degrading]
            if degrading:
                console.print(
                    f"\n[bold red]⚠️  {len(degrading)} metric(s) degrading![/bold red]"
                )
                table = Table(title="Degradation Alerts")
                table.add_column("Metric", style="cyan")
                table.add_column("Slope/Run", justify="right")
                table.add_column("Total Change", justify="right")

                for a in degrading:
                    table.add_row(
                        a.metric,
                        f"+{a.slope_per_run:.2f} ms",
                        f"[red]{a.total_change_pct:+.1%}[/red]",
                    )
                console.print(table)
            else:
                console.print("\n[bold green]✅ No degradation trends detected.[/bold green]")
        else:
            console.print("[red]Error: specify 'add', 'show', or 'check'[/red]")
            sys.exit(1)
    finally:
        tracker.close()
