"""CLI budget-tracker command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_budget_tracker(args: argparse.Namespace) -> None:
    """Handle the 'budget-tracker' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.budget_tracker import LatencyBudgetTracker

    console = Console()
    data = load_benchmark_auto(args.benchmark)

    tracker = LatencyBudgetTracker(
        sla_ttft_ms=getattr(args, "sla_ttft", None),
        sla_tpot_ms=getattr(args, "sla_tpot", None),
        sla_total_ms=getattr(args, "sla_total", None),
        near_miss_threshold=args.near_miss_threshold,
    )
    report = tracker.analyze(data, top_n=args.top_n)

    if args.output_format == "json":
        print(report.model_dump_json(indent=2))
        return

    # Summary
    console.print("\n[bold]📊 Latency Budget Tracker[/bold]")
    console.print(f"  Total requests: {report.total_requests}")
    console.print(f"  Near-miss threshold: {report.near_miss_threshold:.0%}")
    console.print(f"  Comfortable: {report.comfortable_count}")
    console.print(f"  Near-miss: {report.near_miss_count}")
    console.print(f"  Exceeded: {report.exceeded_count}")

    # Distribution table
    if report.distributions:
        table = Table(title="Budget Consumption Distribution")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("P99", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Near-miss", justify="right")
        table.add_column("Exceeded", justify="right")

        for dist in report.distributions:
            table.add_row(
                dist.metric,
                f"{dist.mean:.2%}",
                f"{dist.p50:.2%}",
                f"{dist.p95:.2%}",
                f"{dist.p99:.2%}",
                f"{dist.max:.2%}",
                str(dist.near_miss_count),
                str(dist.exceeded_count),
            )
        console.print(table)

    # Worst requests
    if report.worst_requests:
        table = Table(title=f"Top {len(report.worst_requests)} Worst Requests")
        table.add_column("Request ID", style="cyan")
        table.add_column("Worst Metric")
        table.add_column("Consumption", justify="right")
        table.add_column("Status")

        for req in report.worst_requests:
            status_color = {
                "COMFORTABLE": "green",
                "MODERATE": "yellow",
                "NEAR_MISS": "bright_red",
                "EXCEEDED": "red bold",
            }.get(req.status.value, "white")
            table.add_row(
                req.request_id,
                req.worst_metric,
                f"{req.worst_ratio:.2%}",
                f"[{status_color}]{req.status.value}[/{status_color}]",
            )
        console.print(table)

    # Alerts
    for alert in report.alerts:
        severity_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(
            alert.severity, "⚪"
        )
        console.print(f"  {severity_icon} [{alert.severity}] {alert.message}")
