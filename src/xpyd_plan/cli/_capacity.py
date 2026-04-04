"""CLI capacity command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from xpyd_plan.models import SLAConfig


def _cmd_plan_capacity(args: argparse.Namespace) -> None:
    """Handle the 'plan-capacity' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.capacity import CapacityPlanner

    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )

    datasets = [load_benchmark_auto(b) for b in args.benchmark]
    planner = CapacityPlanner()
    planner.fit(datasets, sla=sla)
    rec = planner.recommend(
        target_qps=args.target_qps,
        sla=sla,
        max_instances=args.max_instances,
    )

    if args.output_format == "json":
        print(rec.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]📐 Capacity Planning — Target QPS: {rec.target_qps:.1f}[/bold]")
    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}[rec.confidence.value]
    console.print(
        f"\n   [bold {confidence_color}]Recommendation: {rec.recommended_ratio}"
        f" ({rec.recommended_instances} instances)[/bold {confidence_color}]"
    )
    console.print(f"   Confidence: [{confidence_color}]{rec.confidence.value}[/{confidence_color}]")
    console.print(f"   Estimated headroom: {rec.estimated_headroom_pct:+.1f}%")

    if rec.notes:
        for note in rec.notes:
            console.print(f"   ⚠️  {note}")

    # Scaling data table
    table = Table(title="\nScaling Data Points")
    table.add_column("Instances", justify="right")
    table.add_column("Config", style="cyan")
    table.add_column("QPS", justify="right")
    table.add_column("Meets SLA", justify="center")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")

    for p in rec.scaling_points:
        sla_str = "✅" if p.max_qps_meeting_sla is not None else "❌"
        table.add_row(
            str(p.total_instances),
            f"{p.num_prefill}P:{p.num_decode}D",
            f"{p.measured_qps:.1f}",
            sla_str,
            f"{p.ttft_p95_ms:.1f}",
            f"{p.tpot_p95_ms:.1f}",
        )

    console.print(table)
