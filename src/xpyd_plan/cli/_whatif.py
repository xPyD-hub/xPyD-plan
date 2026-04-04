"""CLI whatif command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.models import SLAConfig


def _cmd_what_if(args: argparse.Namespace) -> None:
    """Handle the 'what-if' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.whatif import WhatIfSimulator

    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )

    data = load_benchmark_auto(args.benchmark)
    sim = WhatIfSimulator()
    sim.load(data)

    # Build scenario specs from CLI args
    scenarios: list[dict] = []
    if args.scale_qps:
        for part in args.scale_qps.split(","):
            part = part.strip().rstrip("xX")
            scenarios.append({"scale_qps": float(part)})

    if args.add_instances is not None:
        if scenarios:
            # Combine with each QPS scenario
            combined = []
            for s in scenarios:
                combined.append({**s, "add_instances": args.add_instances})
            # Also add instance-only scenario
            combined.append({"add_instances": args.add_instances})
            scenarios = combined
        else:
            scenarios.append({"add_instances": args.add_instances})

    if not scenarios:
        console.print("[red]Error: specify --scale-qps and/or --add-instances[/red]")
        sys.exit(1)

    comparison = sim.compare(scenarios, sla)

    if args.output_format == "json":
        print(comparison.model_dump_json(indent=2))
        return

    # Rich table output
    meta = data.metadata
    console.print("\n[bold]🔮 What-If Analysis[/bold]")
    console.print(
        f"   Baseline: {meta.num_prefill_instances}P:{meta.num_decode_instances}D"
        f"  ({meta.total_instances} instances, QPS: {meta.measured_qps:.1f})"
    )

    table = Table(title="\nScenario Comparison")
    table.add_column("Scenario", style="cyan")
    table.add_column("Instances", justify="right")
    table.add_column("Best P:D", style="green")
    table.add_column("Waste", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("SLA", justify="center")

    # Baseline row
    b = comparison.baseline
    if b.best and b.best.sla_check:
        table.add_row(
            "[bold]Baseline[/bold]",
            str(b.total_instances),
            b.best.ratio_str,
            f"{b.best.waste_rate:.1%}",
            f"{b.best.sla_check.ttft_p95_ms:.1f}",
            f"{b.best.sla_check.tpot_p95_ms:.1f}",
            "✅" if b.best.meets_sla else "❌",
        )
    else:
        table.add_row(
            "[bold]Baseline[/bold]",
            str(b.total_instances),
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "❌",
        )

    # Scenario rows
    for s in comparison.scenarios:
        if s.best and s.best.sla_check:
            table.add_row(
                s.label,
                str(s.total_instances),
                s.best.ratio_str,
                f"{s.best.waste_rate:.1%}",
                f"{s.best.sla_check.ttft_p95_ms:.1f}",
                f"{s.best.sla_check.tpot_p95_ms:.1f}",
                "✅" if s.best.meets_sla else "❌",
            )
        else:
            table.add_row(
                s.label,
                str(s.total_instances),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "❌",
            )

    console.print(table)
