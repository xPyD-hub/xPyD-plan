"""CLI workload command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import SLAConfig
from xpyd_plan.workload import classify_workload


def _cmd_workload(args: argparse.Namespace) -> None:
    """Handle the 'workload' subcommand."""
    console = Console()

    benchmarks = [load_benchmark_auto(p) for p in args.benchmark]
    if not benchmarks:
        console.print("[red]Error:[/red] No benchmark files provided.")
        sys.exit(1)

    data = benchmarks[0]

    sla = None
    ttft = getattr(args, "sla_ttft", None)
    tpot = getattr(args, "sla_tpot", None)
    max_lat = getattr(args, "sla_max_latency", None)
    pctl = getattr(args, "sla_percentile", 95.0)
    if any(v is not None for v in (ttft, tpot, max_lat)):
        sla = SLAConfig(ttft_ms=ttft, tpot_ms=tpot, max_latency_ms=max_lat, sla_percentile=pctl)

    report = classify_workload(data=data, sla=sla)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        console.print(json.dumps(report.model_dump(), indent=2))
        return

    # Table output
    console.print(
        f"\n[bold]Workload Characterization[/bold] "
        f"({report.profile.total_requests} requests, "
        f"{report.profile.num_classes} classes)\n"
    )

    table = Table()
    table.add_column("Category", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Fraction", justify="right")
    table.add_column("Avg Prompt", justify="right")
    table.add_column("Avg Output", justify="right")
    table.add_column("P:O Ratio", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("Latency P95", justify="right")
    if sla:
        table.add_column("SLA", justify="center")
        table.add_column("Margin", justify="right")

    for wc in report.classes:
        if wc.request_count == 0:
            continue
        row = [
            wc.category.value,
            str(wc.request_count),
            f"{wc.fraction:.1%}",
            f"{wc.avg_prompt_tokens:.0f}",
            f"{wc.avg_output_tokens:.0f}",
            f"{wc.prompt_output_ratio:.2f}",
            f"{wc.ttft_p95_ms:.1f}",
            f"{wc.tpot_p95_ms:.1f}",
            f"{wc.total_latency_p95_ms:.1f}",
        ]
        if sla:
            if wc.meets_sla is None:
                row.extend(["—", "—"])
            elif wc.meets_sla:
                row.extend(["[green]PASS[/green]", f"{wc.sla_margin_ms:+.1f} ms"])
            else:
                row.extend(["[red]FAIL[/red]", f"{wc.sla_margin_ms:+.1f} ms"])
        table.add_row(*row)

    console.print(table)

    console.print(f"\n[bold]Dominant class:[/bold] {report.profile.dominant_class.value}")

    if report.bottleneck:
        style = "[red]" if report.bottleneck.meets_sla is False else "[yellow]"
        console.print(
            f"[bold]Bottleneck class:[/bold] {style}{report.bottleneck.category.value}[/] "
            f"(margin: {report.bottleneck.sla_margin_ms:+.1f} ms)"
        )
