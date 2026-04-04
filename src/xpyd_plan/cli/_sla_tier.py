"""CLI subcommand for multi-SLA tier analysis."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table


def add_sla_tier_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the sla-tier subcommand."""
    p = subparsers.add_parser(
        "sla-tier",
        help="Analyze benchmark data against multiple SLA tiers",
    )
    p.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    p.add_argument("--tiers", required=True, help="Path to YAML tier definitions")
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def _run_sla_tier(args: argparse.Namespace) -> None:
    """Execute sla-tier subcommand."""
    from xpyd_plan.analyzer import BenchmarkAnalyzer
    from xpyd_plan.sla_tier import SLATierAnalyzer, load_tiers_from_yaml

    tiers = load_tiers_from_yaml(args.tiers)

    analyzer = BenchmarkAnalyzer()
    data = analyzer.load_data(args.benchmark)

    tier_analyzer = SLATierAnalyzer(data)
    report = tier_analyzer.analyze(tiers)

    if args.output_format == "json":
        print(json.dumps(report.model_dump(), indent=2))
        return

    console = Console()

    table = Table(title="Multi-SLA Tier Analysis")
    table.add_column("Tier", style="cyan")
    table.add_column("SLA (TTFT/TPOT/Total)", style="white")
    table.add_column("Percentile", style="white")
    table.add_column("Best Ratio", style="green")
    table.add_column("Waste", style="yellow")
    table.add_column("Meets SLA", style="white")

    for tr in report.tier_results:
        sla = tr.tier.sla
        sla_str = "/".join(
            str(v) if v is not None else "-"
            for v in [sla.ttft_ms, sla.tpot_ms, sla.max_latency_ms]
        )
        if tr.best:
            ratio_str = tr.best.ratio_str
            waste_str = f"{tr.best.waste_rate:.1%}"
            meets = "✓"
        else:
            ratio_str = "—"
            waste_str = "—"
            meets = "✗"
        table.add_row(
            tr.tier.name,
            sla_str,
            f"P{sla.sla_percentile:g}",
            ratio_str,
            waste_str,
            meets,
        )

    console.print(table)

    if report.unified_best:
        console.print(
            f"\n[bold green]Unified recommendation:[/] {report.unified_best.ratio_str} "
            f"(waste: {report.unified_best.waste_rate:.1%}) — meets ALL tiers"
        )
    else:
        console.print(
            "\n[bold red]No single P:D ratio satisfies all tiers simultaneously.[/]"
        )
