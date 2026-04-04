"""CLI recommend command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import SLAConfig


def _cmd_recommend(args: argparse.Namespace) -> None:
    """Handle 'recommend' subcommand."""
    import json as json_mod

    from xpyd_plan.recommender import RecommendationEngine

    console = Console()

    data = load_benchmark_auto(args.benchmark[0])

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )
    total = args.total_instances or data.metadata.total_instances

    analyzer = BenchmarkAnalyzer(data)
    analysis = analyzer.find_optimal_ratio(total, sla)
    measured_qps = data.metadata.measured_qps

    # Cost config
    cost_config = None
    if args.cost_model:
        from xpyd_plan.cost import CostConfig

        cost_config = CostConfig.from_yaml(args.cost_model)

    engine = RecommendationEngine(
        cost_config=cost_config,
        waste_threshold=args.waste_threshold,
    )
    report = engine.analyze(analysis, measured_qps=measured_qps)

    if args.output_format == "json":
        console.print(json_mod.dumps(report.model_dump(), indent=2))
        return

    # Table output
    console.print("\n[bold]📋 Recommendation Report[/bold]")
    console.print(f"   {report.analysis_summary}")
    if report.current_ratio:
        console.print(f"   Current: {report.current_ratio}")
    if report.optimal_ratio:
        console.print(f"   Optimal: {report.optimal_ratio}")

    if not report.recommendations:
        console.print("[green]No recommendations.[/green]")
        return

    table = Table(title="Recommendations")
    table.add_column("Priority", style="bold")
    table.add_column("Action")
    table.add_column("Title")
    table.add_column("Detail")
    table.add_column("Suggested Ratio", style="cyan")

    priority_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
    }

    for rec in report.recommendations:
        style = priority_styles.get(rec.priority.value, "")
        table.add_row(
            f"[{style}]{rec.priority.value.upper()}[/{style}]",
            rec.action.value,
            rec.title,
            rec.detail,
            rec.suggested_ratio or "-",
        )

    console.print(table)
