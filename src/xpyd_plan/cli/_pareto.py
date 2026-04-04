"""CLI pareto command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.models import SLAConfig


def _cmd_pareto(args: argparse.Namespace) -> None:
    """Handle 'pareto' subcommand."""
    import json as json_mod

    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.pareto import ParetoAnalyzer, ParetoObjective

    console = Console()

    # Load benchmark data
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

    # Parse objectives
    objectives = None
    if args.objectives:
        objectives = [ParetoObjective(o) for o in args.objectives]

    # Parse weights
    weights = None
    if args.weights:
        weights = {}
        for pair in args.weights.split(","):
            k, v = pair.strip().split("=")
            weights[k.strip()] = float(v.strip())

    pareto_analyzer = ParetoAnalyzer(cost_config=cost_config)
    frontier = pareto_analyzer.analyze(
        analysis,
        measured_qps=measured_qps,
        objectives=objectives,
        weights=weights,
    )

    if args.output_format == "json":
        console.print(json_mod.dumps(frontier.model_dump(), indent=2))
        return

    # Table output
    if not frontier.frontier:
        console.print("[yellow]No Pareto-optimal candidates found.[/yellow]")
        return

    count = len(frontier.frontier)
    console.print(f"\n[bold]🎯 Pareto Frontier ({count} optimal candidates)[/bold]")
    console.print(f"   Objectives: {', '.join(frontier.objectives_used)}")
    console.print(f"   Weights: {frontier.weights}")

    table = Table(title="Pareto-Optimal P:D Ratios")
    table.add_column("Ratio", style="cyan")
    table.add_column("Latency P95 (ms)", justify="right")
    if "cost" in frontier.objectives_used:
        table.add_column("Hourly Cost", justify="right")
    table.add_column("Waste Rate", justify="right")
    table.add_column("Score", justify="right", style="green")

    for c in frontier.frontier:
        row = [
            c.ratio_str,
            f"{c.latency_ms:.1f}",
        ]
        if "cost" in frontier.objectives_used:
            row.append(f"{c.hourly_cost:.2f}" if c.hourly_cost is not None else "N/A")
        row.extend([
            f"{c.waste_rate:.3f}",
            f"{c.weighted_score:.4f}" if c.weighted_score is not None else "N/A",
        ])
        table.add_row(*row)

    console.print(table)

    if frontier.best_weighted:
        console.print(
            f"\n   [bold green]✅ Best weighted:[/bold green] {frontier.best_weighted.ratio_str}"
            f" (score: {frontier.best_weighted.weighted_score:.4f})"
        )

    if args.include_dominated and frontier.dominated:
        console.print(f"\n[dim]Dominated candidates: {len(frontier.dominated)}[/dim]")
        for c in frontier.dominated:
            console.print(
                f"   [dim]{c.ratio_str}  latency={c.latency_ms:.1f}ms"
                f"  waste={c.waste_rate:.3f}[/dim]"
            )
