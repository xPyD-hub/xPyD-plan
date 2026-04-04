"""Shared CLI helper functions."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.models import SLAConfig


def _print_cost_analysis(
    console: Console,
    analyzer: BenchmarkAnalyzer,
    sla: SLAConfig,
    total: int,
    args: argparse.Namespace,
) -> None:
    """Print cost-aware analysis output."""
    from xpyd_plan.cost import CostAnalyzer, CostConfig

    cost_config = CostConfig.from_yaml(args.cost_model)
    cost_analyzer = CostAnalyzer(cost_config)
    result = analyzer.find_optimal_ratio(total, sla)
    qps = analyzer.data.metadata.measured_qps

    comparison = cost_analyzer.compare(result, qps, args.budget_ceiling)

    console.print(f"\n[bold]💰 Cost Analysis ({cost_config.currency})[/bold]")
    console.print(f"   GPU hourly rate: {cost_config.currency} {cost_config.gpu_hourly_rate:.2f}")
    if args.budget_ceiling is not None:
        console.print(f"   Budget ceiling: {cost_config.currency} {args.budget_ceiling:.2f}/hr")

    if comparison.sla_optimal:
        s = comparison.sla_optimal
        console.print(
            f"\n   [bold]SLA-Optimal:[/bold] {s.ratio_str}"
            f"  — {s.currency} {s.hourly_cost:.2f}/hr"
        )
        if s.cost_per_request is not None:
            console.print(f"   Cost/request: {s.currency} {s.cost_per_request:.6f}")

    if comparison.cost_optimal:
        c = comparison.cost_optimal
        console.print(
            f"   [bold]Cost-Optimal:[/bold] {c.ratio_str}"
            f"  — {c.currency} {c.hourly_cost:.2f}/hr"
        )
        if c.cost_per_request is not None:
            console.print(f"   Cost/request: {c.currency} {c.cost_per_request:.6f}")

    if comparison.sla_optimal and comparison.cost_optimal:
        diff = comparison.sla_optimal.hourly_cost - comparison.cost_optimal.hourly_cost
        if abs(diff) > 0.01:
            console.print(
                f"\n   💡 Switching to cost-optimal saves"
                f" {comparison.sla_optimal.currency} {diff:.2f}/hr"
            )
        else:
            console.print("\n   ✅ SLA-optimal and cost-optimal are the same!")

    # Cost table
    if comparison.all_costs:
        table = Table(title="\nCost Breakdown by P:D Ratio")
        table.add_column("Config", style="cyan")
        table.add_column("Instances", justify="right")
        table.add_column("Hourly Cost", justify="right")
        table.add_column("Cost/Request", justify="right")
        table.add_column("SLA", justify="center")

        for cost in comparison.all_costs:
            sla_str = "✅" if cost.meets_sla else "❌"
            cpr = f"{cost.currency} {cost.cost_per_request:.6f}" if cost.cost_per_request else "N/A"
            table.add_row(
                cost.ratio_str,
                str(cost.total_instances),
                f"{cost.currency} {cost.hourly_cost:.2f}",
                cpr,
                sla_str,
            )
        console.print(table)


def _print_single_analysis(
    console: Console, analyzer: BenchmarkAnalyzer, sla: SLAConfig, total: int, top: int
) -> dict:
    """Print single-scenario analysis and return the result dict."""
    # Show current config analysis
    console.print("\n[bold]📊 Current Configuration Analysis[/bold]")
    meta = analyzer.data.metadata
    console.print(
        f"   Config: {meta.num_prefill_instances}P:{meta.num_decode_instances}D"
        f"  (total: {meta.total_instances} instances, QPS: {meta.measured_qps:.1f})"
    )

    current_sla = analyzer.check_sla(sla)
    sla_status = "[green]✅ PASS[/green]" if current_sla.meets_all else "[red]❌ FAIL[/red]"
    console.print(f"   SLA: {sla_status}")
    console.print(
        f"   TTFT P95: {current_sla.ttft_p95_ms:.1f}ms  P99: {current_sla.ttft_p99_ms:.1f}ms"
    )
    console.print(
        f"   TPOT P95: {current_sla.tpot_p95_ms:.1f}ms  P99: {current_sla.tpot_p99_ms:.1f}ms"
    )

    current_util = analyzer.compute_utilization()
    console.print(
        f"   Utilization: P={current_util.prefill_utilization:.1%}"
        f"  D={current_util.decode_utilization:.1%}"
        f"  Waste={current_util.waste_rate:.1%}"
    )

    result = analyzer.find_optimal_ratio(total, sla)

    console.print(f"\n[bold]🔍 Optimal P:D Ratio Search (total={total} instances)[/bold]")

    if result.best:
        b = result.best
        console.print(
            f"\n[bold green]✅ Recommended: {b.ratio_str}[/bold green]"
            f"  (waste: {b.waste_rate:.1%})"
        )
        console.print(
            f"   P util: {b.prefill_utilization:.1%}"
            f"  D util: {b.decode_utilization:.1%}"
        )
        if b.sla_check:
            console.print(
                f"   TTFT P95: {b.sla_check.ttft_p95_ms:.1f}ms"
                f"  TPOT P95: {b.sla_check.tpot_p95_ms:.1f}ms"
            )
    else:
        console.print(
            "\n[bold red]❌ No P:D ratio meets SLA constraints"
            f" with {total} instances.[/bold red]"
        )

    # Candidates table
    table = Table(title=f"\nTop {top} Candidates")
    table.add_column("Config", style="cyan")
    table.add_column("P Util", justify="right")
    table.add_column("D Util", justify="right")
    table.add_column("Waste", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("SLA", justify="center")

    for c in result.candidates[:top]:
        sla_str = "✅" if c.meets_sla else "❌"
        ttft_str = f"{c.sla_check.ttft_p95_ms:.1f}" if c.sla_check else "N/A"
        tpot_str = f"{c.sla_check.tpot_p95_ms:.1f}" if c.sla_check else "N/A"
        table.add_row(
            c.ratio_str,
            f"{c.prefill_utilization:.1%}",
            f"{c.decode_utilization:.1%}",
            f"{c.waste_rate:.1%}",
            ttft_str,
            tpot_str,
            sla_str,
        )

    console.print(table)
    return result.model_dump()


def _print_sensitivity(
    console: Console, analyzer: BenchmarkAnalyzer, sla: SLAConfig, total: int
) -> None:
    """Print sensitivity analysis output."""
    from xpyd_plan.sensitivity import analyze_sensitivity

    sens = analyze_sensitivity(analyzer, total, sla)

    table = Table(title="\n📈 Sensitivity Analysis — P:D vs SLA Margin")
    table.add_column("Config", style="cyan")
    table.add_column("SLA", justify="center")
    table.add_column("Waste", justify="right")
    if sla.ttft_ms is not None:
        table.add_column("TTFT Margin", justify="right")
    if sla.tpot_ms is not None:
        table.add_column("TPOT Margin", justify="right")
    if sla.max_latency_ms is not None:
        table.add_column("Latency Margin", justify="right")

    for p in sens.points:
        sla_str = "[green]✅[/green]" if p.meets_sla else "[red]❌[/red]"
        row = [p.ratio_str, sla_str, f"{p.waste_rate:.1%}"]
        if sla.ttft_ms is not None and p.ttft_margin_pct is not None:
            color = "green" if p.ttft_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.ttft_margin_pct:+.1%}[/{color}]")
        if sla.tpot_ms is not None and p.tpot_margin_pct is not None:
            color = "green" if p.tpot_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.tpot_margin_pct:+.1%}[/{color}]")
        if sla.max_latency_ms is not None and p.total_latency_margin_pct is not None:
            color = "green" if p.total_latency_margin_pct >= 0 else "red"
            row.append(f"[{color}]{p.total_latency_margin_pct:+.1%}[/{color}]")
        table.add_row(*row)

    console.print(table)

    if sens.cliffs:
        console.print("\n[bold yellow]⚠️  SLA Cliff Points[/bold yellow]")
        for cliff in sens.cliffs:
            if cliff.last_pass and cliff.first_fail:
                console.print(
                    f"   {cliff.last_pass.ratio_str} → {cliff.first_fail.ratio_str}"
                    f"  ({cliff.direction}, fails on: {cliff.failing_metric})"
                )

    if sens.recommendation and sens.recommendation.recommended:
        rec = sens.recommendation
        console.print("\n[bold]🛡️  Safety Recommendation[/bold]")
        console.print(
            f"   Recommended: [bold green]{rec.recommended.ratio_str}[/bold green]"
            f"  (cliff distance: {rec.cliff_distance} steps)"
        )
        if rec.min_margin_pct is not None:
            console.print(f"   Min SLA margin: {rec.min_margin_pct:.1%}")
        if rec.optimal and rec.recommended.num_prefill != rec.optimal.num_prefill:
            console.print(
                f"   (Optimal by waste: {rec.optimal.ratio_str},"
                f" but too close to cliff)"
            )
