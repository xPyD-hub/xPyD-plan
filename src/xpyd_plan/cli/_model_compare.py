"""CLI model-compare command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_model_compare(args: argparse.Namespace) -> None:
    """Execute the model-compare subcommand."""

    from xpyd_plan.model_compare import compare_models

    console = Console()

    benchmarks = args.benchmarks
    models = args.models.split(",")

    if len(benchmarks) != len(models):
        console.print(
            f"[red]Error: {len(benchmarks)} benchmark file(s) but {len(models)} model name(s). "
            "Counts must match.[/red]"
        )
        raise SystemExit(1)

    gpu_rate = getattr(args, "gpu_hourly_rate", None)
    result = compare_models(benchmarks, models, gpu_hourly_rate=gpu_rate)

    if args.output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
        return

    # --- Table output ---
    console.print("\n[bold]🔬 Multi-Model Comparison Matrix[/bold]")

    # Profiles summary
    profile_table = Table(title="Model Profiles")
    profile_table.add_column("Model", style="cyan")
    profile_table.add_column("P:D Ratio", justify="center")
    profile_table.add_column("QPS", justify="right")
    profile_table.add_column("TTFT P95", justify="right")
    profile_table.add_column("TPOT P95", justify="right")
    profile_table.add_column("Total P95", justify="right")
    profile_table.add_column("Requests", justify="right")
    if any(p.cost_per_request is not None for p in result.profiles):
        profile_table.add_column("$/req", justify="right")

    for p in result.profiles:
        row = [
            p.model_name,
            f"{p.num_prefill_instances}:{p.num_decode_instances}",
            f"{p.measured_qps:.1f}",
            f"{p.ttft_p95_ms:.1f}",
            f"{p.tpot_p95_ms:.1f}",
            f"{p.total_latency_p95_ms:.1f}",
            str(p.request_count),
        ]
        if any(pr.cost_per_request is not None for pr in result.profiles):
            row.append(f"{p.cost_per_request:.6f}" if p.cost_per_request is not None else "N/A")
        profile_table.add_row(*row)

    console.print(profile_table)

    # Rankings
    rank_table = Table(title="\nModel Rankings")
    rank_table.add_column("Rank", justify="center", style="bold")
    rank_table.add_column("Model", style="cyan")
    rank_table.add_column("Latency Score", justify="right")
    rank_table.add_column("Cost Score", justify="right")
    rank_table.add_column("Recommendation")

    for r in result.rankings:
        rank_table.add_row(
            str(r.rank),
            r.model_name,
            f"{r.latency_score:.3f}",
            f"{r.cost_efficiency_score:.3f}" if r.cost_efficiency_score is not None else "N/A",
            r.recommendation,
        )

    console.print(rank_table)

    console.print(f"\n[bold green]🏆 Best latency: {result.best_latency_model}[/bold green]")
    if result.best_cost_model:
        console.print(f"[bold green]💰 Best cost: {result.best_cost_model}[/bold green]")
