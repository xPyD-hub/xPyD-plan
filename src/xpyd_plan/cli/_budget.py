"""CLI budget command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_budget(args: argparse.Namespace) -> None:
    """Handle 'budget' subcommand."""
    import json as json_mod

    from xpyd_plan.benchmark_models import BenchmarkData
    from xpyd_plan.budget import AllocationStrategy, BudgetAllocator

    console = Console()

    bench_path = args.benchmark
    with open(bench_path) as f:
        data = json_mod.load(f)
    benchmark = BenchmarkData.model_validate(data)

    strategy = AllocationStrategy(args.strategy)
    allocator = BudgetAllocator(benchmark)
    result = allocator.allocate(
        total_budget_ms=args.total_budget_ms,
        strategy=strategy,
        percentile=getattr(args, "sla_percentile", 95.0),
    )

    if args.output_format == "json":
        console.print_json(json_mod.dumps(result.model_dump()))
    else:
        table = Table(title="SLA Budget Allocation")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total budget", f"{result.total_budget_ms} ms")
        table.add_row("Strategy", result.strategy.value)
        table.add_row("Percentile", f"P{result.percentile}")
        table.add_row("Observed TTFT:TPOT ratio", f"{result.observed_ratio:.2%}")
        table.add_row("Feasible", "✅" if result.feasible else "❌")
        table.add_row("", "")
        table.add_row("TTFT budget", f"{result.ttft.budget_ms} ms ({result.ttft.share:.0%})")
        table.add_row("TTFT observed", f"{result.ttft.observed_ms} ms")
        table.add_row("TTFT headroom", f"{result.ttft.headroom_ms} ms")
        table.add_row("", "")
        table.add_row("TPOT budget", f"{result.tpot.budget_ms} ms ({result.tpot.share:.0%})")
        table.add_row("TPOT observed", f"{result.tpot.observed_ms} ms")
        table.add_row("TPOT headroom", f"{result.tpot.headroom_ms} ms")
        console.print(table)

    raise SystemExit(0)
