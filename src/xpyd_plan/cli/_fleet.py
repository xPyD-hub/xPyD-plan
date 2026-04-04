"""CLI fleet command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.models import SLAConfig


def _cmd_fleet(args: argparse.Namespace) -> None:
    """Handle 'fleet' subcommand."""
    import json as json_mod

    import yaml

    from xpyd_plan.fleet import FleetCalculator, GPUTypeConfig

    console = Console()

    # Load GPU configs from YAML

    gpu_configs_data = yaml.safe_load(Path(args.gpu_configs).read_text())
    if not isinstance(gpu_configs_data, list):
        gpu_configs_data = gpu_configs_data.get("gpu_types", [])

    gpu_configs = []
    for entry in gpu_configs_data:
        bench_path = entry.pop("benchmark_file", None)
        if bench_path is None:
            console.print(f"[red]Missing 'benchmark_file' in GPU config: {entry}[/red]")
            sys.exit(1)
        bench_data = load_benchmark_auto(bench_path)
        gpu_configs.append(GPUTypeConfig(benchmark=bench_data, **entry))

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )

    calculator = FleetCalculator(
        gpu_configs,
        sla,
        budget_ceiling=args.budget_ceiling,
        max_options=args.max_options,
    )
    report = calculator.calculate(args.target_qps)

    if args.output_format == "json":
        console.print(json_mod.dumps(report.model_dump(), indent=2))
        return

    # Table output
    console.print("\n[bold]🚀 Fleet Sizing Report[/bold]")
    console.print(f"   Target QPS: {report.target_qps}")
    console.print(f"   GPU types: {', '.join(report.gpu_types)}")
    if report.budget_ceiling is not None:
        console.print(f"   Budget ceiling: {report.budget_ceiling} {report.currency}/hr")

    if not report.options:
        console.print("[red]No fleet options found meeting constraints.[/red]")
        return

    if report.best:
        console.print(f"\n[green bold]Best option:[/green bold] "
                       f"{report.best.total_instances} instances, "
                       f"{report.best.total_hourly_cost} {report.currency}/hr, "
                       f"{report.best.total_qps} QPS")

    table = Table(title="Fleet Options")
    table.add_column("#", style="dim")
    table.add_column("GPU Type")
    table.add_column("Instances")
    table.add_column("P:D Ratio")
    table.add_column("Est. QPS", style="cyan")
    table.add_column("Cost/hr", style="yellow")
    table.add_column("SLA", style="green")

    for idx, option in enumerate(report.options[:10], 1):
        for alloc in option.allocations:
            sla_str = "✅" if alloc.meets_sla else "❌"
            table.add_row(
                str(idx),
                alloc.gpu_type,
                str(alloc.total_instances),
                alloc.ratio_str,
                f"{alloc.estimated_qps:.1f}",
                f"{alloc.hourly_cost:.2f}",
                sla_str,
            )
        # Summary row
        table.add_row(
            "",
            "[bold]TOTAL[/bold]",
            f"[bold]{option.total_instances}[/bold]",
            "",
            f"[bold]{option.total_qps:.1f}[/bold]",
            f"[bold]{option.total_hourly_cost:.2f}[/bold]",
            "✅" if option.meets_sla else "❌",
        )
        table.add_section()

    console.print(table)
