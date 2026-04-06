"""CLI subcommand for workload mix optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from xpyd_plan.workload_mix import (
    AllocationMode,
    MixOptimizationResult,
    WorkloadMixOptimizer,
    WorkloadSpec,
)


def register(subparsers: Any) -> None:
    """Register the workload-mix subcommand."""
    p = subparsers.add_parser(
        "workload-mix",
        help="Optimize GPU allocation across multiple workloads",
        description=(
            "Given benchmark data for multiple workloads, find the minimum "
            "total GPU instances while meeting per-workload SLA constraints."
        ),
    )
    p.add_argument(
        "--workload",
        action="append",
        required=True,
        metavar="YAML",
        help="Workload spec YAML file (repeatable, one per workload)",
    )
    p.add_argument(
        "--total-gpus",
        type=int,
        default=None,
        help="Total GPU budget (default: unlimited)",
    )
    p.add_argument(
        "--max-per-workload",
        type=int,
        default=32,
        help="Max instances per workload role (default: 32)",
    )
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output as JSON",
    )
    p.set_defaults(func=_run)


def _load_workload(path: str) -> WorkloadSpec:
    """Load a WorkloadSpec from a YAML file.

    Expected YAML format:
        name: "workload-a"
        benchmark: "path/to/benchmark.json"
        sla:
          ttft_p99_ms: 200
          tpot_p99_ms: 50
        min_prefill: 1
        min_decode: 1
        weight: 1.0
    """
    from xpyd_plan.benchmark_models import BenchmarkData
    from xpyd_plan.models import SLAConfig

    data = yaml.safe_load(Path(path).read_text())
    bench_path = Path(path).parent / data["benchmark"]
    bench_data = BenchmarkData.model_validate_json(bench_path.read_text())
    sla = SLAConfig(**data.get("sla", {}))
    return WorkloadSpec(
        name=data.get("name", bench_path.stem),
        benchmark_data=bench_data,
        sla=sla,
        min_prefill=data.get("min_prefill", 1),
        min_decode=data.get("min_decode", 1),
        weight=data.get("weight", 1.0),
    )


def _print_table(result: MixOptimizationResult) -> None:
    """Print results as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        # Fallback plain text
        print(result.summary)
        for a in result.allocations:
            print(f"  {a.name}: {a.ratio_str} waste={a.weighted_waste:.3f} sla={a.meets_sla}")
        return

    console = Console()
    console.print(f"\n[bold]{result.summary}[/bold]\n")

    if not result.feasible:
        return

    table = Table(title="Workload Allocations")
    table.add_column("Workload", style="cyan")
    table.add_column("P:D Ratio", style="green")
    table.add_column("Instances", justify="right")
    table.add_column("P Waste", justify="right")
    table.add_column("D Waste", justify="right")
    table.add_column("Weighted Waste", justify="right")
    table.add_column("SLA Met", justify="center")

    for a in result.allocations:
        table.add_row(
            a.name,
            a.ratio_str,
            str(a.total_instances),
            f"{a.prefill_waste:.1%}",
            f"{a.decode_waste:.1%}",
            f"{a.weighted_waste:.4f}",
            "✅" if a.meets_sla else "❌",
        )

    console.print(table)
    console.print(f"\nCandidates evaluated: {result.candidates_evaluated}")


def _run(args: argparse.Namespace) -> None:
    """Execute workload-mix optimization."""
    workloads: list[WorkloadSpec] = []
    for wpath in args.workload:
        workloads.append(_load_workload(wpath))

    optimizer = WorkloadMixOptimizer(max_instances_per_workload=args.max_per_workload)
    result = optimizer.optimize(
        workloads,
        total_gpu_budget=args.total_gpus,
        mode=AllocationMode.DEDICATED,
    )

    if args.json_output:
        print(result.model_dump_json(indent=2))
    else:
        _print_table(result)

    if not result.feasible:
        sys.exit(1)
