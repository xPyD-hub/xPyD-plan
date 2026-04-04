"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.models import DatasetStats, GPUProfile, SLAConfig


def _load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _load_dataset(path: str) -> list[dict[str, int]]:
    """Load dataset from JSON lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' subcommand."""
    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
    )

    analyzer = BenchmarkAnalyzer()
    analyzer.load_data(args.benchmark)

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

    # Find optimal ratio
    total = args.total_instances or meta.total_instances
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
    top = args.top
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

    # JSON output
    if args.output:
        Path(args.output).write_text(result.model_dump_json(indent=2))
        console.print(f"\n[dim]Result written to {args.output}[/dim]")


def _cmd_plan_legacy(args: argparse.Namespace) -> None:
    """Handle the legacy 'plan' subcommand (deprecated)."""
    warnings.warn(
        "The 'plan' subcommand is deprecated. Use 'analyze' with benchmark data instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    from xpyd_plan.planner import plan

    cfg = _load_config(args.config)
    sla = SLAConfig(**(cfg.get("sla") or {}))
    gpu = GPUProfile(**(cfg["gpu"]))
    budget: int = cfg["budget"]

    if args.dataset:
        records = _load_dataset(args.dataset)
        num_requests = cfg.get("num_requests", 1000)
        dataset = DatasetStats.from_records(records, num_requests=num_requests)
    elif "dataset" in cfg:
        dataset = DatasetStats(**cfg["dataset"])
    else:
        print("Error: provide --dataset or include 'dataset' section in config", file=sys.stderr)
        sys.exit(1)

    result = plan(sla=sla, dataset=dataset, gpu=gpu, budget=budget)

    console = Console()
    if result.best:
        console.print(
            f"\n[bold green]✅ Best: {result.best.config.ratio_str}[/bold green]"
            f"  (score: {result.best.score:.2f})"
        )
    else:
        console.print("\n[bold red]❌ No configuration meets SLA constraints.[/bold red]")

    table = Table(title=f"\nTop {args.top} Candidates (budget={budget} GPUs)")
    table.add_column("Config", style="cyan")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TPOT (ms)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Cost/h", justify="right")
    table.add_column("SLA", justify="center")
    table.add_column("Score", justify="right")

    for c in result.candidates[: args.top]:
        sla_str = "✅" if c.meets_sla else "❌"
        table.add_row(
            c.config.ratio_str,
            f"{c.performance.ttft_ms:.1f}",
            f"{c.performance.tpot_ms:.1f}",
            f"{c.performance.throughput_rps:.1f}",
            f"${c.performance.total_cost_per_hour:.2f}",
            sla_str,
            f"{c.score:.2f}",
        )

    console.print(table)

    if args.output:
        Path(args.output).write_text(result.model_dump_json(indent=2))
        console.print(f"\n[dim]Result written to {args.output}[/dim]")


def main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Analyze benchmark data to find optimal Prefill:Decode instance ratio",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze subcommand (primary)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze benchmark data to find optimal P:D ratio",
    )
    analyze_parser.add_argument(
        "--benchmark", type=str, required=True, help="Path to benchmark JSON file"
    )
    analyze_parser.add_argument(
        "--sla-ttft", type=float, default=None, help="SLA: max TTFT P95 (ms)"
    )
    analyze_parser.add_argument(
        "--sla-tpot", type=float, default=None, help="SLA: max TPOT P95 (ms)"
    )
    analyze_parser.add_argument(
        "--sla-max-latency", type=float, default=None, help="SLA: max total latency P95 (ms)"
    )
    analyze_parser.add_argument(
        "--total-instances", type=int, default=None,
        help="Total instances to optimize for (default: same as benchmark)",
    )
    analyze_parser.add_argument("--top", type=int, default=5, help="Top N candidates to show")
    analyze_parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    # plan subcommand (legacy, deprecated)
    plan_parser = subparsers.add_parser(
        "plan",
        help="[DEPRECATED] Estimate-based planning (use 'analyze' instead)",
    )
    plan_parser.add_argument("--config", type=str, required=True, help="YAML config path")
    plan_parser.add_argument("--dataset", type=str, default=None, help="Dataset JSON lines file")
    plan_parser.add_argument("--top", type=int, default=5, help="Top N candidates")
    plan_parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "plan":
        _cmd_plan_legacy(args)
    else:
        parser.print_help()
        sys.exit(1)
