"""CLI entry point for xpyd-plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from xpyd_plan.models import DatasetStats, GPUProfile, SLAConfig
from xpyd_plan.planner import plan


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


def main(argv: list[str] | None = None) -> None:
    """Entry point for `xpyd-plan` command."""
    parser = argparse.ArgumentParser(
        prog="xpyd-plan",
        description="Recommend optimal Prefill:Decode GPU ratio under SLA constraints",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (SLA, GPU profile, budget)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (JSON lines: {prompt_len, output_len})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for recommendation (JSON)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top candidates to display (default: 5)",
    )

    args = parser.parse_args(argv)

    # Load config
    cfg = _load_config(args.config)

    sla = SLAConfig(**(cfg.get("sla") or {}))
    gpu = GPUProfile(**(cfg["gpu"]))
    budget: int = cfg["budget"]

    # Load or build dataset stats
    if args.dataset:
        records = _load_dataset(args.dataset)
        num_requests = cfg.get("num_requests", 1000)
        dataset = DatasetStats.from_records(records, num_requests=num_requests)
    elif "dataset" in cfg:
        dataset = DatasetStats(**cfg["dataset"])
    else:
        print("Error: provide --dataset or include 'dataset' section in config", file=sys.stderr)
        sys.exit(1)

    # Run planner
    result = plan(sla=sla, dataset=dataset, gpu=gpu, budget=budget)

    # Output
    console = Console()

    if result.best:
        console.print(
            f"\n[bold green]✅ Best: {result.best.config.ratio_str}[/bold green]"
            f"  (score: {result.best.score:.2f})"
        )
        console.print(
            f"   TTFT: {result.best.performance.ttft_ms:.1f}ms"
            f"  TPOT: {result.best.performance.tpot_ms:.1f}ms"
            f"  Throughput: {result.best.performance.throughput_rps:.1f} req/s"
            f"  Cost: ${result.best.performance.total_cost_per_hour:.2f}/h"
        )
    else:
        console.print("\n[bold red]❌ No configuration meets SLA constraints.[/bold red]")

    # Top candidates table
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

    # Write JSON output
    if args.output:
        Path(args.output).write_text(result.model_dump_json(indent=2))
        console.print(f"\n[dim]Result written to {args.output}[/dim]")
