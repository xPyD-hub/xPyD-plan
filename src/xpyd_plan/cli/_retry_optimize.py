"""CLI retry-optimize command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.retry_optimizer import RetryOptimizer, RetryOptimizerConfig


def _cmd_retry_optimize(args: argparse.Namespace) -> None:
    """Handle the 'retry-optimize' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    config = RetryOptimizerConfig(
        max_amplification=args.max_amplification,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )
    optimizer = RetryOptimizer(config)
    report = optimizer.optimize(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    table = Table(title="Retry Policy Optimization")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Candidates Evaluated", str(report.candidates_evaluated))
    table.add_row("Within Budget", str(report.candidates_within_budget))
    table.add_row("Pareto-Optimal", str(len(report.pareto_frontier)))
    console.print(table)

    # Best policy
    if report.best_policy:
        bp = report.best_policy
        bc = bp.config
        best_table = Table(title="Best Policy")
        best_table.add_column("Parameter", justify="left")
        best_table.add_column("Value", justify="right")
        best_table.add_row("Max Retries", str(bc.max_retries))
        best_table.add_row("Backoff", f"{bc.backoff_ms:.0f}ms ({bc.backoff_type.value})")
        if bc.retry_threshold_ttft_ms is not None:
            best_table.add_row("TTFT Threshold", f"{bc.retry_threshold_ttft_ms:.1f}ms")
        if bc.retry_threshold_tpot_ms is not None:
            best_table.add_row("TPOT Threshold", f"{bc.retry_threshold_tpot_ms:.1f}ms")
        if bc.retry_threshold_total_ms is not None:
            best_table.add_row("Total Threshold", f"{bc.retry_threshold_total_ms:.1f}ms")
        best_table.add_row("Effective Goodput", f"{bp.effective_goodput:.1%}")
        best_table.add_row("Improvement", f"+{bp.goodput_improvement:.1%}")
        best_table.add_row("Amplification", f"{bp.amplification_factor:.2f}x")
        console.print()
        console.print(best_table)

    # Pareto frontier
    if report.pareto_frontier:
        pareto_table = Table(title="Pareto Frontier")
        pareto_table.add_column("Max Retries", justify="right")
        pareto_table.add_column("Backoff", justify="left")
        pareto_table.add_column("Goodput", justify="right")
        pareto_table.add_column("Improvement", justify="right")
        pareto_table.add_column("Amplification", justify="right")
        pareto_table.add_column("Within Budget", justify="center")
        for c in report.pareto_frontier:
            pareto_table.add_row(
                str(c.config.max_retries),
                f"{c.config.backoff_ms:.0f}ms ({c.config.backoff_type.value})",
                f"{c.effective_goodput:.1%}",
                f"+{c.goodput_improvement:.1%}",
                f"{c.amplification_factor:.2f}x",
                "✓" if c.within_budget else "✗",
            )
        console.print()
        console.print(pareto_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_retry_optimize_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the retry-optimize subcommand parser."""
    parser = subparsers.add_parser(
        "retry-optimize",
        help="Find optimal retry policy for benchmark data",
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    parser.add_argument(
        "--max-amplification", type=float, default=2.0,
        help="Maximum allowed load amplification factor (default: 2.0)",
    )
    parser.add_argument("--sla-ttft", type=float, default=None, help="SLA TTFT threshold (ms)")
    parser.add_argument("--sla-tpot", type=float, default=None, help="SLA TPOT threshold (ms)")
    parser.add_argument(
        "--sla-total", type=float, default=None,
        help="SLA total latency threshold (ms)",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_retry_optimize)
