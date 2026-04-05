"""CLI token-budget command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.token_budget import TokenBudgetEstimator


def _cmd_token_budget(args: argparse.Namespace) -> None:
    """Handle the 'token-budget' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    estimator = TokenBudgetEstimator(data)
    report = estimator.estimate(
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Token limits table
    table = Table(title="Token Budget Estimates")
    table.add_column("Metric", justify="left")
    table.add_column("SLA (ms)", justify="right")
    table.add_column("Predictor", justify="left")
    table.add_column("Max Tokens", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("Confidence", justify="center")

    for limit in report.limits:
        conf_style = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(
            limit.confidence, ""
        )
        table.add_row(
            limit.metric,
            f"{limit.sla_threshold_ms:.1f}",
            limit.predictor,
            str(limit.max_tokens),
            f"{limit.r_squared:.4f}",
            f"[{conf_style}]{limit.confidence}[/{conf_style}]",
        )

    console.print(table)

    # Effective limits
    if report.effective_max_prompt is not None:
        console.print(f"\n[bold]Effective max prompt tokens:[/bold] {report.effective_max_prompt}")
    if report.effective_max_output is not None:
        console.print(f"[bold]Effective max output tokens:[/bold] {report.effective_max_output}")

    # Warnings
    for warning in report.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")


def add_token_budget_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the token-budget subcommand parser."""
    parser = subparsers.add_parser(
        "token-budget",
        help="Estimate max token counts within SLA thresholds (invert regression)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="TTFT SLA threshold (ms)",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold (ms)",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold (ms)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_token_budget)
