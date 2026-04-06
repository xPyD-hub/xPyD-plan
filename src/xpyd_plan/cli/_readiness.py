"""CLI readiness command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.readiness import ReadinessAssessor, ReadinessConfig


def _cmd_readiness(args: argparse.Namespace) -> None:
    """Handle the 'readiness' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)

    config = ReadinessConfig()
    assessor = ReadinessAssessor(config=config)
    report = assessor.assess(
        data,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
        cost_per_request=args.cost_per_request,
        optimal_cost_per_request=args.optimal_cost,
        measured_qps=args.measured_qps,
        max_safe_qps=args.max_safe_qps,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Verdict banner
    verdict_style = {
        "ready": "[bold green]READY[/bold green]",
        "caution": "[bold yellow]CAUTION[/bold yellow]",
        "not_ready": "[bold red]NOT READY[/bold red]",
    }
    styled = verdict_style.get(report.verdict.value, report.verdict.value)
    console.print(f"\nDeployment Readiness: {styled}")
    console.print()

    # Checks table
    table = Table(title="Readiness Checks")
    table.add_column("Check", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="left")
    table.add_column("Detail", justify="left")

    status_style = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }

    for c in report.checks:
        table.add_row(
            c.name,
            status_style.get(c.status.value, c.status.value),
            c.value,
            c.threshold,
            c.detail,
        )

    console.print(table)
    console.print()

    if report.blockers:
        console.print(f"[bold red]Blockers:[/bold red] {', '.join(report.blockers)}")
    if report.warnings:
        console.print(f"[yellow]Warnings:[/yellow] {', '.join(report.warnings)}")

    console.print(f"\n[bold]{report.recommendation}[/bold]")

    if report.verdict.value == "not_ready":
        sys.exit(1)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the readiness subcommand."""
    parser = subparsers.add_parser(
        "readiness",
        help="Unified deployment readiness assessment (go/no-go)",
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
        help="TTFT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold in ms",
    )
    parser.add_argument(
        "--cost-per-request",
        type=float,
        default=None,
        help="Actual cost per request",
    )
    parser.add_argument(
        "--optimal-cost",
        type=float,
        default=None,
        help="Optimal cost per request baseline",
    )
    parser.add_argument(
        "--measured-qps",
        type=float,
        default=None,
        help="Current measured QPS",
    )
    parser.add_argument(
        "--max-safe-qps",
        type=float,
        default=None,
        help="Maximum safe QPS from rate limit analysis",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
