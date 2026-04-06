"""CLI subcommand for benchmark quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.quality_gate import GateConfig, GateVerdict, QualityGate, load_gate_config


def register(subparsers: Any) -> None:
    """Register the quality-gate subcommand."""
    parser = subparsers.add_parser(
        "quality-gate",
        help="Composite pass/fail gate for benchmark quality (CI/CD friendly)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML gate config file (overrides other flags)",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=100,
        help="Minimum request count (default: 100)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.7,
        help="Minimum data quality score 0-1 (default: 0.7)",
    )
    parser.add_argument(
        "--max-outlier-pct",
        type=float,
        default=10.0,
        help="Maximum outlier percentage (default: 10.0)",
    )
    parser.add_argument(
        "--require-stable-convergence",
        action="store_true",
        default=True,
        help="Require stable percentile convergence (default: True)",
    )
    parser.add_argument(
        "--no-require-stable-convergence",
        action="store_false",
        dest="require_stable_convergence",
        help="Allow marginal convergence",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def _run(args: argparse.Namespace) -> None:
    """Execute the quality-gate subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)

    if args.config:
        config = load_gate_config(args.config)
    else:
        config = GateConfig(
            min_requests=args.min_requests,
            min_quality_score=args.min_quality_score,
            max_outlier_pct=args.max_outlier_pct,
            require_stable_convergence=args.require_stable_convergence,
        )

    gate = QualityGate(config)
    result = gate.evaluate(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(result.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        if result.verdict == GateVerdict.FAIL:
            sys.exit(1)
        return

    verdict_style = {
        GateVerdict.PASS: ("[green]PASS[/green]", "✅"),
        GateVerdict.WARN: ("[yellow]WARN[/yellow]", "⚠️"),
        GateVerdict.FAIL: ("[red]FAIL[/red]", "❌"),
    }

    styled, emoji = verdict_style[result.verdict]
    console.print(f"\n{emoji} Quality Gate: {styled}  ({result.request_count} requests)\n")

    table = Table(title="Gate Checks")
    table.add_column("Check", justify="left")
    table.add_column("Verdict", justify="center")
    table.add_column("Detail", justify="left")
    table.add_column("Threshold", justify="left")

    for check in result.checks:
        s, e = verdict_style[check.verdict]
        table.add_row(check.name, f"{e} {s}", check.detail, check.threshold or "—")

    console.print(table)

    if result.verdict == GateVerdict.FAIL:
        sys.exit(1)
