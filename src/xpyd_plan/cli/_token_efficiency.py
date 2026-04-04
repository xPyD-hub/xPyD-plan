"""CLI handler for the token-efficiency subcommand."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.benchmark_models import BenchmarkData
from xpyd_plan.token_efficiency import TokenEfficiencyAnalyzer


def add_token_efficiency_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the token-efficiency subcommand."""
    p = subparsers.add_parser(
        "token-efficiency",
        help="Analyze token generation efficiency from benchmark data",
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    p.add_argument(
        "--details",
        action="store_true",
        default=False,
        help="Include per-request efficiency breakdown",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def handle_token_efficiency(args: argparse.Namespace) -> None:
    """Execute the token-efficiency subcommand."""
    import json as _json
    from pathlib import Path

    path = Path(args.benchmark)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = _json.loads(path.read_text())
    data = BenchmarkData(**raw)
    analyzer = TokenEfficiencyAnalyzer(data)
    report = analyzer.analyze(include_details=args.details)

    if args.output_format == "json":
        print(_json.dumps(report.model_dump(), indent=2))
        return

    console = Console()

    # Aggregate table
    t = Table(title="Token Efficiency — Aggregate")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")

    agg = report.aggregate
    t.add_row("Output TPS (mean)", f"{agg.output_tps_mean:.2f}")
    t.add_row("Output TPS (P50)", f"{agg.output_tps_p50:.2f}")
    t.add_row("Output TPS (P95)", f"{agg.output_tps_p95:.2f}")
    t.add_row("Output TPS (P99)", f"{agg.output_tps_p99:.2f}")
    t.add_row("Output TPS (min)", f"{agg.output_tps_min:.2f}")
    t.add_row("Output TPS (max)", f"{agg.output_tps_max:.2f}")
    t.add_row("Decode TPS (mean)", f"{agg.decode_tps_mean:.2f}")
    t.add_row("Decode TPS (P50)", f"{agg.decode_tps_p50:.2f}")
    t.add_row("Total tokens", str(agg.total_tokens))
    t.add_row("Total output tokens", str(agg.total_output_tokens))
    t.add_row("Total prompt tokens", str(agg.total_prompt_tokens))
    console.print(t)

    # Instance efficiency
    ie = report.instance_efficiency
    t2 = Table(title="Instance Efficiency")
    t2.add_column("Metric", style="cyan")
    t2.add_column("Value", justify="right")
    t2.add_row("Output tokens / instance", f"{ie.output_tokens_per_instance:.2f}")
    t2.add_row(
        "Output tokens / decode instance",
        f"{ie.output_tokens_per_decode_instance:.2f}",
    )
    t2.add_row(
        "Prompt tokens / prefill instance",
        f"{ie.prompt_tokens_per_prefill_instance:.2f}",
    )
    console.print(t2)

    # Grade
    console.print(f"\n[bold]Grade:[/bold] {report.grade.value}")
    console.print(f"[dim]{report.grade_reason}[/dim]")
