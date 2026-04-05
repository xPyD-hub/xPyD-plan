"""CLI subcommand for weighted goodput analysis."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the weighted-goodput subcommand."""
    p = subparsers.add_parser(
        "weighted-goodput",
        help="Weighted goodput analysis with continuous SLA scoring",
    )
    p.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    p.add_argument("--sla-ttft", type=float, default=None, help="TTFT SLA (ms)")
    p.add_argument("--sla-tpot", type=float, default=None, help="TPOT SLA (ms)")
    p.add_argument(
        "--sla-total", type=float, default=None, help="Total latency SLA (ms)"
    )
    p.add_argument(
        "--grace-factor",
        type=float,
        default=0.5,
        help="Grace window as fraction of threshold (default: 0.5)",
    )
    p.add_argument(
        "--aggregation",
        choices=["min", "mean"],
        default="min",
        help="Score aggregation method (default: min)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    from ..analyzer import load_benchmark_auto
    from ..weighted_goodput import ScoringConfig, WeightedGoodputAnalyzer

    if args.sla_ttft is None and args.sla_tpot is None and args.sla_total is None:
        print("Error: at least one SLA threshold required", file=sys.stderr)
        sys.exit(1)

    data = load_benchmark_auto(args.benchmark)
    config = ScoringConfig(
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_latency_ms=args.sla_total,
        grace_factor=args.grace_factor,
        aggregation=args.aggregation,
    )
    analyzer = WeightedGoodputAnalyzer(config)
    report = analyzer.analyze(data)

    if args.output_format == "json":
        print(json.dumps(report.model_dump(), indent=2))
        return

    console = Console()

    t = Table(title="Weighted Goodput Analysis")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")

    t.add_row("Total Requests", str(report.total_requests))
    t.add_row("Weighted Goodput", f"{report.weighted_goodput:.4f}")
    t.add_row("Binary Goodput Ratio", f"{report.binary_goodput_ratio:.4f}")
    t.add_row("Raw QPS", f"{report.raw_qps:.2f}")
    t.add_row("Weighted Goodput QPS", f"{report.weighted_goodput_qps:.2f}")
    t.add_row("Near-Miss Requests", str(report.near_miss_count))
    t.add_row("Near-Miss Fraction", f"{report.near_miss_fraction:.4f}")
    console.print(t)

    # Score distribution
    dt = Table(title="Score Distribution")
    dt.add_column("Range", style="cyan")
    dt.add_column("Count", style="green")
    dt.add_column("Fraction", style="yellow")
    for b in report.score_distribution.buckets:
        dt.add_row(
            f"[{b.lower:.1f}, {b.upper:.1f}]",
            str(b.count),
            f"{b.fraction:.4f}",
        )
    console.print(dt)
    console.print(f"\n[bold]{report.recommendation}[/bold]")
