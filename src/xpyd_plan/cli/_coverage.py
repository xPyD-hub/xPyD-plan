"""CLI subcommand for benchmark coverage analysis."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.coverage import CoverageAnalyzer


def add_coverage_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the coverage subcommand."""
    p = subparsers.add_parser(
        "coverage",
        help="Assess P:D ratio space exploration completeness",
    )
    p.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files to analyze",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def _cmd_coverage(args: argparse.Namespace) -> None:
    """Execute the coverage subcommand."""
    datasets = []
    for path in args.benchmark:
        data = load_benchmark_auto(path)
        datasets.append(data)

    analyzer = CoverageAnalyzer(datasets)
    report = analyzer.analyze()

    if args.output_format == "json":
        print(json.dumps(report.model_dump(), indent=2, default=str))
        return

    console = Console()

    # Summary table
    t = Table(title="Benchmark Coverage Report")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")

    m = report.metrics
    t.add_row("Total instances", str(report.total_instances))
    t.add_row("Possible ratios", str(m.total_possible_ratios))
    t.add_row("Benchmarked ratios", str(m.benchmarked_ratios))
    t.add_row("Coverage", f"{m.coverage_fraction:.1%}")
    t.add_row("Boundaries covered", "✓" if m.has_boundaries else "✗")
    t.add_row("Balanced covered", "✓" if m.has_balanced else "✗")
    t.add_row("Max gap size", str(m.max_gap_size))
    t.add_row("Spread score", f"{m.spread_score:.2f}")
    t.add_row("Score", f"{report.score}/100")
    t.add_row("Grade", report.grade.value)
    console.print(t)

    # Benchmarked ratios
    if report.benchmarked:
        bt = Table(title="Benchmarked Ratios")
        bt.add_column("Prefill", style="cyan")
        bt.add_column("Decode", style="cyan")
        bt.add_column("Ratio", style="green")
        for p, d in report.benchmarked:
            bt.add_row(str(p), str(d), f"{p}P:{d}D")
        console.print(bt)

    # Top gaps
    top_gaps = [g for g in report.gaps if g.priority in ("critical", "high")][:10]
    if top_gaps:
        gt = Table(title="Top Gaps (suggested next benchmarks)")
        gt.add_column("Ratio", style="cyan")
        gt.add_column("Distance", style="yellow")
        gt.add_column("Priority", style="red")
        for g in top_gaps:
            gt.add_row(g.ratio_str, str(g.distance_to_nearest), g.priority)
        console.print(gt)

    # Recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")
