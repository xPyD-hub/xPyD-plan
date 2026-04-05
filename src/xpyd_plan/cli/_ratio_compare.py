"""CLI ratio-compare command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.ratio_compare import RatioComparer


def _cmd_ratio_compare(args: argparse.Namespace) -> None:
    """Handle the 'ratio-compare' subcommand."""
    console = Console()

    datasets = [load_benchmark_auto(path) for path in args.benchmark]
    comparer = RatioComparer()
    report = comparer.compare(datasets)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Comparison matrix
    console.print("\n[bold]P:D Ratio Latency Comparison[/bold]\n")

    table = Table(title="Latency Percentiles by P:D Ratio")
    table.add_column("Ratio", justify="center")
    table.add_column("Requests", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("TTFT P50", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TTFT P99", justify="right")
    table.add_column("TPOT P50", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("TPOT P99", justify="right")
    table.add_column("Total P50", justify="right")
    table.add_column("Total P95", justify="right")
    table.add_column("Total P99", justify="right")

    # Build set of best values for highlighting
    best_set: set[tuple[str, str]] = set()
    for br in report.best_ratios:
        best_set.add((br.ratio_label, f"{br.metric}_{br.percentile}"))

    for r in report.ratios:
        def _fmt(metric: str, pct: str, val: float) -> str:
            s = f"{val:.1f}"
            if (r.ratio_label, f"{metric}_{pct}") in best_set:
                return f"[bold green]{s}[/bold green]"
            return s

        table.add_row(
            r.ratio_label,
            str(r.request_count),
            f"{r.measured_qps:.1f}",
            _fmt("ttft", "p50", r.ttft.p50),
            _fmt("ttft", "p95", r.ttft.p95),
            _fmt("ttft", "p99", r.ttft.p99),
            _fmt("tpot", "p50", r.tpot.p50),
            _fmt("tpot", "p95", r.tpot.p95),
            _fmt("tpot", "p99", r.tpot.p99),
            _fmt("total_latency", "p50", r.total_latency.p50),
            _fmt("total_latency", "p95", r.total_latency.p95),
            _fmt("total_latency", "p99", r.total_latency.p99),
        )

    console.print(table)

    # Best ratios summary
    console.print(f"\n[bold]Overall Best Ratio: {report.overall_best}[/bold]")
    console.print("(Wins the most metric+percentile combinations)\n")

    best_table = Table(title="Best Ratio per Metric")
    best_table.add_column("Metric")
    best_table.add_column("Percentile")
    best_table.add_column("Best Ratio")
    best_table.add_column("Value (ms)", justify="right")

    for br in report.best_ratios:
        best_table.add_row(br.metric, br.percentile, br.ratio_label, f"{br.value_ms:.1f}")

    console.print(best_table)


def add_ratio_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ratio-compare subcommand parser."""
    parser = subparsers.add_parser(
        "ratio-compare",
        help="Compare latency percentiles across P:D ratio configurations",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        help="Paths to benchmark JSON files (one per P:D ratio)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_ratio_compare)
