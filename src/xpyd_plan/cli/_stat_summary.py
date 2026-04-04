"""CLI stat-summary command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.stat_summary import StatSummaryAnalyzer


def _cmd_stat_summary(args: argparse.Namespace) -> None:
    """Handle the 'stat-summary' subcommand."""
    console = Console()

    datasets = [load_benchmark_auto(f) for f in args.benchmark]
    analyzer = StatSummaryAnalyzer(datasets)
    report = analyzer.summarize()

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Aggregated stats table
    agg = report.aggregated
    summary = Table(title=f"Multi-Benchmark Statistical Summary ({agg.num_runs} runs)")
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="right")
    summary.add_row("Total Runs", str(agg.num_runs))
    summary.add_row("Total Requests", str(agg.total_requests))
    summary.add_row("Mean QPS", f"{agg.mean_qps:.2f}")
    summary.add_row("QPS Std Dev", f"{agg.std_qps:.2f}")
    summary.add_row("QPS CV", f"{agg.cv_qps:.4f}")
    summary.add_row("Repeatability CV", f"{agg.repeatability_cv:.4f}")
    console.print(summary)
    console.print()

    # Per-metric table
    metrics = [("TTFT", agg.ttft), ("TPOT", agg.tpot), ("Total Latency", agg.total_latency)]
    for name, metric_agg in metrics:
        t = Table(title=f"{name} Cross-Run Statistics")
        t.add_column("Statistic", justify="left")
        t.add_column("Value", justify="right")
        t.add_row("Mean of Means", f"{metric_agg.mean_of_means_ms:.2f} ms")
        t.add_row("Std of Means", f"{metric_agg.std_of_means_ms:.2f} ms")
        t.add_row("Mean of P95s", f"{metric_agg.mean_of_p95s_ms:.2f} ms")
        t.add_row("Std of P95s", f"{metric_agg.std_of_p95s_ms:.2f} ms")
        t.add_row("CV of P95s", f"{metric_agg.cv_of_p95s:.4f}")
        console.print(t)
        console.print()

    # Per-run table
    runs_table = Table(title="Per-Run Summary")
    runs_table.add_column("Run", justify="right")
    runs_table.add_column("Requests", justify="right")
    runs_table.add_column("QPS", justify="right")
    runs_table.add_column("TTFT P95", justify="right")
    runs_table.add_column("TPOT P95", justify="right")
    runs_table.add_column("Total P95", justify="right")
    runs_table.add_column("Stability", justify="center")

    stability_styles = {
        "MOST_STABLE": "bold green",
        "LEAST_STABLE": "bold red",
        "NORMAL": "",
    }
    for run in report.runs:
        style = stability_styles.get(run.stability.value, "")
        label = run.stability.value
        runs_table.add_row(
            str(run.index),
            str(run.request_count),
            f"{run.measured_qps:.2f}",
            f"{run.ttft_p95_ms:.2f} ms",
            f"{run.tpot_p95_ms:.2f} ms",
            f"{run.total_latency_p95_ms:.2f} ms",
            f"[{style}]{label}[/{style}]" if style else label,
        )
    console.print(runs_table)


def add_stat_summary_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the stat-summary subcommand parser."""
    parser = subparsers.add_parser(
        "stat-summary",
        help="Cross-run statistical summary of multiple benchmark files",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        help="Benchmark JSON files (at least 2)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_stat_summary)
