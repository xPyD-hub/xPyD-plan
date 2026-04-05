"""CLI subcommand for benchmark diff report."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.diff_report import DiffReporter


def _run(args: argparse.Namespace) -> None:
    """Handle the 'diff-report' subcommand."""
    console = Console()

    baseline_data = load_benchmark_auto(args.baseline)
    target_data = load_benchmark_auto(args.target)

    reporter = DiffReporter(
        regression_threshold_pct=args.regression_threshold,
    )
    report = reporter.compare(
        baseline_data,
        target_data,
        baseline_name=args.baseline,
        target_name=args.target,
    )

    if args.output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif args.output_format == "markdown":
        console.print(reporter.to_markdown(report))
    else:
        table = Table(title="Benchmark Diff Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Status", justify="center")

        qd = report.qps_diff
        rel = (
            f"{qd.relative_delta_pct:+.1f}%"
            if qd.relative_delta_pct is not None
            else "N/A"
        )
        table.add_row(
            "QPS",
            f"{qd.baseline_value:.2f}",
            f"{qd.target_value:.2f}",
            rel,
            qd.direction.value,
        )

        for section in report.latency_diffs:
            for label, diff in [
                ("P50", section.p50),
                ("P95", section.p95),
                ("P99", section.p99),
            ]:
                rel = (
                    f"{diff.relative_delta_pct:+.1f}%"
                    if diff.relative_delta_pct is not None
                    else "N/A"
                )
                table.add_row(
                    f"{section.metric} {label}",
                    f"{diff.baseline_value:.2f}",
                    f"{diff.target_value:.2f}",
                    rel,
                    diff.direction.value,
                )

        console.print(table)
        console.print(
            f"\nVerdict: [bold]{report.summary.verdict}[/bold] — "
            f"{report.summary.improvements} improved, "
            f"{report.summary.regressions} regressed, "
            f"{report.summary.unchanged} unchanged"
        )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the diff-report subcommand."""
    p = subparsers.add_parser(
        "diff-report",
        help="Comprehensive side-by-side benchmark diff report",
    )
    p.add_argument("--baseline", required=True, help="Baseline benchmark JSON file")
    p.add_argument("--target", required=True, help="Target benchmark JSON file")
    p.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_run)
