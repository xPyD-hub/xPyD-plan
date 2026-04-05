"""CLI subcommand for benchmark duration advisor."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.duration_advisor import DurationAdvisor


def add_duration_advisor_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the duration-advisor subcommand."""
    p = subparsers.add_parser(
        "duration-advisor",
        help="Analyze whether a benchmark ran long enough for reliable results",
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Target percentile for convergence check (default: 95)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for stabilization (default: 0.05)",
    )
    p.add_argument(
        "--window-steps",
        type=int,
        default=20,
        help="Number of cumulative windows (default: 20)",
    )
    p.add_argument(
        "--safety-multiplier",
        type=float,
        default=1.5,
        help="Safety multiplier on stabilization point (default: 1.5)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_run_duration_advisor)


def _run_duration_advisor(args: argparse.Namespace) -> None:
    """Execute duration-advisor subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(args.benchmark)

    advisor = DurationAdvisor(
        data,
        percentile=args.percentile,
        tolerance=args.tolerance,
        window_steps=args.window_steps,
        safety_multiplier=args.safety_multiplier,
    )
    report = advisor.analyze()

    if args.output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        print()
        return

    console = Console()

    # Summary table
    summary = Table(title="Benchmark Duration Analysis")
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    summary.add_row("Actual Duration", f"{report.actual_duration_s:.1f}s")
    summary.add_row("Actual Requests", str(report.actual_request_count))
    summary.add_row("Recommended Duration", f"{report.recommended_duration_s:.1f}s")
    summary.add_row("Recommended Requests", str(report.recommended_request_count))

    verdict_colors = {
        "SUFFICIENT": "green",
        "MARGINAL": "yellow",
        "INSUFFICIENT": "red",
        "TOO_SHORT": "red bold",
    }
    color = verdict_colors.get(report.verdict.value, "white")
    summary.add_row("Verdict", f"[{color}]{report.verdict.value}[/{color}]")
    console.print(summary)

    if report.metrics:
        console.print()
        detail = Table(title="Per-Metric Stabilization")
        detail.add_column("Metric")
        detail.add_column("Stabilized")
        detail.add_column("Stab. Request #")
        detail.add_column("Stab. Time (s)")
        detail.add_column("Final CV")
        detail.add_column("Rec. Requests")

        for m in report.metrics:
            stab_str = "✅" if m.stabilized else "❌"
            idx_str = (
                str(m.stabilization_request_index)
                if m.stabilization_request_index is not None
                else "—"
            )
            time_str = (
                f"{m.stabilization_time_s:.1f}"
                if m.stabilization_time_s is not None
                else "—"
            )
            detail.add_row(
                m.metric,
                stab_str,
                idx_str,
                time_str,
                f"{m.final_cv:.4f}",
                str(m.recommended_requests),
            )
        console.print(detail)
