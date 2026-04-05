"""CLI baseline command — save and compare latency baselines."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.baseline import BaselineManager
from xpyd_plan.bench_adapter import load_benchmark_auto


def _cmd_baseline(args: argparse.Namespace) -> None:
    """Handle the 'baseline' subcommand."""
    console = Console()
    manager = BaselineManager()

    action = args.baseline_action
    output_format = getattr(args, "output_format", "table")

    if action == "save":
        data = load_benchmark_auto(args.benchmark)
        profile = manager.save_to_file(data, args.output)
        if output_format == "json":
            json.dump(profile.model_dump(), sys.stdout, indent=2)
            sys.stdout.write("\n")
            return
        console.print(f"[green]Baseline saved to {args.output}[/green]")
        table = Table(title="Baseline Profile")
        table.add_column("Metric", justify="left")
        table.add_column("P50 (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("P99 (ms)", justify="right")
        for name, m in [
            ("TTFT", profile.ttft),
            ("TPOT", profile.tpot),
            ("Total Latency", profile.total_latency),
        ]:
            table.add_row(name, f"{m.p50_ms:.1f}", f"{m.p95_ms:.1f}", f"{m.p99_ms:.1f}")
        console.print(table)
        console.print(f"QPS: {profile.qps:.1f}  |  Requests: {profile.request_count}")

    elif action == "compare":
        data = load_benchmark_auto(args.benchmark)
        baseline = manager.load(args.baseline_file)
        report = manager.compare(
            data, baseline, regression_threshold_pct=args.regression_threshold
        )
        if output_format == "json":
            json.dump(report.model_dump(), sys.stdout, indent=2)
            sys.stdout.write("\n")
            return
        table = Table(title="Baseline Comparison")
        table.add_column("Metric", justify="left")
        table.add_column("Percentile", justify="center")
        table.add_column("Baseline (ms)", justify="right")
        table.add_column("Current (ms)", justify="right")
        table.add_column("Delta (ms)", justify="right")
        table.add_column("Delta (%)", justify="right")
        table.add_column("Verdict", justify="center")

        verdict_style = {
            "pass": "[green]PASS[/green]",
            "warn": "[yellow]WARN[/yellow]",
            "fail": "[bold red]FAIL[/bold red]",
        }
        for d in report.deltas:
            table.add_row(
                d.metric,
                d.percentile.upper(),
                f"{d.baseline_ms:.1f}",
                f"{d.current_ms:.1f}",
                f"{d.delta_ms:+.1f}",
                f"{d.delta_pct:+.1f}%",
                verdict_style.get(d.verdict.value, d.verdict.value),
            )
        console.print(table)
        overall_style = verdict_style.get(
            report.overall_verdict.value, report.overall_verdict.value
        )
        console.print(f"\nOverall: {overall_style}")
        console.print(f"[bold]{report.summary}[/bold]")

        if report.overall_verdict.value == "fail":
            sys.exit(1)


def add_baseline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the baseline subcommand parser."""
    parser = subparsers.add_parser(
        "baseline",
        help="Save and compare latency baselines",
    )
    sub = parser.add_subparsers(dest="baseline_action", required=True)

    # baseline save
    save_p = sub.add_parser("save", help="Save a baseline from benchmark data")
    save_p.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    save_p.add_argument(
        "--output", required=True, help="Output path for baseline JSON"
    )
    save_p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # baseline compare
    cmp_p = sub.add_parser("compare", help="Compare benchmark against saved baseline")
    cmp_p.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    cmp_p.add_argument(
        "--baseline-file", required=True, help="Path to saved baseline JSON"
    )
    cmp_p.add_argument(
        "--regression-threshold",
        type=float,
        default=10.0,
        help="Regression threshold %% (default: 10.0)",
    )
    cmp_p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.set_defaults(func=_cmd_baseline)
