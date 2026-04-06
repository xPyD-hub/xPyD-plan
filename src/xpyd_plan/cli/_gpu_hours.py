"""CLI subcommand for GPU hour calculation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from xpyd_plan.gpu_hours import (
    GPUHourCalculator,
    GPUHourReport,
    HourlyTraffic,
    TrafficProfile,
)


def register(subparsers: Any) -> None:
    """Register the gpu-hours subcommand."""
    p = subparsers.add_parser(
        "gpu-hours",
        help="Estimate GPU hours and costs from traffic profiles",
        description=(
            "Given benchmark data and a daily traffic profile (hourly QPS), "
            "estimate total GPU hours, costs, and auto-scaling savings."
        ),
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    p.add_argument(
        "--traffic-profile",
        required=True,
        help="Traffic profile YAML file (hourly QPS schedule)",
    )
    p.add_argument(
        "--gpu-cost",
        type=float,
        default=2.0,
        help="GPU cost per instance per hour (default: 2.0)",
    )
    p.add_argument(
        "--currency",
        default="USD",
        help="Currency label (default: USD)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    """Execute gpu-hours subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto

    data = load_benchmark_auto(Path(args.benchmark))

    # Load traffic profile
    profile_path = Path(args.traffic_profile)
    with open(profile_path) as f:
        profile_data = yaml.safe_load(f)

    hours = [
        HourlyTraffic(hour=h["hour"], qps=h["qps"]) for h in profile_data["hours"]
    ]
    profile = TrafficProfile(
        hours=hours,
        name=profile_data.get("name", profile_path.stem),
    )

    calc = GPUHourCalculator(data)
    report = calc.calculate(
        profile,
        gpu_cost_per_hour=args.gpu_cost,
        currency=args.currency,
    )

    if args.output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(report)


def _print_table(report: GPUHourReport) -> None:
    """Print report as Rich table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Summary
    console.print(f"\n[bold]GPU Hour Report: {report.profile_name}[/bold]\n")

    summary = Table(title="Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("QPS per Instance", f"{report.qps_per_instance:.2f}")
    summary.add_row("Peak QPS", f"{report.peak_qps:.1f}")
    summary.add_row("Peak Instances", str(report.peak_instances))
    summary.add_row("Off-Peak QPS", f"{report.off_peak_qps:.1f}")
    summary.add_row("Off-Peak Instances", str(report.off_peak_instances))
    summary.add_row("Avg Utilization", f"{report.avg_utilization:.1%}")
    summary.add_row("Daily GPU Hours", f"{report.daily_gpu_hours:.1f}")
    summary.add_row("Monthly GPU Hours", f"{report.monthly_gpu_hours:.1f}")
    summary.add_row(
        "Daily Cost", f"{report.daily_cost:.2f} {report.currency}"
    )
    summary.add_row(
        "Monthly Cost", f"{report.monthly_cost:.2f} {report.currency}"
    )
    console.print(summary)

    # Scaling savings
    s = report.scaling_savings
    savings = Table(title="Auto-Scaling Savings")
    savings.add_column("Metric", style="cyan")
    savings.add_column("Fixed", justify="right")
    savings.add_column("Dynamic", justify="right")
    savings.add_column("Saved", justify="right")
    savings.add_row(
        "Daily GPU Hours",
        f"{s.fixed_daily_gpu_hours:.1f}",
        f"{s.dynamic_daily_gpu_hours:.1f}",
        f"{s.saved_gpu_hours:.1f} ({s.savings_percent:.1f}%)",
    )
    savings.add_row(
        f"Daily Cost ({report.currency})",
        f"{s.fixed_daily_cost:.2f}",
        f"{s.dynamic_daily_cost:.2f}",
        f"{s.saved_cost:.2f}",
    )
    console.print(savings)

    # Hourly breakdown
    hourly = Table(title="Hourly Breakdown")
    hourly.add_column("Hour", justify="right")
    hourly.add_column("QPS", justify="right")
    hourly.add_column("Instances", justify="right")
    hourly.add_column("GPU Hours", justify="right")
    hourly.add_column(f"Cost ({report.currency})", justify="right")
    for hb in report.hourly_breakdown:
        hourly.add_row(
            f"{hb.hour:02d}:00",
            f"{hb.qps:.1f}",
            str(hb.required_instances),
            f"{hb.gpu_hours:.1f}",
            f"{hb.cost:.2f}",
        )
    console.print(hourly)
