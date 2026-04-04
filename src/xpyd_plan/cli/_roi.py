"""CLI roi command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.cost import CostConfig
from xpyd_plan.models import SLAConfig
from xpyd_plan.roi import ROICalculator


def _cmd_roi(args: argparse.Namespace) -> None:
    """Handle the 'roi' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    cost_config = CostConfig.from_yaml(args.cost_model)
    sla = SLAConfig(
        ttft_ms=getattr(args, "sla_ttft", None),
        tpot_ms=getattr(args, "sla_tpot", None),
        max_latency_ms=getattr(args, "sla_total", None),
    )
    migration_cost = getattr(args, "migration_cost", 0.0)

    calculator = ROICalculator(cost_config, sla, migration_cost=migration_cost)
    report = calculator.calculate(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Current vs Optimal projections
    proj_table = Table(title="Cost Projections")
    proj_table.add_column("", justify="left")
    proj_table.add_column("Current", justify="right")
    proj_table.add_column("Optimal", justify="right")

    cur = report.current_projection
    opt = report.optimal_projection
    proj_table.add_row("P:D Ratio", cur.ratio_str, opt.ratio_str)
    proj_table.add_row("Instances", str(cur.total_instances), str(opt.total_instances))
    proj_table.add_row("Meets SLA", str(cur.meets_sla), str(opt.meets_sla))
    proj_table.add_row(
        "Hourly",
        f"{cur.currency} {cur.hourly_cost:.2f}",
        f"{opt.currency} {opt.hourly_cost:.2f}",
    )
    proj_table.add_row(
        "Daily",
        f"{cur.currency} {cur.daily_cost:.2f}",
        f"{opt.currency} {opt.daily_cost:.2f}",
    )
    proj_table.add_row(
        "Monthly (30d)",
        f"{cur.currency} {cur.monthly_cost:.2f}",
        f"{opt.currency} {opt.monthly_cost:.2f}",
    )
    proj_table.add_row(
        "Yearly",
        f"{cur.currency} {cur.yearly_cost:.2f}",
        f"{opt.currency} {opt.yearly_cost:.2f}",
    )
    console.print(proj_table)
    console.print()

    # Savings table
    s = report.savings
    sav_table = Table(title="Savings Estimate")
    sav_table.add_column("Metric", justify="left")
    sav_table.add_column("Value", justify="right")
    sav_table.add_row("Migration", f"{s.current_ratio} → {s.optimal_ratio}")
    sav_table.add_row("Hourly Savings", f"{s.currency} {s.hourly_savings:.2f}")
    sav_table.add_row("Monthly Savings", f"{s.currency} {s.monthly_savings:.2f}")
    sav_table.add_row("Yearly Savings", f"{s.currency} {s.yearly_savings:.2f}")
    sav_table.add_row("Savings %", f"{s.savings_percent:.1f}%")
    if s.migration_cost > 0:
        sav_table.add_row("Migration Cost", f"{s.currency} {s.migration_cost:.2f}")
        if s.break_even_hours is not None:
            sav_table.add_row("Break-Even", f"{s.break_even_hours:.1f} hours")
        else:
            sav_table.add_row("Break-Even", "N/A (no savings)")
    console.print(sav_table)


def add_roi_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the roi subcommand parser."""
    parser = subparsers.add_parser(
        "roi",
        help="Cost projection and ROI calculator",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    parser.add_argument(
        "--cost-model",
        required=True,
        help="YAML file with GPU cost configuration",
    )
    parser.add_argument(
        "--migration-cost",
        type=float,
        default=0.0,
        help="One-time migration cost (default: 0)",
    )
    parser.add_argument("--sla-ttft", type=float, default=None, help="SLA TTFT (ms)")
    parser.add_argument("--sla-tpot", type=float, default=None, help="SLA TPOT (ms)")
    parser.add_argument("--sla-total", type=float, default=None, help="SLA total latency (ms)")
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_roi)
