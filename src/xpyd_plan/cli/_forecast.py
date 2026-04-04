"""CLI subcommand: forecast — capacity forecasting from trend data."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.forecaster import CapacityForecaster, ForecastMethod
from xpyd_plan.trend import TrendTracker


def add_forecast_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``forecast`` subcommand."""
    p = subparsers.add_parser(
        "forecast",
        help="Forecast capacity from historical trend data",
    )
    p.add_argument(
        "--trend-db",
        required=True,
        help="Path to TrendTracker SQLite database",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=30,
        help="Planning horizon in days (default: 30)",
    )
    p.add_argument(
        "--method",
        choices=["linear", "exponential"],
        default="linear",
        help="Extrapolation method (default: linear)",
    )
    p.add_argument("--sla-ttft", type=float, default=None, help="TTFT P95 SLA threshold (ms)")
    p.add_argument("--sla-tpot", type=float, default=None, help="TPOT P95 SLA threshold (ms)")
    p.add_argument(
        "--sla-total", type=float, default=None,
        help="Total latency P95 SLA threshold (ms)",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_run_forecast)


def _run_forecast(args: argparse.Namespace) -> None:
    tracker = TrendTracker(db_path=args.trend_db)
    try:
        forecaster = CapacityForecaster(tracker)
        report = forecaster.forecast(
            horizon_days=args.horizon_days,
            method=ForecastMethod(args.method),
            sla_ttft_ms=args.sla_ttft,
            sla_tpot_ms=args.sla_tpot,
            sla_total_ms=args.sla_total,
        )
    finally:
        tracker.close()

    if args.output_format == "json":
        print(json.dumps(report.model_dump(), indent=2))
        return

    console = Console()

    # Summary
    title = (
        f"\n[bold]Capacity Forecast[/bold]"
        f" ({report.method.value}, {report.horizon_days}-day horizon)"
    )
    console.print(title)
    console.print(f"Historical entries: {report.num_historical_entries}")

    if report.num_historical_entries < 2:
        console.print("[yellow]Not enough historical data (need >= 2 entries).[/yellow]")
        return

    # Projections table
    if report.projections:
        tbl = Table(title="Projected Latency (P95)")
        tbl.add_column("Days", justify="right")
        tbl.add_column("TTFT (ms)", justify="right")
        tbl.add_column("TPOT (ms)", justify="right")
        tbl.add_column("Total (ms)", justify="right")
        for p in report.projections:
            tbl.add_row(
                str(p.days_from_now),
                f"{p.ttft_p95_ms:.1f}",
                f"{p.tpot_p95_ms:.1f}",
                f"{p.total_latency_p95_ms:.1f}",
            )
        console.print(tbl)

    # Exhaustions
    if report.exhaustions:
        tbl2 = Table(title="SLA Breach Forecast")
        tbl2.add_column("Metric")
        tbl2.add_column("Current (ms)", justify="right")
        tbl2.add_column("Threshold (ms)", justify="right")
        tbl2.add_column("Days to Breach", justify="right")
        tbl2.add_column("Status")
        for e in report.exhaustions:
            dtb = f"{e.days_to_breach:.1f}" if e.days_to_breach is not None else "—"
            status = "[red]BREACH[/red]" if e.breaches_within_horizon else "[green]SAFE[/green]"
            tbl2.add_row(
                e.metric, f"{e.current_value_ms:.1f}",
                f"{e.threshold_ms:.1f}", dtb, status,
            )
        console.print(tbl2)

    if report.has_breach:
        console.print(
            f"\n[red bold]⚠ Earliest projected breach in"
            f" {report.earliest_breach_days:.1f} days[/red bold]"
        )
    else:
        console.print("\n[green]✓ No SLA breach projected within planning horizon.[/green]")
