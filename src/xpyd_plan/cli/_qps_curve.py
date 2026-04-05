"""CLI qps-curve command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.qps_curve import FitMethod, QPSCurveFitter


def _cmd_qps_curve(args: argparse.Namespace) -> None:
    """Handle the 'qps-curve' subcommand."""
    console = Console()

    if len(args.benchmark) < 2:
        console.print("[red]Error: need at least 2 benchmark files for QPS curve fitting[/red]")
        sys.exit(1)

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    predict_qps = None
    if args.predict_qps:
        predict_qps = [float(q) for q in args.predict_qps]

    fitter = QPSCurveFitter(
        method=FitMethod(args.method),
        percentile=args.percentile,
    )
    report = fitter.fit(
        benchmarks,
        predict_qps=predict_qps,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Fits table
    fit_table = Table(title="QPS-Latency Curve Fits")
    fit_table.add_column("Metric", justify="left")
    fit_table.add_column("Method", justify="center")
    fit_table.add_column("R²", justify="right")
    fit_table.add_column("MSE", justify="right")
    fit_table.add_column("QPS Range", justify="center")

    for fit in report.fits:
        fit_table.add_row(
            fit.metric,
            fit.method,
            f"{fit.r_squared:.4f}",
            f"{fit.residual_error:.2f}",
            f"{fit.qps_range[0]:.1f}–{fit.qps_range[1]:.1f}",
        )

    console.print(fit_table)

    # Predictions table
    if report.predictions:
        console.print()
        pred_table = Table(title="Latency Predictions")
        pred_table.add_column("QPS", justify="right")
        pred_table.add_column("Metric", justify="left")
        pred_table.add_column("Predicted Latency (ms)", justify="right")
        pred_table.add_column("Confidence", justify="center")

        for pred in report.predictions:
            pred_table.add_row(
                f"{pred.qps:.1f}",
                pred.metric,
                f"{pred.predicted_latency_ms:.2f}",
                pred.confidence.value,
            )

        console.print(pred_table)

    # Max sustainable QPS table
    if report.max_sustainable:
        console.print()
        sus_table = Table(title="Maximum Sustainable QPS")
        sus_table.add_column("Metric", justify="left")
        sus_table.add_column("Max QPS", justify="right")
        sus_table.add_column("SLA Threshold (ms)", justify="right")
        sus_table.add_column("Predicted Latency (ms)", justify="right")
        sus_table.add_column("Confidence", justify="center")

        for m in report.max_sustainable:
            sus_table.add_row(
                m.metric,
                f"{m.max_qps:.1f}",
                f"{m.sla_threshold_ms:.1f}",
                f"{m.predicted_latency_at_max_ms:.2f}",
                m.confidence.value,
            )

        console.print(sus_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_qps_curve_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the qps-curve subcommand parser."""
    parser = subparsers.add_parser(
        "qps-curve",
        help="Fit QPS-latency curves and predict latency at untested QPS levels",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 2, different QPS levels)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "linear", "poly", "exp"],
        default="auto",
        help="Curve fitting method (default: auto — selects best fit)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Latency percentile to fit (default: 95)",
    )
    parser.add_argument(
        "--predict-qps",
        nargs="+",
        type=float,
        help="QPS values to predict latency for",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="TTFT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="TPOT SLA threshold in ms",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="Total latency SLA threshold in ms",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_qps_curve)
