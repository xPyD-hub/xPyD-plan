"""CLI ensemble command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.ensemble import EnsemblePredictor


def _cmd_ensemble(args: argparse.Namespace) -> None:
    """Handle the 'ensemble' subcommand."""
    console = Console()

    if len(args.benchmark) < 2:
        console.print("[red]Error: need at least 2 benchmark files[/red]")
        sys.exit(1)

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    predict_ratio = None
    if args.predict_ratio:
        parts = args.predict_ratio.split(":")
        if len(parts) != 2:
            console.print("[red]Error: --predict-ratio must be P:D format (e.g., 2:6)[/red]")
            sys.exit(1)
        predict_ratio = (int(parts[0]), int(parts[1]))

    predictor = EnsemblePredictor(percentile=getattr(args, "percentile", 95.0))
    report = predictor.predict(
        benchmarks,
        predict_qps=getattr(args, "predict_qps", None),
        predict_ratio=predict_ratio,
        sla_ttft_ms=getattr(args, "sla_ttft", None),
        sla_tpot_ms=getattr(args, "sla_tpot", None),
        sla_total_ms=getattr(args, "sla_total", None),
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Method predictions table
    method_table = Table(title="Per-Method Predictions")
    method_table.add_column("Method", justify="left")
    method_table.add_column("Available", justify="center")
    method_table.add_column("Confidence", justify="right")
    method_table.add_column("TTFT P95 (ms)", justify="right")
    method_table.add_column("TPOT P95 (ms)", justify="right")
    method_table.add_column("Total P95 (ms)", justify="right")
    method_table.add_column("QPS", justify="right")

    for mp in report.method_predictions:
        method_table.add_row(
            mp.method,
            "✓" if mp.available else f"✗ ({mp.error})" if mp.error else "✗",
            f"{mp.confidence:.2f}" if mp.available else "—",
            f"{mp.ttft_p95_ms:.2f}" if mp.ttft_p95_ms is not None else "—",
            f"{mp.tpot_p95_ms:.2f}" if mp.tpot_p95_ms is not None else "—",
            f"{mp.total_latency_p95_ms:.2f}" if mp.total_latency_p95_ms is not None else "—",
            f"{mp.throughput_qps:.1f}" if mp.throughput_qps is not None else "—",
        )

    console.print(method_table)

    # Ensemble result table
    console.print()
    ens = report.ensemble
    ens_table = Table(title="Ensemble Prediction")
    ens_table.add_column("Metric", justify="left")
    ens_table.add_column("Predicted (ms)", justify="right")
    ens_table.add_column("Disagreement", justify="center")

    rows = [
        ("TTFT P95", ens.ttft_p95_ms, ens.ttft_disagreement),
        ("TPOT P95", ens.tpot_p95_ms, ens.tpot_disagreement),
        ("Total Latency P95", ens.total_latency_p95_ms, ens.total_latency_disagreement),
    ]

    for label, val, disagree in rows:
        color = ""
        if disagree.value in ("moderate", "high"):
            color = "[yellow]" if disagree.value == "moderate" else "[red]"
        end_color = "[/]" if color else ""
        ens_table.add_row(
            label,
            f"{val:.2f}" if val is not None else "—",
            f"{color}{disagree.value}{end_color}",
        )

    if ens.throughput_qps is not None:
        ens_table.add_row(
            "Throughput (QPS)",
            f"{ens.throughput_qps:.1f}",
            ens.qps_disagreement.value,
        )

    console.print(ens_table)

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_ensemble_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ensemble subcommand parser."""
    parser = subparsers.add_parser(
        "ensemble",
        help="Combine multiple prediction methods with confidence-weighted averaging",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 2)",
    )
    parser.add_argument(
        "--predict-qps",
        type=float,
        default=None,
        help="QPS value to predict latency for (used by QPS curve fitter)",
    )
    parser.add_argument(
        "--predict-ratio",
        type=str,
        default=None,
        help="P:D ratio to predict (e.g., 2:6, used by interpolator)",
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
        "--percentile",
        type=float,
        default=95.0,
        help="Latency percentile (default: 95)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_ensemble)
