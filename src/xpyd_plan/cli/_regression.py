"""CLI regression command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.regression import RegressionAnalyzer


def _cmd_regression(args: argparse.Namespace) -> None:
    """Handle the 'regression' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = RegressionAnalyzer(data)

    predict_prompt = getattr(args, "predict_prompt", None)
    predict_output = getattr(args, "predict_output", None)

    report = analyzer.analyze(
        predict_prompt=predict_prompt,
        predict_output=predict_output,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Regression fits table
    fit_table = Table(title="Latency-Token Regression Fits")
    fit_table.add_column("Predictor", justify="left")
    fit_table.add_column("Target", justify="left")
    fit_table.add_column("Slope (ms/tok)", justify="right")
    fit_table.add_column("Intercept (ms)", justify="right")
    fit_table.add_column("R²", justify="right")
    fit_table.add_column("Slope SE", justify="right")
    fit_table.add_column("Samples", justify="right")

    for fit in report.fits:
        fit_table.add_row(
            fit.predictor,
            fit.target,
            f"{fit.slope:.4f}",
            f"{fit.intercept:.2f}",
            f"{fit.r_squared:.4f}",
            f"{fit.slope_std_err:.4f}",
            str(fit.n_samples),
        )

    console.print(fit_table)

    if report.predictions:
        pred_table = Table(title="Latency Predictions (95% CI)")
        pred_table.add_column("Metric", justify="left")
        pred_table.add_column("Token Count", justify="right")
        pred_table.add_column("Predicted (ms)", justify="right")
        pred_table.add_column("CI Lower (ms)", justify="right")
        pred_table.add_column("CI Upper (ms)", justify="right")

        for pred in report.predictions:
            pred_table.add_row(
                pred.metric,
                str(pred.token_count),
                f"{pred.predicted_ms:.2f}",
                f"{pred.ci_lower_ms:.2f}",
                f"{pred.ci_upper_ms:.2f}",
            )

        console.print(pred_table)


def add_regression_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the regression subcommand parser."""
    parser = subparsers.add_parser(
        "regression",
        help="Latency-token regression analysis (fit linear models, predict latency)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--predict-prompt",
        type=int,
        default=None,
        help="Predict TTFT for this prompt token count",
    )
    parser.add_argument(
        "--predict-output",
        type=int,
        default=None,
        help="Predict TPOT for this output token count",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_regression)
