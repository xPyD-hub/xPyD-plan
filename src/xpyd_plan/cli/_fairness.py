"""CLI fairness command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.fairness import FairnessAnalyzer, FairnessClassification


def _cmd_fairness(args: argparse.Namespace) -> None:
    """Handle the 'fairness' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = FairnessAnalyzer()
    report = analyzer.analyze(data, num_buckets=args.buckets)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Bucket stats table
    table = Table(title="Request Fairness Analysis — Bucket Stats")
    table.add_column("Bucket", justify="center")
    table.add_column("Token Range", justify="center")
    table.add_column("Requests", justify="right")
    table.add_column("TTFT P95", justify="right")
    table.add_column("TPOT P95", justify="right")
    table.add_column("Total P95", justify="right")

    for b in report.buckets:
        table.add_row(
            str(b.bucket_index),
            f"{b.min_tokens}–{b.max_tokens}",
            str(b.request_count),
            f"{b.ttft_p95_ms:.1f}",
            f"{b.tpot_p95_ms:.1f}",
            f"{b.total_latency_p95_ms:.1f}",
        )

    console.print(table)
    console.print()

    # Fairness indices table
    fi_table = Table(title="Jain's Fairness Index")
    fi_table.add_column("Metric", justify="left")
    fi_table.add_column("Jain's Index", justify="right")
    fi_table.add_column("Classification", justify="center")

    for fi in report.fairness_indices:
        style = {
            FairnessClassification.FAIR: "green",
            FairnessClassification.MODERATE: "yellow",
            FairnessClassification.UNFAIR: "bold red",
        }[fi.classification]
        fi_table.add_row(
            fi.metric,
            f"[{style}]{fi.jain_index:.4f}[/{style}]",
            f"[{style}]{fi.classification.value}[/{style}]",
        )

    console.print(fi_table)
    console.print()
    style_map = {
        FairnessClassification.FAIR: "green",
        FairnessClassification.MODERATE: "yellow",
        FairnessClassification.UNFAIR: "bold red",
    }
    overall_style = style_map[report.overall_classification]
    console.print(
        f"Overall: [{overall_style}]"
        f"{report.overall_classification.value}[/]"
    )
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_fairness_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the fairness subcommand parser."""
    parser = subparsers.add_parser(
        "fairness",
        help="Analyze latency fairness across request token-count buckets",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=4,
        help="Number of quantile-based buckets (default: 4)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_fairness)
