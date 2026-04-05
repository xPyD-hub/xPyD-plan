"""CLI sample command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.sampler import BenchmarkSampler, SamplingMethod


def _cmd_sample(args: argparse.Namespace) -> None:
    """Handle the 'sample' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    method = SamplingMethod(args.method)
    seed = getattr(args, "seed", None)
    bins = getattr(args, "bins", 4)
    tolerance = getattr(args, "tolerance", 5.0)

    sampler = BenchmarkSampler(seed=seed, tolerance_pct=tolerance)

    if method == SamplingMethod.RANDOM:
        result = sampler.random_sample(data, args.size)
    elif method == SamplingMethod.STRATIFIED:
        result = sampler.stratified_sample(data, args.size, bins=bins)
    else:
        result = sampler.reservoir_sample(data, args.size)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        out = result.model_dump()
        # Don't dump the full data in JSON summary — too large
        out.pop("data", None)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    console.print(f"\n🎲 Sampling: [bold]{result.method.value.upper()}[/bold]")
    console.print(
        f"   {result.original_count} → {result.sample_count} requests "
        f"({result.sample_fraction:.1%})"
    )

    v = result.validation
    status = "✅ Representative" if v.is_representative else "⚠️ NOT Representative"
    console.print(
        f"   Max deviation: {v.max_deviation_pct:.2f}% (tolerance: {v.tolerance_pct}%) — {status}\n"
    )

    table = Table(title="Metric Deviations")
    table.add_column("Metric", justify="left")
    table.add_column("P50 Δ%", justify="right")
    table.add_column("P95 Δ%", justify="right")
    table.add_column("P99 Δ%", justify="right")

    for d in v.deviations:
        table.add_row(
            d.metric,
            f"{d.p50_deviation_pct:.2f}",
            f"{d.p95_deviation_pct:.2f}",
            f"{d.p99_deviation_pct:.2f}",
        )

    console.print(table)

    # Write sampled data if --output specified
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.data.model_dump(), f, indent=2)
        console.print(f"\n💾 Sampled data written to {output_path}")


def add_sample_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the sample subcommand parser."""
    parser = subparsers.add_parser(
        "sample",
        help="Downsample benchmark data while preserving statistical properties",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--method",
        choices=["random", "stratified", "reservoir"],
        default="random",
        help="Sampling method (default: random)",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Target sample size (number of requests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=4,
        help="Number of bins for stratified sampling (default: 4)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Max acceptable P50/P95/P99 deviation %% (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write sampled benchmark data to this JSON file",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_sample)
