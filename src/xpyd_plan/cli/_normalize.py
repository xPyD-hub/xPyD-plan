"""CLI normalize command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.normalizer import BenchmarkNormalizer, NormalizationConfig


def _cmd_normalize(args: argparse.Namespace) -> None:
    """Handle the 'normalize' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    config = NormalizationConfig(
        source_gpu=args.source_gpu,
        target_gpu=args.target_gpu,
    )
    normalizer = BenchmarkNormalizer(config)
    report = normalizer.normalize(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary table
    summary = Table(title="Benchmark Normalization")
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="right")
    summary.add_row("Source GPU", config.source_gpu)
    summary.add_row("Target GPU", config.target_gpu)
    summary.add_row("Source Factor", f"{report.source_factor:.2f}")
    summary.add_row("Target Factor", f"{report.target_factor:.2f}")
    summary.add_row("Scaling Factor", f"{report.scaling_factor:.4f}")
    summary.add_row("Requests", str(report.request_count))
    summary.add_row("Original QPS", f"{report.original_qps:.2f}")
    summary.add_row("Normalized QPS", f"{report.normalized_qps:.2f}")
    console.print(summary)

    # Latency comparison
    lat_table = Table(title="Latency Comparison")
    lat_table.add_column("Metric", justify="left")
    lat_table.add_column("Orig P50", justify="right")
    lat_table.add_column("Orig P95", justify="right")
    lat_table.add_column("Orig P99", justify="right")
    lat_table.add_column("Norm P50", justify="right")
    lat_table.add_column("Norm P95", justify="right")
    lat_table.add_column("Norm P99", justify="right")

    for name, stats in [
        ("TTFT", report.ttft_stats),
        ("TPOT", report.tpot_stats),
        ("Total", report.total_latency_stats),
    ]:
        lat_table.add_row(
            name,
            f"{stats.original_p50:.1f}ms",
            f"{stats.original_p95:.1f}ms",
            f"{stats.original_p99:.1f}ms",
            f"{stats.normalized_p50:.1f}ms",
            f"{stats.normalized_p95:.1f}ms",
            f"{stats.normalized_p99:.1f}ms",
        )
    console.print()
    console.print(lat_table)

    # Known GPUs
    gpu_table = Table(title="Known GPU Performance Factors")
    gpu_table.add_column("GPU Type", justify="left")
    gpu_table.add_column("Factor", justify="right")
    for gpu in report.known_gpus:
        marker = " ←" if gpu.gpu_type in (config.source_gpu, config.target_gpu) else ""
        gpu_table.add_row(gpu.gpu_type, f"{gpu.factor:.2f}{marker}")
    console.print()
    console.print(gpu_table)

    # Output normalized data if requested
    if args.output:
        norm_data = normalizer.normalize_data(data)
        with open(args.output, "w") as f:
            json.dump(norm_data.model_dump(), f, indent=2)
        console.print(f"\n[green]Normalized benchmark written to {args.output}[/green]")


def add_normalize_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the normalize subcommand parser."""
    parser = subparsers.add_parser(
        "normalize",
        help="Normalize benchmark latencies across GPU types",
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    parser.add_argument(
        "--source-gpu", required=True,
        help="GPU type the benchmark was run on (e.g., H100-80G)",
    )
    parser.add_argument(
        "--target-gpu", default="A100-80G",
        help="GPU type to normalize to (default: A100-80G)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write normalized benchmark data to file",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_normalize)
