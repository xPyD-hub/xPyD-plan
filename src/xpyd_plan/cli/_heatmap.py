"""CLI heatmap command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.heatmap import (
    AggregationMetric,
    HeatmapConfig,
    HeatmapGenerator,
    LatencyField,
)


def add_heatmap_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the heatmap subcommand."""
    parser = subparsers.add_parser(
        "heatmap",
        help="Generate latency heatmap data from benchmark results",
    )
    parser.add_argument(
        "--benchmark", required=True, help="Path to benchmark JSON file"
    )
    parser.add_argument(
        "--prompt-bins", type=int, default=10, help="Number of prompt token bins (default: 10)"
    )
    parser.add_argument(
        "--output-bins", type=int, default=10, help="Number of output token bins (default: 10)"
    )
    parser.add_argument(
        "--metric",
        choices=["mean", "p50", "p95", "p99"],
        default="p95",
        help="Aggregation metric (default: p95)",
    )
    parser.add_argument(
        "--field",
        choices=["ttft_ms", "tpot_ms", "total_latency_ms"],
        default="total_latency_ms",
        help="Latency field to analyze (default: total_latency_ms)",
    )
    parser.add_argument(
        "--sla-threshold",
        type=float,
        default=None,
        help="SLA threshold in ms for hotspot detection",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def _cmd_heatmap(args: argparse.Namespace) -> None:
    """Handle the 'heatmap' subcommand."""
    import json as json_mod
    from pathlib import Path

    from xpyd_plan.benchmark_models import BenchmarkData

    console = Console()

    path = Path(args.benchmark)
    if not path.exists():
        console.print(f"[red]Error: file not found: {path}[/red]")
        sys.exit(1)

    raw = json_mod.loads(path.read_text())
    data = BenchmarkData.model_validate(raw)

    config = HeatmapConfig(
        prompt_bins=args.prompt_bins,
        output_bins=args.output_bins,
        metric=AggregationMetric(args.metric),
        field=LatencyField(args.field),
        sla_threshold_ms=args.sla_threshold,
    )

    generator = HeatmapGenerator()
    report = generator.generate(data, config)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    console.print(f"\n[bold]Latency Heatmap — {config.field.value} ({config.metric.value})[/bold]")
    console.print(f"Total requests: {report.total_requests}")
    if report.min_value is not None:
        console.print(f"Range: {report.min_value:.1f}ms – {report.max_value:.1f}ms")
    if report.hotspot_count > 0:
        console.print(f"[red]Hotspots: {report.hotspot_count}[/red]")
    console.print()

    table = Table(title="Heatmap Cells (non-empty)")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Count", justify="right")
    table.add_column(f"{config.metric.value} (ms)", justify="right")
    table.add_column("Hotspot", justify="center")

    for cell in report.grid.cells:
        if cell.count == 0:
            continue
        hotspot_str = "[red]●[/red]" if cell.is_hotspot else ""
        table.add_row(
            f"{cell.prompt_bin_start}–{cell.prompt_bin_end}",
            f"{cell.output_bin_start}–{cell.output_bin_end}",
            str(cell.count),
            f"{cell.value:.1f}",
            hotspot_str,
        )

    console.print(table)
    console.print(f"\n{report.recommendation}")
