"""CLI size-distribution command."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.size_distribution import SizeDistributionAnalyzer


def _cmd_size_distribution(args: argparse.Namespace) -> None:
    """Handle the 'size-distribution' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = SizeDistributionAnalyzer(bins=args.bins)
    report = analyzer.analyze(data)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        console.print(json.dumps(report.model_dump(), indent=2))
        return

    # Table output
    console.print(f"\n[bold]Request Size Distribution[/bold] ({report.total_requests} requests)\n")

    for hist in [report.prompt_histogram, report.output_histogram]:
        console.print(
            f"[bold]{hist.dimension.title()} Tokens[/bold] — "
            f"shape: {hist.shape.value}, "
            f"mean: {hist.mean_tokens:.0f}, std: {hist.std_tokens:.0f}, "
            f"range: [{hist.min_tokens:.0f}, {hist.max_tokens:.0f}]"
        )
        table = Table()
        table.add_column("Bin Range", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Fraction", justify="right")
        table.add_column("TTFT P50", justify="right")
        table.add_column("TTFT P95", justify="right")
        table.add_column("TPOT P50", justify="right")
        table.add_column("TPOT P95", justify="right")
        table.add_column("Latency P50", justify="right")
        table.add_column("Latency P95", justify="right")
        for b in hist.bins:
            if b.count == 0:
                table.add_row(
                    f"{b.bin_start:.0f}–{b.bin_end:.0f}", "0", "0.0%",
                    "—", "—", "—", "—", "—", "—",
                )
            else:
                table.add_row(
                    f"{b.bin_start:.0f}–{b.bin_end:.0f}",
                    str(b.count),
                    f"{b.fraction:.1%}",
                    f"{b.ttft_p50_ms:.1f}",
                    f"{b.ttft_p95_ms:.1f}",
                    f"{b.tpot_p50_ms:.1f}",
                    f"{b.tpot_p95_ms:.1f}",
                    f"{b.total_latency_p50_ms:.1f}",
                    f"{b.total_latency_p95_ms:.1f}",
                )
        console.print(table)
        console.print()

    # Correlations
    console.print("[bold]Size-Latency Correlations[/bold]")
    corr_table = Table()
    corr_table.add_column("Dimension", style="cyan")
    corr_table.add_column("TTFT r", justify="right")
    corr_table.add_column("TPOT r", justify="right")
    corr_table.add_column("Total Latency r", justify="right")
    for c in report.correlations:
        corr_table.add_row(
            c.dimension,
            f"{c.ttft_pearson_r:.3f}",
            f"{c.tpot_pearson_r:.3f}",
            f"{c.total_latency_pearson_r:.3f}",
        )
    console.print(corr_table)


def add_size_distribution_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the size-distribution subcommand parser."""
    parser = subparsers.add_parser(
        "size-distribution",
        help="Analyze request token size distributions and latency correlations",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of histogram bins (default: 10)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
