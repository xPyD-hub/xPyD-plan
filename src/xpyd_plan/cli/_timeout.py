"""CLI timeout command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.timeout import TimeoutAnalyzer


def _cmd_timeout(args: argparse.Namespace) -> None:
    """Handle the 'timeout' subcommand."""
    console = Console()

    data = load_benchmark_auto(args.benchmark)
    analyzer = TimeoutAnalyzer()
    report = analyzer.analyze(
        data,
        ttft_ms=args.timeout_ttft,
        tpot_ms=args.timeout_tpot,
        total_latency_ms=args.timeout_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print("\n[bold]Request Timeout Analysis[/bold]")
    console.print(f"Total requests: {report.total_requests}")
    console.print(f"Timed out: {report.timeout_count}")
    console.print(f"Timeout rate: {report.timeout_rate:.2%}")
    console.print(f"Severity: {report.severity.value}")

    if report.per_metric_counts:
        console.print("\n[bold]Per-Metric Breakdown[/bold]")
        for metric, count in report.per_metric_counts.items():
            if count > 0:
                console.print(f"  {metric}: {count} requests")

    if report.token_characterization:
        tc = report.token_characterization
        console.print("\n[bold]Timed-Out Request Characteristics[/bold]")
        console.print(
            f"  Prompt tokens: mean={tc.prompt_tokens_mean:.0f}, "
            f"median={tc.prompt_tokens_median:.0f}, "
            f"range=[{tc.prompt_tokens_min}, {tc.prompt_tokens_max}]"
        )
        console.print(
            f"  Output tokens: mean={tc.output_tokens_mean:.0f}, "
            f"median={tc.output_tokens_median:.0f}, "
            f"range=[{tc.output_tokens_min}, {tc.output_tokens_max}]"
        )

    if report.temporal_clusters:
        console.print(f"\n[bold]Temporal Clusters ({len(report.temporal_clusters)})[/bold]")
        cluster_table = Table()
        cluster_table.add_column("Cluster", justify="right")
        cluster_table.add_column("Count", justify="right")
        cluster_table.add_column("Duration (s)", justify="right")
        for i, c in enumerate(report.temporal_clusters[:10]):
            cluster_table.add_row(str(i + 1), str(c.count), f"{c.duration_seconds:.1f}")
        console.print(cluster_table)

    console.print(f"\n[italic]{report.recommendation}[/italic]\n")

    if report.events:
        table = Table(title="Timeout Events (first 20)")
        table.add_column("Index", justify="right")
        table.add_column("Request ID")
        table.add_column("Prompt Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("TPOT (ms)", justify="right")
        table.add_column("Total (ms)", justify="right")
        table.add_column("Exceeded")

        for e in report.events[:20]:
            table.add_row(
                str(e.index),
                e.request_id,
                str(e.prompt_tokens),
                str(e.output_tokens),
                f"{e.ttft_ms:.1f}",
                f"{e.tpot_ms:.1f}",
                f"{e.total_latency_ms:.1f}",
                ", ".join(e.exceeded_metrics),
            )

        if len(report.events) > 20:
            console.print(f"  ... and {len(report.events) - 20} more timeout events")

        console.print(table)


def add_timeout_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the timeout subcommand parser."""
    parser = subparsers.add_parser(
        "timeout",
        help="Identify and characterize requests exceeding latency thresholds",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--timeout-ttft",
        type=float,
        default=None,
        help="TTFT timeout threshold in ms",
    )
    parser.add_argument(
        "--timeout-tpot",
        type=float,
        default=None,
        help="TPOT timeout threshold in ms",
    )
    parser.add_argument(
        "--timeout-total",
        type=float,
        default=None,
        help="Total latency timeout threshold in ms",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_timeout)
