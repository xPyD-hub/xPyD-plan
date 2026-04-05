"""CLI ranking command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.ranking import BenchmarkRanker, RankWeights


def _cmd_ranking(args: argparse.Namespace) -> None:
    """Handle the 'ranking' subcommand."""
    console = Console()

    datasets = [(f, load_benchmark_auto(f)) for f in args.benchmark]

    weights = RankWeights(
        sla_compliance=args.sla_weight,
        latency=args.latency_weight,
        throughput=args.throughput_weight,
    )

    ranker = BenchmarkRanker(
        sla_ttft_ms=getattr(args, "sla_ttft", None),
        sla_tpot_ms=getattr(args, "sla_tpot", None),
        sla_total_ms=getattr(args, "sla_total", None),
        weights=weights,
    )
    report = ranker.rank(datasets)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    t = Table(
        title=f"Benchmark Ranking ({report.benchmark_count} configs)",
    )
    t.add_column("Rank", justify="right")
    t.add_column("File", justify="left", max_width=40)
    t.add_column("P:D", justify="center")
    t.add_column("QPS", justify="right")
    t.add_column("TTFT P95", justify="right")
    t.add_column("TPOT P95", justify="right")
    t.add_column("Total P95", justify="right")
    t.add_column("SLA", justify="right")
    t.add_column("Latency", justify="right")
    t.add_column("Throughput", justify="right")
    t.add_column("Score", justify="right")

    for r in report.rankings:
        style = "bold green" if r.rank == 1 else ""
        rank_str = f"[{style}]{r.rank}[/{style}]" if style else str(r.rank)
        score_str = (
            f"[{style}]{r.composite_score:.1f}[/{style}]"
            if style else f"{r.composite_score:.1f}"
        )
        t.add_row(
            rank_str,
            r.file,
            f"{r.prefill_instances}:{r.decode_instances}",
            f"{r.measured_qps:.1f}",
            f"{r.ttft_p95_ms:.1f}",
            f"{r.tpot_p95_ms:.1f}",
            f"{r.total_latency_p95_ms:.1f}",
            f"{r.sla_score:.1f}",
            f"{r.latency_score:.1f}",
            f"{r.throughput_score:.1f}",
            score_str,
        )
    console.print(t)

    if report.best:
        b = report.best
        console.print(
            f"\n[bold green]Best:[/bold green] {b.file} "
            f"(P:D {b.prefill_instances}:{b.decode_instances}, "
            f"score {b.composite_score:.1f})"
        )


def add_ranking_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add the ranking subcommand parser."""
    parser = subparsers.add_parser(
        "ranking",
        help="Rank multiple benchmark configs by composite score",
    )
    parser.add_argument(
        "--benchmark", required=True, nargs="+",
        help="Benchmark JSON files to rank",
    )
    parser.add_argument(
        "--sla-ttft", type=float, default=None,
        help="SLA TTFT threshold (ms)",
    )
    parser.add_argument(
        "--sla-tpot", type=float, default=None,
        help="SLA TPOT threshold (ms)",
    )
    parser.add_argument(
        "--sla-total", type=float, default=None,
        help="SLA total latency threshold (ms)",
    )
    parser.add_argument(
        "--sla-weight", type=float, default=0.4,
        help="Weight for SLA score (default: 0.4)",
    )
    parser.add_argument(
        "--latency-weight", type=float, default=0.4,
        help="Weight for latency score (default: 0.4)",
    )
    parser.add_argument(
        "--throughput-weight", type=float, default=0.2,
        help="Weight for throughput score (default: 0.2)",
    )
    parser.add_argument(
        "--output-format", choices=["table", "json"],
        default="table", help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_ranking)
