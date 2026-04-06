"""CLI compare-backends command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_compare_backends(args: argparse.Namespace) -> None:
    """Execute the compare-backends subcommand."""

    from xpyd_plan.backend_compare import compare_backends

    console = Console()

    benchmarks = args.benchmarks
    labels = args.labels.split(",")

    if len(benchmarks) != len(labels):
        console.print(
            f"[red]Error: {len(benchmarks)} benchmark file(s) but {len(labels)} label(s). "
            "Counts must match.[/red]"
        )
        raise SystemExit(1)

    formats_list = None
    if args.formats:
        formats_list = args.formats.split(",")
        if len(formats_list) != len(benchmarks):
            console.print(
                f"[red]Error: {len(formats_list)} format(s) but {len(benchmarks)} benchmark(s). "
                "Counts must match.[/red]"
            )
            raise SystemExit(1)

    result = compare_backends(
        benchmark_paths=benchmarks,
        backend_labels=labels,
        rank_by=args.rank_by,
        formats=formats_list,
        sla_ttft_p99_ms=getattr(args, "sla_ttft_p99", None),
        sla_tpot_p99_ms=getattr(args, "sla_tpot_p99", None),
        sla_total_latency_p99_ms=getattr(args, "sla_total_latency_p99", None),
    )

    if args.output_format == "json":
        console.print_json(result.model_dump_json(indent=2))
        return

    # --- Table output ---
    console.print("\n[bold]🔬 Multi-Backend Comparison Report[/bold]")

    # Metrics table
    metrics_table = Table(title="Backend Metrics")
    metrics_table.add_column("Backend", style="cyan")
    metrics_table.add_column("Format", style="dim")
    metrics_table.add_column("P:D Ratio", justify="center")
    metrics_table.add_column("QPS", justify="right")
    metrics_table.add_column("TTFT P99", justify="right")
    metrics_table.add_column("TPOT P99", justify="right")
    metrics_table.add_column("Total P99", justify="right")
    metrics_table.add_column("Requests", justify="right")

    for m in result.metrics:
        metrics_table.add_row(
            m.backend_label,
            m.format_detected,
            f"{m.num_prefill_instances}:{m.num_decode_instances}",
            f"{m.measured_qps:.1f}",
            f"{m.ttft_p99_ms:.1f}",
            f"{m.tpot_p99_ms:.1f}",
            f"{m.total_latency_p99_ms:.1f}",
            str(m.request_count),
        )

    console.print(metrics_table)

    # SLA results (if any SLA configured)
    if any(not s.meets_all for s in result.sla_results) or any(
        s.meets_all for s in result.sla_results
    ):
        has_sla = any(
            getattr(args, a, None) is not None
            for a in ("sla_ttft_p99", "sla_tpot_p99", "sla_total_latency_p99")
        )
        if has_sla:
            sla_table = Table(title="\nSLA Compliance")
            sla_table.add_column("Backend", style="cyan")
            sla_table.add_column("TTFT P99", justify="center")
            sla_table.add_column("TPOT P99", justify="center")
            sla_table.add_column("Total P99", justify="center")
            sla_table.add_column("Overall", justify="center")

            for s in result.sla_results:
                sla_table.add_row(
                    s.backend_label,
                    "✅" if s.ttft_p99_pass else "❌",
                    "✅" if s.tpot_p99_pass else "❌",
                    "✅" if s.total_latency_p99_pass else "❌",
                    "[green]PASS[/green]" if s.meets_all else "[red]FAIL[/red]",
                )

            console.print(sla_table)

    # Rankings
    rank_table = Table(title=f"\nRankings (by {result.rank_criteria})")
    rank_table.add_column("Rank", justify="center", style="bold")
    rank_table.add_column("Backend", style="cyan")
    rank_table.add_column("Score", justify="right")
    rank_table.add_column("Recommendation")

    for r in result.rankings:
        rank_table.add_row(
            str(r.rank),
            r.backend_label,
            f"{r.score:.1f}",
            r.recommendation,
        )

    console.print(rank_table)

    console.print(f"\n[bold green]🏆 Best backend: {result.best_backend}[/bold green]")


def register_compare_backends(subparsers: argparse._SubParsersAction) -> None:
    """Register the compare-backends subcommand."""

    p = subparsers.add_parser(
        "compare-backends",
        help="Compare benchmark results across serving backends",
    )
    p.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        required=True,
        help="Path to benchmark JSON file (repeat for each backend)",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="Comma-separated backend labels (e.g., 'vllm,sglang,trtllm')",
    )
    p.add_argument(
        "--formats",
        default=None,
        help="Comma-separated format hints (auto,native,vllm,sglang,trtllm)",
    )
    p.add_argument(
        "--rank-by",
        dest="rank_by",
        default="ttft_p99",
        choices=["ttft_p99", "tpot_p99", "total_latency_p99", "throughput"],
        help="Ranking criteria (default: ttft_p99)",
    )
    p.add_argument(
        "--sla-ttft-p99",
        dest="sla_ttft_p99",
        type=float,
        default=None,
        help="SLA threshold for TTFT P99 (ms)",
    )
    p.add_argument(
        "--sla-tpot-p99",
        dest="sla_tpot_p99",
        type=float,
        default=None,
        help="SLA threshold for TPOT P99 (ms)",
    )
    p.add_argument(
        "--sla-total-latency-p99",
        dest="sla_total_latency_p99",
        type=float,
        default=None,
        help="SLA threshold for total latency P99 (ms)",
    )
    p.add_argument(
        "--format",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_cmd_compare_backends)
