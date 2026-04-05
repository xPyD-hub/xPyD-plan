"""CLI rate-limit command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.rate_limit import RateLimitRecommender


def _cmd_rate_limit(args: argparse.Namespace) -> None:
    """Handle the 'rate-limit' subcommand."""
    console = Console()

    if len(args.benchmark) < 2:
        console.print("[red]Error: need at least 2 benchmark files for rate limit analysis[/red]")
        sys.exit(1)

    if args.sla_ttft is None and args.sla_tpot is None and args.sla_total is None:
        console.print(
            "[red]Error: specify at least one SLA threshold "
            "(--sla-ttft, --sla-tpot, --sla-total)[/red]"
        )
        sys.exit(1)

    benchmarks = []
    for path in args.benchmark:
        benchmarks.append(load_benchmark_auto(path))

    recommender = RateLimitRecommender(
        safety_margin=args.safety_margin,
        burst_window_seconds=args.burst_window,
    )
    report = recommender.recommend(
        benchmarks,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
    )

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(report.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Summary
    console.print("\n[bold]Rate Limit Recommendation[/bold]")
    console.print(f"  Benchmarks analyzed: {report.data_points}")
    if report.qps_range:
        console.print(f"  QPS range: {report.qps_range[0]:.1f} – {report.qps_range[1]:.1f}")
    console.print(f"  Safety margin: {report.safety_margin * 100:.0f}%")
    console.print()

    if report.max_sla_compliant_qps is not None:
        console.print(
            f"  Max SLA-compliant QPS: "
            f"[green]{report.max_sla_compliant_qps:.1f}[/green]"
        )
        console.print(
            f"  [bold]Recommended limit: "
            f"[cyan]{report.recommended_qps:.1f} QPS[/cyan][/bold]"
        )
    else:
        console.print("  [red]No SLA-compliant QPS found[/red]")

    # Headroom table
    if report.headroom:
        console.print()
        ht = Table(title="Headroom at Recommended QPS")
        ht.add_column("Metric", justify="left")
        ht.add_column("SLA (ms)", justify="right")
        ht.add_column("Est. Latency (ms)", justify="right")
        ht.add_column("Headroom (ms)", justify="right")
        ht.add_column("Headroom %", justify="right")
        for h in report.headroom:
            color = "green" if h.headroom_pct >= 20 else (
                "yellow" if h.headroom_pct >= 10 else "red"
            )
            ht.add_row(
                h.metric,
                f"{h.sla_threshold_ms:.1f}",
                f"{h.estimated_latency_ms:.1f}",
                f"{h.headroom_ms:.1f}",
                f"[{color}]{h.headroom_pct:.1f}%[/{color}]",
            )
        console.print(ht)

    # Burst info
    if report.burst:
        console.print()
        burst = report.burst
        console.print(
            f"  Burst allowance: {burst.burst_qps:.1f} QPS "
            f"(×{burst.burst_ratio:.1f} for "
            f"{burst.burst_window_seconds:.0f}s window)"
        )

    console.print()
    console.print(f"[bold]{report.recommendation}[/bold]")


def add_rate_limit_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the rate-limit subcommand parser."""
    parser = subparsers.add_parser(
        "rate-limit",
        help="Recommend request rate limits from multi-QPS benchmark data",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        required=True,
        help="Benchmark JSON files (at least 2, different QPS levels)",
    )
    parser.add_argument(
        "--sla-ttft",
        type=float,
        default=None,
        help="SLA threshold for TTFT P95 (ms)",
    )
    parser.add_argument(
        "--sla-tpot",
        type=float,
        default=None,
        help="SLA threshold for TPOT P95 (ms)",
    )
    parser.add_argument(
        "--sla-total",
        type=float,
        default=None,
        help="SLA threshold for total latency P95 (ms)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.10,
        help="Safety margin fraction (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--burst-window",
        type=float,
        default=10.0,
        help="Burst analysis window in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_rate_limit)
