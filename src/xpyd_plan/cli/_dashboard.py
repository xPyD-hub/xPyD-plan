"""CLI dashboard command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Handle the 'dashboard' subcommand."""
    from xpyd_plan.benchmark_models import BenchmarkMetadata
    from xpyd_plan.dashboard import Dashboard
    from xpyd_plan.models import SLAConfig

    console = Console()

    if not args.benchmark and not args.stream:
        console.print("[red]Error: --benchmark or --stream is required.[/red]")
        sys.exit(1)

    sla = SLAConfig(
        ttft_ms=args.max_ttft_ms,
        tpot_ms=args.max_tpot_ms,
        max_latency_ms=args.max_total_latency_ms,
    )

    dashboard = Dashboard(
        sla=sla,
        total_instances=args.total_instances,
        refresh_interval=args.refresh_interval,
    )

    if args.benchmark:
        dashboard.load_file(args.benchmark)
        dashboard.run(console=console)
    else:
        # Streaming mode
        metadata = None
        if args.num_prefill and args.num_decode:
            total = args.total_instances or (args.num_prefill + args.num_decode)
            metadata = BenchmarkMetadata(
                num_prefill_instances=args.num_prefill,
                num_decode_instances=args.num_decode,
                total_instances=total,
                measured_qps=0.0,
            )
        dashboard.run(stream=sys.stdin, metadata=metadata, console=console)
