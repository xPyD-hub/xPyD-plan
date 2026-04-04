"""CLI export command."""

from __future__ import annotations

import argparse

from xpyd_plan.models import SLAConfig


def _cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand for batch export."""
    from xpyd_plan.export import export_batch

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )
    output = export_batch(
        benchmark_dir=args.dir,
        sla_config=sla,
        output_format=args.output_format,
        total_instances=args.total_instances,
    )
    print(output, end="" if args.output_format == "csv" else "\n")
