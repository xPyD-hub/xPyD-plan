"""CLI handler for the scaling-policy subcommand."""

from __future__ import annotations

import argparse
import json
import sys

from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.scaling_policy import ScalingPolicyGenerator


def _cmd_scaling_policy(args: argparse.Namespace) -> None:
    """Execute scaling-policy subcommand."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    data = [load_benchmark_auto(p) for p in args.benchmark]

    gen = ScalingPolicyGenerator(
        data,
        sla_ttft_ms=args.sla_ttft,
        sla_tpot_ms=args.sla_tpot,
        sla_total_ms=args.sla_total,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )
    policy = gen.generate()

    if args.output_format == "json":
        json.dump(policy.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    # Table output
    table = Table(title="Auto-Scaling Policy Rules")
    table.add_column("Priority", justify="right")
    table.add_column("Name")
    table.add_column("Trigger")
    table.add_column("Direction")
    table.add_column("Action")
    table.add_column("Reason")

    for rule in policy.rules:
        trigger_str = (
            f"{rule.trigger.metric} {rule.trigger.comparator} "
            f"{rule.trigger.threshold} for {rule.trigger.sustained_seconds}s"
        )
        action_str = f"P:{rule.action.prefill_delta:+d} D:{rule.action.decode_delta:+d}"
        table.add_row(
            str(rule.priority),
            rule.name,
            trigger_str,
            rule.action.direction.value,
            action_str,
            rule.action.reason,
        )

    console.print(table)
    console.print(f"\nCooldown: {policy.cooldown_seconds}s")
    p_min = policy.min_prefill_instances
    p_max = policy.max_prefill_instances
    d_min = policy.min_decode_instances
    d_max = policy.max_decode_instances
    console.print(
        f"Instance bounds: prefill [{p_min}-{p_max}], "
        f"decode [{d_min}-{d_max}]"
    )

    if policy.notes:
        console.print("\n[bold]Notes:[/bold]")
        for note in policy.notes:
            console.print(f"  • {note}")


def add_scaling_policy_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add scaling-policy subcommand parser."""
    parser = subparsers.add_parser(
        "scaling-policy",
        help="Generate auto-scaling policy rules from benchmark data",
    )
    parser.add_argument(
        "--benchmark", "-b", nargs="+", required=True,
        help="Benchmark JSON files (multiple QPS levels recommended)",
    )
    parser.add_argument("--sla-ttft", type=float, default=500.0, help="TTFT SLA (ms)")
    parser.add_argument("--sla-tpot", type=float, default=100.0, help="TPOT SLA (ms)")
    parser.add_argument("--sla-total", type=float, default=5000.0, help="Total latency SLA (ms)")
    parser.add_argument("--min-instances", type=int, default=1, help="Min instances per type")
    parser.add_argument("--max-instances", type=int, default=16, help="Max instances per type")
    parser.add_argument(
        "--output-format", "-o", choices=["table", "json"], default="table",
        help="Output format",
    )
