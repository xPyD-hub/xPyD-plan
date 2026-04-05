"""CLI fingerprint command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_fingerprint(args: argparse.Namespace) -> None:
    """Handle the 'fingerprint' subcommand."""
    from xpyd_plan.bench_adapter import load_benchmark_auto
    from xpyd_plan.fingerprint import EnvironmentFingerprinter

    console = Console()
    fp = EnvironmentFingerprinter()

    data = load_benchmark_auto(args.benchmark)
    baseline = fp.extract(data)

    if args.compare:
        data2 = load_benchmark_auto(args.compare)
        current = fp.extract(data2)
        comparison = fp.compare(baseline, current)

        if args.output_format == "json":
            print(comparison.model_dump_json(indent=2))
            return

        console.print("\n[bold]🔍 Environment Comparison[/bold]")
        console.print(f"  Baseline hash: {comparison.baseline_hash}")
        console.print(f"  Current hash:  {comparison.current_hash}")
        console.print(f"  Compatibility: {comparison.compatibility.value}")

        if comparison.differences:
            table = Table(title="Differences")
            table.add_column("Field", style="cyan")
            table.add_column("Baseline", style="red")
            table.add_column("Current", style="green")

            for diff in comparison.differences:
                table.add_row(diff.field, diff.baseline_value, diff.current_value)
            console.print(table)
        else:
            console.print("  [green]✅ Environments are identical[/green]")
    else:
        if args.output_format == "json":
            print(baseline.model_dump_json(indent=2))
            return

        console.print("\n[bold]🔑 Environment Fingerprint[/bold]")
        console.print(f"  Hash: {baseline.hash}")
        table = Table(title="Environment Details")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        for field_name in [
            "num_prefill_instances",
            "num_decode_instances",
            "total_instances",
            "measured_qps",
            "num_requests",
            "prompt_tokens_min",
            "prompt_tokens_max",
            "output_tokens_min",
            "output_tokens_max",
        ]:
            table.add_row(field_name, str(getattr(baseline, field_name)))
        console.print(table)
