"""CLI filter command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_filter(args: argparse.Namespace) -> None:
    """Handle 'filter' subcommand."""
    import json as json_mod

    from xpyd_plan.benchmark_models import BenchmarkData
    from xpyd_plan.filter import BenchmarkFilter, FilterConfig

    console = Console()

    with open(args.benchmark) as f:
        data = BenchmarkData.model_validate(json_mod.load(f))

    config = FilterConfig(
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        min_ttft_ms=args.min_ttft_ms,
        max_ttft_ms=args.max_ttft_ms,
        min_tpot_ms=args.min_tpot_ms,
        max_tpot_ms=args.max_tpot_ms,
        min_total_latency_ms=args.min_total_latency_ms,
        max_total_latency_ms=args.max_total_latency_ms,
        time_start=args.time_start,
        time_end=args.time_end,
        sample_count=args.sample_count,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    bf = BenchmarkFilter(config)
    result = bf.apply(data)

    # Write filtered benchmark to output file
    with open(args.output, "w") as f:
        f.write(result.data.model_dump_json(indent=2))

    if args.output_format == "json":
        summary = {
            "original_count": result.original_count,
            "filtered_count": result.filtered_count,
            "removed_count": result.removed_count,
            "retention_rate": round(result.retention_rate, 4),
            "filters_applied": result.filters_applied,
            "output_file": args.output,
        }
        console.print_json(json_mod.dumps(summary))
    else:
        table = Table(title="Benchmark Filter Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Original requests", str(result.original_count))
        table.add_row("Filtered requests", str(result.filtered_count))
        table.add_row("Removed", str(result.removed_count))
        table.add_row("Retention rate", f"{result.retention_rate:.1%}")
        table.add_row("Filters applied", ", ".join(result.filters_applied) or "none")
        table.add_row("Output file", args.output)
        console.print(table)

    raise SystemExit(0)
