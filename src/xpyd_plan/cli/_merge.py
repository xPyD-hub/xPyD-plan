"""CLI merge command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_merge(args: argparse.Namespace) -> None:
    """Handle 'merge' subcommand."""
    import json as json_mod

    from xpyd_plan.benchmark_models import BenchmarkData
    from xpyd_plan.merger import BenchmarkMerger, MergeConfig, MergeStrategy

    console = Console()

    datasets = []
    for path in args.benchmark:
        with open(path) as f:
            data = json_mod.load(f)
        datasets.append(BenchmarkData.model_validate(data))

    config = MergeConfig(
        strategy=MergeStrategy(args.strategy),
        require_same_config=not args.no_config_check,
    )
    merger = BenchmarkMerger(config)
    result = merger.merge(datasets)

    # Write merged benchmark to output file
    with open(args.output, "w") as f:
        f.write(result.merged.model_dump_json(indent=2))

    if args.output_format == "json":
        summary = {
            "source_count": result.source_count,
            "total_requests_before": result.total_requests_before,
            "total_requests_after": result.total_requests_after,
            "duplicates_removed": result.duplicates_removed,
            "strategy_used": result.strategy_used.value,
            "output_file": args.output,
        }
        console.print_json(json_mod.dumps(summary))
    else:
        table = Table(title="Benchmark Merge Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Sources merged", str(result.source_count))
        table.add_row("Strategy", result.strategy_used.value)
        table.add_row("Requests before", str(result.total_requests_before))
        table.add_row("Requests after", str(result.total_requests_after))
        table.add_row("Duplicates removed", str(result.duplicates_removed))
        table.add_row("Output file", args.output)
        console.print(table)

    raise SystemExit(0)
