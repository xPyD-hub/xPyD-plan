"""CLI pipeline command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Handle 'pipeline' subcommand."""
    import json as json_mod

    from xpyd_plan.pipeline import PipelineRunner, load_pipeline_config

    console = Console()

    cfg = load_pipeline_config(args.config)
    benchmarks = args.benchmark or []
    runner = PipelineRunner(cfg, benchmark_paths=benchmarks)
    result = runner.run(dry_run=args.dry_run)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print(json_mod.dumps(result.model_dump(), indent=2, default=str))
        return

    # Table output
    status = "[green]✅ PASSED[/green]" if result.success else "[red]❌ FAILED[/red]"
    console.print(f"\n[bold]Pipeline: {result.name}[/bold]  {status}")
    if result.dry_run:
        console.print("[yellow](dry run)[/yellow]")
    console.print(
        f"Steps: {result.completed_steps}/{result.total_steps} completed, "
        f"{result.failed_steps} failed, {result.duration_ms:.1f}ms\n"
    )


    table = Table(title="Pipeline Steps")
    table.add_column("Step", style="cyan")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Duration (ms)", justify="right")
    table.add_column("Details")

    for sr in result.step_results:
        if sr.success:
            status_str = "[green]✅[/green]"
            details = str(sr.output)[:80] if sr.output else ""
        else:
            status_str = "[red]❌[/red]"
            details = sr.error or ""
        table.add_row(sr.name, sr.type.value, status_str, f"{sr.duration_ms:.1f}", details)

    console.print(table)

    if not result.success:
        import sys

        sys.exit(1)
