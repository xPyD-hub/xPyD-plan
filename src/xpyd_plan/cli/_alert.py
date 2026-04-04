"""CLI alert command."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table


def _cmd_alert(args: argparse.Namespace) -> None:
    """Execute the alert subcommand."""
    import json as json_mod

    from xpyd_plan.alerting import AlertEngine
    from xpyd_plan.benchmark_models import BenchmarkData

    console = Console()

    # Load benchmark data
    with open(args.benchmark) as f:
        raw = json_mod.load(f)
    data = BenchmarkData(**raw)

    # Load rules and evaluate
    engine = AlertEngine.from_yaml(args.rules)
    report = engine.evaluate(data)

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        console.print_json(report.model_dump_json(indent=2))
    else:
        # Table output
        table = Table(title="Alert Evaluation")
        table.add_column("Rule", style="bold")
        table.add_column("Metric")
        table.add_column("Actual")
        table.add_column("Threshold")
        table.add_column("Severity")
        table.add_column("Status")

        for result in report.results:
            severity_style = {
                "critical": "red bold",
                "warning": "yellow",
                "info": "blue",
            }.get(result.severity.value, "")
            status = "[red]TRIGGERED[/red]" if result.triggered else "[green]OK[/green]"
            table.add_row(
                result.rule_name,
                result.metric,
                f"{result.actual_value:.2f}",
                f"{result.threshold:.2f}",
                f"[{severity_style}]{result.severity.value.upper()}[/{severity_style}]",
                status,
            )

        console.print(table)
        console.print()
        if report.passed:
            console.print(f"[green bold]✓ {report.summary}[/green bold]")
        else:
            console.print(f"[red bold]✗ {report.summary}[/red bold]")

    if not report.passed:
        sys.exit(1)
