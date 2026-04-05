"""CLI replay command."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from xpyd_plan.replay import ReplayConfig, ReplayGenerator


def _cmd_replay(args: argparse.Namespace) -> None:
    """Handle the 'replay' subcommand."""
    console = Console()
    output_format = getattr(args, "output_format", "table")
    output_path = getattr(args, "output", None)

    config = ReplayConfig(
        time_scale=args.time_scale,
        target_qps=args.target_qps,
    )
    generator = ReplayGenerator(config)

    try:
        schedule = generator.generate_from_file(args.benchmark)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if output_format == "json":
        out = {
            "request_count": schedule.request_count,
            "total_duration_ms": schedule.total_duration_ms,
            "effective_qps": schedule.effective_qps,
            "config": {
                "time_scale": schedule.config.time_scale,
                "target_qps": schedule.config.target_qps,
            },
            "entries": [e.model_dump() for e in schedule.entries],
        }
        if output_path:
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
            console.print(f"[green]Replay schedule written to {output_path}[/green]")
        else:
            console.print_json(json.dumps(out))
        return

    # Table output — summary + first/last entries
    summary = Table(title="Replay Schedule Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Request count", str(schedule.request_count))
    summary.add_row("Total duration (ms)", f"{schedule.total_duration_ms:.1f}")
    summary.add_row("Effective QPS", f"{schedule.effective_qps:.2f}")
    summary.add_row("Time scale", f"{schedule.config.time_scale:.2f}")
    if schedule.config.target_qps:
        summary.add_row("Target QPS", f"{schedule.config.target_qps:.2f}")
    console.print(summary)

    if schedule.entries:
        detail = Table(title="Replay Entries (first 20)")
        detail.add_column("#", justify="right")
        detail.add_column("Offset (ms)", justify="right")
        detail.add_column("Prompt Tokens", justify="right")
        detail.add_column("Output Tokens", justify="right")

        for i, entry in enumerate(schedule.entries[:20]):
            detail.add_row(
                str(i + 1),
                f"{entry.offset_ms:.1f}",
                str(entry.prompt_tokens),
                str(entry.output_tokens),
            )
        if len(schedule.entries) > 20:
            detail.add_row("...", "...", "...", "...")
        console.print(detail)

    if output_path:
        out = {
            "request_count": schedule.request_count,
            "total_duration_ms": schedule.total_duration_ms,
            "effective_qps": schedule.effective_qps,
            "entries": [e.model_dump() for e in schedule.entries],
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        console.print(f"[green]Replay schedule written to {output_path}[/green]")


def add_replay_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the 'replay' subcommand parser."""
    p = subparsers.add_parser(
        "replay",
        help="Generate a replay schedule from benchmark data",
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark JSON file",
    )
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Time scaling factor (2.0 = 2x faster, 0.5 = 2x slower; default: 1.0)",
    )
    p.add_argument(
        "--target-qps",
        type=float,
        default=None,
        help="Override QPS with uniform distribution",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output file path for the replay schedule JSON",
    )
    p.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    p.set_defaults(func=_cmd_replay)
