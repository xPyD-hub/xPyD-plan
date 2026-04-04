"""CLI config subcommand and config helpers."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console


def _add_config_flag(parser: argparse.ArgumentParser) -> None:
    """Add --config flag to a subparser."""
    parser.add_argument(
        "--config", type=str, default=None, dest="config_file",
        help="Path to YAML config profile (default: auto-detect xpyd-plan.yaml)",
    )


def _apply_config_defaults(args: argparse.Namespace) -> None:
    """Apply config profile defaults to args where CLI didn't set a value."""
    from xpyd_plan.config import load_config

    config_path = getattr(args, "config_file", None)
    cfg = load_config(config_path)

    # SLA defaults
    if getattr(args, "sla_ttft", None) is None and cfg.sla.ttft_ms is not None:
        args.sla_ttft = cfg.sla.ttft_ms
    if getattr(args, "sla_tpot", None) is None and cfg.sla.tpot_ms is not None:
        args.sla_tpot = cfg.sla.tpot_ms
    if getattr(args, "sla_max_latency", None) is None and cfg.sla.max_latency_ms is not None:
        args.sla_max_latency = cfg.sla.max_latency_ms
    if getattr(args, "sla_percentile", None) == 95.0 and cfg.sla.percentile is not None:
        args.sla_percentile = cfg.sla.percentile

    # Output defaults
    if hasattr(args, "output_format") and args.output_format == "table":
        args.output_format = cfg.output.format
    if hasattr(args, "top") and args.top == 5:
        args.top = cfg.output.top

    # Defaults
    if getattr(args, "total_instances", None) is None and cfg.defaults.total_instances is not None:
        args.total_instances = cfg.defaults.total_instances
    if hasattr(args, "format") and args.format == "auto":
        args.format = cfg.defaults.benchmark_format

    # Cost defaults
    if getattr(args, "budget_ceiling", None) is None and cfg.cost.budget_ceiling is not None:
        args.budget_ceiling = cfg.cost.budget_ceiling

    # Store resolved config for potential use
    args._resolved_config = cfg


def _cmd_config(args: argparse.Namespace) -> None:
    """Handle the 'config' subcommand."""
    from xpyd_plan.config import init_config, load_config

    console = Console()

    if args.config_action == "init":
        path = init_config(args.output_path)
        console.print(f"[green]✅ Config file created: {path}[/green]")
    elif args.config_action == "show":
        config_path = getattr(args, "config_file", None)
        cfg = load_config(config_path)
        console.print("[bold]Resolved configuration:[/bold]\n")
        console.print(cfg.to_yaml())
    else:
        console.print("[red]Error: specify 'init' or 'show'[/red]")
        sys.exit(1)
