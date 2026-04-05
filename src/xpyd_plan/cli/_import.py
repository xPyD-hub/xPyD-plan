"""CLI import command — import vLLM benchmark data."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console

from xpyd_plan.vllm_import import import_vllm


def _cmd_import(args: argparse.Namespace) -> None:
    """Handle the 'import' subcommand."""
    console = Console()

    if args.prefill_instances is None or args.decode_instances is None:
        console.print(
            "[red]Error: --prefill-instances and --decode-instances are required[/red]"
        )
        sys.exit(1)

    try:
        result = import_vllm(
            input_path=args.input,
            num_prefill_instances=args.prefill_instances,
            num_decode_instances=args.decode_instances,
            output_path=args.output,
            format=args.format,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)

    output_format = getattr(args, "output_format", "table")
    if output_format == "json":
        json.dump(result.model_dump(), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    console.print("\n[bold]Import Result[/bold]")
    console.print(f"  Source format: {result.source_format}")
    console.print(f"  Requests imported: {result.num_requests}")
    meta = result.benchmark_data.metadata
    console.print(f"  Prefill instances: {meta.num_prefill_instances}")
    console.print(f"  Decode instances: {meta.num_decode_instances}")
    console.print(f"  Measured QPS: {meta.measured_qps:.2f}")

    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for w in result.warnings[:10]:
            console.print(f"  ⚠ {w}")
        if len(result.warnings) > 10:
            console.print(f"  ... and {len(result.warnings) - 10} more")

    if args.output:
        console.print(f"\n  Output saved to: {args.output}")

    console.print()


def add_import_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the import subcommand parser."""
    parser = subparsers.add_parser(
        "import",
        help="Import vLLM benchmark data to native format",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input benchmark JSON file",
    )
    parser.add_argument(
        "--format",
        choices=["vllm", "native", "auto"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save converted native-format JSON",
    )
    parser.add_argument(
        "--prefill-instances",
        type=int,
        required=True,
        help="Number of prefill instances in the cluster",
    )
    parser.add_argument(
        "--decode-instances",
        type=int,
        required=True,
        help="Number of decode instances in the cluster",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.set_defaults(func=_cmd_import)
