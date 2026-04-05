"""CLI vllm-commands subcommand."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from xpyd_plan.vllm_commands import CommandConfig, CommandGenerator


def _cmd_vllm_commands(args: argparse.Namespace) -> None:
    """Handle the 'vllm-commands' subcommand."""
    console = Console()

    qps_levels = [float(q) for q in args.qps.split(",")]

    config = CommandConfig(
        model=args.model,
        total_instances=args.total_instances,
        qps_levels=qps_levels,
        tp_size=getattr(args, "tp_size", 1),
        max_model_len=getattr(args, "max_model_len", None),
        dataset=getattr(args, "dataset", None),
        num_prompts=getattr(args, "num_prompts", 1000),
        host=getattr(args, "host", "localhost"),
        port=getattr(args, "port", 8000),
    )

    gen = CommandGenerator()
    result = gen.generate(config)

    if args.output_script:
        script = gen.to_shell_script(result)
        with open(args.output_script, "w") as f:
            f.write(script)
        console.print(
            f"[green]Shell script written to {args.output_script} "
            f"({len(result)} ratios)[/green]"
        )
        return

    output_format = getattr(args, "output_format", "table")

    if output_format == "json":
        print(json.dumps([cs.model_dump() for cs in result], indent=2, default=str))
        return

    # Table output
    console.print("\n[bold]vLLM Benchmark Commands[/bold]")
    console.print(
        f"Model: {config.model} | Instances: {config.total_instances} | "
        f"Ratios: {len(result)}\n"
    )

    table = Table(title="Benchmark Runs")
    table.add_column("P:D Ratio")
    table.add_column("Prefill", justify="right")
    table.add_column("Decode", justify="right")
    table.add_column("QPS Levels")

    for cs in result:
        table.add_row(
            cs.server.ratio,
            str(cs.server.prefill_instances),
            str(cs.server.decode_instances),
            ", ".join(f"{b.qps}" for b in cs.benchmarks),
        )

    console.print(table)
    console.print(
        "\n[dim]Use --output-script to generate an executable shell script[/dim]"
    )


def register_vllm_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the vllm-commands subcommand."""
    p = subparsers.add_parser(
        "vllm-commands",
        help="Generate vLLM benchmark commands for P:D ratio exploration",
    )
    p.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    p.add_argument(
        "--total-instances", type=int, required=True,
        help="Total instances (prefill + decode)",
    )
    p.add_argument(
        "--qps", type=str, required=True,
        help="Comma-separated QPS levels (e.g. 1,2,4)",
    )
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--max-model-len", type=int, default=None, help="Max model length")
    p.add_argument("--dataset", type=str, default=None, help="Dataset path")
    p.add_argument("--num-prompts", type=int, default=1000, help="Prompts per run")
    p.add_argument("--host", type=str, default="localhost", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument("--output-script", type=str, default=None, help="Write shell script")
    p.add_argument(
        "--output-format", choices=["table", "json"], default="table",
        help="Output format",
    )
    p.set_defaults(func=_cmd_vllm_commands)
