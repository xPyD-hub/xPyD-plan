"""CLI import command — import vLLM/SGLang/TensorRT-LLM benchmark data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from xpyd_plan.sglang_import import (
    SGLangImportConfig,
    _detect_sglang_format,
    import_sglang,
    import_sglang_data,
)
from xpyd_plan.trtllm_import import (
    TRTLLMImportConfig,
    _detect_trtllm_format,
    import_trtllm,
    import_trtllm_data,
)
from xpyd_plan.vllm_import import import_vllm


def _cmd_import(args: argparse.Namespace) -> None:
    """Handle the 'import' subcommand."""
    console = Console()

    if args.prefill_instances is None or args.decode_instances is None:
        console.print(
            "[red]Error: --prefill-instances and --decode-instances are required[/red]"
        )
        sys.exit(1)

    fmt = args.format

    try:
        if fmt == "trtllm":
            config = TRTLLMImportConfig(
                num_prefill_instances=args.prefill_instances,
                num_decode_instances=args.decode_instances,
                format="trtllm",
            )
            result = import_trtllm(args.input, config)
            if args.output:
                Path(args.output).write_text(
                    result.benchmark_data.model_dump_json(indent=2),
                    encoding="utf-8",
                )
        elif fmt == "sglang":
            config = SGLangImportConfig(
                num_prefill_instances=args.prefill_instances,
                num_decode_instances=args.decode_instances,
                format="sglang",
            )
            result = import_sglang(args.input, config)
            if args.output:
                Path(args.output).write_text(
                    result.benchmark_data.model_dump_json(indent=2),
                    encoding="utf-8",
                )
        elif fmt == "auto":
            raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
            if _detect_trtllm_format(raw):
                trtllm_config = TRTLLMImportConfig(
                    num_prefill_instances=args.prefill_instances,
                    num_decode_instances=args.decode_instances,
                    format="trtllm",
                )
                result = import_trtllm_data(raw, trtllm_config)
                if args.output:
                    Path(args.output).write_text(
                        result.benchmark_data.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
            elif _detect_sglang_format(raw):
                sglang_config = SGLangImportConfig(
                    num_prefill_instances=args.prefill_instances,
                    num_decode_instances=args.decode_instances,
                    format="sglang",
                )
                result = import_sglang_data(raw, sglang_config)
                if args.output:
                    Path(args.output).write_text(
                        result.benchmark_data.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
            else:
                result = import_vllm(
                    input_path=args.input,
                    num_prefill_instances=args.prefill_instances,
                    num_decode_instances=args.decode_instances,
                    output_path=args.output,
                    format="auto",
                )
        else:
            result = import_vllm(
                input_path=args.input,
                num_prefill_instances=args.prefill_instances,
                num_decode_instances=args.decode_instances,
                output_path=args.output,
                format=fmt,
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
        help="Import vLLM/SGLang/TensorRT-LLM benchmark data to native format",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input benchmark JSON file",
    )
    parser.add_argument(
        "--format",
        choices=["vllm", "sglang", "trtllm", "native", "auto"],
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
