"""CLI generate command."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _cmd_generate(args: argparse.Namespace) -> None:
    """Handle 'generate' subcommand."""

    from xpyd_plan.generator import (
        BenchmarkGenerator,
        GeneratorConfig,
        load_generator_config,
    )

    console = Console()

    if args.config:
        cfg = load_generator_config(args.config)
    else:
        cfg = GeneratorConfig()

    # Apply CLI overrides
    overrides: dict = {}
    if args.num_requests is not None:
        overrides["num_requests"] = args.num_requests
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    generator = BenchmarkGenerator(cfg)
    data = generator.generate()
    generator2 = BenchmarkGenerator(cfg)
    generator2.to_json(args.output)

    if args.output_format == "json":
        import json

        summary = {
            "output_file": args.output,
            "num_requests": len(data.requests),
            "metadata": data.metadata.model_dump(),
            "latency_distribution": cfg.latency.distribution.value,
            "anomalies": len(cfg.anomalies),
        }
        console.print_json(json.dumps(summary))
    else:
        table = Table(title="Generated Benchmark Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Output file", args.output)
        table.add_row("Requests", str(len(data.requests)))
        table.add_row("Prefill instances", str(data.metadata.num_prefill_instances))
        table.add_row("Decode instances", str(data.metadata.num_decode_instances))
        table.add_row("Measured QPS", str(data.metadata.measured_qps))
        table.add_row("Distribution", cfg.latency.distribution.value)
        table.add_row("Anomalies configured", str(len(cfg.anomalies)))
        if cfg.seed is not None:
            table.add_row("Seed", str(cfg.seed))
        console.print(table)
