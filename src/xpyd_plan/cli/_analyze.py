"""CLI analyze command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.bench_adapter import load_benchmark_auto
from xpyd_plan.cli._helpers import _print_cost_analysis, _print_sensitivity, _print_single_analysis
from xpyd_plan.models import SLAConfig


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' subcommand."""
    console = Console()

    sla = SLAConfig(
        ttft_ms=args.sla_ttft,
        tpot_ms=args.sla_tpot,
        max_latency_ms=args.sla_max_latency,
        sla_percentile=args.sla_percentile,
    )

    # Handle streaming mode
    if args.stream:
        from xpyd_plan.streaming import stream_from_stdin

        console.print("[bold]📡 Streaming mode — reading JSONL from stdin...[/bold]")
        snapshot_interval = args.snapshot_interval or 10
        stream_from_stdin(sla=sla, snapshot_interval=snapshot_interval)
        return

    if not args.benchmark:
        console.print("[red]Error: --benchmark is required unless --stream is used.[/red]")
        sys.exit(1)

    analyzer = BenchmarkAnalyzer()
    benchmarks = args.benchmark
    fmt = args.format

    # Load with format detection
    if fmt == "auto":
        load_fn = load_benchmark_auto
    elif fmt == "xpyd-bench":
        from xpyd_plan.bench_adapter import XpydBenchAdapter

        adapter = XpydBenchAdapter()
        load_fn = adapter.load
    else:
        # native — use analyzer's built-in loader
        load_fn = None

    if len(benchmarks) == 1:
        # Single-file mode (original behavior)
        if load_fn:
            data = load_fn(benchmarks[0])
            analyzer._data = data
        else:
            analyzer.load_data(benchmarks[0])

        # Validate and filter outliers if requested
        if getattr(args, "validate", False):
            from xpyd_plan.validator import DataValidator, OutlierMethod

            method = OutlierMethod(getattr(args, "outlier_method", "iqr"))
            validator = DataValidator(method=method)
            vr = validator.validate(analyzer.data, filter_outliers=True)
            console.print(
                f"\n[bold]🔍 Validation:[/bold] {vr.outlier_count}/{vr.total_requests} "
                f"outliers detected (method={vr.method.value}, "
                f"quality={vr.quality.overall:.2f})"
            )
            if vr.filtered_data is not None:
                analyzer._data = vr.filtered_data
                console.print(
                    f"[dim]   Filtered to {len(vr.filtered_data.requests)} clean requests[/dim]"
                )

        total = args.total_instances or analyzer.data.metadata.total_instances
        _print_single_analysis(console, analyzer, sla, total, args.top)

        if args.cost_model:
            _print_cost_analysis(console, analyzer, sla, total, args)

        if args.sensitivity:
            _print_sensitivity(console, analyzer, sla, total)

        if args.report:
            result = analyzer.find_optimal_ratio(total, sla)
            report_fmt = getattr(args, "report_format", "html")
            if report_fmt == "markdown":
                from xpyd_plan.md_report import MarkdownReporter

                gen = MarkdownReporter()
                md = gen.generate_single(result, total_instances=total)
                gen.write(md, args.report)
                console.print(f"\n[dim]Markdown report written to {args.report}[/dim]")
            else:
                from xpyd_plan.report import ReportGenerator

                gen = ReportGenerator()
                html = gen.generate_single(result, total_instances=total)
                gen.write(html, args.report)
                console.print(f"\n[dim]HTML report written to {args.report}[/dim]")

        if args.output:
            result = analyzer.find_optimal_ratio(total, sla)
            Path(args.output).write_text(result.model_dump_json(indent=2))
            console.print(f"\n[dim]Result written to {args.output}[/dim]")

        # Machine-readable output format
        if args.output_format in ("json", "csv"):
            from xpyd_plan.export import result_to_csv, result_to_json

            result = analyzer.find_optimal_ratio(total, sla)
            if args.output_format == "json":
                print(result_to_json(result))
            else:
                print(result_to_csv(result), end="")
    else:
        # Multi-file mode
        if load_fn:
            datasets = [load_fn(b) for b in benchmarks]
            datasets.sort(key=lambda d: d.metadata.measured_qps)
            analyzer._multi_data = datasets
            analyzer._data = datasets[0]
        else:
            analyzer.load_multi_data(benchmarks)
        total = args.total_instances or analyzer.multi_data[0].metadata.total_instances

        console.print(f"\n[bold]📊 Multi-Scenario Analysis ({len(benchmarks)} scenarios)[/bold]")

        multi_result = analyzer.find_optimal_ratio_multi(total, sla)

        # Per-scenario summary table
        summary_table = Table(title="\nPer-Scenario Summary")
        summary_table.add_column("QPS", justify="right")
        summary_table.add_column("Config", style="cyan")
        summary_table.add_column("Best P:D", style="green")
        summary_table.add_column("Waste", justify="right")
        summary_table.add_column("SLA", justify="center")

        for scenario in multi_result.scenarios:
            sla_check = scenario.analysis.current_sla_check
            best = scenario.analysis.best
            sla_str = "✅" if sla_check and sla_check.meets_all else "❌"
            best_str = best.ratio_str if best else "N/A"
            waste_str = f"{best.waste_rate:.1%}" if best else "N/A"
            summary_table.add_row(
                f"{scenario.qps:.1f}",
                f"{scenario.analysis.total_instances}",
                best_str,
                waste_str,
                sla_str,
            )

        console.print(summary_table)

        # Unified recommendation
        console.print(
            f"\n[bold]🎯 Unified Recommendation (total={total} instances)[/bold]"
        )
        if multi_result.unified_best:
            u = multi_result.unified_best
            console.print(
                f"\n[bold green]✅ Unified Best: {u.ratio_str}[/bold green]"
                f"  (worst-case waste: {u.waste_rate:.1%})"
            )
        else:
            console.print(
                "\n[bold red]❌ No single P:D ratio meets SLA"
                " across all scenarios.[/bold red]"
            )
            console.print("   Consider per-scenario scaling:")
            for i, scenario in enumerate(multi_result.scenarios):
                best = scenario.analysis.best
                if best:
                    console.print(f"   QPS {scenario.qps:.1f}: {best.ratio_str}")
                else:
                    console.print(f"   QPS {scenario.qps:.1f}: No ratio meets SLA")

        if args.sensitivity:
            for dataset in analyzer.multi_data:
                analyzer._data = dataset
                meta = dataset.metadata
                console.print(
                    f"\n[bold]📈 Sensitivity Analysis"
                    f" (QPS={meta.measured_qps:.1f})[/bold]"
                )
                _print_sensitivity(console, analyzer, sla, total)
            analyzer._data = analyzer.multi_data[0]

        if args.report:
            report_fmt = getattr(args, "report_format", "html")
            if report_fmt == "markdown":
                from xpyd_plan.md_report import MarkdownReporter

                gen = MarkdownReporter()
                md = gen.generate_multi(multi_result)
                gen.write(md, args.report)
                console.print(f"\n[dim]Markdown report written to {args.report}[/dim]")
            else:
                from xpyd_plan.report import ReportGenerator

                gen = ReportGenerator()
                html = gen.generate_multi(multi_result)
                gen.write(html, args.report)
                console.print(f"\n[dim]HTML report written to {args.report}[/dim]")

        if args.output:
            Path(args.output).write_text(multi_result.model_dump_json(indent=2))
            console.print(f"\n[dim]Result written to {args.output}[/dim]")

        # Machine-readable output format
        if args.output_format in ("json", "csv"):
            from xpyd_plan.export import result_to_csv, result_to_json

            if args.output_format == "json":
                print(result_to_json(multi_result))
            else:
                print(result_to_csv(multi_result), end="")
