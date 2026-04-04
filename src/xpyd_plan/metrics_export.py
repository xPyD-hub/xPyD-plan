"""Prometheus/OpenMetrics text format exporter for benchmark analysis results."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class MetricLine(BaseModel):
    """A single metric line in OpenMetrics format."""

    name: str = Field(..., description="Metric name")
    labels: dict[str, str] = Field(default_factory=dict, description="Label key-value pairs")
    value: float = Field(..., description="Metric value")
    metric_type: str = Field("gauge", description="Metric type (gauge, counter, info)")
    help_text: str = Field("", description="HELP description for this metric")


class MetricsReport(BaseModel):
    """Complete metrics export result."""

    metrics: list[MetricLine] = Field(default_factory=list)
    text: str = Field("", description="Rendered OpenMetrics text")


class MetricsExporter:
    """Export benchmark analysis results in Prometheus/OpenMetrics text format."""

    METRIC_PREFIX = "xpyd_plan"

    def export(self, benchmarks: list[BenchmarkData]) -> MetricsReport:
        """Export one or more benchmark datasets as OpenMetrics metrics.

        Args:
            benchmarks: Benchmark datasets to export.

        Returns:
            MetricsReport with structured metrics and rendered text.

        Raises:
            ValueError: If benchmarks list is empty.
        """
        if not benchmarks:
            msg = "Need at least 1 benchmark dataset"
            raise ValueError(msg)

        metrics: list[MetricLine] = []

        for bench in benchmarks:
            meta = bench.metadata
            labels = {
                "num_prefill": str(meta.num_prefill_instances),
                "num_decode": str(meta.num_decode_instances),
                "total_instances": str(meta.total_instances),
                "pd_ratio": f"{meta.num_prefill_instances}:{meta.num_decode_instances}",
            }

            # Instance counts
            metrics.append(MetricLine(
                name=f"{self.METRIC_PREFIX}_prefill_instances",
                labels=labels,
                value=float(meta.num_prefill_instances),
                help_text="Number of prefill instances",
            ))
            metrics.append(MetricLine(
                name=f"{self.METRIC_PREFIX}_decode_instances",
                labels=labels,
                value=float(meta.num_decode_instances),
                help_text="Number of decode instances",
            ))
            metrics.append(MetricLine(
                name=f"{self.METRIC_PREFIX}_total_instances",
                labels=labels,
                value=float(meta.total_instances),
                help_text="Total instance count",
            ))

            # QPS
            metrics.append(MetricLine(
                name=f"{self.METRIC_PREFIX}_measured_qps",
                labels=labels,
                value=meta.measured_qps,
                help_text="Measured queries per second",
            ))

            # Request count
            metrics.append(MetricLine(
                name=f"{self.METRIC_PREFIX}_request_count",
                labels=labels,
                value=float(len(bench.requests)),
                metric_type="counter",
                help_text="Total number of benchmark requests",
            ))

            # Latency percentiles
            ttft_vals = [r.ttft_ms for r in bench.requests]
            tpot_vals = [r.tpot_ms for r in bench.requests]
            total_vals = [r.total_latency_ms for r in bench.requests]

            for metric_name, values in [
                ("ttft_ms", ttft_vals),
                ("tpot_ms", tpot_vals),
                ("total_latency_ms", total_vals),
            ]:
                for pct in [50, 95, 99]:
                    pct_val = float(np.percentile(values, pct))
                    pct_labels = {**labels, "percentile": str(pct)}
                    metrics.append(MetricLine(
                        name=f"{self.METRIC_PREFIX}_{metric_name}",
                        labels=pct_labels,
                        value=round(pct_val, 4),
                        help_text=f"{metric_name} at p{pct}",
                    ))

        text = self._render_openmetrics(metrics)
        return MetricsReport(metrics=metrics, text=text)

    def _render_openmetrics(self, metrics: list[MetricLine]) -> str:
        """Render metrics as OpenMetrics text format."""
        lines: list[str] = []
        seen_help: set[str] = set()

        for m in metrics:
            if m.name not in seen_help:
                lines.append(f"# HELP {m.name} {m.help_text}")
                lines.append(f"# TYPE {m.name} {m.metric_type}")
                seen_help.add(m.name)

            if m.labels:
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in sorted(m.labels.items())
                )
                lines.append(f"{m.name}{{{label_str}}} {m.value}")
            else:
                lines.append(f"{m.name} {m.value}")

        lines.append("# EOF")
        return "\n".join(lines) + "\n"


def export_metrics(
    benchmarks: list[BenchmarkData],
) -> dict:
    """Programmatic API for metrics export.

    Args:
        benchmarks: List of benchmark datasets.

    Returns:
        Dictionary representation of MetricsReport.
    """
    exporter = MetricsExporter()
    report = exporter.export(benchmarks)
    return report.model_dump()
