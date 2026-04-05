"""Latency anomaly classifier — label each request as NORMAL, SLOW, OUTLIER, or TIMEOUT.

Rule-based per-request classification using IQR-derived thresholds and optional
absolute timeout limits. Produces a labeled dataset for downstream analysis.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from .benchmark_models import BenchmarkData


class AnomalyClass(str, Enum):
    """Request anomaly classification."""

    NORMAL = "normal"
    SLOW = "slow"
    OUTLIER = "outlier"
    TIMEOUT = "timeout"


class RequestLabel(BaseModel):
    """Per-request anomaly label."""

    request_id: str = Field(..., description="Request identifier")
    ttft_class: AnomalyClass = Field(..., description="TTFT classification")
    tpot_class: AnomalyClass = Field(..., description="TPOT classification")
    total_class: AnomalyClass = Field(..., description="Total latency classification")
    worst_class: AnomalyClass = Field(..., description="Worst classification across metrics")


class ClassDistribution(BaseModel):
    """Distribution of anomaly classes for a metric."""

    metric: str = Field(..., description="Metric name")
    normal_count: int = Field(..., ge=0)
    slow_count: int = Field(..., ge=0)
    outlier_count: int = Field(..., ge=0)
    timeout_count: int = Field(..., ge=0)
    total: int = Field(..., ge=1)
    normal_pct: float = Field(..., ge=0, le=100)
    slow_pct: float = Field(..., ge=0, le=100)
    outlier_pct: float = Field(..., ge=0, le=100)
    timeout_pct: float = Field(..., ge=0, le=100)


class AnomalyReport(BaseModel):
    """Complete anomaly classification report."""

    labels: list[RequestLabel] = Field(..., description="Per-request labels")
    distributions: list[ClassDistribution] = Field(
        ..., description="Per-metric class distributions"
    )
    worst_metric: str = Field(..., description="Metric with highest anomaly rate")
    total_anomalous: int = Field(..., ge=0, description="Requests with any non-NORMAL label")
    anomaly_rate: float = Field(..., ge=0, le=100, description="Percentage of anomalous requests")
    recommendation: str = Field(..., description="Human-readable summary")


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    k = (pct / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def _classify_value(
    value: float,
    q75: float,
    iqr: float,
    slow_mult: float,
    outlier_mult: float,
    timeout_threshold: float | None,
) -> AnomalyClass:
    """Classify a single value."""
    if timeout_threshold is not None and value > timeout_threshold:
        return AnomalyClass.TIMEOUT
    if iqr > 0:
        if value > q75 + outlier_mult * iqr:
            return AnomalyClass.OUTLIER
        if value > q75 + slow_mult * iqr:
            return AnomalyClass.SLOW
    return AnomalyClass.NORMAL


_SEVERITY_ORDER = {
    AnomalyClass.NORMAL: 0,
    AnomalyClass.SLOW: 1,
    AnomalyClass.OUTLIER: 2,
    AnomalyClass.TIMEOUT: 3,
}

_METRICS = ["ttft_ms", "tpot_ms", "total_latency_ms"]


class LatencyAnomalyClassifier:
    """Classify benchmark requests into anomaly categories."""

    def classify(
        self,
        data: BenchmarkData,
        slow_multiplier: float = 1.0,
        outlier_multiplier: float = 3.0,
        timeout_ttft: float | None = None,
        timeout_tpot: float | None = None,
        timeout_total: float | None = None,
    ) -> AnomalyReport:
        """Classify each request in the benchmark data.

        Args:
            data: Loaded benchmark data.
            slow_multiplier: IQR multiplier for SLOW threshold (default 1.0).
            outlier_multiplier: IQR multiplier for OUTLIER threshold (default 3.0).
            timeout_ttft: Absolute TTFT timeout threshold in ms (optional).
            timeout_tpot: Absolute TPOT timeout threshold in ms (optional).
            timeout_total: Absolute total latency timeout threshold in ms (optional).

        Returns:
            AnomalyReport with per-request labels and aggregate distributions.

        Raises:
            ValueError: If fewer than 1 request or invalid multipliers.
        """
        if not data.requests:
            raise ValueError("Need at least 1 request for classification")
        if slow_multiplier < 0 or outlier_multiplier < 0:
            raise ValueError("Multipliers must be non-negative")
        if slow_multiplier > outlier_multiplier:
            raise ValueError("slow_multiplier must be <= outlier_multiplier")

        timeout_map = {
            "ttft_ms": timeout_ttft,
            "tpot_ms": timeout_tpot,
            "total_latency_ms": timeout_total,
        }

        # Compute IQR thresholds per metric
        thresholds: dict[str, tuple[float, float]] = {}
        for metric in _METRICS:
            values = [getattr(r, metric) for r in data.requests]
            q75 = _percentile(values, 75.0)
            q25 = _percentile(values, 25.0)
            thresholds[metric] = (q75, q75 - q25)

        # Classify each request
        labels: list[RequestLabel] = []
        for req in data.requests:
            classes: dict[str, AnomalyClass] = {}
            for metric in _METRICS:
                q75, iqr = thresholds[metric]
                value = getattr(req, metric)
                classes[metric] = _classify_value(
                    value, q75, iqr, slow_multiplier, outlier_multiplier, timeout_map[metric]
                )

            worst = max(classes.values(), key=lambda c: _SEVERITY_ORDER[c])
            labels.append(
                RequestLabel(
                    request_id=req.request_id,
                    ttft_class=classes["ttft_ms"],
                    tpot_class=classes["tpot_ms"],
                    total_class=classes["total_latency_ms"],
                    worst_class=worst,
                )
            )

        # Compute distributions
        distributions: list[ClassDistribution] = []
        for metric, attr in [
            ("ttft_ms", "ttft_class"),
            ("tpot_ms", "tpot_class"),
            ("total_latency_ms", "total_class"),
        ]:
            counts = {c: 0 for c in AnomalyClass}
            for label in labels:
                counts[getattr(label, attr)] += 1
            total = len(labels)
            distributions.append(
                ClassDistribution(
                    metric=metric,
                    normal_count=counts[AnomalyClass.NORMAL],
                    slow_count=counts[AnomalyClass.SLOW],
                    outlier_count=counts[AnomalyClass.OUTLIER],
                    timeout_count=counts[AnomalyClass.TIMEOUT],
                    total=total,
                    normal_pct=round(100.0 * counts[AnomalyClass.NORMAL] / total, 2),
                    slow_pct=round(100.0 * counts[AnomalyClass.SLOW] / total, 2),
                    outlier_pct=round(100.0 * counts[AnomalyClass.OUTLIER] / total, 2),
                    timeout_pct=round(100.0 * counts[AnomalyClass.TIMEOUT] / total, 2),
                )
            )

        # Worst metric = highest non-normal rate
        worst_metric = max(
            distributions,
            key=lambda d: d.slow_pct + d.outlier_pct + d.timeout_pct,
        ).metric

        total_anomalous = sum(
            1 for label in labels if label.worst_class != AnomalyClass.NORMAL
        )
        anomaly_rate = round(100.0 * total_anomalous / len(labels), 2)

        # Recommendation
        if anomaly_rate > 20:
            recommendation = (
                f"High anomaly rate ({anomaly_rate}%). "
                f"Worst metric: {worst_metric}. "
                f"Investigate system stability — consider filtering outliers before analysis."
            )
        elif anomaly_rate > 5:
            recommendation = (
                f"Moderate anomaly rate ({anomaly_rate}%). "
                f"Some requests are slow or outliers on {worst_metric}. "
                f"Review tail latency causes."
            )
        else:
            recommendation = (
                f"Low anomaly rate ({anomaly_rate}%). "
                f"Request latencies are well-behaved."
            )

        return AnomalyReport(
            labels=labels,
            distributions=distributions,
            worst_metric=worst_metric,
            total_anomalous=total_anomalous,
            anomaly_rate=anomaly_rate,
            recommendation=recommendation,
        )


def classify_anomalies(
    data: BenchmarkData,
    slow_multiplier: float = 1.0,
    outlier_multiplier: float = 3.0,
    timeout_ttft: float | None = None,
    timeout_tpot: float | None = None,
    timeout_total: float | None = None,
) -> dict:
    """Programmatic API for anomaly classification.

    Args:
        data: Benchmark data to classify.
        slow_multiplier: IQR multiplier for SLOW threshold.
        outlier_multiplier: IQR multiplier for OUTLIER threshold.
        timeout_ttft: Absolute TTFT timeout in ms.
        timeout_tpot: Absolute TPOT timeout in ms.
        timeout_total: Absolute total latency timeout in ms.

    Returns:
        Dictionary representation of the AnomalyReport.
    """
    classifier = LatencyAnomalyClassifier()
    report = classifier.classify(
        data,
        slow_multiplier=slow_multiplier,
        outlier_multiplier=outlier_multiplier,
        timeout_ttft=timeout_ttft,
        timeout_tpot=timeout_tpot,
        timeout_total=timeout_total,
    )
    return report.model_dump()
