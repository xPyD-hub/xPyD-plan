"""Benchmark data validation and outlier detection.

Assess data quality and detect anomalous requests using statistical methods
(IQR and Z-score). Supports automatic filtering and quality scoring.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class OutlierMethod(str, Enum):
    """Outlier detection method."""

    IQR = "iqr"
    ZSCORE = "zscore"


class OutlierInfo(BaseModel):
    """Information about a detected outlier."""

    request_id: str
    index: int = Field(..., description="Index in the requests list")
    metric: str = Field(..., description="Metric where outlier was detected")
    value: float = Field(..., description="Observed value")
    reason: str = Field(..., description="Why it was flagged")


class DataQualityScore(BaseModel):
    """Data quality assessment on a 0-1 scale."""

    completeness: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of non-null required fields"
    )
    consistency: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of requests with consistent latency relationships",
    )
    outlier_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of requests flagged as outliers"
    )
    overall: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted average: 0.3*completeness + 0.3*consistency + 0.4*(1-outlier_ratio)",
    )


class ValidationResult(BaseModel):
    """Result of benchmark data validation."""

    total_requests: int
    outlier_count: int
    outlier_indices: list[int] = Field(default_factory=list)
    outliers: list[OutlierInfo] = Field(default_factory=list)
    method: OutlierMethod
    quality: DataQualityScore
    filtered_data: BenchmarkData | None = Field(
        None, description="Data with outliers removed (only when filter=True)"
    )


def _iqr_bounds(values: list[float], factor: float = 1.5) -> tuple[float, float]:
    """Return (lower, upper) bounds using IQR method."""
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def _zscore_outlier(values: list[float], threshold: float = 3.0) -> list[bool]:
    """Return boolean mask where True = outlier via Z-score."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0:
        return [False] * len(values)
    return [abs((v - mean) / std) > threshold for v in values]


class DataValidator:
    """Validate benchmark data quality and detect outliers.

    Args:
        method: Outlier detection method (iqr or zscore).
        iqr_factor: IQR multiplier for IQR method (default 1.5).
        zscore_threshold: Z-score threshold for zscore method (default 3.0).
    """

    def __init__(
        self,
        method: OutlierMethod = OutlierMethod.IQR,
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> None:
        self.method = method
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold

    def validate(
        self, data: BenchmarkData, *, filter_outliers: bool = False
    ) -> ValidationResult:
        """Validate benchmark data and detect outliers.

        Args:
            data: Benchmark dataset to validate.
            filter_outliers: If True, produce filtered_data with outliers removed.

        Returns:
            ValidationResult with outlier info and quality scores.
        """
        requests = data.requests
        n = len(requests)

        # Extract metric arrays
        metrics: dict[str, list[float]] = {
            "ttft_ms": [r.ttft_ms for r in requests],
            "tpot_ms": [r.tpot_ms for r in requests],
            "total_latency_ms": [r.total_latency_ms for r in requests],
        }

        # Detect outliers per metric
        outlier_set: set[int] = set()
        all_outliers: list[OutlierInfo] = []

        for metric_name, values in metrics.items():
            if self.method == OutlierMethod.IQR:
                lower, upper = _iqr_bounds(values, self.iqr_factor)
                for i, v in enumerate(values):
                    if v < lower or v > upper:
                        outlier_set.add(i)
                        reason = (
                            f"below IQR lower bound {lower:.2f}"
                            if v < lower
                            else f"above IQR upper bound {upper:.2f}"
                        )
                        all_outliers.append(
                            OutlierInfo(
                                request_id=requests[i].request_id,
                                index=i,
                                metric=metric_name,
                                value=v,
                                reason=reason,
                            )
                        )
            else:  # zscore
                flags = _zscore_outlier(values, self.zscore_threshold)
                arr = np.array(values)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                for i, (is_out, v) in enumerate(zip(flags, values)):
                    if is_out:
                        outlier_set.add(i)
                        z = (v - mean) / std if std > 0 else 0.0
                        all_outliers.append(
                            OutlierInfo(
                                request_id=requests[i].request_id,
                                index=i,
                                metric=metric_name,
                                value=v,
                                reason=f"Z-score {z:.2f} exceeds ±{self.zscore_threshold}",
                            )
                        )

        # Quality scoring
        completeness = 1.0  # All fields are required by Pydantic; if we got here, 100%

        # Consistency: total_latency should be >= ttft for each request
        consistent_count = sum(
            1 for r in requests if r.total_latency_ms >= r.ttft_ms
        )
        consistency = consistent_count / n if n > 0 else 1.0

        outlier_ratio = len(outlier_set) / n if n > 0 else 0.0
        overall = 0.3 * completeness + 0.3 * consistency + 0.4 * (1.0 - outlier_ratio)

        quality = DataQualityScore(
            completeness=completeness,
            consistency=consistency,
            outlier_ratio=outlier_ratio,
            overall=round(overall, 4),
        )

        # Build filtered data if requested
        filtered = None
        if filter_outliers and outlier_set:
            clean_requests = [r for i, r in enumerate(requests) if i not in outlier_set]
            if clean_requests:
                filtered = BenchmarkData(
                    metadata=data.metadata,
                    requests=clean_requests,
                )

        sorted_indices = sorted(outlier_set)
        return ValidationResult(
            total_requests=n,
            outlier_count=len(outlier_set),
            outlier_indices=sorted_indices,
            outliers=all_outliers,
            method=self.method,
            quality=quality,
            filtered_data=filtered,
        )


def validate_benchmark(
    path: str | Path,
    method: str = "iqr",
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
    filter_outliers: bool = False,
) -> ValidationResult:
    """Programmatic API: validate a benchmark file.

    Args:
        path: Path to benchmark JSON file.
        method: Outlier detection method ('iqr' or 'zscore').
        iqr_factor: IQR multiplier (default 1.5).
        zscore_threshold: Z-score threshold (default 3.0).
        filter_outliers: If True, include filtered_data in result.

    Returns:
        ValidationResult with quality scores and outlier details.
    """
    raw = json.loads(Path(path).read_text())
    data = BenchmarkData(**raw)
    validator = DataValidator(
        method=OutlierMethod(method),
        iqr_factor=iqr_factor,
        zscore_threshold=zscore_threshold,
    )
    return validator.validate(data, filter_outliers=filter_outliers)
