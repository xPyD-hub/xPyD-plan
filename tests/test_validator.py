"""Tests for benchmark data validation and outlier detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.validator import (
    DataValidator,
    OutlierMethod,
    validate_benchmark,
)


def _make_requests(
    n: int = 50,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 0.2,
) -> list[BenchmarkRequest]:
    """Generate benchmark requests with controlled latency distribution."""
    requests = []
    for i in range(n):
        factor = 1.0 + spread * (i / max(n - 1, 1))
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=ttft_base * factor,
                tpot_ms=tpot_base * factor,
                total_latency_ms=total_base * factor,
                timestamp=1700000000.0 + i,
            )
        )
    return requests


def _make_benchmark(
    n: int = 50,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 0.2,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=_make_requests(n, ttft_base, tpot_base, total_base, spread),
    )


def _make_benchmark_with_outliers() -> BenchmarkData:
    """Create benchmark with a few extreme outliers."""
    reqs = _make_requests(50, ttft_base=50.0, tpot_base=10.0, total_base=200.0, spread=0.2)
    # Add extreme outliers
    for i in range(3):
        reqs.append(
            BenchmarkRequest(
                request_id=f"outlier-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=5000.0,  # 100x normal
                tpot_ms=1000.0,
                total_latency_ms=20000.0,
                timestamp=1700000100.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


# --- DataValidator basic tests ---


class TestDataValidatorIQR:
    """Tests for IQR-based outlier detection."""

    def test_no_outliers_clean_data(self) -> None:
        data = _make_benchmark(n=50, spread=0.1)
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data)
        assert result.total_requests == 50
        assert result.method == OutlierMethod.IQR
        assert result.quality.completeness == 1.0

    def test_detects_outliers(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data)
        assert result.outlier_count > 0
        assert len(result.outlier_indices) > 0
        assert len(result.outliers) > 0
        # The 3 extreme outliers should be detected
        outlier_ids = {o.request_id for o in result.outliers}
        for i in range(3):
            assert f"outlier-{i}" in outlier_ids

    def test_outlier_indices_sorted(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data)
        assert result.outlier_indices == sorted(result.outlier_indices)

    def test_filter_outliers_produces_filtered_data(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data, filter_outliers=True)
        assert result.filtered_data is not None
        assert len(result.filtered_data.requests) < len(data.requests)
        # No outlier IDs in filtered data
        filtered_ids = {r.request_id for r in result.filtered_data.requests}
        for i in range(3):
            assert f"outlier-{i}" not in filtered_ids

    def test_no_filter_when_no_outliers(self) -> None:
        data = _make_benchmark(n=50, spread=0.05)
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data, filter_outliers=True)
        # If no outliers, filtered_data may be None
        if result.outlier_count == 0:
            assert result.filtered_data is None

    def test_iqr_factor_adjustable(self) -> None:
        data = _make_benchmark_with_outliers()
        # Very tight factor → more outliers
        tight = DataValidator(method=OutlierMethod.IQR, iqr_factor=0.5)
        result_tight = tight.validate(data)
        # Loose factor → fewer outliers
        loose = DataValidator(method=OutlierMethod.IQR, iqr_factor=3.0)
        result_loose = loose.validate(data)
        assert result_tight.outlier_count >= result_loose.outlier_count


class TestDataValidatorZScore:
    """Tests for Z-score-based outlier detection."""

    def test_detects_outliers_zscore(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator(method=OutlierMethod.ZSCORE)
        result = validator.validate(data)
        assert result.method == OutlierMethod.ZSCORE
        assert result.outlier_count > 0
        outlier_ids = {o.request_id for o in result.outliers}
        for i in range(3):
            assert f"outlier-{i}" in outlier_ids

    def test_zscore_threshold_adjustable(self) -> None:
        data = _make_benchmark_with_outliers()
        tight = DataValidator(method=OutlierMethod.ZSCORE, zscore_threshold=1.5)
        result_tight = tight.validate(data)
        loose = DataValidator(method=OutlierMethod.ZSCORE, zscore_threshold=5.0)
        result_loose = loose.validate(data)
        assert result_tight.outlier_count >= result_loose.outlier_count

    def test_zscore_no_outliers_uniform_data(self) -> None:
        # All identical values → std=0 → no outliers
        reqs = [
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=50.0,
                tpot_ms=10.0,
                total_latency_ms=200.0,
                timestamp=1700000000.0 + i,
            )
            for i in range(20)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=2,
                num_decode_instances=2,
                total_instances=4,
                measured_qps=10.0,
            ),
            requests=reqs,
        )
        validator = DataValidator(method=OutlierMethod.ZSCORE)
        result = validator.validate(data)
        assert result.outlier_count == 0

    def test_filter_outliers_zscore(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator(method=OutlierMethod.ZSCORE)
        result = validator.validate(data, filter_outliers=True)
        assert result.filtered_data is not None
        assert len(result.filtered_data.requests) < len(data.requests)


# --- Quality scoring tests ---


class TestDataQualityScore:
    """Tests for data quality scoring."""

    def test_perfect_quality_no_outliers(self) -> None:
        data = _make_benchmark(n=50, spread=0.05)
        validator = DataValidator(method=OutlierMethod.IQR)
        result = validator.validate(data)
        q = result.quality
        assert q.completeness == 1.0
        assert q.consistency == 1.0
        if result.outlier_count == 0:
            assert q.overall == pytest.approx(1.0, abs=0.01)

    def test_outliers_reduce_quality(self) -> None:
        clean = _make_benchmark(n=50, spread=0.05)
        dirty = _make_benchmark_with_outliers()
        v = DataValidator(method=OutlierMethod.IQR)
        q_clean = v.validate(clean).quality
        q_dirty = v.validate(dirty).quality
        assert q_dirty.overall <= q_clean.overall

    def test_consistency_detects_bad_latency(self) -> None:
        """Requests where total_latency < ttft should reduce consistency."""
        reqs = [
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=100.0,
                tpot_ms=10.0,
                total_latency_ms=50.0,  # Less than TTFT — inconsistent
                timestamp=1700000000.0 + i,
            )
            for i in range(20)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=2,
                num_decode_instances=2,
                total_instances=4,
                measured_qps=10.0,
            ),
            requests=reqs,
        )
        validator = DataValidator()
        result = validator.validate(data)
        assert result.quality.consistency == 0.0

    def test_quality_overall_formula(self) -> None:
        """Overall = 0.3*completeness + 0.3*consistency + 0.4*(1-outlier_ratio)."""
        data = _make_benchmark_with_outliers()
        validator = DataValidator()
        result = validator.validate(data)
        q = result.quality
        expected = 0.3 * q.completeness + 0.3 * q.consistency + 0.4 * (1.0 - q.outlier_ratio)
        assert q.overall == pytest.approx(expected, abs=0.001)


# --- OutlierInfo model tests ---


class TestOutlierInfo:
    """Tests for OutlierInfo fields."""

    def test_outlier_info_fields(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator()
        result = validator.validate(data)
        assert len(result.outliers) > 0
        o = result.outliers[0]
        assert isinstance(o.request_id, str)
        assert isinstance(o.index, int)
        assert isinstance(o.metric, str)
        assert isinstance(o.value, float)
        assert isinstance(o.reason, str)

    def test_outlier_metrics_are_valid(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator()
        result = validator.validate(data)
        valid_metrics = {"ttft_ms", "tpot_ms", "total_latency_ms"}
        for o in result.outliers:
            assert o.metric in valid_metrics


# --- ValidationResult model tests ---


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_serializable(self) -> None:
        data = _make_benchmark_with_outliers()
        validator = DataValidator()
        result = validator.validate(data)
        # Should be JSON-serializable (excluding filtered_data for brevity)
        j = result.model_dump_json(exclude={"filtered_data"})
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["total_requests"] == 53

    def test_method_recorded(self) -> None:
        data = _make_benchmark()
        for m in [OutlierMethod.IQR, OutlierMethod.ZSCORE]:
            v = DataValidator(method=m)
            r = v.validate(data)
            assert r.method == m


# --- Programmatic API tests ---


class TestValidateBenchmark:
    """Tests for the validate_benchmark() programmatic API."""

    def test_validate_from_file(self, tmp_path: Path) -> None:
        data = _make_benchmark_with_outliers()
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        result = validate_benchmark(str(p))
        assert result.total_requests == 53
        assert result.outlier_count > 0

    def test_validate_from_file_zscore(self, tmp_path: Path) -> None:
        data = _make_benchmark_with_outliers()
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        result = validate_benchmark(str(p), method="zscore")
        assert result.method == OutlierMethod.ZSCORE
        assert result.outlier_count > 0

    def test_validate_with_filter(self, tmp_path: Path) -> None:
        data = _make_benchmark_with_outliers()
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        result = validate_benchmark(str(p), filter_outliers=True)
        assert result.filtered_data is not None

    def test_validate_clean_data(self, tmp_path: Path) -> None:
        data = _make_benchmark(n=50, spread=0.05)
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        result = validate_benchmark(str(p))
        assert result.quality.completeness == 1.0
