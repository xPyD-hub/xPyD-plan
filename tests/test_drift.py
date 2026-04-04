"""Tests for distribution drift detection."""

from __future__ import annotations

import json
import random
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.drift import (
    DriftDetector,
    DriftSeverity,
    _bisect_right,
    _classify_severity,
    _ks_two_sample,
    detect_drift,
)


def _make_benchmark(
    num_requests: int = 100,
    ttft_base: float = 20.0,
    tpot_base: float = 10.0,
    total_base: float = 70.0,
    noise: float = 5.0,
    seed: int = 42,
) -> BenchmarkData:
    """Create a benchmark dataset with configurable latency distributions."""
    rng = random.Random(seed)
    requests_list = []
    for i in range(num_requests):
        requests_list.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=max(0.1, ttft_base + rng.gauss(0, noise)),
                tpot_ms=max(0.1, tpot_base + rng.gauss(0, noise)),
                total_latency_ms=max(0.1, total_base + rng.gauss(0, noise)),
                timestamp=1000.0 + i,
            )
        )

    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests_list,
    )


class TestKSTwoSample:
    """Tests for the KS two-sample implementation."""

    def test_identical_samples(self) -> None:
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        d, p = _ks_two_sample(s, s)
        assert d == 0.0
        assert p == 1.0

    def test_completely_separated(self) -> None:
        s1 = [1.0, 2.0, 3.0]
        s2 = [10.0, 11.0, 12.0]
        d, p = _ks_two_sample(s1, s2)
        assert d == 1.0
        assert p < 0.05

    def test_empty_sample(self) -> None:
        d, p = _ks_two_sample([], [1.0, 2.0])
        assert d == 0.0
        assert p == 1.0

    def test_similar_samples(self) -> None:
        rng = random.Random(42)
        s1 = [rng.gauss(10, 1) for _ in range(100)]
        s2 = [rng.gauss(10, 1) for _ in range(100)]
        d, p = _ks_two_sample(s1, s2)
        assert p > 0.05  # Should not reject null

    def test_different_distributions(self) -> None:
        rng = random.Random(42)
        s1 = [rng.gauss(10, 1) for _ in range(200)]
        s2 = [rng.gauss(20, 1) for _ in range(200)]
        d, p = _ks_two_sample(s1, s2)
        assert d > 0.5
        assert p < 0.01


class TestBisectRight:
    def test_basic(self) -> None:
        assert _bisect_right([1, 2, 3, 4, 5], 3) == 3

    def test_value_not_present(self) -> None:
        assert _bisect_right([1, 3, 5], 2) == 1

    def test_empty(self) -> None:
        assert _bisect_right([], 1) == 0


class TestClassifySeverity:
    def test_none(self) -> None:
        assert _classify_severity(0.1, 0.5) == DriftSeverity.NONE

    def test_minor(self) -> None:
        assert _classify_severity(0.15, 0.01) == DriftSeverity.MINOR

    def test_moderate(self) -> None:
        assert _classify_severity(0.25, 0.01) == DriftSeverity.MODERATE

    def test_major(self) -> None:
        assert _classify_severity(0.5, 0.001) == DriftSeverity.MAJOR


class TestDriftDetector:
    def test_no_drift(self) -> None:
        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)
        detector = DriftDetector()
        report = detector.detect(baseline, current)
        assert report.overall_severity == DriftSeverity.NONE
        assert len(report.drifted_metrics) == 0

    def test_major_drift(self) -> None:
        baseline = _make_benchmark(total_base=70.0, seed=42)
        current = _make_benchmark(total_base=200.0, seed=43)
        detector = DriftDetector()
        report = detector.detect(baseline, current)
        assert report.overall_severity in (DriftSeverity.MODERATE, DriftSeverity.MAJOR)
        assert "total_latency" in report.drifted_metrics

    def test_ttft_drift(self) -> None:
        baseline = _make_benchmark(ttft_base=20.0, seed=42)
        current = _make_benchmark(ttft_base=100.0, seed=43)
        detector = DriftDetector()
        report = detector.detect(baseline, current)
        assert "ttft" in report.drifted_metrics

    def test_three_metrics_reported(self) -> None:
        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)
        detector = DriftDetector()
        report = detector.detect(baseline, current)
        assert len(report.results) == 3
        metrics = {r.metric for r in report.results}
        assert metrics == {"ttft", "tpot", "total_latency"}

    def test_too_few_baseline_requests(self) -> None:
        baseline = _make_benchmark(num_requests=1)
        current = _make_benchmark(num_requests=100)
        detector = DriftDetector()
        with pytest.raises(ValueError, match="Baseline must have at least 2"):
            detector.detect(baseline, current)

    def test_too_few_current_requests(self) -> None:
        baseline = _make_benchmark(num_requests=100)
        current = _make_benchmark(num_requests=1)
        detector = DriftDetector()
        with pytest.raises(ValueError, match="Current must have at least 2"):
            detector.detect(baseline, current)

    def test_recommendation_no_drift(self) -> None:
        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)
        report = DriftDetector().detect(baseline, current)
        assert "no significant" in report.recommendation.lower()

    def test_recommendation_with_drift(self) -> None:
        baseline = _make_benchmark(total_base=70.0, seed=42)
        current = _make_benchmark(total_base=200.0, seed=43)
        report = DriftDetector().detect(baseline, current)
        assert "drift detected" in report.recommendation.lower()

    def test_mean_shift_positive(self) -> None:
        baseline = _make_benchmark(total_base=50.0, seed=42)
        current = _make_benchmark(total_base=150.0, seed=43)
        report = DriftDetector().detect(baseline, current)
        total_result = next(r for r in report.results if r.metric == "total_latency")
        assert total_result.mean_shift_ms > 0

    def test_mean_shift_negative(self) -> None:
        baseline = _make_benchmark(total_base=150.0, seed=42)
        current = _make_benchmark(total_base=50.0, seed=43)
        report = DriftDetector().detect(baseline, current)
        total_result = next(r for r in report.results if r.metric == "total_latency")
        assert total_result.mean_shift_ms < 0

    def test_ks_statistic_range(self) -> None:
        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)
        report = DriftDetector().detect(baseline, current)
        for r in report.results:
            assert 0.0 <= r.ks_statistic <= 1.0
            assert 0.0 <= r.p_value <= 1.0


class TestDetectDriftAPI:
    def test_returns_dict(self) -> None:
        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)
        result = detect_drift(baseline, current)
        assert isinstance(result, dict)
        assert "results" in result
        assert "overall_severity" in result
        assert "drifted_metrics" in result
        assert "recommendation" in result

    def test_drift_detected_in_api(self) -> None:
        baseline = _make_benchmark(total_base=70.0, seed=42)
        current = _make_benchmark(total_base=200.0, seed=43)
        result = detect_drift(baseline, current)
        assert result["overall_severity"] != "none"


class TestDriftCLI:
    def test_table_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)

        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as bf,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf,
        ):
            bf.write(baseline.model_dump_json(indent=2))
            bf.flush()
            cf.write(current.model_dump_json(indent=2))
            cf.flush()

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main(["drift", "--baseline", bf.name, "--current", cf.name])
                except SystemExit:
                    pass

    def test_json_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        baseline = _make_benchmark(seed=42)
        current = _make_benchmark(seed=43)

        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as bf,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf,
        ):
            bf.write(baseline.model_dump_json(indent=2))
            bf.flush()
            cf.write(current.model_dump_json(indent=2))
            cf.flush()

            buf = io.StringIO()
            with redirect_stdout(buf):
                try:
                    main([
                        "drift", "--baseline", bf.name,
                        "--current", cf.name,
                        "--output-format", "json",
                    ])
                except SystemExit:
                    pass

            output = buf.getvalue()
            if output.strip():
                result = json.loads(output)
                assert "results" in result
                assert "overall_severity" in result
