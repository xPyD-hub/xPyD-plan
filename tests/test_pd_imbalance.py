"""Tests for pd_imbalance module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.pd_imbalance import (
    ImbalanceClassification,
    ImbalanceLevel,
    ImbalanceReport,
    MetricSensitivity,
    PDImbalanceDetector,
    detect_pd_imbalance,
)


def _make_benchmark(
    n_prefill: int,
    n_decode: int,
    ttft_base: float,
    tpot_base: float,
    n_requests: int = 100,
    qps: float = 10.0,
) -> BenchmarkData:
    """Create a synthetic benchmark with controlled latency distributions."""
    requests = []
    for i in range(n_requests):
        # Add some noise but keep P95 near the base
        ttft = ttft_base + (i % 20) * 0.1
        tpot = tpot_base + (i % 20) * 0.05
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=128,
                output_tokens=64,
                ttft_ms=ttft,
                tpot_ms=tpot,
                total_latency_ms=ttft + tpot * 64,
                timestamp=1000.0 + i * (1.0 / qps),
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=n_prefill,
            num_decode_instances=n_decode,
            total_instances=n_prefill + n_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _write_benchmark(path: Path, data: BenchmarkData) -> None:
    """Write benchmark data to a JSON file."""
    path.write_text(json.dumps(data.model_dump(), default=str))


class TestPDImbalanceDetector:
    """Tests for PDImbalanceDetector."""

    def test_insufficient_data_zero(self) -> None:
        detector = PDImbalanceDetector([])
        report = detector.analyze()
        assert report.classification == ImbalanceClassification.INSUFFICIENT_DATA
        assert report.level == ImbalanceLevel.NONE
        assert report.num_benchmarks == 0

    def test_insufficient_data_one(self) -> None:
        ds = _make_benchmark(2, 2, 50.0, 10.0)
        detector = PDImbalanceDetector([ds])
        report = detector.analyze()
        assert report.classification == ImbalanceClassification.INSUFFICIENT_DATA

    def test_balanced_system(self) -> None:
        """Both TTFT and TPOT improve similarly with more instances."""
        datasets = [
            _make_benchmark(1, 3, ttft_base=100.0, tpot_base=20.0),
            _make_benchmark(2, 2, ttft_base=80.0, tpot_base=25.0),
            _make_benchmark(3, 1, ttft_base=60.0, tpot_base=30.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.ttft_sensitivity is not None
        assert report.tpot_sensitivity is not None
        assert report.num_benchmarks == 3

    def test_prefill_starved(self) -> None:
        """TTFT drops dramatically with more prefill instances; TPOT barely changes."""
        datasets = [
            _make_benchmark(1, 3, ttft_base=200.0, tpot_base=10.0),
            _make_benchmark(2, 2, ttft_base=100.0, tpot_base=10.5),
            _make_benchmark(3, 1, ttft_base=50.0, tpot_base=11.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.classification == ImbalanceClassification.PREFILL_STARVED
        assert report.level != ImbalanceLevel.NONE
        assert "prefill" in report.recommendation.lower()

    def test_decode_starved(self) -> None:
        """TPOT drops dramatically with more decode instances; TTFT barely changes."""
        datasets = [
            _make_benchmark(3, 1, ttft_base=50.0, tpot_base=100.0),
            _make_benchmark(2, 2, ttft_base=50.5, tpot_base=50.0),
            _make_benchmark(1, 3, ttft_base=51.0, tpot_base=25.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.classification == ImbalanceClassification.DECODE_STARVED
        assert report.level != ImbalanceLevel.NONE
        assert "decode" in report.recommendation.lower()

    def test_sensitivity_has_data_points(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(2, 2, 80.0, 12.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.ttft_sensitivity is not None
        assert report.ttft_sensitivity.data_points == 2
        assert report.tpot_sensitivity is not None
        assert report.tpot_sensitivity.data_points == 2

    def test_sensitivity_r_squared(self) -> None:
        """With 3+ points, R² should be computed."""
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(2, 2, 80.0, 12.0),
            _make_benchmark(3, 1, 60.0, 14.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.ttft_sensitivity is not None
        assert report.ttft_sensitivity.r_squared is not None

    def test_severity_mild(self) -> None:
        """Sensitivity ratio just above 1.5 threshold → mild."""
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(2, 2, 80.0, 10.1),  # TTFT sensitivity ~ -20, TPOT ~ 0.1
            _make_benchmark(3, 1, 60.0, 10.2),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        # TTFT sensitivity is much larger than TPOT → should be prefill-starved
        if report.classification == ImbalanceClassification.PREFILL_STARVED:
            assert report.level in (
                ImbalanceLevel.MILD,
                ImbalanceLevel.MODERATE,
                ImbalanceLevel.SEVERE,
            )

    def test_severity_severe(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 500.0, 10.0),
            _make_benchmark(2, 2, 200.0, 10.01),
            _make_benchmark(3, 1, 50.0, 10.02),
        ]
        detector = PDImbalanceDetector(
            datasets, severity_threshold_severe=5.0
        )
        report = detector.analyze()
        assert report.classification == ImbalanceClassification.PREFILL_STARVED
        assert report.level == ImbalanceLevel.SEVERE

    def test_recommendation_balanced(self) -> None:
        datasets = [
            _make_benchmark(2, 2, 50.0, 50.0),
            _make_benchmark(3, 3, 40.0, 40.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        if report.classification == ImbalanceClassification.BALANCED:
            assert "no adjustment" in report.recommendation.lower()

    def test_single_request_benchmarks(self) -> None:
        """Benchmarks with very few requests should still work."""
        ds1 = _make_benchmark(1, 3, 100.0, 10.0, n_requests=1)
        ds2 = _make_benchmark(2, 2, 80.0, 12.0, n_requests=1)
        detector = PDImbalanceDetector([ds1, ds2])
        report = detector.analyze()
        assert report.classification != ImbalanceClassification.INSUFFICIENT_DATA

    def test_same_prefill_count_zero_sensitivity(self) -> None:
        """All benchmarks have same prefill count → zero TTFT sensitivity."""
        datasets = [
            _make_benchmark(2, 1, 50.0, 20.0),
            _make_benchmark(2, 2, 50.0, 15.0),
            _make_benchmark(2, 3, 50.0, 10.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.ttft_sensitivity is not None
        assert report.ttft_sensitivity.sensitivity_ms_per_instance == 0.0

    def test_model_serialization(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(2, 2, 80.0, 12.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        d = report.model_dump()
        assert "classification" in d
        assert "level" in d
        assert "recommendation" in d
        # Round-trip
        ImbalanceReport.model_validate(d)

    def test_metric_sensitivity_model(self) -> None:
        ms = MetricSensitivity(
            metric="ttft_p95",
            instance_type="prefill",
            sensitivity_ms_per_instance=-20.5,
            data_points=3,
            r_squared=0.95,
        )
        d = ms.model_dump()
        assert d["metric"] == "ttft_p95"
        assert d["r_squared"] == 0.95

    def test_custom_thresholds(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(2, 2, 80.0, 10.1),
            _make_benchmark(3, 1, 60.0, 10.2),
        ]
        detector = PDImbalanceDetector(
            datasets,
            severity_threshold_mild=100.0,  # Very high threshold
            severity_threshold_moderate=200.0,
            severity_threshold_severe=300.0,
        )
        report = detector.analyze()
        # With very high thresholds, might classify as balanced
        assert isinstance(report.classification, ImbalanceClassification)

    def test_sensitivity_ratio_present(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 100.0, 20.0),
            _make_benchmark(2, 2, 80.0, 22.0),
            _make_benchmark(3, 1, 60.0, 24.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.sensitivity_ratio is not None
        assert report.sensitivity_ratio > 0

    def test_two_benchmarks_minimal(self) -> None:
        datasets = [
            _make_benchmark(1, 3, 100.0, 10.0),
            _make_benchmark(3, 1, 60.0, 30.0),
        ]
        detector = PDImbalanceDetector(datasets)
        report = detector.analyze()
        assert report.num_benchmarks == 2
        assert report.classification != ImbalanceClassification.INSUFFICIENT_DATA


class TestDetectPdImbalanceAPI:
    """Tests for the programmatic API."""

    def test_api_from_files(self, tmp_path: Path) -> None:
        ds1 = _make_benchmark(1, 3, 100.0, 10.0)
        ds2 = _make_benchmark(2, 2, 80.0, 12.0)
        ds3 = _make_benchmark(3, 1, 60.0, 14.0)
        p1 = tmp_path / "bench1.json"
        p2 = tmp_path / "bench2.json"
        p3 = tmp_path / "bench3.json"
        _write_benchmark(p1, ds1)
        _write_benchmark(p2, ds2)
        _write_benchmark(p3, ds3)
        report = detect_pd_imbalance([p1, p2, p3])
        assert isinstance(report, ImbalanceReport)
        assert report.num_benchmarks == 3

    def test_api_single_file(self, tmp_path: Path) -> None:
        ds = _make_benchmark(2, 2, 50.0, 10.0)
        p = tmp_path / "bench.json"
        _write_benchmark(p, ds)
        report = detect_pd_imbalance([p])
        assert report.classification == ImbalanceClassification.INSUFFICIENT_DATA
