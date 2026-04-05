"""Tests for latency anomaly classifier (M100)."""

from __future__ import annotations

import argparse
import json
import tempfile

import pytest

from xpyd_plan.anomaly_classifier import (
    AnomalyClass,
    AnomalyReport,
    LatencyAnomalyClassifier,
    _classify_value,
    _percentile,
    classify_anomalies,
)
from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli._anomaly_classify import _cmd_anomaly_classify


def _make_data(
    n: int = 100,
    ttft_base: float = 100.0,
    ttft_spread: float = 5.0,
) -> BenchmarkData:
    """Generate benchmark data with controlled spread."""
    import random

    rng = random.Random(42)
    requests = []
    for i in range(n):
        ttft = ttft_base + rng.uniform(-ttft_spread, ttft_spread)
        tpot = 20.0 + rng.uniform(-2.0, 2.0)
        total = ttft + tpot * 50
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


def _make_data_with_outliers(n: int = 100) -> BenchmarkData:
    """Generate data with some extreme outliers."""
    import random

    rng = random.Random(99)
    requests = []
    for i in range(n):
        if i >= n - 5:
            # Last 5 are extreme outliers
            ttft = 1000.0 + rng.uniform(0, 200)
        else:
            ttft = 100.0 + rng.uniform(-5.0, 5.0)
        tpot = 20.0 + rng.uniform(-2.0, 2.0)
        total = ttft + tpot * 50
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestPercentile:
    """Test _percentile helper."""

    def test_single_value(self):
        assert _percentile([5.0], 50.0) == 5.0

    def test_known_percentile(self):
        values = list(range(1, 101))  # 1..100
        assert _percentile(values, 50.0) == pytest.approx(50.5, abs=0.1)

    def test_empty_returns_zero(self):
        assert _percentile([], 50.0) == 0.0


class TestClassifyValue:
    """Test _classify_value helper."""

    def test_normal(self):
        assert _classify_value(50.0, 75.0, 20.0, 1.0, 3.0, None) == AnomalyClass.NORMAL

    def test_slow(self):
        # q75=75, iqr=20, slow threshold = 75 + 1.0*20 = 95
        assert _classify_value(96.0, 75.0, 20.0, 1.0, 3.0, None) == AnomalyClass.SLOW

    def test_outlier(self):
        # q75=75, iqr=20, outlier threshold = 75 + 3.0*20 = 135
        assert _classify_value(140.0, 75.0, 20.0, 1.0, 3.0, None) == AnomalyClass.OUTLIER

    def test_timeout_overrides(self):
        # Even if below outlier threshold, timeout wins
        assert _classify_value(100.0, 75.0, 20.0, 1.0, 3.0, 90.0) == AnomalyClass.TIMEOUT

    def test_zero_iqr_all_normal(self):
        # When IQR is 0, only timeout can trigger
        assert _classify_value(100.0, 75.0, 0.0, 1.0, 3.0, None) == AnomalyClass.NORMAL

    def test_zero_iqr_with_timeout(self):
        assert _classify_value(100.0, 75.0, 0.0, 1.0, 3.0, 90.0) == AnomalyClass.TIMEOUT


class TestLatencyAnomalyClassifier:
    """Test LatencyAnomalyClassifier."""

    def test_basic_classification(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)

        assert isinstance(report, AnomalyReport)
        assert len(report.labels) == len(data.requests)
        assert len(report.distributions) == 3

    def test_all_normal_tight_data(self):
        """Very tight data should have mostly NORMAL labels."""
        data = _make_data(n=100, ttft_spread=0.1)
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)

        # With very tight data, most should be normal
        normal_count = sum(
            1 for lbl in report.labels if lbl.worst_class == AnomalyClass.NORMAL
        )
        assert normal_count >= 50  # Most should be normal

    def test_outliers_detected(self):
        data = _make_data_with_outliers()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)

        # The last 5 requests should be outliers or worse on TTFT
        outlier_ids = {
            lbl.request_id
            for lbl in report.labels
            if lbl.ttft_class in (AnomalyClass.OUTLIER, AnomalyClass.TIMEOUT)
        }
        for i in range(95, 100):
            assert f"req-{i}" in outlier_ids

    def test_timeout_detection(self):
        data = _make_data_with_outliers()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data, timeout_ttft=500.0)

        timeout_count = sum(
            1 for lbl in report.labels if lbl.ttft_class == AnomalyClass.TIMEOUT
        )
        assert timeout_count >= 5  # At least the outlier requests

    def test_custom_multipliers(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()

        # Very tight multipliers should flag more anomalies
        report_tight = classifier.classify(data, slow_multiplier=0.1, outlier_multiplier=0.5)
        report_loose = classifier.classify(data, slow_multiplier=5.0, outlier_multiplier=10.0)

        tight_anomalies = sum(
            1 for lbl in report_tight.labels if lbl.worst_class != AnomalyClass.NORMAL
        )
        loose_anomalies = sum(
            1 for lbl in report_loose.labels if lbl.worst_class != AnomalyClass.NORMAL
        )
        assert tight_anomalies >= loose_anomalies

    def test_invalid_multipliers(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()

        with pytest.raises(ValueError, match="non-negative"):
            classifier.classify(data, slow_multiplier=-1.0)

        with pytest.raises(ValueError, match="<="):
            classifier.classify(data, slow_multiplier=5.0, outlier_multiplier=3.0)

    def test_distribution_sums_to_total(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)

        for d in report.distributions:
            assert d.normal_count + d.slow_count + d.outlier_count + d.timeout_count == d.total

    def test_anomaly_rate(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)

        assert 0 <= report.anomaly_rate <= 100
        expected = 100.0 * report.total_anomalous / len(report.labels)
        assert report.anomaly_rate == pytest.approx(expected, abs=0.01)

    def test_worst_class_is_max(self):
        data = _make_data_with_outliers()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data, timeout_ttft=500.0)

        for label in report.labels:
            classes = [label.ttft_class, label.tpot_class, label.total_class]
            severity_order = {
                AnomalyClass.NORMAL: 0,
                AnomalyClass.SLOW: 1,
                AnomalyClass.OUTLIER: 2,
                AnomalyClass.TIMEOUT: 3,
            }
            max_severity = max(classes, key=lambda c: severity_order[c])
            assert label.worst_class == max_severity

    def test_recommendation_high(self):
        data = _make_data_with_outliers()
        classifier = LatencyAnomalyClassifier()
        # Very tight multipliers to trigger high anomaly rate
        report = classifier.classify(data, slow_multiplier=0.01, outlier_multiplier=0.05)
        # The recommendation should exist
        assert len(report.recommendation) > 0

    def test_recommendation_low(self):
        data = _make_data(n=100, ttft_spread=0.01)
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data, slow_multiplier=5.0, outlier_multiplier=10.0)
        assert "low" in report.recommendation.lower() or "well" in report.recommendation.lower()


class TestClassifyAnomaliesAPI:
    """Test the programmatic API."""

    def test_returns_dict(self):
        data = _make_data()
        result = classify_anomalies(data)
        assert isinstance(result, dict)
        assert "labels" in result
        assert "distributions" in result
        assert "anomaly_rate" in result

    def test_with_timeout_params(self):
        data = _make_data()
        result = classify_anomalies(
            data,
            timeout_ttft=90.0,
            timeout_tpot=15.0,
            timeout_total=800.0,
        )
        assert isinstance(result, dict)

    def test_json_serializable(self):
        data = _make_data()
        result = classify_anomalies(data)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


class TestAnomalyModels:
    """Test Pydantic model serialization."""

    def test_report_serializable(self):
        data = _make_data()
        classifier = LatencyAnomalyClassifier()
        report = classifier.classify(data)
        d = report.model_dump()
        assert isinstance(d, dict)
        report2 = AnomalyReport.model_validate(d)
        assert report2.anomaly_rate == report.anomaly_rate

    def test_anomaly_class_values(self):
        assert AnomalyClass.NORMAL.value == "normal"
        assert AnomalyClass.SLOW.value == "slow"
        assert AnomalyClass.OUTLIER.value == "outlier"
        assert AnomalyClass.TIMEOUT.value == "timeout"


class TestAnomalyClassifyCLI:
    """Test CLI integration."""

    def test_cli_table_output(self):
        data = _make_data()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data.model_dump(), f)
            f.flush()



            args = argparse.Namespace(
                benchmark=f.name,
                slow_multiplier=1.0,
                outlier_multiplier=3.0,
                timeout_ttft=None,
                timeout_tpot=None,
                timeout_total=None,
                output_format="table",
            )
            _cmd_anomaly_classify(args)  # Should not raise

    def test_cli_json_output(self, capsys):
        data = _make_data()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data.model_dump(), f)
            f.flush()



            args = argparse.Namespace(
                benchmark=f.name,
                slow_multiplier=1.0,
                outlier_multiplier=3.0,
                timeout_ttft=None,
                timeout_tpot=None,
                timeout_total=None,
                output_format="json",
            )
            _cmd_anomaly_classify(args)
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert "labels" in parsed
            assert "distributions" in parsed
