"""Tests for threshold_advisor module."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.threshold_advisor import (
    AdvisorReport,
    SweetSpot,
    ThresholdAdvisor,
    ThresholdSuggestion,
    advise_thresholds,
)


def _make_requests(n: int = 100, seed: int = 42) -> list[BenchmarkRequest]:
    """Generate synthetic benchmark requests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    requests = []
    for i in range(n):
        ttft = float(rng.lognormal(mean=3.0, sigma=0.5))
        tpot = float(rng.lognormal(mean=1.5, sigma=0.3))
        total = ttft + tpot * rng.integers(10, 200)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=int(rng.integers(10, 500)),
                output_tokens=int(rng.integers(10, 200)),
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1000.0 + i,
            )
        )
    return requests


def _make_data(n: int = 100, seed: int = 42) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=_make_requests(n, seed),
    )


class TestThresholdAdvisor:
    """Tests for ThresholdAdvisor class."""

    def test_basic_advise(self) -> None:
        data = _make_data(100)
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()

        assert isinstance(report, AdvisorReport)
        assert report.total_requests == 100
        # 3 metrics × 3 pass rates = 9 suggestions
        assert len(report.suggestions) == 9

    def test_suggestions_have_correct_metrics(self) -> None:
        data = _make_data(100)
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()

        metrics = {s.metric for s in report.suggestions}
        assert metrics == {"ttft_ms", "tpot_ms", "total_latency_ms"}

    def test_suggestions_increase_with_pass_rate(self) -> None:
        data = _make_data(200)
        advisor = ThresholdAdvisor(data, pass_rates=[0.50, 0.90, 0.99])
        report = advisor.advise()

        for metric in ThresholdAdvisor.METRICS:
            metric_suggestions = [
                s for s in report.suggestions if s.metric == metric
            ]
            thresholds = [s.threshold_ms for s in metric_suggestions]
            # Higher pass rate requires higher (looser) threshold
            assert thresholds == sorted(thresholds), (
                f"{metric}: thresholds should increase with pass rate"
            )

    def test_custom_pass_rates(self) -> None:
        data = _make_data(50)
        advisor = ThresholdAdvisor(data, pass_rates=[0.80, 0.95])
        report = advisor.advise()

        # 3 metrics × 2 pass rates = 6
        assert len(report.suggestions) == 6
        pass_rates = {s.pass_rate for s in report.suggestions}
        assert pass_rates == {0.80, 0.95}

    def test_small_data(self) -> None:
        """With very few requests, advisor still works."""
        data = _make_data(3, seed=1)
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()
        assert report.total_requests == 3
        assert len(report.suggestions) == 9

    def test_invalid_pass_rate_zero(self) -> None:
        data = _make_data(10)
        with pytest.raises(ValueError, match="pass_rate must be in"):
            ThresholdAdvisor(data, pass_rates=[0.0])

    def test_invalid_pass_rate_negative(self) -> None:
        data = _make_data(10)
        with pytest.raises(ValueError, match="pass_rate must be in"):
            ThresholdAdvisor(data, pass_rates=[-0.1])

    def test_suggestion_model_fields(self) -> None:
        data = _make_data(50)
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()

        for s in report.suggestions:
            assert isinstance(s, ThresholdSuggestion)
            assert s.threshold_ms >= 0
            assert 0 < s.pass_rate <= 1.0
            assert s.metric in ThresholdAdvisor.METRICS

    def test_sweet_spot_model(self) -> None:
        """Sweet spots, if found, have valid structure."""
        data = _make_data(500, seed=7)
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()

        for spot in report.sweet_spots:
            assert isinstance(spot, SweetSpot)
            assert spot.gain > 0
            assert spot.pass_rate_above > spot.pass_rate_below
            assert spot.threshold_ms >= 0

    def test_single_request(self) -> None:
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="r1",
                    prompt_tokens=10,
                    output_tokens=10,
                    ttft_ms=50.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=1000.0,
                )
            ],
        )
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()
        assert report.total_requests == 1
        assert len(report.suggestions) == 9

    def test_uniform_latencies(self) -> None:
        """All identical latencies should produce same threshold for all pass rates."""
        requests = [
            BenchmarkRequest(
                request_id=f"r{i}",
                prompt_tokens=10,
                output_tokens=10,
                ttft_ms=50.0,
                tpot_ms=5.0,
                total_latency_ms=100.0,
                timestamp=1000.0 + i,
            )
            for i in range(100)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10.0,
            ),
            requests=requests,
        )
        advisor = ThresholdAdvisor(data)
        report = advisor.advise()

        for metric in ThresholdAdvisor.METRICS:
            thresholds = [
                s.threshold_ms for s in report.suggestions if s.metric == metric
            ]
            # All thresholds should be equal (same value)
            assert len(set(thresholds)) == 1

    def test_large_dataset(self) -> None:
        data = _make_data(1000, seed=99)
        advisor = ThresholdAdvisor(data, pass_rates=[0.90, 0.95, 0.99])
        report = advisor.advise()
        assert report.total_requests == 1000
        assert len(report.suggestions) == 9


class TestAdviseThresholds:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        data = _make_data(50)
        result = advise_thresholds(data)
        assert isinstance(result, dict)
        assert "total_requests" in result
        assert "suggestions" in result
        assert "sweet_spots" in result

    def test_custom_pass_rates(self) -> None:
        data = _make_data(50)
        result = advise_thresholds(data, pass_rates=[0.80])
        assert len(result["suggestions"]) == 3  # 3 metrics × 1 pass rate

    def test_json_serializable(self) -> None:
        data = _make_data(50)
        result = advise_thresholds(data)
        # Should not raise
        json.dumps(result)

    def test_result_structure(self) -> None:
        data = _make_data(100)
        result = advise_thresholds(data)
        for s in result["suggestions"]:
            assert "metric" in s
            assert "pass_rate" in s
            assert "threshold_ms" in s

    def test_pass_rate_1_0(self) -> None:
        """Pass rate 1.0 should return the max value."""
        data = _make_data(100)
        result = advise_thresholds(data, pass_rates=[1.0])
        assert len(result["suggestions"]) == 3
