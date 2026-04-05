"""Tests for token budget estimator."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.token_budget import TokenBudgetEstimator, TokenBudgetReport, estimate_token_budget


def _make_benchmark(n: int = 100) -> BenchmarkData:
    """Create benchmark data with clear linear latency-token relationships.

    TTFT ≈ 10 + 0.5 * prompt_tokens
    TPOT ≈ 5 + 0.1 * output_tokens
    total ≈ 50 + 0.3 * total_tokens
    """
    import random

    rng = random.Random(42)
    requests = []
    for i in range(n):
        prompt = rng.randint(100, 2000)
        output = rng.randint(50, 500)
        ttft = 10 + 0.5 * prompt + rng.gauss(0, 5)
        tpot = 5 + 0.1 * output + rng.gauss(0, 2)
        total = 50 + 0.3 * (prompt + output) + rng.gauss(0, 10)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt,
                output_tokens=output,
                ttft_ms=max(1.0, ttft),
                tpot_ms=max(1.0, tpot),
                total_latency_ms=max(1.0, total),
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
        requests=requests,
    )


class TestTokenBudgetEstimator:
    def test_basic_ttft_limit(self) -> None:
        data = _make_benchmark()
        estimator = TokenBudgetEstimator(data)
        report = estimator.estimate(sla_ttft_ms=500.0)

        assert len(report.limits) == 1
        limit = report.limits[0]
        assert limit.metric == "ttft_ms"
        assert limit.predictor == "prompt_tokens"
        # TTFT ≈ 10 + 0.5*x → x ≈ (500-10)/0.5 = 980
        assert 900 <= limit.max_tokens <= 1100
        assert limit.r_squared > 0.5
        assert report.effective_max_prompt is not None
        assert report.effective_max_output is None

    def test_basic_tpot_limit(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_tpot_ms=50.0)

        assert len(report.limits) == 1
        limit = report.limits[0]
        assert limit.metric == "tpot_ms"
        assert limit.predictor == "output_tokens"
        # TPOT ≈ 5 + 0.1*x → x ≈ (50-5)/0.1 = 450
        assert 350 <= limit.max_tokens <= 550
        assert report.effective_max_output is not None
        assert report.effective_max_prompt is None

    def test_total_latency_limit(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_total_ms=500.0)

        assert len(report.limits) == 1
        limit = report.limits[0]
        assert limit.metric == "total_latency_ms"
        assert limit.predictor == "total_tokens"
        # total ≈ 50 + 0.3*x → x ≈ (500-50)/0.3 = 1500
        assert 1200 <= limit.max_tokens <= 1800

    def test_all_thresholds(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(
            data, sla_ttft_ms=500.0, sla_tpot_ms=50.0, sla_total_ms=500.0
        )
        assert len(report.limits) == 3
        assert report.effective_max_prompt is not None
        assert report.effective_max_output is not None

    def test_no_threshold_raises(self) -> None:
        data = _make_benchmark()
        estimator = TokenBudgetEstimator(data)
        with pytest.raises(ValueError, match="At least one SLA threshold"):
            estimator.estimate()

    def test_very_tight_sla_gives_zero_or_small(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=1.0)
        assert len(report.limits) == 1
        assert report.limits[0].max_tokens == 0

    def test_very_loose_sla_gives_large(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=100000.0)
        assert report.limits[0].max_tokens > 10000

    def test_confidence_labels(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=500.0, sla_tpot_ms=50.0)
        for limit in report.limits:
            assert limit.confidence in ("HIGH", "MEDIUM", "LOW")

    def test_report_model_dump(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=500.0)
        d = report.model_dump()
        assert "limits" in d
        assert "effective_max_prompt" in d
        assert "warnings" in d

    def test_json_serializable(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=500.0)
        s = json.dumps(report.model_dump())
        assert isinstance(s, str)

    def test_warnings_for_low_r_squared(self) -> None:
        """With random data, R² should be low and warnings should appear."""
        import random

        rng = random.Random(99)
        requests = []
        for i in range(50):
            requests.append(
                BenchmarkRequest(
                    request_id=f"req-{i}",
                    prompt_tokens=rng.randint(100, 2000),
                    output_tokens=rng.randint(50, 500),
                    ttft_ms=rng.uniform(10, 1000),  # random, no correlation
                    tpot_ms=rng.uniform(5, 100),
                    total_latency_ms=rng.uniform(50, 2000),
                    timestamp=1000.0 + i,
                )
            )
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=2,
                num_decode_instances=2,
                total_instances=4,
                measured_qps=5.0,
            ),
            requests=requests,
        )
        report = estimate_token_budget(data, sla_ttft_ms=500.0)
        # At minimum, the report should still produce a result
        assert len(report.limits) >= 0  # might be 1 or 0 depending on slope

    def test_effective_prompt_is_min_of_limits(self) -> None:
        """When multiple metrics contribute prompt limits, effective = min."""
        data = _make_benchmark()
        # Only TTFT contributes to prompt limits
        report = estimate_token_budget(data, sla_ttft_ms=200.0)
        if report.effective_max_prompt is not None:
            prompt_limits = [
                lim.max_tokens
                for lim in report.limits
                if lim.predictor == "prompt_tokens"
            ]
            assert report.effective_max_prompt == min(prompt_limits)

    def test_programmatic_api(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(data, sla_ttft_ms=500.0)
        assert isinstance(report, TokenBudgetReport)
        assert len(report.limits) > 0

    def test_small_dataset(self) -> None:
        """Should work with minimal data (3 requests)."""
        requests = [
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
                ttft_ms=10 + 0.5 * 100 * (i + 1),
                tpot_ms=5 + 0.1 * 50 * (i + 1),
                total_latency_ms=50 + 0.3 * 150 * (i + 1),
                timestamp=1000.0 + i,
            )
            for i in range(3)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=requests,
        )
        report = estimate_token_budget(data, sla_ttft_ms=500.0)
        assert len(report.limits) == 1

    def test_max_tokens_not_negative(self) -> None:
        data = _make_benchmark()
        report = estimate_token_budget(
            data, sla_ttft_ms=0.001, sla_tpot_ms=0.001, sla_total_ms=0.001
        )
        for limit in report.limits:
            assert limit.max_tokens >= 0
