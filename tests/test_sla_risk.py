"""Tests for SLA Risk Score (M118)."""

from __future__ import annotations

import os
import tempfile

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.sla_risk import (
    RiskFactor,
    RiskLevel,
    SLARiskReport,
    SLARiskScorer,
    _burn_rate_score,
    _classify_risk,
    _convergence_score,
    _headroom_score,
    _jitter_score,
    _tail_score,
    assess_sla_risk,
)


def _make_data(
    n: int = 200,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 0.5,
) -> BenchmarkData:
    """Generate benchmark data with controllable latency distributions."""
    requests = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=ttft_base + frac * spread * ttft_base,
                tpot_ms=tpot_base + frac * spread * tpot_base,
                total_latency_ms=total_base + frac * spread * total_base,
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


def _make_high_risk_data(n: int = 200) -> BenchmarkData:
    """Generate data that triggers high risk scores."""
    import random
    random.seed(42)
    requests = []
    for i in range(n):
        # High jitter, heavy tails, some violations
        base = 50.0
        if i % 10 == 0:
            # Spike every 10th request
            mult = random.uniform(5.0, 15.0)
        else:
            mult = random.uniform(0.5, 2.0)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=base * mult,
                tpot_ms=10.0 * mult,
                total_latency_ms=200.0 * mult,
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


class TestClassifyRisk:
    def test_low(self):
        assert _classify_risk(0.0) == RiskLevel.LOW
        assert _classify_risk(24.9) == RiskLevel.LOW

    def test_moderate(self):
        assert _classify_risk(25.0) == RiskLevel.MODERATE
        assert _classify_risk(49.9) == RiskLevel.MODERATE

    def test_high(self):
        assert _classify_risk(50.0) == RiskLevel.HIGH
        assert _classify_risk(74.9) == RiskLevel.HIGH

    def test_critical(self):
        assert _classify_risk(75.0) == RiskLevel.CRITICAL
        assert _classify_risk(100.0) == RiskLevel.CRITICAL


class TestHeadroomScore:
    def test_comfortable_headroom(self):
        values = [10.0] * 100
        score = _headroom_score(values, 100.0)
        assert score == 0.0  # 90% headroom

    def test_no_headroom(self):
        values = [100.0] * 100
        score = _headroom_score(values, 50.0)
        assert score == 100.0  # violated

    def test_tight_headroom(self):
        values = list(range(1, 101))  # P95 ≈ 95.05
        score = _headroom_score(values, 100.0)
        assert score > 0  # some risk

    def test_zero_threshold(self):
        assert _headroom_score([1.0], 0.0) == 0.0


class TestTailScore:
    def test_light_tail(self):
        # All same → ratio = 1
        values = [10.0] * 100
        assert _tail_score(values) == 0.0

    def test_heavy_tail(self):
        # 99 values at 10, 1 at 200 → P99 ≈ 200, P50 = 10 → ratio 20
        values = [10.0] * 90 + [200.0] * 10
        score = _tail_score(values)
        assert score > 0

    def test_single_value(self):
        assert _tail_score([10.0]) == 0.0


class TestJitterScore:
    def test_stable(self):
        values = [10.0] * 100
        assert _jitter_score(values) == 0.0

    def test_high_jitter(self):
        import random
        random.seed(123)
        values = [random.uniform(1, 100) for _ in range(200)]
        score = _jitter_score(values)
        assert score > 0

    def test_single_value(self):
        assert _jitter_score([10.0]) == 0.0


class TestConvergenceScore:
    def test_well_converged(self):
        values = [10.0 + i * 0.001 for i in range(1000)]
        score = _convergence_score(values)
        assert score < 50

    def test_too_few_samples(self):
        assert _convergence_score([1.0] * 5) == 100.0


class TestBurnRateScore:
    def test_no_violations(self):
        data = _make_data(100, spread=0.1)
        score = _burn_rate_score(data, sla_ttft_ms=1000.0, sla_tpot_ms=None, sla_total_ms=None)
        assert score == 0.0

    def test_all_violations(self):
        data = _make_data(100, ttft_base=100.0)
        score = _burn_rate_score(data, sla_ttft_ms=1.0, sla_tpot_ms=None, sla_total_ms=None)
        assert score == 100.0

    def test_no_thresholds(self):
        data = _make_data(100)
        score = _burn_rate_score(data, None, None, None)
        assert score == 0.0


class TestSLARiskScorer:
    def test_low_risk(self):
        data = _make_data(200, spread=0.1)  # tight distribution
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0, sla_tpot_ms=50.0, sla_total_ms=1000.0)
        assert isinstance(report, SLARiskReport)
        assert report.risk_score.risk_level == RiskLevel.LOW
        assert report.risk_score.total_score < 25
        assert len(report.factors) == 5
        assert report.total_requests == 200

    def test_high_risk(self):
        data = _make_high_risk_data(200)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=100.0, sla_tpot_ms=20.0, sla_total_ms=400.0)
        assert report.risk_score.total_score > 25  # at least moderate

    def test_no_sla_thresholds(self):
        data = _make_data(100)
        scorer = SLARiskScorer()
        report = scorer.assess(data)
        # headroom and burn rate should be 0 without thresholds
        headroom_factor = next(f for f in report.factors if f.name == "headroom_tightness")
        burn_factor = next(f for f in report.factors if f.name == "error_budget_burn_rate")
        assert headroom_factor.score == 0.0
        assert burn_factor.score == 0.0

    def test_custom_weights(self):
        data = _make_data(200, spread=0.5)
        scorer = SLARiskScorer()
        report = scorer.assess(
            data,
            sla_ttft_ms=100.0,
            weight_headroom=0.50,
            weight_tail=0.10,
            weight_jitter=0.10,
            weight_convergence=0.15,
            weight_burn_rate=0.15,
        )
        headroom_factor = next(f for f in report.factors if f.name == "headroom_tightness")
        assert headroom_factor.weight == 0.50

    def test_factors_have_required_fields(self):
        data = _make_data(100)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0)
        for f in report.factors:
            assert isinstance(f, RiskFactor)
            assert f.name
            assert 0.0 <= f.score <= 100.0
            assert f.weight > 0
            assert f.detail

    def test_recommendation_present(self):
        data = _make_data(100)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0)
        assert report.recommendation
        assert isinstance(report.recommendation, str)

    def test_critical_recommendation(self):
        data = _make_high_risk_data(200)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=50.0, sla_tpot_ms=10.0, sla_total_ms=200.0)
        if report.risk_score.risk_level == RiskLevel.CRITICAL:
            assert "CRITICAL" in report.recommendation

    def test_model_dump(self):
        data = _make_data(100)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0)
        d = report.model_dump()
        assert "risk_score" in d
        assert "factors" in d
        assert d["risk_score"]["risk_level"] == report.risk_score.risk_level.value

    def test_score_bounded(self):
        data = _make_data(100)
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0)
        assert 0.0 <= report.risk_score.total_score <= 100.0


class TestAssessSLARiskAPI:
    def test_programmatic_api(self):
        data = _make_data(100)
        # Write benchmark to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            f.write(data.model_dump_json())
            path = f.name

        try:
            result = assess_sla_risk(
                path,
                sla_ttft_ms=200.0,
                sla_tpot_ms=50.0,
                sla_total_ms=1000.0,
            )
            assert isinstance(result, dict)
            assert "risk_score" in result
            assert "factors" in result
            assert result["risk_score"]["risk_level"] in ["low", "moderate", "high", "critical"]
        finally:
            os.unlink(path)


class TestEdgeCases:
    def test_single_request(self):
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="req-0000",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=50.0,
                    tpot_ms=10.0,
                    total_latency_ms=200.0,
                    timestamp=1000.0,
                ),
            ],
        )
        scorer = SLARiskScorer()
        report = scorer.assess(data, sla_ttft_ms=200.0)
        assert report.total_requests == 1
