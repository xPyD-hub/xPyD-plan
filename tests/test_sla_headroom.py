"""Tests for SLA headroom calculator."""

from __future__ import annotations

import json

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.sla_headroom import (
    HeadroomReport,
    MetricHeadroom,
    SafetyLevel,
    SLAHeadroomCalculator,
    analyze_sla_headroom,
)


def _make_data(
    n: int = 100,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 1.0,
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


class TestSLAHeadroomCalculator:
    """Tests for SLAHeadroomCalculator."""

    def test_no_thresholds(self) -> None:
        data = _make_data()
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data)
        assert report.all_pass is True
        assert report.metrics == []
        assert report.tightest_metric is None

    def test_comfortable_headroom(self) -> None:
        data = _make_data(n=100, ttft_base=50.0, spread=0.5)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=200.0, percentile=95.0)
        assert len(report.metrics) == 1
        m = report.metrics[0]
        assert m.metric == "ttft"
        assert m.passes_sla is True
        assert m.headroom_pct > 30.0
        assert m.safety_level == SafetyLevel.COMFORTABLE
        assert report.all_pass is True

    def test_tight_headroom(self) -> None:
        # Actual P95 will be near ttft_base * (1 + 0.95*spread) = 50*(1+0.95) = 97.5
        data = _make_data(n=100, ttft_base=50.0, spread=1.0)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=100.0, percentile=95.0)
        m = report.metrics[0]
        assert m.passes_sla is True
        assert m.safety_level == SafetyLevel.TIGHT
        assert m.headroom_pct < 10.0

    def test_sla_violation(self) -> None:
        data = _make_data(n=100, ttft_base=50.0, spread=2.0)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=80.0, percentile=95.0)
        m = report.metrics[0]
        assert m.passes_sla is False
        assert m.safety_level == SafetyLevel.CRITICAL
        assert m.headroom_ms < 0
        assert report.all_pass is False
        assert "VIOLATION" in report.recommendation

    def test_multiple_metrics(self) -> None:
        data = _make_data(n=100, ttft_base=50.0, tpot_base=10.0, total_base=200.0, spread=0.5)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(
            data,
            sla_ttft_ms=200.0,
            sla_tpot_ms=50.0,
            sla_total_ms=500.0,
            percentile=95.0,
        )
        assert len(report.metrics) == 3
        assert report.all_pass is True
        assert report.tightest_metric is not None

    def test_tightest_metric_identified(self) -> None:
        data = _make_data(n=100, spread=1.0)
        calc = SLAHeadroomCalculator()
        # TTFT threshold tight, TPOT threshold generous
        report = calc.calculate(
            data,
            sla_ttft_ms=100.0,
            sla_tpot_ms=1000.0,
            percentile=95.0,
        )
        assert report.tightest_metric == "ttft"

    def test_percentile_parameter(self) -> None:
        data = _make_data(n=100, ttft_base=50.0, spread=1.0)
        calc = SLAHeadroomCalculator()
        r50 = calc.calculate(data, sla_ttft_ms=100.0, percentile=50.0)
        r99 = calc.calculate(data, sla_ttft_ms=100.0, percentile=99.0)
        # P50 should have more headroom than P99
        assert r50.metrics[0].headroom_ms > r99.metrics[0].headroom_ms

    def test_adequate_headroom(self) -> None:
        # P95 ≈ 50*(1+0.95*0.8) = 50*1.76 = 88. Threshold 120 → headroom ~26.7%
        data = _make_data(n=100, ttft_base=50.0, spread=0.8)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=120.0, percentile=95.0)
        m = report.metrics[0]
        assert m.passes_sla is True
        assert m.safety_level == SafetyLevel.ADEQUATE

    def test_model_serialization(self) -> None:
        data = _make_data(n=50)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=200.0)
        d = report.model_dump()
        assert "metrics" in d
        assert "tightest_metric" in d
        assert "recommendation" in d

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
                    output_tokens=5,
                    ttft_ms=30.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=1000.0,
                )
            ],
        )
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=50.0)
        assert len(report.metrics) == 1
        assert report.metrics[0].actual_ms == 30.0
        assert report.metrics[0].headroom_ms == 20.0


class TestProgrammaticAPI:
    """Tests for analyze_sla_headroom function."""

    def test_api_returns_dict(self, tmp_path) -> None:
        data = _make_data(n=50)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))
        result = analyze_sla_headroom(
            str(path), sla_ttft_ms=200.0, sla_tpot_ms=50.0
        )
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "all_pass" in result

    def test_api_with_percentile(self, tmp_path) -> None:
        data = _make_data(n=50)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))
        result = analyze_sla_headroom(
            str(path), sla_ttft_ms=200.0, percentile=99.0
        )
        assert result["metrics"][0]["percentile"] == 99.0


class TestHeadroomReportModel:
    """Test Pydantic model behavior."""

    def test_safety_level_enum(self) -> None:
        assert SafetyLevel.CRITICAL.value == "critical"
        assert SafetyLevel.COMFORTABLE.value == "comfortable"

    def test_metric_headroom_model(self) -> None:
        m = MetricHeadroom(
            metric="ttft",
            sla_threshold_ms=100.0,
            actual_ms=80.0,
            percentile=95.0,
            headroom_ms=20.0,
            headroom_pct=20.0,
            passes_sla=True,
            safety_level=SafetyLevel.ADEQUATE,
        )
        assert m.metric == "ttft"
        d = m.model_dump()
        assert d["safety_level"] == "adequate"

    def test_headroom_report_model(self) -> None:
        r = HeadroomReport(
            metrics=[],
            tightest_metric=None,
            tightest_headroom_pct=None,
            all_pass=True,
            recommendation="No thresholds.",
        )
        assert r.all_pass is True


class TestEdgeCases:
    """Edge case tests."""

    def test_all_same_latency(self) -> None:
        requests = [
            BenchmarkRequest(
                request_id=f"r{i}",
                prompt_tokens=10,
                output_tokens=5,
                ttft_ms=50.0,
                tpot_ms=10.0,
                total_latency_ms=200.0,
                timestamp=1000.0 + i,
            )
            for i in range(20)
        ]
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=5.0,
            ),
            requests=requests,
        )
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=100.0)
        assert report.metrics[0].actual_ms == 50.0
        assert report.metrics[0].headroom_ms == 50.0

    def test_mixed_pass_fail(self) -> None:
        data = _make_data(n=100, ttft_base=50.0, tpot_base=10.0, spread=1.0)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(
            data,
            sla_ttft_ms=200.0,  # comfortable
            sla_tpot_ms=10.0,   # will fail (P95 near 19.5)
        )
        assert report.all_pass is False
        assert any(m.passes_sla for m in report.metrics)
        assert any(not m.passes_sla for m in report.metrics)

    def test_zero_threshold(self) -> None:
        data = _make_data(n=10)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=0.0)
        assert report.metrics[0].passes_sla is False

    def test_large_dataset(self) -> None:
        data = _make_data(n=10000, spread=0.3)
        calc = SLAHeadroomCalculator()
        report = calc.calculate(data, sla_ttft_ms=100.0)
        assert report.metrics[0].passes_sla is True
