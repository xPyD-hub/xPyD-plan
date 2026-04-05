"""Tests for SLO error budget burn rate analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.error_budget import (
    BudgetStatus,
    BurnRateLevel,
    BurnRateWindow,
    ErrorBudgetAnalyzer,
    ErrorBudgetConfig,
    ErrorBudgetReport,
    _classify_burn_rate,
    _request_passes_sla,
    analyze_error_budget,
)

# --- helpers ---

def _make_request(
    request_id: str = "r1",
    ttft_ms: float = 50.0,
    tpot_ms: float = 20.0,
    total_latency_ms: float = 200.0,
    timestamp: float = 1000.0,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=timestamp,
    )


def _make_data(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=10.0,
        ),
        requests=requests,
    )


def _write_benchmark(path: Path, data: BenchmarkData) -> None:
    path.write_text(json.dumps(data.model_dump(), default=str))


# --- unit tests ---


class TestRequestPassesSLA:
    def test_all_pass(self):
        req = _make_request(ttft_ms=50, tpot_ms=20, total_latency_ms=200)
        assert _request_passes_sla(req, 100, 50, 500) is True

    def test_ttft_fail(self):
        req = _make_request(ttft_ms=150)
        assert _request_passes_sla(req, 100, None, None) is False

    def test_tpot_fail(self):
        req = _make_request(tpot_ms=60)
        assert _request_passes_sla(req, None, 50, None) is False

    def test_total_fail(self):
        req = _make_request(total_latency_ms=600)
        assert _request_passes_sla(req, None, None, 500) is False

    def test_no_sla_always_passes(self):
        req = _make_request(ttft_ms=9999, tpot_ms=9999, total_latency_ms=9999)
        assert _request_passes_sla(req, None, None, None) is True


class TestClassifyBurnRate:
    def test_safe(self):
        assert _classify_burn_rate(0.5, 2.0, 10.0, 0.9) == BurnRateLevel.SAFE

    def test_warning(self):
        assert _classify_burn_rate(3.0, 2.0, 10.0, 0.5) == BurnRateLevel.WARNING

    def test_critical(self):
        assert _classify_burn_rate(15.0, 2.0, 10.0, 0.3) == BurnRateLevel.CRITICAL

    def test_exhausted(self):
        assert _classify_burn_rate(1.0, 2.0, 10.0, -0.1) == BurnRateLevel.EXHAUSTED


class TestErrorBudgetConfig:
    def test_defaults(self):
        cfg = ErrorBudgetConfig()
        assert cfg.slo_target == 0.999
        assert cfg.window_size == 10.0
        assert cfg.warning_burn_rate == 2.0
        assert cfg.critical_burn_rate == 10.0

    def test_custom(self):
        cfg = ErrorBudgetConfig(slo_target=0.99, sla_ttft_ms=100)
        assert cfg.slo_target == 0.99
        assert cfg.sla_ttft_ms == 100


class TestErrorBudgetAnalyzer:
    def test_single_passing_request(self):
        requests = [_make_request(request_id="r0", ttft_ms=50)]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert report.total_requests == 1
        assert report.overall_level == BurnRateLevel.SAFE
        assert report.budget_status.is_exhausted is False

    def test_all_passing(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(100)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(slo_target=0.999, sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_requests == 100
        assert report.total_failures == 0
        assert report.overall_error_rate == 0.0
        assert report.overall_burn_rate == 0.0
        assert report.overall_level == BurnRateLevel.SAFE
        assert report.budget_status.remaining == 1.0
        assert report.budget_status.is_exhausted is False

    def test_some_failures_safe_level(self):
        # 99.9% SLO → 0.1% error budget. 1000 requests, 1 failure = 0.1% error rate
        # burn rate = 0.001 / 0.001 = 1.0 → SAFE
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i * 0.1)
            for i in range(999)
        ] + [
            _make_request(request_id="rfail", ttft_ms=150, timestamp=1099.9)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(slo_target=0.999, sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_failures == 1
        assert report.overall_burn_rate == pytest.approx(1.0, abs=0.01)
        assert report.overall_level == BurnRateLevel.SAFE

    def test_high_failure_rate_critical(self):
        # 99.9% SLO → 0.1% budget. 100 requests, 5 fail = 5% error rate
        # burn rate = 0.05 / 0.001 = 50.0 → CRITICAL
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(95)
        ] + [
            _make_request(request_id=f"rfail{i}", ttft_ms=150, timestamp=1095 + i)
            for i in range(5)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(slo_target=0.999, sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_failures == 5
        assert report.overall_burn_rate == pytest.approx(50.0, abs=0.1)
        assert report.overall_level == BurnRateLevel.EXHAUSTED  # budget consumed > 100%

    def test_budget_exhausted(self):
        # All requests fail
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=150, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(slo_target=0.999, sla_ttft_ms=100)
        report = analyzer.analyze(data)

        assert report.total_failures == 10
        assert report.budget_status.is_exhausted is True
        assert report.overall_level == BurnRateLevel.EXHAUSTED

    def test_windows_created(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(30)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100, window_size=10.0)
        report = analyzer.analyze(data)

        assert len(report.windows) >= 3

    def test_window_burn_rates(self):
        # Window 1 (0-10s): all pass. Window 2 (10-20s): half fail.
        good = [
            _make_request(request_id=f"g{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(10)
        ]
        mixed = [
            _make_request(request_id=f"m{i}", ttft_ms=50 if i < 5 else 150, timestamp=1010 + i)
            for i in range(10)
        ]
        data = _make_data(good + mixed)
        analyzer = ErrorBudgetAnalyzer(slo_target=0.999, sla_ttft_ms=100, window_size=10.0)
        report = analyzer.analyze(data)

        assert len(report.windows) >= 2
        # First window should have lower burn rate
        assert report.windows[0].burn_rate < report.windows[1].burn_rate
        assert report.worst_window_burn_rate == max(w.burn_rate for w in report.windows)

    def test_safe_window_fraction(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(50)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100, window_size=10.0)
        report = analyzer.analyze(data)

        assert report.safe_window_fraction == 1.0

    def test_multiple_sla_metrics(self):
        req_good = _make_request(request_id="good", ttft_ms=50, tpot_ms=20, total_latency_ms=200)
        req_ttft_fail = _make_request(
            request_id="fail1", ttft_ms=150, tpot_ms=20, total_latency_ms=200,
        )
        req_tpot_fail = _make_request(
            request_id="fail2", ttft_ms=50, tpot_ms=60, total_latency_ms=200,
        )
        data = _make_data([req_good, req_ttft_fail, req_tpot_fail])
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100, sla_tpot_ms=50)
        report = analyzer.analyze(data)

        assert report.total_failures == 2

    def test_custom_thresholds(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(100)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(
            slo_target=0.95,
            sla_ttft_ms=100,
            warning_burn_rate=5.0,
            critical_burn_rate=20.0,
        )
        report = analyzer.analyze(data)
        assert report.config.slo_target == 0.95
        assert report.config.warning_burn_rate == 5.0

    def test_recommendation_safe(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(100)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert "healthy" in report.recommendation.lower()

    def test_recommendation_exhausted(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=150, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(requests)
        analyzer = ErrorBudgetAnalyzer(sla_ttft_ms=100)
        report = analyzer.analyze(data)
        assert "exhausted" in report.recommendation.lower()


class TestBudgetStatus:
    def test_model(self):
        bs = BudgetStatus(
            total_budget=0.001,
            consumed=0.5,
            remaining=0.5,
            is_exhausted=False,
        )
        assert bs.total_budget == 0.001
        assert bs.remaining == 0.5


class TestBurnRateWindow:
    def test_model(self):
        w = BurnRateWindow(
            window_start=0.0,
            window_end=10.0,
            total_requests=100,
            failing_requests=1,
            error_rate=0.01,
            burn_rate=10.0,
            level=BurnRateLevel.CRITICAL,
            budget_consumed_fraction=10.0,
        )
        assert w.level == BurnRateLevel.CRITICAL


class TestProgrammaticAPI:
    def test_analyze_error_budget(self, tmp_path):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=50, timestamp=1000 + i)
            for i in range(50)
        ]
        data = _make_data(requests)
        benchmark_path = tmp_path / "bench.json"
        _write_benchmark(benchmark_path, data)

        result = analyze_error_budget(
            str(benchmark_path),
            slo_target=0.999,
            sla_ttft_ms=100,
        )
        assert isinstance(result, dict)
        assert result["total_requests"] == 50
        assert result["total_failures"] == 0
        assert result["overall_level"] == "SAFE"

    def test_analyze_error_budget_with_failures(self, tmp_path):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=150, timestamp=1000 + i)
            for i in range(10)
        ]
        data = _make_data(requests)
        benchmark_path = tmp_path / "bench.json"
        _write_benchmark(benchmark_path, data)

        result = analyze_error_budget(
            str(benchmark_path),
            sla_ttft_ms=100,
        )
        assert result["total_failures"] == 10
        assert result["budget_status"]["is_exhausted"] is True


class TestErrorBudgetReport:
    def test_serialization(self):
        report = ErrorBudgetReport(
            config=ErrorBudgetConfig(),
            total_requests=100,
            total_failures=1,
            overall_error_rate=0.01,
            overall_burn_rate=10.0,
            overall_level=BurnRateLevel.CRITICAL,
            budget_status=BudgetStatus(
                total_budget=0.001,
                consumed=10.0,
                remaining=-9.0,
                is_exhausted=True,
            ),
            windows=[],
            worst_window_burn_rate=10.0,
            safe_window_fraction=0.0,
            recommendation="Scale up.",
        )
        d = report.model_dump()
        assert d["overall_level"] == "CRITICAL"
        assert d["budget_status"]["is_exhausted"] is True
