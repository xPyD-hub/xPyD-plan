"""Tests for Deployment Readiness Report (M119)."""

from __future__ import annotations

import tempfile

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.readiness import (
    CheckStatus,
    ReadinessAssessor,
    ReadinessCheck,
    ReadinessConfig,
    ReadinessReport,
    ReadinessVerdict,
    _derive_verdict,
    assess_readiness,
)


def _make_data(
    n: int = 200,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    spread: float = 0.1,
) -> BenchmarkData:
    """Generate benchmark data with low variance (healthy system)."""
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


def _make_unhealthy_data(n: int = 200) -> BenchmarkData:
    """Generate data with high jitter and spikes."""
    import random

    random.seed(99)
    requests = []
    for i in range(n):
        mult = random.uniform(0.5, 8.0) if i % 5 == 0 else random.uniform(0.8, 1.5)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=50.0 * mult,
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


def _check(name: str, status: CheckStatus, detail: str = "ok") -> ReadinessCheck:
    return ReadinessCheck(
        name=name, status=status, detail=detail, value="1", threshold="1",
    )


class TestDeriveVerdict:
    def test_all_pass(self):
        checks = [_check("a", CheckStatus.PASS), _check("b", CheckStatus.PASS)]
        assert _derive_verdict(checks) == ReadinessVerdict.READY

    def test_any_warn(self):
        checks = [_check("a", CheckStatus.PASS), _check("b", CheckStatus.WARN, "meh")]
        assert _derive_verdict(checks) == ReadinessVerdict.CAUTION

    def test_any_fail(self):
        checks = [_check("a", CheckStatus.PASS), _check("b", CheckStatus.FAIL, "bad")]
        assert _derive_verdict(checks) == ReadinessVerdict.NOT_READY

    def test_fail_overrides_warn(self):
        checks = [_check("a", CheckStatus.WARN, "meh"), _check("b", CheckStatus.FAIL, "bad")]
        assert _derive_verdict(checks) == ReadinessVerdict.NOT_READY


class TestReadinessConfig:
    def test_defaults(self):
        cfg = ReadinessConfig()
        assert cfg.max_risk_score == 50.0
        assert cfg.min_headroom_pct == 10.0
        assert cfg.min_cost_efficiency == 0.7
        assert cfg.min_rate_headroom_pct == 15.0

    def test_custom(self):
        cfg = ReadinessConfig(max_risk_score=30.0, min_headroom_pct=20.0)
        assert cfg.max_risk_score == 30.0
        assert cfg.min_headroom_pct == 20.0


class TestReadinessAssessor:
    def test_basic_quality_gate_only(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(data)
        assert isinstance(report, ReadinessReport)
        assert report.total_requests == 200
        assert len(report.checks) >= 1
        # quality gate should be first check
        assert report.checks[0].name == "quality_gate"

    def test_ready_with_sla(self):
        data = _make_data(n=200, spread=0.1)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            sla_ttft_ms=200.0,  # very generous
            sla_tpot_ms=50.0,
            sla_total_ms=500.0,
        )
        # Should have quality_gate + sla_risk_score + sla_headroom
        names = [c.name for c in report.checks]
        assert "quality_gate" in names
        assert "sla_risk_score" in names
        assert "sla_headroom" in names

    def test_no_sla_checks_without_sla(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(data)
        names = [c.name for c in report.checks]
        assert "sla_risk_score" not in names
        assert "sla_headroom" not in names

    def test_cost_efficiency_pass(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            cost_per_request=0.01,
            optimal_cost_per_request=0.009,
        )
        names = [c.name for c in report.checks]
        assert "cost_efficiency" in names
        cost_check = [c for c in report.checks if c.name == "cost_efficiency"][0]
        assert cost_check.status == CheckStatus.PASS

    def test_cost_efficiency_fail(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            cost_per_request=0.10,
            optimal_cost_per_request=0.02,
        )
        cost_check = [c for c in report.checks if c.name == "cost_efficiency"][0]
        assert cost_check.status == CheckStatus.FAIL

    def test_cost_efficiency_warn(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            cost_per_request=0.10,
            optimal_cost_per_request=0.075,
        )
        cost_check = [c for c in report.checks if c.name == "cost_efficiency"][0]
        assert cost_check.status == CheckStatus.WARN

    def test_rate_headroom_pass(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            measured_qps=50.0,
            max_safe_qps=100.0,
        )
        rate_check = [c for c in report.checks if c.name == "rate_headroom"][0]
        assert rate_check.status == CheckStatus.PASS

    def test_rate_headroom_fail(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            measured_qps=95.0,
            max_safe_qps=100.0,
        )
        rate_check = [c for c in report.checks if c.name == "rate_headroom"][0]
        assert rate_check.status == CheckStatus.FAIL

    def test_rate_headroom_warn(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            measured_qps=78.0,
            max_safe_qps=100.0,
        )
        rate_check = [c for c in report.checks if c.name == "rate_headroom"][0]
        assert rate_check.status == CheckStatus.WARN

    def test_rate_headroom_zero_max(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            measured_qps=10.0,
            max_safe_qps=0.0,
        )
        rate_check = [c for c in report.checks if c.name == "rate_headroom"][0]
        assert rate_check.status == CheckStatus.FAIL

    def test_verdict_ready(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(data)
        # Only quality gate, should pass on clean data
        assert report.verdict in (ReadinessVerdict.READY, ReadinessVerdict.CAUTION)

    def test_recommendation_ready(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(data)
        assert report.recommendation  # non-empty

    def test_blockers_populated_on_fail(self):
        data = _make_data(n=200)
        assessor = ReadinessAssessor()
        report = assessor.assess(
            data,
            measured_qps=99.0,
            max_safe_qps=100.0,
        )
        if report.verdict == ReadinessVerdict.NOT_READY:
            assert len(report.blockers) > 0

    def test_custom_config(self):
        cfg = ReadinessConfig(max_risk_score=10.0)
        assessor = ReadinessAssessor(config=cfg)
        assert assessor.config.max_risk_score == 10.0


class TestReadinessReport:
    def test_model_serialization(self):
        report = ReadinessReport(
            verdict=ReadinessVerdict.READY,
            checks=[
                ReadinessCheck(
                    name="test",
                    status=CheckStatus.PASS,
                    detail="ok",
                    value="1",
                    threshold="1",
                )
            ],
            blockers=[],
            warnings=[],
            recommendation="Ship it.",
            total_requests=100,
        )
        d = report.model_dump()
        assert d["verdict"] == "ready"
        assert len(d["checks"]) == 1

    def test_model_roundtrip(self):
        report = ReadinessReport(
            verdict=ReadinessVerdict.CAUTION,
            checks=[
                ReadinessCheck(
                    name="x",
                    status=CheckStatus.WARN,
                    detail="meh",
                    value="42",
                    threshold="50",
                )
            ],
            blockers=[],
            warnings=["x"],
            recommendation="Proceed with care.",
            total_requests=50,
        )
        json_str = report.model_dump_json()
        loaded = ReadinessReport.model_validate_json(json_str)
        assert loaded.verdict == ReadinessVerdict.CAUTION


class TestAssessReadinessConvenience:
    def test_from_file(self):
        data = _make_data(n=150)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(data.model_dump_json())
            path = f.name

        report = assess_readiness(path)
        assert isinstance(report, ReadinessReport)
        assert report.total_requests == 150

    def test_from_file_with_sla(self):
        data = _make_data(n=150)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(data.model_dump_json())
            path = f.name

        report = assess_readiness(
            path,
            sla_ttft_ms=200.0,
            sla_tpot_ms=50.0,
            sla_total_ms=500.0,
        )
        assert len(report.checks) >= 3


class TestCheckStatusEnum:
    def test_values(self):
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"


class TestReadinessVerdictEnum:
    def test_values(self):
        assert ReadinessVerdict.READY.value == "ready"
        assert ReadinessVerdict.CAUTION.value == "caution"
        assert ReadinessVerdict.NOT_READY.value == "not_ready"
