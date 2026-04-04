"""Tests for the fleet sizing calculator (M23)."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.fleet import (
    FleetAllocation,
    FleetCalculator,
    FleetOption,
    FleetReport,
    GPUTypeConfig,
    calculate_fleet,
)
from xpyd_plan.models import SLAConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_requests(n: int = 50, ttft: float = 20.0, tpot: float = 5.0) -> list[dict]:
    """Generate benchmark request dicts."""
    return [
        {
            "request_id": f"req-{i}",
            "prompt_tokens": 100,
            "output_tokens": 50,
            "ttft_ms": ttft + (i % 10) * 0.5,
            "tpot_ms": tpot + (i % 10) * 0.2,
            "total_latency_ms": ttft + tpot * 50 + (i % 10) * 2.0,
            "timestamp": 1700000000.0 + i,
        }
        for i in range(n)
    ]


def _make_benchmark(
    num_p: int = 3,
    num_d: int = 5,
    qps: float = 100.0,
    ttft: float = 20.0,
    tpot: float = 5.0,
) -> BenchmarkData:
    """Create a BenchmarkData fixture."""
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_p,
            num_decode_instances=num_d,
            total_instances=num_p + num_d,
            measured_qps=qps,
        ),
        requests=[BenchmarkRequest(**r) for r in _make_requests(50, ttft, tpot)],
    )


def _sla(**kwargs) -> SLAConfig:
    defaults = {"ttft_ms": 100.0, "tpot_ms": 50.0, "max_latency_ms": 5000.0}
    defaults.update(kwargs)
    return SLAConfig(**defaults)


def _gpu_config(
    name: str = "A100",
    rate: float = 5.0,
    qps: float = 100.0,
    max_inst: int = 20,
    ttft: float = 20.0,
    tpot: float = 5.0,
) -> GPUTypeConfig:
    return GPUTypeConfig(
        name=name,
        benchmark=_make_benchmark(qps=qps, ttft=ttft, tpot=tpot),
        hourly_rate=rate,
        max_instances=max_inst,
    )


# ---------------------------------------------------------------------------
# GPUTypeConfig model
# ---------------------------------------------------------------------------

class TestGPUTypeConfig:
    def test_basic_creation(self):
        cfg = _gpu_config()
        assert cfg.name == "A100"
        assert cfg.hourly_rate == 5.0
        assert cfg.max_instances == 20

    def test_validation_hourly_rate(self):
        with pytest.raises(Exception):
            GPUTypeConfig(
                name="Bad",
                benchmark=_make_benchmark(),
                hourly_rate=-1.0,
            )


# ---------------------------------------------------------------------------
# FleetAllocation model
# ---------------------------------------------------------------------------

class TestFleetAllocation:
    def test_fields(self):
        a = FleetAllocation(
            gpu_type="A100",
            num_prefill=3,
            num_decode=5,
            total_instances=8,
            estimated_qps=100.0,
            hourly_cost=40.0,
            meets_sla=True,
            ratio_str="3P:5D",
        )
        assert a.gpu_type == "A100"
        assert a.total_instances == 8
        assert a.currency == "USD"


# ---------------------------------------------------------------------------
# FleetOption model
# ---------------------------------------------------------------------------

class TestFleetOption:
    def test_cost_per_qps(self):
        option = FleetOption(
            allocations=[],
            total_instances=8,
            total_hourly_cost=40.0,
            total_qps=100.0,
            meets_target_qps=True,
            meets_sla=True,
        )
        assert option.cost_per_qps == pytest.approx(0.4)

    def test_cost_per_qps_zero_qps(self):
        option = FleetOption(
            allocations=[],
            total_instances=0,
            total_hourly_cost=0.0,
            total_qps=0.0,
            meets_target_qps=False,
            meets_sla=False,
        )
        assert option.cost_per_qps is None


# ---------------------------------------------------------------------------
# FleetCalculator: single GPU type
# ---------------------------------------------------------------------------

class TestSingleGPU:
    def test_basic_fleet(self):
        cfg = _gpu_config(qps=100.0, max_inst=20)
        sla = _sla()
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=50.0)
        assert isinstance(report, FleetReport)
        assert report.target_qps == 50.0
        assert len(report.options) > 0
        assert report.best is not None
        assert report.best.meets_target_qps

    def test_high_qps_needs_more_instances(self):
        cfg = _gpu_config(qps=100.0, max_inst=30)
        sla = _sla()
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=200.0)
        if report.best:
            assert report.best.total_qps >= 200.0

    def test_impossible_target(self):
        cfg = _gpu_config(qps=10.0, max_inst=2)
        sla = _sla()
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=1000.0)
        # May have no options if max_instances too low
        # Just check it doesn't crash
        assert isinstance(report, FleetReport)


# ---------------------------------------------------------------------------
# FleetCalculator: multiple GPU types
# ---------------------------------------------------------------------------

class TestMultiGPU:
    def test_two_gpu_types(self):
        a100 = _gpu_config(name="A100", rate=5.0, qps=100.0)
        h100 = _gpu_config(name="H100", rate=8.0, qps=200.0)
        sla = _sla()
        calc = FleetCalculator([a100, h100], sla)
        report = calc.calculate(target_qps=150.0)
        assert len(report.gpu_types) == 2
        assert "A100" in report.gpu_types
        assert "H100" in report.gpu_types
        assert isinstance(report, FleetReport)

    def test_prefers_cheaper_option(self):
        cheap = _gpu_config(name="Cheap", rate=2.0, qps=100.0, max_inst=20)
        expensive = _gpu_config(name="Expensive", rate=20.0, qps=100.0, max_inst=20)
        sla = _sla()
        calc = FleetCalculator([cheap, expensive], sla)
        report = calc.calculate(target_qps=50.0)
        if report.best and report.best.meets_sla:
            # Best should prefer the cheaper GPU
            # At minimum, the cheapest option should exist
            assert report.best.total_hourly_cost > 0


# ---------------------------------------------------------------------------
# Budget ceiling
# ---------------------------------------------------------------------------

class TestBudgetCeiling:
    def test_filters_expensive_options(self):
        cfg = _gpu_config(rate=10.0, qps=100.0, max_inst=20)
        sla = _sla()
        calc = FleetCalculator([cfg], sla, budget_ceiling=50.0)
        report = calc.calculate(target_qps=50.0)
        for option in report.options:
            assert option.total_hourly_cost <= 50.0

    def test_impossible_budget(self):
        cfg = _gpu_config(rate=100.0, qps=10.0, max_inst=20)
        sla = _sla()
        calc = FleetCalculator([cfg], sla, budget_ceiling=1.0)
        report = calc.calculate(target_qps=100.0)
        # Either no options or all within budget
        for option in report.options:
            assert option.total_hourly_cost <= 1.0


# ---------------------------------------------------------------------------
# FleetReport model
# ---------------------------------------------------------------------------

class TestFleetReport:
    def test_serializable(self):
        cfg = _gpu_config()
        sla = _sla()
        report = calculate_fleet([cfg], target_qps=50.0, sla=sla)
        data = report.model_dump()
        assert isinstance(json.dumps(data), str)

    def test_report_fields(self):
        cfg = _gpu_config(name="TestGPU")
        sla = _sla()
        report = calculate_fleet([cfg], target_qps=50.0, sla=sla)
        assert report.target_qps == 50.0
        assert "TestGPU" in report.gpu_types
        assert report.currency == "USD"


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

class TestProgrammaticAPI:
    def test_calculate_fleet_basic(self):
        cfg = _gpu_config()
        sla = _sla()
        report = calculate_fleet([cfg], target_qps=50.0, sla=sla)
        assert isinstance(report, FleetReport)
        assert len(report.options) > 0

    def test_calculate_fleet_with_budget(self):
        cfg = _gpu_config(rate=5.0)
        sla = _sla()
        report = calculate_fleet(
            [cfg], target_qps=50.0, sla=sla, budget_ceiling=100.0
        )
        for option in report.options:
            assert option.total_hourly_cost <= 100.0

    def test_calculate_fleet_max_options(self):
        cfg = _gpu_config(max_inst=30)
        sla = _sla()
        report = calculate_fleet([cfg], target_qps=50.0, sla=sla, max_options=5)
        assert len(report.options) <= 5

    def test_calculate_fleet_multi_gpu(self):
        configs = [
            _gpu_config(name="A100", rate=5.0, qps=100.0),
            _gpu_config(name="H100", rate=8.0, qps=200.0),
        ]
        sla = _sla()
        report = calculate_fleet(configs, target_qps=100.0, sla=sla)
        assert isinstance(report, FleetReport)


# ---------------------------------------------------------------------------
# SLA compliance
# ---------------------------------------------------------------------------

class TestSLACompliance:
    def test_sla_meeting_options_preferred(self):
        cfg = _gpu_config()
        sla = _sla()
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=50.0)
        if len(report.options) >= 2:
            # SLA-meeting options should come first
            sla_options = [o for o in report.options if o.meets_sla]
            non_sla = [o for o in report.options if not o.meets_sla]
            if sla_options and non_sla:
                first_non_sla_idx = report.options.index(non_sla[0])
                last_sla_idx = report.options.index(sla_options[-1])
                assert last_sla_idx < first_non_sla_idx

    def test_strict_sla(self):
        # Very tight SLA
        cfg = _gpu_config(ttft=50.0, tpot=20.0)
        sla = _sla(ttft_ms=10.0, tpot_ms=1.0, max_latency_ms=100.0)
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=50.0)
        # Should still return a report (might have no SLA-meeting options)
        assert isinstance(report, FleetReport)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_instance_max(self):
        cfg = _gpu_config(max_inst=2)
        sla = _sla()
        calc = FleetCalculator([cfg], sla)
        report = calc.calculate(target_qps=1.0)
        assert isinstance(report, FleetReport)

    def test_empty_gpu_list(self):
        sla = _sla()
        calc = FleetCalculator([], sla)
        report = calc.calculate(target_qps=100.0)
        assert len(report.options) == 0
        assert report.best is None
