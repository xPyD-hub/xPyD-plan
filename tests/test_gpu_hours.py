"""Tests for GPU Hour Calculator (M116)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.gpu_hours import (
    GPUHourCalculator,
    HourlyTraffic,
    TrafficProfile,
    calculate_gpu_hours,
)


def _make_data(
    measured_qps: float = 100.0,
    num_prefill: int = 2,
    num_decode: int = 2,
) -> BenchmarkData:
    """Create minimal benchmark data."""
    requests = [
        BenchmarkRequest(
            request_id=f"r{i}",
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=20.0,
            tpot_ms=10.0,
            total_latency_ms=30.0,
            timestamp=float(i),
        )
        for i in range(100)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=measured_qps,
        ),
        requests=requests,
    )


def _make_profile(hours: list[tuple[int, float]] | None = None) -> TrafficProfile:
    """Create a traffic profile."""
    if hours is None:
        hours = [(h, 50.0) for h in range(24)]
    return TrafficProfile(
        hours=[HourlyTraffic(hour=h, qps=q) for h, q in hours],
        name="test-profile",
    )


# --- Model tests ---


class TestHourlyTraffic:
    def test_valid(self):
        ht = HourlyTraffic(hour=0, qps=10.0)
        assert ht.hour == 0
        assert ht.qps == 10.0

    def test_invalid_hour(self):
        with pytest.raises(Exception):
            HourlyTraffic(hour=25, qps=10.0)

    def test_negative_qps(self):
        with pytest.raises(Exception):
            HourlyTraffic(hour=0, qps=-1.0)


class TestTrafficProfile:
    def test_valid(self):
        profile = _make_profile([(0, 10.0), (12, 50.0)])
        assert len(profile.hours) == 2

    def test_duplicate_hours_rejected(self):
        with pytest.raises(Exception):
            TrafficProfile(
                hours=[
                    HourlyTraffic(hour=0, qps=10.0),
                    HourlyTraffic(hour=0, qps=20.0),
                ],
            )

    def test_sorted_by_hour(self):
        profile = TrafficProfile(
            hours=[
                HourlyTraffic(hour=12, qps=50.0),
                HourlyTraffic(hour=0, qps=10.0),
            ],
        )
        assert profile.hours[0].hour == 0
        assert profile.hours[1].hour == 12


# --- Calculator tests ---


class TestGPUHourCalculator:
    def test_qps_per_instance(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        assert calc._qps_per_instance == 25.0  # 100 / 4

    def test_instances_for_qps(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        # 25 QPS per instance, so 50 QPS needs 2 instances
        assert calc._instances_for_qps(50.0) == 2
        # 51 needs 3
        assert calc._instances_for_qps(51.0) == 3
        # 0 needs 0
        assert calc._instances_for_qps(0.0) == 0

    def test_uniform_traffic(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        profile = _make_profile()  # 50 QPS every hour
        report = calc.calculate(profile, gpu_cost_per_hour=2.0)
        # 50 QPS / 25 per instance = 2 instances every hour
        assert report.peak_instances == 2
        assert report.off_peak_instances == 2
        assert report.daily_gpu_hours == 48.0  # 2 * 24
        assert report.monthly_gpu_hours == 48.0 * 30

    def test_variable_traffic(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        # Peak at hour 12 (100 QPS = 4 instances), off-peak at hour 0 (10 QPS = 1 instance)
        hours = [(0, 10.0), (12, 100.0)]
        profile = _make_profile(hours)
        report = calc.calculate(profile, gpu_cost_per_hour=3.0)
        assert report.peak_qps == 100.0
        assert report.peak_instances == 4
        assert report.off_peak_qps == 0.0  # unspecified hours = 0
        assert report.off_peak_instances == 0

    def test_scaling_savings(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        # Only 1 hour of peak traffic
        hours = [(12, 100.0)]
        profile = _make_profile(hours)
        report = calc.calculate(profile, gpu_cost_per_hour=2.0)
        s = report.scaling_savings
        # Fixed = 4 * 24 = 96 GPU hours
        assert s.fixed_daily_gpu_hours == 96.0
        # Dynamic = 4 (hour 12) + 0 (other 23 hours) = 4
        assert s.dynamic_daily_gpu_hours == 4.0
        assert s.saved_gpu_hours == 92.0
        assert s.savings_percent > 95.0

    def test_hourly_breakdown_has_24_entries(self):
        data = _make_data()
        calc = GPUHourCalculator(data)
        profile = _make_profile([(6, 50.0)])
        report = calc.calculate(profile)
        assert len(report.hourly_breakdown) == 24

    def test_zero_qps_hours(self):
        data = _make_data()
        calc = GPUHourCalculator(data)
        profile = _make_profile([(12, 50.0)])
        report = calc.calculate(profile)
        # Hour 0 should have 0 instances
        h0 = report.hourly_breakdown[0]
        assert h0.required_instances == 0
        assert h0.gpu_hours == 0.0

    def test_cost_calculation(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        profile = _make_profile()  # 50 QPS * 24h
        report = calc.calculate(profile, gpu_cost_per_hour=5.0, currency="EUR")
        assert report.currency == "EUR"
        assert report.daily_cost == 48.0 * 5.0  # 2 instances * 24h * $5
        assert report.monthly_cost == report.daily_cost * 30

    def test_report_model_dump(self):
        data = _make_data()
        calc = GPUHourCalculator(data)
        profile = _make_profile([(12, 50.0)])
        report = calc.calculate(profile)
        d = report.model_dump()
        assert "hourly_breakdown" in d
        assert "scaling_savings" in d
        assert "qps_per_instance" in d

    def test_low_measured_qps(self):
        data = _make_data(measured_qps=0.5, num_prefill=1, num_decode=1)
        calc = GPUHourCalculator(data)
        assert calc._qps_per_instance == 0.25

    def test_low_instances(self):
        data = _make_data(measured_qps=100.0, num_prefill=1, num_decode=1)
        calc = GPUHourCalculator(data)
        assert calc._qps_per_instance == 50.0


# --- Convenience function ---


class TestCalculateGPUHours:
    def test_returns_dict(self):
        data = _make_data()
        profile = _make_profile([(12, 50.0)])
        result = calculate_gpu_hours(data, profile, gpu_cost_per_hour=2.0)
        assert isinstance(result, dict)
        assert "daily_gpu_hours" in result
        assert "scaling_savings" in result

    def test_custom_currency(self):
        data = _make_data()
        profile = _make_profile([(12, 50.0)])
        result = calculate_gpu_hours(data, profile, currency="CNY")
        assert result["currency"] == "CNY"


# --- Public imports ---


class TestPublicImports:
    def test_imports_from_package(self):
        import xpyd_plan

        assert hasattr(xpyd_plan, "GPUHourCalculator")
        assert hasattr(xpyd_plan, "GPUHourReport")
        assert hasattr(xpyd_plan, "HourBreakdown")
        assert hasattr(xpyd_plan, "HourlyTraffic")
        assert hasattr(xpyd_plan, "ScalingSavings")
        assert hasattr(xpyd_plan, "TrafficProfile")
        assert hasattr(xpyd_plan, "calculate_gpu_hours")


# --- CLI test ---


class TestCLI:
    def test_gpu_hours_json_output(self, tmp_path: Path):
        """Test CLI produces valid JSON output."""
        data = _make_data()
        bench_path = tmp_path / "bench.json"
        bench_path.write_text(json.dumps(data.model_dump()))

        profile_data = {
            "name": "test",
            "hours": [{"hour": 8, "qps": 50.0}, {"hour": 20, "qps": 10.0}],
        }
        import yaml

        profile_path = tmp_path / "traffic.yaml"
        profile_path.write_text(yaml.dump(profile_data))

        import argparse

        from xpyd_plan.cli._gpu_hours import _run

        args = argparse.Namespace(
            benchmark=str(bench_path),
            traffic_profile=str(profile_path),
            gpu_cost=2.0,
            currency="USD",
            output_format="json",
        )

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _run(args)
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert "daily_gpu_hours" in output
        assert "scaling_savings" in output


# --- Edge cases ---


class TestEdgeCases:
    def test_all_24_hours_specified(self):
        data = _make_data(measured_qps=100.0, num_prefill=2, num_decode=2)
        calc = GPUHourCalculator(data)
        hours = [(h, float(h * 5 + 10)) for h in range(24)]
        profile = _make_profile(hours)
        report = calc.calculate(profile)
        assert len(report.hourly_breakdown) == 24
        assert report.peak_qps == 23 * 5 + 10  # hour 23

    def test_single_hour_profile(self):
        data = _make_data()
        calc = GPUHourCalculator(data)
        profile = _make_profile([(12, 100.0)])
        report = calc.calculate(profile)
        assert report.off_peak_qps == 0.0
        assert report.peak_qps == 100.0

    def test_avg_utilization_range(self):
        data = _make_data()
        calc = GPUHourCalculator(data)
        profile = _make_profile()
        report = calc.calculate(profile)
        assert 0.0 <= report.avg_utilization <= 1.0
