"""Tests for the ROI calculator module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.cost import CostConfig
from xpyd_plan.models import SLAConfig
from xpyd_plan.roi import (
    CostProjection,
    ROICalculator,
    ROIReport,
    SavingsEstimate,
    calculate_roi,
)


def _make_requests(n: int = 100, base_ts: float = 1000.0) -> list[BenchmarkRequest]:
    """Generate n benchmark requests."""
    return [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=50 + (i % 50),
            output_tokens=20 + (i % 30),
            ttft_ms=10.0 + (i % 20) * 0.5,
            tpot_ms=5.0 + (i % 10) * 0.3,
            total_latency_ms=50.0 + (i % 30) * 1.0,
            timestamp=base_ts + i * 0.1,
        )
        for i in range(n)
    ]


def _make_benchmark(
    num_prefill: int = 2,
    num_decode: int = 2,
    qps: float = 10.0,
    n_requests: int = 100,
) -> BenchmarkData:
    total = num_prefill + num_decode
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=total,
            measured_qps=qps,
        ),
        requests=_make_requests(n_requests),
    )


def _make_cost_config(rate: float = 3.0) -> CostConfig:
    return CostConfig(gpu_hourly_rate=rate, currency="USD")


def _make_sla() -> SLAConfig:
    return SLAConfig(ttft_ms=50.0, tpot_ms=20.0, max_latency_ms=200.0)


class TestCostProjection:
    """Tests for CostProjection model."""

    def test_ratio_str(self) -> None:
        proj = CostProjection(
            num_prefill=3,
            num_decode=5,
            total_instances=8,
            hourly_cost=24.0,
            daily_cost=576.0,
            monthly_cost=17280.0,
            yearly_cost=210240.0,
        )
        assert proj.ratio_str == "3P:5D"

    def test_time_projections_consistent(self) -> None:
        proj = CostProjection(
            num_prefill=2,
            num_decode=2,
            total_instances=4,
            hourly_cost=12.0,
            daily_cost=288.0,
            monthly_cost=8640.0,
            yearly_cost=105120.0,
        )
        assert proj.daily_cost == proj.hourly_cost * 24
        assert proj.monthly_cost == proj.hourly_cost * 24 * 30
        assert proj.yearly_cost == proj.hourly_cost * 24 * 365


class TestSavingsEstimate:
    """Tests for SavingsEstimate model."""

    def test_positive_savings(self) -> None:
        s = SavingsEstimate(
            current_ratio="3P:5D",
            optimal_ratio="2P:2D",
            hourly_savings=6.0,
            daily_savings=144.0,
            monthly_savings=4320.0,
            yearly_savings=52560.0,
            savings_percent=25.0,
            migration_cost=100.0,
            break_even_hours=16.67,
        )
        assert s.hourly_savings > 0
        assert s.break_even_hours is not None

    def test_no_savings(self) -> None:
        s = SavingsEstimate(
            current_ratio="2P:2D",
            optimal_ratio="2P:2D",
            hourly_savings=0.0,
            daily_savings=0.0,
            monthly_savings=0.0,
            yearly_savings=0.0,
            savings_percent=0.0,
        )
        assert s.break_even_hours is None


class TestROICalculator:
    """Tests for ROICalculator."""

    def test_basic_calculation(self) -> None:
        data = _make_benchmark(num_prefill=2, num_decode=2)
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        calc = ROICalculator(cost_config, sla)
        report = calc.calculate(data)

        assert isinstance(report, ROIReport)
        assert report.current_projection is not None
        assert report.optimal_projection is not None
        assert report.savings is not None

    def test_projections_are_consistent(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(5.0)
        sla = _make_sla()

        report = ROICalculator(cost_config, sla).calculate(data)

        cur = report.current_projection
        assert cur.daily_cost == pytest.approx(cur.hourly_cost * 24)
        assert cur.monthly_cost == pytest.approx(cur.hourly_cost * 24 * 30)
        assert cur.yearly_cost == pytest.approx(cur.hourly_cost * 24 * 365)

    def test_savings_consistent_with_projections(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = ROICalculator(cost_config, sla).calculate(data)
        s = report.savings
        cur = report.current_projection
        opt = report.optimal_projection

        assert s.hourly_savings == pytest.approx(cur.hourly_cost - opt.hourly_cost)
        assert s.daily_savings == pytest.approx(s.hourly_savings * 24)
        assert s.monthly_savings == pytest.approx(s.hourly_savings * 24 * 30)

    def test_migration_cost_break_even(self) -> None:
        data = _make_benchmark(num_prefill=3, num_decode=3)
        cost_config = _make_cost_config(10.0)
        sla = SLAConfig(ttft_ms=500.0, tpot_ms=500.0, max_latency_ms=5000.0)

        report = ROICalculator(
            cost_config, sla, migration_cost=1000.0
        ).calculate(data)

        s = report.savings
        if s.hourly_savings > 0:
            assert s.break_even_hours is not None
            assert s.break_even_hours == pytest.approx(
                1000.0 / s.hourly_savings
            )

    def test_no_migration_cost_no_break_even(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = ROICalculator(cost_config, sla, migration_cost=0.0).calculate(data)
        # break_even_hours should be None when migration_cost = 0
        assert report.savings.break_even_hours is None

    def test_all_projections_populated(self) -> None:
        data = _make_benchmark(num_prefill=2, num_decode=3)
        cost_config = _make_cost_config(2.0)
        sla = SLAConfig(ttft_ms=500.0, tpot_ms=500.0, max_latency_ms=5000.0)

        report = ROICalculator(cost_config, sla).calculate(data)
        assert len(report.all_projections) >= 1

    def test_currency_propagated(self) -> None:
        data = _make_benchmark()
        cost_config = CostConfig(gpu_hourly_rate=3.0, currency="EUR")
        sla = _make_sla()

        report = ROICalculator(cost_config, sla).calculate(data)
        assert report.currency == "EUR"
        assert report.current_projection.currency == "EUR"
        assert report.savings.currency == "EUR"

    def test_same_config_zero_savings(self) -> None:
        """When current == optimal, savings should be zero."""
        data = _make_benchmark(num_prefill=1, num_decode=1)
        cost_config = _make_cost_config(5.0)
        # Very relaxed SLA so all configs pass
        sla = SLAConfig(ttft_ms=9999.0, tpot_ms=9999.0, max_latency_ms=99999.0)

        report = ROICalculator(cost_config, sla).calculate(data)
        # With only total_instances=2, there's only one split (1P:1D)
        assert report.savings.hourly_savings == pytest.approx(0.0)

    def test_report_model_dump(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = ROICalculator(cost_config, sla).calculate(data)
        d = report.model_dump()
        assert "current_projection" in d
        assert "optimal_projection" in d
        assert "savings" in d
        assert "all_projections" in d

    def test_savings_percent_computed(self) -> None:
        data = _make_benchmark(num_prefill=3, num_decode=3)
        cost_config = _make_cost_config(10.0)
        sla = SLAConfig(ttft_ms=500.0, tpot_ms=500.0, max_latency_ms=5000.0)

        report = ROICalculator(cost_config, sla).calculate(data)
        s = report.savings
        cur = report.current_projection
        if cur.hourly_cost > 0:
            expected_pct = s.hourly_savings / cur.hourly_cost * 100
            assert s.savings_percent == pytest.approx(expected_pct)


class TestCalculateROIAPI:
    """Tests for the programmatic calculate_roi() API."""

    def test_returns_roi_report(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = calculate_roi(data, cost_config, sla)
        assert isinstance(report, ROIReport)

    def test_with_migration_cost(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = calculate_roi(data, cost_config, sla, migration_cost=500.0)
        assert report.savings.migration_cost == 500.0

    def test_json_serializable(self) -> None:
        data = _make_benchmark()
        cost_config = _make_cost_config(3.0)
        sla = _make_sla()

        report = calculate_roi(data, cost_config, sla)
        dumped = json.dumps(report.model_dump())
        assert isinstance(dumped, str)


class TestROICLI:
    """Tests for the ROI CLI subcommand."""

    def _write_benchmark(self, path: Path, data: BenchmarkData) -> None:
        path.write_text(json.dumps(data.model_dump()))

    def _write_cost_config(self, path: Path, rate: float = 3.0) -> None:
        path.write_text(f"gpu_hourly_rate: {rate}\ncurrency: USD\n")

    def test_cli_table_output(self) -> None:
        from unittest.mock import patch

        data = _make_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "bench.json"
            cost_path = Path(tmpdir) / "cost.yaml"
            self._write_benchmark(bench_path, data)
            self._write_cost_config(cost_path)

            with patch(
                "sys.argv",
                [
                    "xpyd-plan",
                    "roi",
                    "--benchmark",
                    str(bench_path),
                    "--cost-model",
                    str(cost_path),
                    "--sla-ttft",
                    "50",
                    "--sla-tpot",
                    "20",
                    "--sla-total",
                    "200",
                ],
            ):
                from xpyd_plan.cli import main

                main()

    def test_cli_json_output(self) -> None:
        import io
        from unittest.mock import patch

        data = _make_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "bench.json"
            cost_path = Path(tmpdir) / "cost.yaml"
            self._write_benchmark(bench_path, data)
            self._write_cost_config(cost_path)

            captured = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "xpyd-plan",
                        "roi",
                        "--benchmark",
                        str(bench_path),
                        "--cost-model",
                        str(cost_path),
                        "--sla-ttft",
                        "50",
                        "--output-format",
                        "json",
                    ],
                ),
                patch("sys.stdout", captured),
            ):
                from xpyd_plan.cli import main

                main()

            output = json.loads(captured.getvalue())
            assert "current_projection" in output
            assert "savings" in output

    def test_cli_with_migration_cost(self) -> None:
        from unittest.mock import patch

        data = _make_benchmark()

        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "bench.json"
            cost_path = Path(tmpdir) / "cost.yaml"
            self._write_benchmark(bench_path, data)
            self._write_cost_config(cost_path)

            with patch(
                "sys.argv",
                [
                    "xpyd-plan",
                    "roi",
                    "--benchmark",
                    str(bench_path),
                    "--cost-model",
                    str(cost_path),
                    "--migration-cost",
                    "500",
                    "--sla-ttft",
                    "50",
                ],
            ):
                from xpyd_plan.cli import main

                main()
