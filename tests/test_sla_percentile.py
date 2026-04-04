"""Tests for configurable SLA percentile feature (M13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import SLACheck
from xpyd_plan.config import ConfigProfile, SLAProfile
from xpyd_plan.models import SLAConfig


def _make_benchmark_data(num_requests: int = 100, seed: int = 42) -> dict:
    """Generate synthetic benchmark data with known latency distributions."""
    rng = np.random.default_rng(seed)
    ttfts = sorted(rng.exponential(scale=50, size=num_requests))
    tpots = sorted(rng.exponential(scale=10, size=num_requests))

    requests = []
    for i in range(num_requests):
        prompt_tokens = int(rng.integers(50, 500))
        output_tokens = int(rng.integers(10, 200))
        ttft = float(ttfts[i])
        tpot = float(tpots[i])
        total = ttft + tpot * output_tokens
        requests.append({
            "request_id": f"req-{i:04d}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "total_latency_ms": total,
            "timestamp": 1000.0 + i * 0.1,
        })

    return {
        "metadata": {
            "num_prefill_instances": 4,
            "num_decode_instances": 4,
            "total_instances": 8,
            "measured_qps": 10.0,
        },
        "requests": requests,
    }


@pytest.fixture
def benchmark_data() -> dict:
    return _make_benchmark_data()


@pytest.fixture
def analyzer(benchmark_data: dict) -> BenchmarkAnalyzer:
    a = BenchmarkAnalyzer()
    a.load_data_from_dict(benchmark_data)
    return a


class TestSLAConfigPercentile:
    def test_default_percentile(self):
        sla = SLAConfig(ttft_ms=200.0)
        assert sla.sla_percentile == 95.0

    def test_custom_percentile(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=99.0)
        assert sla.sla_percentile == 99.0

    def test_percentile_p90(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        assert sla.sla_percentile == 90.0

    def test_percentile_min_boundary(self):
        sla = SLAConfig(sla_percentile=1.0)
        assert sla.sla_percentile == 1.0

    def test_percentile_max_boundary(self):
        sla = SLAConfig(sla_percentile=100.0)
        assert sla.sla_percentile == 100.0

    def test_percentile_below_min_raises(self):
        with pytest.raises(Exception):
            SLAConfig(sla_percentile=0.5)

    def test_percentile_above_max_raises(self):
        with pytest.raises(Exception):
            SLAConfig(sla_percentile=100.1)


class TestSLACheckEvaluatedPercentile:
    def test_default_evaluated_percentile(self):
        check = SLACheck(
            ttft_p95_ms=100, ttft_p99_ms=150,
            tpot_p95_ms=20, tpot_p99_ms=30,
            total_latency_p95_ms=500, total_latency_p99_ms=700,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
        )
        assert check.evaluated_percentile == 95.0

    def test_custom_evaluated_percentile(self):
        check = SLACheck(
            ttft_p95_ms=100, ttft_p99_ms=150,
            tpot_p95_ms=20, tpot_p99_ms=30,
            total_latency_p95_ms=500, total_latency_p99_ms=700,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True, evaluated_percentile=99.0,
        )
        assert check.evaluated_percentile == 99.0

    def test_evaluated_values_present(self):
        check = SLACheck(
            ttft_p95_ms=100, ttft_p99_ms=150,
            tpot_p95_ms=20, tpot_p99_ms=30,
            total_latency_p95_ms=500, total_latency_p99_ms=700,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
            ttft_evaluated_ms=110.0,
            tpot_evaluated_ms=25.0,
            total_latency_evaluated_ms=550.0,
        )
        assert check.ttft_evaluated_ms == 110.0
        assert check.tpot_evaluated_ms == 25.0
        assert check.total_latency_evaluated_ms == 550.0


class TestCheckSLAPercentile:
    def test_p95_default_backward_compat(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=500.0, tpot_ms=100.0)
        check = analyzer.check_sla(sla)
        assert check.evaluated_percentile == 95.0
        assert abs(check.ttft_evaluated_ms - check.ttft_p95_ms) < 0.01
        assert abs(check.tpot_evaluated_ms - check.tpot_p95_ms) < 0.01

    def test_p99_stricter_than_p95(self, analyzer: BenchmarkAnalyzer):
        sla_95 = SLAConfig(ttft_ms=500.0, sla_percentile=95.0)
        sla_99 = SLAConfig(ttft_ms=500.0, sla_percentile=99.0)
        check_95 = analyzer.check_sla(sla_95)
        check_99 = analyzer.check_sla(sla_99)
        assert check_99.ttft_evaluated_ms >= check_95.ttft_evaluated_ms
        assert check_99.tpot_evaluated_ms >= check_95.tpot_evaluated_ms

    def test_p90_more_lenient_than_p95(self, analyzer: BenchmarkAnalyzer):
        sla_90 = SLAConfig(ttft_ms=500.0, sla_percentile=90.0)
        sla_95 = SLAConfig(ttft_ms=500.0, sla_percentile=95.0)
        check_90 = analyzer.check_sla(sla_90)
        check_95 = analyzer.check_sla(sla_95)
        assert check_90.ttft_evaluated_ms <= check_95.ttft_evaluated_ms

    def test_p90_passes_where_p99_fails(self, analyzer: BenchmarkAnalyzer):
        sla_90 = SLAConfig(ttft_ms=500.0, sla_percentile=90.0)
        check_90 = analyzer.check_sla(sla_90)
        threshold = check_90.ttft_evaluated_ms + 1.0
        sla_p90 = SLAConfig(ttft_ms=threshold, sla_percentile=90.0)
        sla_p99 = SLAConfig(ttft_ms=threshold, sla_percentile=99.0)
        check_p90 = analyzer.check_sla(sla_p90)
        check_p99 = analyzer.check_sla(sla_p99)
        assert check_p90.meets_ttft is True
        assert check_p99.ttft_evaluated_ms >= check_p90.ttft_evaluated_ms

    def test_reference_p95_p99_always_present(self, analyzer: BenchmarkAnalyzer):
        sla = SLAConfig(ttft_ms=500.0, sla_percentile=90.0)
        check = analyzer.check_sla(sla)
        assert check.ttft_p95_ms > 0
        assert check.ttft_p99_ms > 0
        assert check.tpot_p95_ms > 0
        assert check.tpot_p99_ms > 0


class TestEstimateRatioPercentile:
    def test_ratio_uses_configured_percentile(self, analyzer: BenchmarkAnalyzer):
        sla_90 = SLAConfig(ttft_ms=500.0, sla_percentile=90.0)
        sla_99 = SLAConfig(ttft_ms=500.0, sla_percentile=99.0)
        cand_90 = analyzer._estimate_ratio_performance(4, 4, sla_90)
        cand_99 = analyzer._estimate_ratio_performance(4, 4, sla_99)
        assert cand_90.sla_check.evaluated_percentile == 90.0
        assert cand_99.sla_check.evaluated_percentile == 99.0
        assert cand_99.sla_check.ttft_evaluated_ms >= cand_90.sla_check.ttft_evaluated_ms

    def test_find_optimal_ratio_p90_vs_p99(self, analyzer: BenchmarkAnalyzer):
        sla_90 = SLAConfig(ttft_ms=200.0, tpot_ms=50.0, sla_percentile=90.0)
        sla_99 = SLAConfig(ttft_ms=200.0, tpot_ms=50.0, sla_percentile=99.0)
        result_90 = analyzer.find_optimal_ratio(8, sla_90)
        result_99 = analyzer.find_optimal_ratio(8, sla_99)
        passing_90 = sum(1 for c in result_90.candidates if c.meets_sla)
        passing_99 = sum(1 for c in result_99.candidates if c.meets_sla)
        assert passing_90 >= passing_99


class TestConfigProfilePercentile:
    def test_sla_profile_percentile(self):
        profile = SLAProfile(ttft_ms=200.0, percentile=99.0)
        assert profile.percentile == 99.0

    def test_sla_profile_percentile_none_default(self):
        profile = SLAProfile()
        assert profile.percentile is None

    def test_config_profile_from_yaml(self, tmp_path: Path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("sla:\n  ttft_ms: 200.0\n  percentile: 99.0\n")
        cfg = ConfigProfile.from_yaml(config_yaml)
        assert cfg.sla.percentile == 99.0

    def test_config_profile_percentile_missing_is_none(self, tmp_path: Path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("sla:\n  ttft_ms: 200.0\n")
        cfg = ConfigProfile.from_yaml(config_yaml)
        assert cfg.sla.percentile is None


class TestStreamingPercentile:
    def test_streaming_snapshot_uses_percentile(self, benchmark_data: dict):
        from xpyd_plan.streaming import StreamingAnalyzer

        sla = SLAConfig(ttft_ms=500.0, sla_percentile=90.0)
        sa = StreamingAnalyzer(sla=sla, snapshot_interval=50)
        for req in benchmark_data["requests"]:
            sa.add_request(req)
        snapshots = sa.snapshots
        assert len(snapshots) >= 1
        assert snapshots[-1].request_count > 0


class TestSensitivityPercentile:
    def test_sensitivity_respects_percentile(self, analyzer: BenchmarkAnalyzer):
        from xpyd_plan.sensitivity import analyze_sensitivity

        sla_90 = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        sla_99 = SLAConfig(ttft_ms=200.0, sla_percentile=99.0)
        sens_90 = analyze_sensitivity(analyzer, 8, sla_90)
        sens_99 = analyze_sensitivity(analyzer, 8, sla_99)
        for p90, p99 in zip(sens_90.points, sens_99.points):
            if p90.ttft_margin_pct is not None and p99.ttft_margin_pct is not None:
                assert p90.ttft_margin_pct >= p99.ttft_margin_pct


class TestCLIPercentileFlag:
    def test_sla_percentile_in_help_output(self, capsys):
        from xpyd_plan.cli import main

        with pytest.raises(SystemExit):
            main(["analyze", "--help"])
        captured = capsys.readouterr()
        assert "--sla-percentile" in captured.out

    def test_whatif_sla_percentile_in_help(self, capsys):
        from xpyd_plan.cli import main

        with pytest.raises(SystemExit):
            main(["what-if", "--help"])
        captured = capsys.readouterr()
        assert "--sla-percentile" in captured.out

    def test_plan_capacity_sla_percentile_in_help(self, capsys):
        from xpyd_plan.cli import main

        with pytest.raises(SystemExit):
            main(["plan-capacity", "--help"])
        captured = capsys.readouterr()
        assert "--sla-percentile" in captured.out

    def test_export_sla_percentile_in_help(self, capsys):
        from xpyd_plan.cli import main

        with pytest.raises(SystemExit):
            main(["export", "--help"])
        captured = capsys.readouterr()
        assert "--sla-percentile" in captured.out
