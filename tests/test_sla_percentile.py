"""Tests for configurable SLA percentile threshold (issue #32)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import SLACheck
from xpyd_plan.config import ConfigProfile, SLAProfile
from xpyd_plan.models import SLAConfig
from xpyd_plan.sensitivity import analyze_sensitivity
from xpyd_plan.streaming import StreamingAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark_data(n: int = 100) -> dict:
    """Generate benchmark data with linearly increasing latencies.

    Request i has ttft = i+1, tpot = (i+1)*0.5, total = (i+1)*10.
    This makes percentile values predictable.
    """
    requests = []
    for i in range(n):
        requests.append({
            "request_id": f"req-{i}",
            "prompt_tokens": 100,
            "output_tokens": 50,
            "ttft_ms": float(i + 1),         # 1..100
            "tpot_ms": float(i + 1) * 0.5,   # 0.5..50
            "total_latency_ms": float(i + 1) * 10.0,  # 10..1000
            "timestamp": 1000.0 + i,
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


def _write_benchmark(tmp_path: Path, data: dict | None = None) -> Path:
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data or _make_benchmark_data()))
    return p


# ---------------------------------------------------------------------------
# SLAConfig validation tests
# ---------------------------------------------------------------------------

class TestSLAConfigPercentile:
    def test_default_percentile_is_95(self):
        sla = SLAConfig(ttft_ms=100.0)
        assert sla.sla_percentile == 95.0

    def test_custom_percentile(self):
        sla = SLAConfig(ttft_ms=100.0, sla_percentile=99.0)
        assert sla.sla_percentile == 99.0

    def test_percentile_90(self):
        sla = SLAConfig(sla_percentile=90.0)
        assert sla.sla_percentile == 90.0

    def test_percentile_lower_bound(self):
        sla = SLAConfig(sla_percentile=1.0)
        assert sla.sla_percentile == 1.0

    def test_percentile_upper_bound(self):
        sla = SLAConfig(sla_percentile=100.0)
        assert sla.sla_percentile == 100.0

    def test_percentile_below_range_rejected(self):
        with pytest.raises(Exception):
            SLAConfig(sla_percentile=0.5)

    def test_percentile_above_range_rejected(self):
        with pytest.raises(Exception):
            SLAConfig(sla_percentile=100.1)


# ---------------------------------------------------------------------------
# SLACheck.evaluated_percentile tests
# ---------------------------------------------------------------------------

class TestSLACheckEvaluatedPercentile:
    def test_default_evaluated_percentile(self):
        check = SLACheck(
            ttft_p95_ms=10.0, ttft_p99_ms=20.0,
            tpot_p95_ms=5.0, tpot_p99_ms=8.0,
            total_latency_p95_ms=100.0, total_latency_p99_ms=200.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
        )
        assert check.evaluated_percentile == 95.0

    def test_custom_evaluated_percentile(self):
        check = SLACheck(
            ttft_p95_ms=10.0, ttft_p99_ms=20.0,
            tpot_p95_ms=5.0, tpot_p99_ms=8.0,
            total_latency_p95_ms=100.0, total_latency_p99_ms=200.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
            evaluated_percentile=99.0,
        )
        assert check.evaluated_percentile == 99.0

    def test_evaluated_values_present(self):
        check = SLACheck(
            ttft_p95_ms=10.0, ttft_p99_ms=20.0,
            tpot_p95_ms=5.0, tpot_p99_ms=8.0,
            total_latency_p95_ms=100.0, total_latency_p99_ms=200.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
            evaluated_percentile=90.0,
            ttft_evaluated_ms=8.0,
            tpot_evaluated_ms=4.0,
            total_latency_evaluated_ms=80.0,
        )
        assert check.ttft_evaluated_ms == 8.0
        assert check.tpot_evaluated_ms == 4.0
        assert check.total_latency_evaluated_ms == 80.0


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.check_sla with different percentiles
# ---------------------------------------------------------------------------

class TestCheckSLAPercentile:
    def setup_method(self):
        self.analyzer = BenchmarkAnalyzer()
        self.analyzer.load_data_from_dict(_make_benchmark_data())

    def test_check_sla_default_p95(self):
        """Default P95 evaluation — backward compatible."""
        sla = SLAConfig(ttft_ms=200.0)
        result = self.analyzer.check_sla(sla)
        assert result.evaluated_percentile == 95.0
        assert result.meets_ttft is True

    def test_check_sla_p90_relaxed_passes(self):
        """P90 is more relaxed than P95 — should pass with a tighter threshold."""
        # P90 of ttft (1..100) ≈ 90.1, P95 ≈ 95.05
        sla = SLAConfig(ttft_ms=92.0, sla_percentile=90.0)
        result = self.analyzer.check_sla(sla)
        assert result.evaluated_percentile == 90.0
        assert result.meets_ttft is True

    def test_check_sla_p95_same_threshold_fails(self):
        """Same threshold that passes P90 should fail at P95."""
        sla = SLAConfig(ttft_ms=92.0, sla_percentile=95.0)
        result = self.analyzer.check_sla(sla)
        assert result.meets_ttft is False

    def test_check_sla_p99_strict(self):
        """P99 is stricter — needs higher threshold to pass."""
        sla = SLAConfig(ttft_ms=98.0, sla_percentile=99.0)
        result = self.analyzer.check_sla(sla)
        assert result.evaluated_percentile == 99.0
        # P99 of 1..100 ≈ 99.01 → 98 threshold should fail
        assert result.meets_ttft is False

    def test_check_sla_p99_passes_with_high_threshold(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=99.0)
        result = self.analyzer.check_sla(sla)
        assert result.meets_ttft is True

    def test_check_sla_evaluated_values_populated(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        result = self.analyzer.check_sla(sla)
        assert result.ttft_evaluated_ms is not None
        assert result.tpot_evaluated_ms is not None
        assert result.total_latency_evaluated_ms is not None
        # Evaluated value should be <= P95 for P90
        assert result.ttft_evaluated_ms <= result.ttft_p95_ms

    def test_p95_and_p99_always_reported(self):
        """P95 and P99 values should always be reported regardless of eval percentile."""
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        result = self.analyzer.check_sla(sla)
        assert result.ttft_p95_ms > 0
        assert result.ttft_p99_ms > 0
        assert result.ttft_p99_ms >= result.ttft_p95_ms


# ---------------------------------------------------------------------------
# _estimate_ratio_performance with different percentiles
# ---------------------------------------------------------------------------

class TestEstimateRatioPercentile:
    def setup_method(self):
        self.analyzer = BenchmarkAnalyzer()
        self.analyzer.load_data_from_dict(_make_benchmark_data())

    def test_estimate_p90_passes_where_p95_fails(self):
        """A tight SLA that passes at P90 but fails at P95."""
        # Find a threshold between P90 and P95 of scaled values
        sla_p90 = SLAConfig(ttft_ms=92.0, sla_percentile=90.0)
        sla_p95 = SLAConfig(ttft_ms=92.0, sla_percentile=95.0)

        result_p90 = self.analyzer._estimate_ratio_performance(4, 4, sla_p90)
        result_p95 = self.analyzer._estimate_ratio_performance(4, 4, sla_p95)

        assert result_p90.sla_check.evaluated_percentile == 90.0
        assert result_p95.sla_check.evaluated_percentile == 95.0
        # P90 should be more lenient
        assert result_p90.meets_sla is True
        assert result_p95.meets_sla is False

    def test_estimate_evaluated_percentile_in_sla_check(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=99.0)
        result = self.analyzer._estimate_ratio_performance(4, 4, sla)
        assert result.sla_check.evaluated_percentile == 99.0

    def test_estimate_evaluated_values_populated(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        result = self.analyzer._estimate_ratio_performance(4, 4, sla)
        assert result.sla_check.ttft_evaluated_ms is not None


# ---------------------------------------------------------------------------
# find_optimal_ratio with different percentiles
# ---------------------------------------------------------------------------

class TestFindOptimalRatioPercentile:
    def setup_method(self):
        self.analyzer = BenchmarkAnalyzer()
        self.analyzer.load_data_from_dict(_make_benchmark_data())

    def test_p90_finds_optimal(self):
        sla = SLAConfig(ttft_ms=200.0, tpot_ms=100.0, sla_percentile=90.0)
        result = self.analyzer.find_optimal_ratio(8, sla)
        assert result.best is not None
        assert result.best.sla_check.evaluated_percentile == 90.0

    def test_p99_may_reject_more_candidates(self):
        """With P99, fewer candidates should meet SLA than with P90."""
        sla_p90 = SLAConfig(ttft_ms=200.0, tpot_ms=100.0, sla_percentile=90.0)
        sla_p99 = SLAConfig(ttft_ms=200.0, tpot_ms=100.0, sla_percentile=99.0)

        result_p90 = self.analyzer.find_optimal_ratio(8, sla_p90)
        result_p99 = self.analyzer.find_optimal_ratio(8, sla_p99)

        passing_p90 = sum(1 for c in result_p90.candidates if c.meets_sla)
        passing_p99 = sum(1 for c in result_p99.candidates if c.meets_sla)
        assert passing_p90 >= passing_p99


# ---------------------------------------------------------------------------
# Config profile sla.percentile
# ---------------------------------------------------------------------------

class TestConfigProfilePercentile:
    def test_sla_profile_percentile_default(self):
        profile = SLAProfile()
        assert profile.percentile is None

    def test_sla_profile_percentile_set(self):
        profile = SLAProfile(percentile=99.0)
        assert profile.percentile == 99.0

    def test_config_profile_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("sla:\n  percentile: 90.0\n  ttft_ms: 200.0\n")
        cfg = ConfigProfile.from_yaml(config_yaml)
        assert cfg.sla.percentile == 90.0
        assert cfg.sla.ttft_ms == 200.0

    def test_config_profile_percentile_in_yaml_output(self):
        cfg = ConfigProfile(sla=SLAProfile(percentile=99.0))
        yaml_str = cfg.to_yaml()
        assert "percentile" in yaml_str


# ---------------------------------------------------------------------------
# CLI --sla-percentile flag
# ---------------------------------------------------------------------------

class TestCLISlaPercentile:
    def test_analyze_sla_percentile_flag(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        output_file = tmp_path / "result.json"
        from xpyd_plan.cli import main

        main([
            "analyze",
            "--benchmark", str(bench),
            "--sla-ttft", "200",
            "--sla-percentile", "90",
            "--output", str(output_file),
        ])
        data = json.loads(output_file.read_text())
        # Check that candidates have the evaluated percentile
        if data.get("candidates"):
            first = data["candidates"][0]
            if first.get("sla_check"):
                assert first["sla_check"]["evaluated_percentile"] == 90.0

    def test_analyze_default_percentile_is_95(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        output_file = tmp_path / "result.json"
        from xpyd_plan.cli import main

        main([
            "analyze",
            "--benchmark", str(bench),
            "--sla-ttft", "200",
            "--output", str(output_file),
        ])
        data = json.loads(output_file.read_text())
        if data.get("candidates"):
            first = data["candidates"][0]
            if first.get("sla_check"):
                assert first["sla_check"]["evaluated_percentile"] == 95.0


# ---------------------------------------------------------------------------
# Sensitivity analysis with percentile
# ---------------------------------------------------------------------------

class TestSensitivityPercentile:
    def setup_method(self):
        self.analyzer = BenchmarkAnalyzer()
        self.analyzer.load_data_from_dict(_make_benchmark_data())

    def test_sensitivity_respects_p90(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        result = analyze_sensitivity(self.analyzer, 8, sla)
        assert len(result.points) > 0
        # All SLA checks should use the configured percentile
        for p in result.points:
            assert p.sla_check.evaluated_percentile == 90.0

    def test_sensitivity_p99_fewer_passing(self):
        sla_p90 = SLAConfig(ttft_ms=200.0, tpot_ms=100.0, sla_percentile=90.0)
        sla_p99 = SLAConfig(ttft_ms=200.0, tpot_ms=100.0, sla_percentile=99.0)

        result_p90 = analyze_sensitivity(self.analyzer, 8, sla_p90)
        result_p99 = analyze_sensitivity(self.analyzer, 8, sla_p99)

        passing_p90 = sum(1 for p in result_p90.points if p.meets_sla)
        passing_p99 = sum(1 for p in result_p99.points if p.meets_sla)
        assert passing_p90 >= passing_p99


# ---------------------------------------------------------------------------
# Streaming analyzer with percentile
# ---------------------------------------------------------------------------

class TestStreamingPercentile:
    def test_streaming_respects_sla_percentile(self):
        sla = SLAConfig(ttft_ms=200.0, sla_percentile=90.0)
        sa = StreamingAnalyzer(sla=sla, snapshot_interval=5)

        for i in range(10):
            sa.add_request({
                "request_id": f"req-{i}",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": float(i + 1) * 10,
                "tpot_ms": float(i + 1),
                "total_latency_ms": float(i + 1) * 100,
                "timestamp": 1000.0 + i,
            })

        snapshot = sa.finalize()
        # With P90 evaluation and ttft_ms=200, should pass
        assert snapshot.meets_sla is True

    def test_streaming_p99_stricter(self):
        """Same data, P99 should be stricter."""
        sla_p90 = SLAConfig(ttft_ms=95.0, sla_percentile=90.0)
        sla_p99 = SLAConfig(ttft_ms=95.0, sla_percentile=99.0)

        records = [
            {
                "request_id": f"req-{i}",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": float(i + 1),
                "tpot_ms": float(i + 1) * 0.5,
                "total_latency_ms": float(i + 1) * 10,
                "timestamp": 1000.0 + i,
            }
            for i in range(100)
        ]

        sa90 = StreamingAnalyzer(sla=sla_p90, snapshot_interval=200)
        for r in records:
            sa90.add_request(r)
        snap90 = sa90.finalize()

        sa99 = StreamingAnalyzer(sla=sla_p99, snapshot_interval=200)
        for r in records:
            sa99.add_request(r)
        snap99 = sa99.finalize()

        # P90 should pass (P90 ≈ 90.1 < 95), P99 should fail (P99 ≈ 99.01 > 95)
        assert snap90.meets_sla is True
        assert snap99.meets_sla is False


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_sla_config_without_percentile(self):
        """SLAConfig without explicit percentile defaults to 95."""
        sla = SLAConfig(ttft_ms=100.0)
        assert sla.sla_percentile == 95.0

    def test_sla_check_without_evaluated_percentile(self):
        """SLACheck without explicit evaluated_percentile defaults to 95."""
        check = SLACheck(
            ttft_p95_ms=10.0, ttft_p99_ms=20.0,
            tpot_p95_ms=5.0, tpot_p99_ms=8.0,
            total_latency_p95_ms=100.0, total_latency_p99_ms=200.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True,
            meets_all=True,
        )
        assert check.evaluated_percentile == 95.0
        assert check.ttft_evaluated_ms is None  # None when not explicitly set
