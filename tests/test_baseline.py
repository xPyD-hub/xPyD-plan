"""Tests for BaselineManager — save, load, compare latency baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.baseline import (
    BaselineComparison,
    BaselineManager,
    BaselineProfile,
    BaselineVerdict,
    MetricDelta,
    compare_baseline,
    save_baseline,
)
from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)


def _make_requests(
    n: int = 100,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
) -> list[BenchmarkRequest]:
    """Generate deterministic benchmark requests with controlled latency spread."""
    reqs = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        reqs.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i,
                output_tokens=50 + i,
                ttft_ms=ttft_base + frac * ttft_base,
                tpot_ms=tpot_base + frac * tpot_base,
                total_latency_ms=total_base + frac * total_base,
                timestamp=1000.0 + i * 0.1,
            )
        )
    return reqs


def _make_data(
    n: int = 100,
    ttft_base: float = 50.0,
    tpot_base: float = 10.0,
    total_base: float = 200.0,
    qps: float = 100.0,
    prefill: int = 2,
    decode: int = 6,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=_make_requests(n, ttft_base, tpot_base, total_base),
    )


class TestBaselineManager:
    """Tests for BaselineManager core operations."""

    def test_save_creates_profile(self) -> None:
        data = _make_data()
        mgr = BaselineManager()
        profile = mgr.save(data)
        assert isinstance(profile, BaselineProfile)
        assert profile.qps == 100.0
        assert profile.request_count == 100
        assert profile.num_prefill_instances == 2
        assert profile.num_decode_instances == 6
        assert profile.ttft.p50_ms > 0
        assert profile.ttft.p95_ms > profile.ttft.p50_ms
        assert profile.ttft.p99_ms >= profile.ttft.p95_ms

    def test_save_to_file_and_load(self, tmp_path: Path) -> None:
        data = _make_data()
        mgr = BaselineManager()
        out_path = tmp_path / "baseline.json"
        profile = mgr.save_to_file(data, out_path)
        assert out_path.exists()

        loaded = mgr.load(out_path)
        assert loaded.qps == profile.qps
        assert loaded.ttft.p95_ms == profile.ttft.p95_ms
        assert loaded.tpot.p99_ms == profile.tpot.p99_ms

    def test_save_file_is_valid_json(self, tmp_path: Path) -> None:
        data = _make_data()
        mgr = BaselineManager()
        out_path = tmp_path / "baseline.json"
        mgr.save_to_file(data, out_path)
        raw = json.loads(out_path.read_text())
        assert "ttft" in raw
        assert "tpot" in raw
        assert "total_latency" in raw
        assert "qps" in raw

    def test_compare_no_regression(self) -> None:
        data = _make_data()
        mgr = BaselineManager()
        baseline = mgr.save(data)
        result = mgr.compare(data, baseline)
        assert result.overall_verdict == BaselineVerdict.PASS
        assert all(d.verdict == BaselineVerdict.PASS for d in result.deltas)
        assert "No regression" in result.summary

    def test_compare_with_regression(self) -> None:
        baseline_data = _make_data(ttft_base=50.0)
        current_data = _make_data(ttft_base=60.0)  # 20% increase
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        result = mgr.compare(current_data, baseline, regression_threshold_pct=10.0)
        assert result.overall_verdict == BaselineVerdict.FAIL
        ttft_fails = [
            d for d in result.deltas if d.metric == "ttft" and d.verdict == BaselineVerdict.FAIL
        ]
        assert len(ttft_fails) > 0

    def test_compare_with_warn(self) -> None:
        baseline_data = _make_data(ttft_base=50.0)
        # ~8% increase — above warn (5%) but below fail (10%)
        current_data = _make_data(ttft_base=54.0)
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        result = mgr.compare(current_data, baseline, regression_threshold_pct=10.0)
        warn_deltas = [d for d in result.deltas if d.verdict == BaselineVerdict.WARN]
        assert len(warn_deltas) > 0

    def test_compare_improvement_is_pass(self) -> None:
        baseline_data = _make_data(ttft_base=60.0)
        current_data = _make_data(ttft_base=50.0)  # improvement
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        result = mgr.compare(current_data, baseline)
        assert result.overall_verdict == BaselineVerdict.PASS

    def test_compare_custom_threshold(self) -> None:
        baseline_data = _make_data(ttft_base=50.0)
        current_data = _make_data(ttft_base=53.0)  # ~6%
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        # With 5% threshold, this should fail
        result = mgr.compare(current_data, baseline, regression_threshold_pct=5.0)
        assert result.overall_verdict == BaselineVerdict.FAIL

    def test_deltas_have_correct_fields(self) -> None:
        data = _make_data()
        mgr = BaselineManager()
        baseline = mgr.save(data)
        result = mgr.compare(data, baseline)
        assert len(result.deltas) == 9  # 3 metrics × 3 percentiles
        for d in result.deltas:
            assert isinstance(d, MetricDelta)
            assert d.metric in ("ttft", "tpot", "total_latency")
            assert d.percentile in ("p50", "p95", "p99")
            assert d.delta_ms == pytest.approx(0.0, abs=0.01)

    def test_compare_report_thresholds(self) -> None:
        data = _make_data()
        mgr = BaselineManager()
        baseline = mgr.save(data)
        result = mgr.compare(data, baseline, regression_threshold_pct=20.0)
        assert result.regression_threshold_pct == 20.0
        assert result.warn_threshold_pct == 10.0

    def test_metric_baseline_values(self) -> None:
        data = _make_data(n=100, ttft_base=100.0)
        mgr = BaselineManager()
        profile = mgr.save(data)
        # With linear spread 100-200, P50 ≈ 150
        assert 140 < profile.ttft.p50_ms < 160
        assert profile.ttft.p95_ms > profile.ttft.p50_ms

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
                    request_id="req-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=42.0,
                    tpot_ms=8.0,
                    total_latency_ms=150.0,
                    timestamp=1000.0,
                )
            ],
        )
        mgr = BaselineManager()
        profile = mgr.save(data)
        assert profile.ttft.p50_ms == 42.0
        assert profile.ttft.p95_ms == 42.0
        assert profile.request_count == 1

    def test_compare_multiple_metric_regression(self) -> None:
        baseline_data = _make_data(ttft_base=50.0, tpot_base=10.0, total_base=200.0)
        current_data = _make_data(ttft_base=60.0, tpot_base=12.0, total_base=250.0)
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        result = mgr.compare(current_data, baseline, regression_threshold_pct=10.0)
        assert result.overall_verdict == BaselineVerdict.FAIL
        fail_metrics = {d.metric for d in result.deltas if d.verdict == BaselineVerdict.FAIL}
        assert len(fail_metrics) >= 2

    def test_summary_fail_message_includes_count(self) -> None:
        baseline_data = _make_data(ttft_base=50.0)
        current_data = _make_data(ttft_base=100.0)
        mgr = BaselineManager()
        baseline = mgr.save(baseline_data)
        result = mgr.compare(current_data, baseline)
        assert "regressed" in result.summary
        assert "10.0%" in result.summary


class TestProgrammaticAPI:
    """Tests for save_baseline() and compare_baseline() functions."""

    def test_save_baseline_api(self) -> None:
        data = _make_data()
        profile = save_baseline(data)
        assert isinstance(profile, BaselineProfile)
        assert profile.request_count == 100

    def test_compare_baseline_api(self) -> None:
        data = _make_data()
        baseline = save_baseline(data)
        result = compare_baseline(data, baseline)
        assert isinstance(result, BaselineComparison)
        assert result.overall_verdict == BaselineVerdict.PASS

    def test_compare_baseline_api_with_threshold(self) -> None:
        baseline_data = _make_data(ttft_base=50.0)
        current_data = _make_data(ttft_base=55.0)
        baseline = save_baseline(baseline_data)
        result = compare_baseline(current_data, baseline, regression_threshold_pct=5.0)
        assert result.overall_verdict == BaselineVerdict.FAIL

    def test_roundtrip_via_file(self, tmp_path: Path) -> None:
        data = _make_data()
        mgr = BaselineManager()
        path = tmp_path / "bl.json"
        mgr.save_to_file(data, path)
        loaded = mgr.load(path)
        result = compare_baseline(data, loaded)
        assert result.overall_verdict == BaselineVerdict.PASS

    def test_zero_baseline_value(self) -> None:
        """Edge case: baseline with zero value should handle division gracefully."""
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="r0",
                    prompt_tokens=10,
                    output_tokens=10,
                    ttft_ms=0.0,
                    tpot_ms=0.0,
                    total_latency_ms=0.0,
                    timestamp=1000.0,
                )
            ],
        )
        baseline = save_baseline(data)
        result = compare_baseline(data, baseline)
        # Should not crash, all zeros → PASS
        assert result.overall_verdict == BaselineVerdict.PASS
