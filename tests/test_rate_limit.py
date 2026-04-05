"""Tests for rate_limit module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.cli import main
from xpyd_plan.rate_limit import (
    BurstConfig,
    HeadroomInfo,
    RateLimitRecommender,
    RateLimitReport,
    recommend_rate_limit,
)


def _make_benchmark(
    qps: float,
    num_prefill: int = 2,
    num_decode: int = 6,
    num_requests: int = 100,
    base_ttft: float = 50.0,
    base_tpot: float = 20.0,
    base_total: float = 200.0,
    latency_scale: float = 1.0,
) -> BenchmarkData:
    """Create a synthetic benchmark with controllable latency."""
    import random
    rng = random.Random(42 + int(qps * 100))
    requests = []
    duration = num_requests / qps if qps > 0 else 10.0
    for i in range(num_requests):
        noise = rng.gauss(0, 0.05)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=rng.randint(50, 500),
                output_tokens=rng.randint(20, 200),
                ttft_ms=max(1.0, base_ttft * latency_scale * (1 + noise)),
                tpot_ms=max(1.0, base_tpot * latency_scale * (1 + noise)),
                total_latency_ms=max(
                    1.0, base_total * latency_scale * (1 + noise)
                ),
                timestamp=i * (duration / num_requests),
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestRateLimitRecommender:
    """Tests for RateLimitRecommender."""

    def test_basic_recommendation(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=1.5),
            _make_benchmark(qps=30.0, latency_scale=2.5),
        ]
        recommender = RateLimitRecommender(safety_margin=0.10)
        report = recommender.recommend(
            benchmarks,
            sla_ttft_ms=100.0,
            sla_tpot_ms=40.0,
            sla_total_ms=400.0,
        )
        assert isinstance(report, RateLimitReport)
        assert report.max_sla_compliant_qps is not None
        assert report.recommended_qps > 0
        assert report.recommended_qps <= report.max_sla_compliant_qps
        assert report.data_points == 3

    def test_safety_margin_applied(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=2.0),
        ]
        r1 = RateLimitRecommender(safety_margin=0.0).recommend(
            benchmarks, sla_total_ms=500.0,
        )
        r2 = RateLimitRecommender(safety_margin=0.20).recommend(
            benchmarks, sla_total_ms=500.0,
        )
        assert r2.recommended_qps < r1.recommended_qps

    def test_insufficient_benchmarks(self):
        benchmarks = [_make_benchmark(qps=10.0)]
        report = RateLimitRecommender().recommend(benchmarks, sla_total_ms=500.0)
        assert report.recommended_qps == 0.0
        assert report.max_sla_compliant_qps is None
        assert "at least 2" in report.recommendation

    def test_no_sla_thresholds(self):
        benchmarks = [
            _make_benchmark(qps=10.0),
            _make_benchmark(qps=20.0),
        ]
        report = RateLimitRecommender().recommend(benchmarks)
        assert report.recommended_qps == 0.0
        assert "No SLA" in report.recommendation

    def test_sla_always_violated(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=5.0),
            _make_benchmark(qps=20.0, latency_scale=10.0),
        ]
        report = RateLimitRecommender().recommend(benchmarks, sla_ttft_ms=1.0)
        assert report.max_sla_compliant_qps is None
        assert report.recommended_qps == 0.0
        assert "violated" in report.recommendation.lower()

    def test_all_compliant(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=1.0),
        ]
        report = RateLimitRecommender(safety_margin=0.10).recommend(
            benchmarks, sla_total_ms=9999.0,
        )
        # Max compliant should be the highest tested QPS
        assert report.max_sla_compliant_qps == 20.0

    def test_headroom_computed(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=1.5),
        ]
        report = RateLimitRecommender().recommend(
            benchmarks, sla_ttft_ms=200.0, sla_total_ms=500.0,
        )
        assert len(report.headroom) > 0
        for h in report.headroom:
            assert isinstance(h, HeadroomInfo)
            assert h.headroom_ms > 0

    def test_burst_config(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=1.5),
        ]
        report = RateLimitRecommender().recommend(
            benchmarks, sla_total_ms=500.0,
        )
        if report.burst is not None:
            assert isinstance(report.burst, BurstConfig)
            assert report.burst.burst_qps >= report.burst.sustained_qps
            assert report.burst.burst_ratio >= 1.0

    def test_qps_range_reported(self):
        benchmarks = [
            _make_benchmark(qps=5.0),
            _make_benchmark(qps=25.0),
        ]
        report = RateLimitRecommender().recommend(benchmarks, sla_total_ms=500.0)
        assert report.qps_range == (5.0, 25.0)

    def test_invalid_safety_margin(self):
        with pytest.raises(ValueError, match="safety_margin"):
            RateLimitRecommender(safety_margin=1.5)

    def test_invalid_burst_window(self):
        with pytest.raises(ValueError, match="burst_window"):
            RateLimitRecommender(burst_window_seconds=0)

    def test_single_sla_metric(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=2.0),
        ]
        report = RateLimitRecommender().recommend(benchmarks, sla_tpot_ms=30.0)
        assert report.max_sla_compliant_qps is not None
        assert len(report.headroom) == 1
        assert report.headroom[0].metric == "tpot_p95"


class TestRateLimitModels:
    """Tests for Pydantic models."""

    def test_report_serializable(self):
        report = RateLimitReport(
            max_sla_compliant_qps=15.0,
            safety_margin=0.10,
            recommended_qps=13.5,
            data_points=3,
            qps_range=(5.0, 25.0),
            recommendation="Test recommendation",
        )
        data = report.model_dump()
        assert data["recommended_qps"] == 13.5
        restored = RateLimitReport.model_validate(data)
        assert restored.recommended_qps == report.recommended_qps

    def test_json_roundtrip(self):
        report = RateLimitReport(
            max_sla_compliant_qps=15.0,
            safety_margin=0.10,
            recommended_qps=13.5,
            burst=BurstConfig(
                sustained_qps=13.5,
                burst_qps=20.25,
                burst_window_seconds=10.0,
                burst_ratio=1.5,
            ),
            headroom=[
                HeadroomInfo(
                    metric="total_p95",
                    sla_threshold_ms=500.0,
                    estimated_latency_ms=350.0,
                    headroom_ms=150.0,
                    headroom_pct=30.0,
                ),
            ],
            data_points=3,
            qps_range=(5.0, 25.0),
            recommendation="Test",
        )
        j = report.model_dump_json()
        restored = RateLimitReport.model_validate_json(j)
        assert restored.burst.burst_ratio == 1.5
        assert len(restored.headroom) == 1


class TestConvenienceFunction:
    """Tests for recommend_rate_limit()."""

    def test_convenience_matches_class(self):
        benchmarks = [
            _make_benchmark(qps=10.0, latency_scale=1.0),
            _make_benchmark(qps=20.0, latency_scale=1.5),
        ]
        report = recommend_rate_limit(
            benchmarks,
            sla_total_ms=400.0,
            safety_margin=0.15,
        )
        assert isinstance(report, RateLimitReport)
        assert report.safety_margin == 0.15


class TestRateLimitCLI:
    """Tests for the CLI rate-limit subcommand."""

    def _write_benchmark(self, tmp: Path, qps: float, scale: float) -> Path:
        bm = _make_benchmark(qps=qps, latency_scale=scale)
        path = tmp / f"bench_{qps}.json"
        path.write_text(bm.model_dump_json(indent=2))
        return path

    def test_cli_json_output(self, tmp_path, capsys):
        f1 = self._write_benchmark(tmp_path, 10.0, 1.0)
        f2 = self._write_benchmark(tmp_path, 20.0, 1.5)
        main([
            "rate-limit",
            "--benchmark", str(f1), str(f2),
            "--sla-total", "500",
            "--output-format", "json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "recommended_qps" in data
        assert data["recommended_qps"] > 0

    def test_cli_table_output(self, tmp_path):
        f1 = self._write_benchmark(tmp_path, 10.0, 1.0)
        f2 = self._write_benchmark(tmp_path, 20.0, 1.5)
        main([
            "rate-limit",
            "--benchmark", str(f1), str(f2),
            "--sla-ttft", "100",
            "--output-format", "table",
        ])

    def test_cli_missing_sla(self, tmp_path):
        f1 = self._write_benchmark(tmp_path, 10.0, 1.0)
        f2 = self._write_benchmark(tmp_path, 20.0, 1.5)
        with pytest.raises(SystemExit):
            main([
                "rate-limit",
                "--benchmark", str(f1), str(f2),
                "--output-format", "table",
            ])

    def test_cli_single_file_error(self, tmp_path):
        f1 = self._write_benchmark(tmp_path, 10.0, 1.0)
        with pytest.raises(SystemExit):
            main([
                "rate-limit",
                "--benchmark", str(f1),
                "--sla-total", "500",
            ])
