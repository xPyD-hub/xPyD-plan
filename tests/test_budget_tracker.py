"""Tests for latency budget tracker (M98)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.budget_tracker import (
    BudgetReport,
    BudgetStatus,
    LatencyBudgetTracker,
    RequestBudget,
    track_latency_budget,
)
from xpyd_plan.cli._main import main


def _make_data(
    n: int = 50,
    ttft_base: float = 50.0,
    tpot_base: float = 20.0,
    total_base: float = 200.0,
) -> BenchmarkData:
    """Create benchmark data with predictable latencies."""
    requests = []
    for i in range(n):
        # Latencies increase linearly so we get a spread of budget consumption
        factor = 1.0 + (i / n)  # 1.0 to ~2.0
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                ttft_ms=ttft_base * factor,
                tpot_ms=tpot_base * factor,
                total_latency_ms=total_base * factor,
                timestamp=1000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestLatencyBudgetTracker:
    def test_basic_analysis(self):
        data = _make_data(50)
        tracker = LatencyBudgetTracker(sla_ttft_ms=100.0, sla_tpot_ms=40.0, sla_total_ms=400.0)
        report = tracker.analyze(data)

        assert isinstance(report, BudgetReport)
        assert report.total_requests == 50
        assert report.near_miss_threshold == 0.8

    def test_all_comfortable(self):
        """With very high SLA, all requests should be comfortable."""
        data = _make_data(20)
        tracker = LatencyBudgetTracker(
            sla_ttft_ms=10000.0, sla_tpot_ms=10000.0, sla_total_ms=10000.0
        )
        report = tracker.analyze(data)

        assert report.comfortable_count == 20
        assert report.near_miss_count == 0
        assert report.exceeded_count == 0

    def test_all_exceeded(self):
        """With very low SLA, all requests should exceed."""
        data = _make_data(20)
        tracker = LatencyBudgetTracker(sla_ttft_ms=1.0)
        report = tracker.analyze(data)

        assert report.exceeded_count == 20
        assert report.comfortable_count == 0

    def test_near_miss_detection(self):
        data = _make_data(100)
        # SLA at 90ms means requests with TTFT 72-90 are near-miss (80-100%)
        tracker = LatencyBudgetTracker(sla_ttft_ms=90.0, near_miss_threshold=0.8)
        report = tracker.analyze(data)

        assert report.near_miss_count > 0

    def test_single_metric(self):
        data = _make_data(20)
        tracker = LatencyBudgetTracker(sla_ttft_ms=100.0)
        report = tracker.analyze(data)

        assert len(report.distributions) == 1
        assert report.distributions[0].metric == "ttft"

    def test_multiple_metrics(self):
        data = _make_data(20)
        tracker = LatencyBudgetTracker(
            sla_ttft_ms=100.0, sla_tpot_ms=40.0, sla_total_ms=400.0
        )
        report = tracker.analyze(data)

        assert len(report.distributions) == 3
        metrics = {d.metric for d in report.distributions}
        assert metrics == {"ttft", "tpot", "total_latency"}

    def test_worst_requests_sorted(self):
        data = _make_data(50)
        tracker = LatencyBudgetTracker(sla_ttft_ms=80.0)
        report = tracker.analyze(data, top_n=5)

        assert len(report.worst_requests) == 5
        ratios = [r.worst_ratio for r in report.worst_requests]
        assert ratios == sorted(ratios, reverse=True)

    def test_distribution_stats(self):
        data = _make_data(50)
        tracker = LatencyBudgetTracker(sla_ttft_ms=100.0)
        report = tracker.analyze(data)

        dist = report.distributions[0]
        assert dist.mean > 0
        assert dist.p50 > 0
        assert dist.p95 >= dist.p50
        assert dist.p99 >= dist.p95
        assert dist.max >= dist.p99

    def test_request_budget_status(self):
        data = _make_data(10)
        tracker = LatencyBudgetTracker(sla_ttft_ms=60.0, near_miss_threshold=0.8)
        report = tracker.analyze(data)

        for req in report.worst_requests:
            assert isinstance(req, RequestBudget)
            assert req.status in BudgetStatus
            assert req.worst_ratio > 0

    def test_worst_metric_identification(self):
        # TTFT is 50*factor, TPOT is 20*factor
        # With SLA_TTFT=60, SLA_TPOT=60: TTFT ratio > TPOT ratio
        data = _make_data(10)
        tracker = LatencyBudgetTracker(sla_ttft_ms=60.0, sla_tpot_ms=60.0)
        report = tracker.analyze(data)

        # For later requests, TTFT (50*factor) should be worse than TPOT (20*factor)
        last_req = report.worst_requests[0]
        assert last_req.worst_metric == "ttft"

    def test_custom_near_miss_threshold(self):
        data = _make_data(50)
        tracker_low = LatencyBudgetTracker(sla_ttft_ms=80.0, near_miss_threshold=0.5)
        tracker_high = LatencyBudgetTracker(sla_ttft_ms=80.0, near_miss_threshold=0.9)

        report_low = tracker_low.analyze(data)
        report_high = tracker_high.analyze(data)

        # Lower threshold means more near-misses
        assert report_low.near_miss_count >= report_high.near_miss_count

    def test_alerts_critical_exceeded(self):
        data = _make_data(20)
        # Very low SLA → many exceeded → CRITICAL alert
        tracker = LatencyBudgetTracker(sla_ttft_ms=30.0)
        report = tracker.analyze(data)

        critical = [a for a in report.alerts if a.severity == "CRITICAL"]
        assert len(critical) > 0

    def test_no_alerts_comfortable(self):
        data = _make_data(20)
        tracker = LatencyBudgetTracker(sla_ttft_ms=10000.0)
        report = tracker.analyze(data)

        assert len(report.alerts) == 0

    def test_invalid_no_sla(self):
        with pytest.raises(ValueError, match="At least one SLA"):
            LatencyBudgetTracker()

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="near_miss_threshold"):
            LatencyBudgetTracker(sla_ttft_ms=100.0, near_miss_threshold=1.5)

    def test_json_serialization(self):
        data = _make_data(20)
        tracker = LatencyBudgetTracker(sla_ttft_ms=100.0)
        report = tracker.analyze(data)

        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert "total_requests" in parsed
        assert "distributions" in parsed
        assert "worst_requests" in parsed

    def test_top_n(self):
        data = _make_data(50)
        tracker = LatencyBudgetTracker(sla_ttft_ms=80.0)
        report3 = tracker.analyze(data, top_n=3)
        report20 = tracker.analyze(data, top_n=20)

        assert len(report3.worst_requests) == 3
        assert len(report20.worst_requests) == 20


class TestTrackLatencyBudgetAPI:
    def test_programmatic_api(self, tmp_path: Path):
        data = _make_data(30)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        result = track_latency_budget(str(f), sla_ttft_ms=100.0, sla_total_ms=400.0)
        assert isinstance(result, dict)
        assert "total_requests" in result
        assert result["total_requests"] == 30


class TestBudgetTrackerCLI:
    def test_cli_table(self, tmp_path: Path):
        data = _make_data(30)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main(["budget-tracker", "--benchmark", str(f), "--sla-ttft", "100"])

    def test_cli_json(self, tmp_path: Path, capsys):
        data = _make_data(30)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main([
            "budget-tracker", "--benchmark", str(f),
            "--sla-ttft", "100", "--output-format", "json",
        ])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["total_requests"] == 30

    def test_cli_all_sla_flags(self, tmp_path: Path):
        data = _make_data(20)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main([
            "budget-tracker", "--benchmark", str(f),
            "--sla-ttft", "100", "--sla-tpot", "40", "--sla-total", "400",
            "--near-miss-threshold", "0.7", "--top-n", "5",
        ])

    def test_cli_custom_threshold(self, tmp_path: Path, capsys):
        data = _make_data(30)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main([
            "budget-tracker", "--benchmark", str(f),
            "--sla-ttft", "80", "--near-miss-threshold", "0.6",
            "--output-format", "json",
        ])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["near_miss_threshold"] == 0.6
