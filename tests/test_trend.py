"""Tests for trend tracking module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.trend import TrendEntry, TrendReport, TrendTracker, track_trend


def _make_benchmark(
    qps: float = 100.0,
    num_prefill: int = 2,
    num_decode: int = 2,
    num_requests: int = 50,
    ttft_base: float = 10.0,
    tpot_base: float = 5.0,
    latency_base: float = 100.0,
) -> BenchmarkData:
    """Generate synthetic benchmark data."""
    requests = []
    for i in range(num_requests):
        # Add some variation
        factor = 1.0 + (i % 10) * 0.05
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + i,
                output_tokens=50 + i,
                ttft_ms=ttft_base * factor,
                tpot_ms=tpot_base * factor,
                total_latency_ms=latency_base * factor,
                timestamp=1000000.0 + i,
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


class TestTrendTracker:
    """Test TrendTracker core functionality."""

    def test_add_entry(self) -> None:
        tracker = TrendTracker(":memory:")
        data = _make_benchmark()
        entry = tracker.add(data, label="v1.0", timestamp=1000000.0)
        assert entry.id is not None
        assert entry.label == "v1.0"
        assert entry.measured_qps == 100.0
        assert entry.num_prefill == 2
        assert entry.num_decode == 2
        assert entry.num_requests == 50
        assert entry.ttft_p50_ms > 0
        assert entry.ttft_p95_ms >= entry.ttft_p50_ms
        assert entry.ttft_p99_ms >= entry.ttft_p95_ms
        tracker.close()

    def test_add_multiple_entries(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(5):
            data = _make_benchmark(qps=100.0 + i * 10)
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i * 3600)
        assert tracker.count() == 5
        tracker.close()

    def test_list_entries_all(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(3):
            data = _make_benchmark(qps=100.0 + i * 10)
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        entries = tracker.list_entries()
        assert len(entries) == 3
        assert entries[0].label == "run-0"
        assert entries[2].label == "run-2"
        # Ordered by timestamp ascending
        assert entries[0].timestamp < entries[1].timestamp < entries[2].timestamp
        tracker.close()

    def test_list_entries_with_limit(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(10):
            data = _make_benchmark()
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        entries = tracker.list_entries(limit=3)
        assert len(entries) == 3
        # Should be the 3 most recent, ordered ascending
        assert entries[0].label == "run-7"
        assert entries[2].label == "run-9"
        tracker.close()

    def test_count(self) -> None:
        tracker = TrendTracker(":memory:")
        assert tracker.count() == 0
        data = _make_benchmark()
        tracker.add(data, label="test")
        assert tracker.count() == 1
        tracker.close()

    def test_auto_timestamp(self) -> None:
        tracker = TrendTracker(":memory:")
        data = _make_benchmark()
        entry = tracker.add(data, label="auto-ts")
        assert entry.timestamp > 0
        tracker.close()


class TestTrendCheck:
    """Test degradation detection."""

    def test_no_degradation_stable(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(5):
            data = _make_benchmark(ttft_base=10.0, tpot_base=5.0, latency_base=100.0)
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        report = tracker.check(lookback=5, threshold=0.1)
        assert not report.has_degradation
        assert report.lookback_count == 5
        assert len(report.entries) == 5
        tracker.close()

    def test_degradation_detected(self) -> None:
        tracker = TrendTracker(":memory:")
        # Each run gets progressively worse latencies
        for i in range(10):
            data = _make_benchmark(
                ttft_base=10.0 + i * 3.0,  # +30% per run
                tpot_base=5.0 + i * 1.5,
                latency_base=100.0 + i * 30.0,
            )
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        report = tracker.check(lookback=10, threshold=0.1)
        assert report.has_degradation
        degrading = [a for a in report.alerts if a.is_degrading]
        assert len(degrading) > 0
        # All latency metrics should be degrading
        degrading_names = {a.metric for a in degrading}
        assert "ttft_p95_ms" in degrading_names
        tracker.close()

    def test_check_too_few_entries(self) -> None:
        tracker = TrendTracker(":memory:")
        data = _make_benchmark()
        tracker.add(data, label="only-one")
        report = tracker.check(lookback=10)
        assert not report.has_degradation
        assert report.lookback_count == 1
        assert len(report.alerts) == 0
        tracker.close()

    def test_check_empty_database(self) -> None:
        tracker = TrendTracker(":memory:")
        report = tracker.check()
        assert not report.has_degradation
        assert report.lookback_count == 0
        tracker.close()

    def test_custom_threshold(self) -> None:
        tracker = TrendTracker(":memory:")
        # Slight degradation
        for i in range(5):
            data = _make_benchmark(ttft_base=10.0 + i * 0.5)
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        # Strict threshold should flag it
        report_strict = tracker.check(lookback=5, threshold=0.01)  # noqa: F841
        # Lenient threshold should not
        report_lenient = tracker.check(lookback=5, threshold=1.0)
        assert report_lenient.has_degradation is False
        tracker.close()

    def test_lookback_limits_entries(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(20):
            data = _make_benchmark()
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        report = tracker.check(lookback=5)
        assert report.lookback_count == 5
        tracker.close()

    def test_degradation_alert_fields(self) -> None:
        tracker = TrendTracker(":memory:")
        for i in range(5):
            data = _make_benchmark(ttft_base=10.0 + i * 5.0)
            tracker.add(data, label=f"run-{i}", timestamp=1000000.0 + i)
        report = tracker.check(lookback=5, threshold=0.05)
        ttft_alerts = [a for a in report.alerts if a.metric == "ttft_p95_ms"]
        assert len(ttft_alerts) == 1
        alert = ttft_alerts[0]
        assert alert.slope_per_run > 0
        assert alert.total_change_pct > 0
        tracker.close()


class TestTrendPersistence:
    """Test SQLite persistence."""

    def test_file_persistence(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test.db"
        tracker = TrendTracker(db_path=db_file)
        data = _make_benchmark()
        tracker.add(data, label="persisted")
        tracker.close()

        # Reopen and verify
        tracker2 = TrendTracker(db_path=db_file)
        assert tracker2.count() == 1
        entries = tracker2.list_entries()
        assert entries[0].label == "persisted"
        tracker2.close()


class TestTrackTrendAPI:
    """Test the programmatic track_trend() API."""

    def test_track_trend(self, tmp_path: Path) -> None:
        # Write a benchmark file
        data = _make_benchmark()
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(data.model_dump_json())

        db_file = tmp_path / "trend.db"
        report = track_trend(
            benchmark_path=str(bench_file),
            label="api-test",
            db_path=str(db_file),
        )
        assert isinstance(report, TrendReport)
        assert report.lookback_count == 1
        assert not report.has_degradation

    def test_track_trend_multiple(self, tmp_path: Path) -> None:
        db_file = tmp_path / "trend.db"
        for i in range(5):
            data = _make_benchmark(ttft_base=10.0 + i * 5.0)
            bench_file = tmp_path / f"bench-{i}.json"
            bench_file.write_text(data.model_dump_json())
            report = track_trend(
                benchmark_path=str(bench_file),
                label=f"run-{i}",
                db_path=str(db_file),
                threshold=0.05,
            )
        # Last report should detect degradation
        assert report.lookback_count == 5


class TestTrendEntryModel:
    """Test TrendEntry model."""

    def test_model_fields(self) -> None:
        entry = TrendEntry(
            label="test",
            timestamp=1000000.0,
            measured_qps=100.0,
            num_prefill=2,
            num_decode=2,
            ttft_p50_ms=10.0,
            ttft_p95_ms=15.0,
            ttft_p99_ms=20.0,
            tpot_p50_ms=5.0,
            tpot_p95_ms=7.0,
            tpot_p99_ms=9.0,
            total_latency_p50_ms=100.0,
            total_latency_p95_ms=150.0,
            total_latency_p99_ms=200.0,
            num_requests=50,
        )
        assert entry.label == "test"
        assert entry.id is None

    def test_json_roundtrip(self) -> None:
        entry = TrendEntry(
            id=1,
            label="test",
            timestamp=1000000.0,
            measured_qps=100.0,
            num_prefill=2,
            num_decode=2,
            ttft_p50_ms=10.0,
            ttft_p95_ms=15.0,
            ttft_p99_ms=20.0,
            tpot_p50_ms=5.0,
            tpot_p95_ms=7.0,
            tpot_p99_ms=9.0,
            total_latency_p50_ms=100.0,
            total_latency_p95_ms=150.0,
            total_latency_p99_ms=200.0,
            num_requests=50,
        )
        data = json.loads(entry.model_dump_json())
        restored = TrendEntry(**data)
        assert restored.label == entry.label
        assert restored.id == entry.id
