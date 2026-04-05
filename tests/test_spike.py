"""Tests for latency spike detection (M79)."""

from __future__ import annotations

import random

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.spike import (
    SpikeDetector,
    SpikeReport,
    SpikeSeverity,
    _classify_severity,
    _rolling_median,
    detect_spikes,
)


def _make_data(
    n: int = 200,
    spike_indices: list[int] | None = None,
    spike_multiplier: float = 5.0,
    base_ttft: float = 50.0,
) -> BenchmarkData:
    """Generate benchmark data with optional spike injections."""
    rng = random.Random(42)
    spike_set = set(spike_indices or [])
    requests = []
    for i in range(n):
        ttft = base_ttft + rng.uniform(-5, 5)
        if i in spike_set:
            ttft *= spike_multiplier
        tpot = 20.0 + rng.uniform(-1, 1)
        total = ttft + tpot * 50
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(total, 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=100.0,
        ),
        requests=requests,
    )


class TestClassifySeverity:
    """Tests for _classify_severity."""

    def test_minor(self) -> None:
        assert _classify_severity(3.0) == SpikeSeverity.MINOR

    def test_moderate(self) -> None:
        assert _classify_severity(5.0) == SpikeSeverity.MODERATE

    def test_moderate_boundary(self) -> None:
        assert _classify_severity(7.5) == SpikeSeverity.MODERATE

    def test_severe(self) -> None:
        assert _classify_severity(10.0) == SpikeSeverity.SEVERE

    def test_severe_high(self) -> None:
        assert _classify_severity(20.0) == SpikeSeverity.SEVERE


class TestRollingMedian:
    """Tests for _rolling_median."""

    def test_basic(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _rolling_median(values, 3, 3) == 2.0  # median of [1, 2, 3]

    def test_start_of_list(self) -> None:
        values = [10.0, 20.0, 30.0]
        # idx=0, window is empty → returns values[0]
        assert _rolling_median(values, 0, 5) == 10.0

    def test_small_window(self) -> None:
        values = [1.0, 100.0, 3.0, 4.0]
        assert _rolling_median(values, 2, 1) == 100.0  # median of [100]


class TestSpikeDetectorNoSpikes:
    """Tests with no spikes present."""

    def test_stable_data(self) -> None:
        data = _make_data(n=200)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        assert isinstance(report, SpikeReport)
        assert report.total_requests == 200
        assert report.has_spikes is False
        assert len(report.events) == 0
        assert "No latency spikes" in report.recommendation

    def test_summaries_all_zero(self) -> None:
        data = _make_data(n=200)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        for s in report.summaries:
            assert s.spike_count == 0
            assert s.total_affected_requests == 0
            assert s.affected_fraction == 0.0


class TestSpikeDetectorWithSpikes:
    """Tests with injected spikes."""

    def test_single_spike(self) -> None:
        data = _make_data(n=200, spike_indices=[150], spike_multiplier=10.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        assert report.has_spikes is True
        # Should detect at least the ttft spike
        ttft_events = [e for e in report.events if e.metric == "ttft_ms"]
        assert len(ttft_events) >= 1

    def test_consecutive_spikes_grouped(self) -> None:
        data = _make_data(n=200, spike_indices=[100, 101, 102], spike_multiplier=8.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        ttft_events = [e for e in report.events if e.metric == "ttft_ms"]
        # Consecutive spikes should be grouped into one event
        grouped = [e for e in ttft_events if e.request_count >= 3]
        assert len(grouped) >= 1
        assert grouped[0].start_index == 100
        assert grouped[0].end_index == 102

    def test_multiple_separated_spikes(self) -> None:
        data = _make_data(n=200, spike_indices=[80, 160], spike_multiplier=6.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        ttft_events = [e for e in report.events if e.metric == "ttft_ms"]
        assert len(ttft_events) >= 2

    def test_spike_magnitude(self) -> None:
        data = _make_data(n=200, spike_indices=[150], spike_multiplier=10.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        ttft_events = [e for e in report.events if e.metric == "ttft_ms"]
        assert len(ttft_events) >= 1
        # Magnitude should be roughly 10x
        assert ttft_events[0].magnitude >= 5.0

    def test_spike_severity_classification(self) -> None:
        data = _make_data(n=200, spike_indices=[150], spike_multiplier=15.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        ttft_events = [e for e in report.events if e.metric == "ttft_ms"]
        assert len(ttft_events) >= 1
        assert ttft_events[0].severity == SpikeSeverity.SEVERE

    def test_summary_counts(self) -> None:
        data = _make_data(n=200, spike_indices=[100, 101, 150], spike_multiplier=8.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        ttft_summary = next(s for s in report.summaries if s.metric == "ttft_ms")
        assert ttft_summary.spike_count >= 1
        assert ttft_summary.total_affected_requests >= 2


class TestSpikeDetectorParameters:
    """Tests for different parameter configurations."""

    def test_small_window(self) -> None:
        data = _make_data(n=200, spike_indices=[50], spike_multiplier=8.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=10, threshold=3.0)
        assert report.window_size == 10

    def test_high_threshold_reduces_spikes(self) -> None:
        data = _make_data(n=200, spike_indices=[100], spike_multiplier=4.0)
        detector = SpikeDetector()
        report_low = detector.detect(data, window_size=50, threshold=3.0)
        report_high = detector.detect(data, window_size=50, threshold=5.0)
        assert len(report_high.events) <= len(report_low.events)

    def test_invalid_window_size(self) -> None:
        data = _make_data(n=50)
        detector = SpikeDetector()
        with pytest.raises(ValueError, match="window_size"):
            detector.detect(data, window_size=0)

    def test_invalid_threshold(self) -> None:
        data = _make_data(n=50)
        detector = SpikeDetector()
        with pytest.raises(ValueError, match="threshold"):
            detector.detect(data, threshold=0.5)


class TestSpikeDetectorEdgeCases:
    """Edge case tests."""

    def test_small_dataset(self) -> None:
        data = _make_data(n=5)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=3, threshold=3.0)
        assert report.total_requests == 5

    def test_spike_at_start(self) -> None:
        """Spike in first few requests (before full window available)."""
        data = _make_data(n=100, spike_indices=[2], spike_multiplier=10.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        # Should still handle gracefully
        assert isinstance(report, SpikeReport)

    def test_all_spikes(self) -> None:
        """Every request is a 'spike' — but relative to rolling window, only early ones differ."""
        data = _make_data(n=100, spike_indices=list(range(100)), spike_multiplier=1.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        # With multiplier 1.0 no actual spikes
        assert report.has_spikes is False


class TestDetectSpikesAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self) -> None:
        data = _make_data(n=100)
        result = detect_spikes(data)
        assert isinstance(result, dict)
        assert "total_requests" in result
        assert "events" in result
        assert "summaries" in result
        assert "has_spikes" in result

    def test_with_spikes(self) -> None:
        data = _make_data(n=200, spike_indices=[150], spike_multiplier=10.0)
        result = detect_spikes(data, window_size=50, threshold=3.0)
        assert result["has_spikes"] is True
        assert len(result["events"]) >= 1

    def test_custom_params(self) -> None:
        data = _make_data(n=100)
        result = detect_spikes(data, window_size=20, threshold=4.0)
        assert result["window_size"] == 20
        assert result["threshold"] == 4.0


class TestSpikeTimestamps:
    """Tests for timestamp fields in spike events."""

    def test_timestamps_present(self) -> None:
        data = _make_data(n=200, spike_indices=[150], spike_multiplier=10.0)
        detector = SpikeDetector()
        report = detector.detect(data, window_size=50, threshold=3.0)
        for event in report.events:
            assert event.start_timestamp is not None
            assert event.end_timestamp is not None
            assert event.end_timestamp >= event.start_timestamp
