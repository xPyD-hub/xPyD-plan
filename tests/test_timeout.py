"""Tests for request timeout analysis."""

from __future__ import annotations

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.timeout import (
    TimeoutAnalyzer,
    TimeoutConfig,
    TimeoutReport,
    TimeoutSeverity,
    analyze_timeouts,
)


def _make_request(
    request_id: str = "r1",
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 20.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 520.0,
    timestamp: float = 1000.0,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=timestamp,
    )


def _make_data(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestTimeoutAnalyzerNoTimeouts:
    """Tests where no requests time out."""

    def test_no_thresholds_set(self):
        data = _make_data([_make_request()])
        report = TimeoutAnalyzer().analyze(data)
        assert report.timeout_count == 0
        assert report.timeout_rate == 0.0
        assert report.severity == TimeoutSeverity.NONE

    def test_all_within_threshold(self):
        data = _make_data([_make_request(ttft_ms=10.0), _make_request(ttft_ms=15.0)])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.timeout_count == 0
        assert report.severity == TimeoutSeverity.NONE

    def test_empty_data(self):
        data = _make_data([_make_request(ttft_ms=10.0)])
        # Single request within threshold
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.timeout_count == 0
        assert report.timeout_rate == 0.0


class TestTimeoutAnalyzerDetection:
    """Tests for timeout detection."""

    def test_ttft_timeout(self):
        requests = [
            _make_request(request_id="r1", ttft_ms=50.0),
            _make_request(request_id="r2", ttft_ms=200.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.timeout_count == 1
        assert report.events[0].request_id == "r2"
        assert "ttft_ms" in report.events[0].exceeded_metrics

    def test_tpot_timeout(self):
        requests = [
            _make_request(request_id="r1", tpot_ms=5.0),
            _make_request(request_id="r2", tpot_ms=50.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, tpot_ms=20.0)
        assert report.timeout_count == 1
        assert "tpot_ms" in report.events[0].exceeded_metrics

    def test_total_latency_timeout(self):
        requests = [
            _make_request(request_id="r1", total_latency_ms=100.0),
            _make_request(request_id="r2", total_latency_ms=5000.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, total_latency_ms=1000.0)
        assert report.timeout_count == 1
        assert "total_latency_ms" in report.events[0].exceeded_metrics

    def test_multiple_metrics_exceeded(self):
        req = _make_request(ttft_ms=200.0, tpot_ms=50.0, total_latency_ms=5000.0)
        data = _make_data([req])
        report = TimeoutAnalyzer().analyze(
            data, ttft_ms=100.0, tpot_ms=20.0, total_latency_ms=1000.0
        )
        assert report.timeout_count == 1
        assert len(report.events[0].exceeded_metrics) == 3

    def test_per_metric_counts(self):
        requests = [
            _make_request(request_id="r1", ttft_ms=200.0, tpot_ms=5.0, total_latency_ms=100.0),
            _make_request(request_id="r2", ttft_ms=10.0, tpot_ms=50.0, total_latency_ms=100.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0, tpot_ms=20.0)
        assert report.per_metric_counts["ttft_ms"] == 1
        assert report.per_metric_counts["tpot_ms"] == 1

    def test_exact_threshold_not_timeout(self):
        """Requests exactly at threshold should NOT be counted as timeouts."""
        req = _make_request(ttft_ms=100.0)
        data = _make_data([req])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.timeout_count == 0


class TestTimeoutSeverity:
    """Tests for severity classification."""

    def test_none_severity(self):
        assert TimeoutAnalyzer._classify_severity(0.0) == TimeoutSeverity.NONE

    def test_low_severity(self):
        assert TimeoutAnalyzer._classify_severity(0.005) == TimeoutSeverity.LOW

    def test_moderate_severity(self):
        assert TimeoutAnalyzer._classify_severity(0.03) == TimeoutSeverity.MODERATE

    def test_high_severity(self):
        assert TimeoutAnalyzer._classify_severity(0.10) == TimeoutSeverity.HIGH

    def test_critical_severity(self):
        assert TimeoutAnalyzer._classify_severity(0.20) == TimeoutSeverity.CRITICAL


class TestTokenCharacterization:
    """Tests for token characterization of timed-out requests."""

    def test_single_timeout_characterization(self):
        req = _make_request(prompt_tokens=200, output_tokens=100, ttft_ms=500.0)
        data = _make_data([req])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.token_characterization is not None
        assert report.token_characterization.prompt_tokens_mean == 200.0
        assert report.token_characterization.output_tokens_mean == 100.0

    def test_multiple_timeout_characterization(self):
        requests = [
            _make_request(request_id="r1", prompt_tokens=100, output_tokens=50, ttft_ms=500.0),
            _make_request(request_id="r2", prompt_tokens=300, output_tokens=150, ttft_ms=600.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        tc = report.token_characterization
        assert tc is not None
        assert tc.prompt_tokens_mean == 200.0
        assert tc.prompt_tokens_min == 100
        assert tc.prompt_tokens_max == 300

    def test_no_characterization_when_no_timeouts(self):
        data = _make_data([_make_request(ttft_ms=10.0)])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.token_characterization is None


class TestTemporalClusters:
    """Tests for temporal clustering."""

    def test_single_cluster(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=500.0, timestamp=1000.0 + i)
            for i in range(5)
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert len(report.temporal_clusters) == 1
        assert report.temporal_clusters[0].count == 5

    def test_two_clusters(self):
        requests = [
            _make_request(request_id="r1", ttft_ms=500.0, timestamp=1000.0),
            _make_request(request_id="r2", ttft_ms=500.0, timestamp=1002.0),
            _make_request(request_id="r3", ttft_ms=500.0, timestamp=1020.0),
            _make_request(request_id="r4", ttft_ms=500.0, timestamp=1022.0),
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert len(report.temporal_clusters) == 2
        assert report.temporal_clusters[0].count == 2
        assert report.temporal_clusters[1].count == 2

    def test_no_clusters_when_no_timeouts(self):
        data = _make_data([_make_request(ttft_ms=10.0)])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert len(report.temporal_clusters) == 0


class TestTimeoutConfig:
    """Tests for TimeoutConfig model."""

    def test_config_passed_to_analyzer(self):
        config = TimeoutConfig(ttft_ms=50.0, tpot_ms=None, total_latency_ms=None)
        req = _make_request(ttft_ms=100.0)
        data = _make_data([req])
        report = TimeoutAnalyzer().analyze(data, config=config)
        assert report.timeout_count == 1
        assert report.config.ttft_ms == 50.0

    def test_args_override_config(self):
        config = TimeoutConfig(ttft_ms=50.0)
        req = _make_request(ttft_ms=100.0)
        data = _make_data([req])
        report = TimeoutAnalyzer().analyze(data, config=config, ttft_ms=200.0)
        assert report.timeout_count == 0
        assert report.config.ttft_ms == 200.0


class TestProgrammaticAPI:
    """Tests for the analyze_timeouts() convenience function."""

    def test_basic(self):
        requests = [
            _make_request(request_id="r1", ttft_ms=10.0),
            _make_request(request_id="r2", ttft_ms=500.0),
        ]
        data = _make_data(requests)
        report = analyze_timeouts(data, ttft_ms=100.0)
        assert isinstance(report, TimeoutReport)
        assert report.timeout_count == 1

    def test_no_thresholds(self):
        data = _make_data([_make_request()])
        report = analyze_timeouts(data)
        assert report.timeout_count == 0


class TestRecommendation:
    """Tests for recommendation generation."""

    def test_no_timeout_recommendation(self):
        data = _make_data([_make_request(ttft_ms=10.0)])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert "No timeouts" in report.recommendation

    def test_critical_recommendation(self):
        requests = [
            _make_request(request_id=f"r{i}", ttft_ms=500.0, timestamp=1000.0 + i)
            for i in range(20)
        ]
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert "CRITICAL" in report.recommendation

    def test_moderate_mentions_bottleneck(self):
        # 3 out of 100 = 3% -> MODERATE
        requests = []
        for i in range(100):
            ttft = 500.0 if i < 3 else 10.0
            requests.append(_make_request(request_id=f"r{i}", ttft_ms=ttft, timestamp=1000.0 + i))
        data = _make_data(requests)
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        assert report.severity == TimeoutSeverity.MODERATE
        assert "prefill" in report.recommendation


class TestReportSerialization:
    """Test that reports serialize correctly."""

    def test_model_dump(self):
        data = _make_data([_make_request(ttft_ms=500.0)])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        dumped = report.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["timeout_count"] == 1
        assert dumped["severity"] == "CRITICAL"

    def test_model_dump_json(self):
        data = _make_data([_make_request()])
        report = TimeoutAnalyzer().analyze(data, ttft_ms=100.0)
        json_str = report.model_dump_json()
        assert isinstance(json_str, str)
