"""Tests for streaming analyzer."""

from __future__ import annotations

from xpyd_plan.benchmark_models import BenchmarkMetadata
from xpyd_plan.models import SLAConfig
from xpyd_plan.streaming import StreamingAnalyzer, StreamingSnapshot


def _make_request(i: int, ttft: float = 50.0, tpot: float = 10.0) -> dict:
    """Create a request record dict."""
    return {
        "request_id": f"r{i}",
        "prompt_tokens": 100,
        "output_tokens": 200,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "total_latency_ms": ttft + tpot * 200,
        "timestamp": 1700000000.0 + i * 0.1,
    }


def _make_xpyd_bench_request(i: int) -> dict:
    """Create request in xpyd-bench field naming."""
    return {
        "id": f"req-{i:03d}",
        "input_tokens": 128,
        "output_tokens": 256,
        "time_to_first_token_ms": 45.0 + i,
        "time_per_output_token_ms": 12.0,
        "end_to_end_latency_ms": 3117.0 + i,
        "start_time": 1700000000.0 + i * 0.1,
    }


class TestStreamingAnalyzer:
    def test_add_request_no_snapshot(self):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=20.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=5)

        result = analyzer.add_request(_make_request(0))
        assert result is None
        assert analyzer.request_count == 1

    def test_snapshot_at_interval(self):
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=20.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=3)

        for i in range(3):
            result = analyzer.add_request(_make_request(i))

        assert result is not None
        assert isinstance(result, StreamingSnapshot)
        assert result.request_count == 3
        assert result.meets_sla is True

    def test_snapshot_sla_fail(self):
        sla = SLAConfig(ttft_ms=10.0)  # Very tight SLA
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=3)

        for i in range(3):
            result = analyzer.add_request(_make_request(i, ttft=50.0))

        assert result is not None
        assert result.meets_sla is False

    def test_finalize(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=100)

        for i in range(5):
            analyzer.add_request(_make_request(i))

        final = analyzer.finalize()
        assert final.request_count == 5
        assert len(analyzer.snapshots) == 1

    def test_multiple_snapshots(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=2)

        for i in range(6):
            analyzer.add_request(_make_request(i))

        assert len(analyzer.snapshots) == 3

    def test_xpyd_bench_field_names(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=2)

        for i in range(2):
            result = analyzer.add_request(_make_xpyd_bench_request(i))

        assert result is not None
        assert result.request_count == 2

    def test_snapshot_to_dict(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=1)

        result = analyzer.add_request(_make_request(0))
        d = result.to_dict()
        assert "request_count" in d
        assert "measured_qps" in d
        assert "meets_sla" in d

    def test_to_benchmark_data(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=100)

        for i in range(5):
            analyzer.add_request(_make_request(i))

        meta = BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=10.0,
        )
        data = analyzer.to_benchmark_data(meta)
        assert len(data.requests) == 5
        assert data.metadata.measured_qps == 10.0

    def test_empty_finalize(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=10)
        # No requests added — finalize should still produce a snapshot
        final = analyzer.finalize()
        assert final.request_count == 0

    def test_measured_qps_calculation(self):
        sla = SLAConfig(ttft_ms=100.0)
        analyzer = StreamingAnalyzer(sla=sla, snapshot_interval=10)

        # 10 requests over 1 second = ~10 QPS
        for i in range(10):
            analyzer.add_request(_make_request(i))

        snapshot = analyzer.finalize()
        # elapsed = 0.9s (10 requests spaced 0.1s apart), so QPS ≈ 10/0.9 ≈ 11.1
        assert snapshot.measured_qps > 0
