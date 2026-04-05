"""Tests for warm-up exclusion filter."""

from __future__ import annotations

import json
from unittest.mock import patch

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli import main
from xpyd_plan.warmup import (
    LatencyComparison,
    WarmupFilter,
    WarmupWindow,
    filter_warmup,
)


def _make_requests(
    n: int = 100,
    base_ts: float = 1000.0,
    warmup_count: int = 0,
    warmup_latency_factor: float = 10.0,
) -> list[BenchmarkRequest]:
    """Generate benchmark requests with optional warm-up spike."""
    requests = []
    for i in range(n):
        if i < warmup_count:
            latency_mult = warmup_latency_factor
        else:
            latency_mult = 1.0
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i * 2,
                output_tokens=50 + i,
                ttft_ms=10.0 * latency_mult + i * 0.1,
                tpot_ms=5.0 * latency_mult + i * 0.05,
                total_latency_ms=100.0 * latency_mult + i * 1.0,
                timestamp=base_ts + i * 0.5,
            )
        )
    return requests


def _make_data(
    n: int = 100,
    warmup_count: int = 0,
    warmup_latency_factor: float = 10.0,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=50.0,
        ),
        requests=_make_requests(
            n,
            warmup_count=warmup_count,
            warmup_latency_factor=warmup_latency_factor,
        ),
    )


class TestWarmupFilter:
    def test_no_warmup_detected(self):
        """Steady data should have no warm-up."""
        data = _make_data(100, warmup_count=0)
        filt = WarmupFilter(window_size_s=5.0)
        filtered, report = filt.filter(data)

        assert not report.warmup_detected
        assert report.filtered_request_count == 100
        assert len(filtered.requests) == 100

    def test_manual_warmup_seconds(self):
        """Manual warm-up duration should exclude requests."""
        data = _make_data(100)
        filt = WarmupFilter(warmup_seconds=5.0)
        filtered, report = filt.filter(data)

        assert report.warmup_detected
        assert report.warmup_window is not None
        assert report.warmup_window.duration_s == 5.0
        assert report.filtered_request_count < 100
        assert len(filtered.requests) == report.filtered_request_count

    def test_manual_warmup_zero(self):
        """Zero warm-up should keep all requests."""
        data = _make_data(100)
        filt = WarmupFilter(warmup_seconds=0.0)
        filtered, report = filt.filter(data)

        assert not report.warmup_detected
        assert report.filtered_request_count == 100

    def test_auto_detect_warmup(self):
        """Should auto-detect warm-up from latency spike."""
        data = _make_data(200, warmup_count=20, warmup_latency_factor=10.0)
        filt = WarmupFilter(window_size_s=5.0, warmup_factor=2.0)
        filtered, report = filt.filter(data)

        assert report.warmup_detected
        assert report.warmup_window is not None
        assert report.warmup_window.requests_excluded > 0
        assert report.filtered_request_count < 200

    def test_qps_adjusted(self):
        """QPS should be adjusted proportionally after filtering."""
        data = _make_data(100)
        filt = WarmupFilter(warmup_seconds=10.0)
        _, report = filt.filter(data)

        assert report.adjusted_qps < report.original_qps

    def test_latency_comparison(self):
        """Should include before/after latency stats."""
        data = _make_data(200, warmup_count=20, warmup_latency_factor=10.0)
        filt = WarmupFilter(warmup_seconds=10.0)
        _, report = filt.filter(data)

        assert report.latency_comparison is not None
        lc = report.latency_comparison
        assert lc.before_p95_ms >= lc.after_p95_ms

    def test_small_dataset(self):
        """Should handle very small benchmark data."""
        data = _make_data(2, warmup_count=0)
        filt = WarmupFilter()
        filtered, report = filt.filter(data)

        assert not report.warmup_detected
        assert report.filtered_request_count == 2

    def test_warmup_exceeds_all_data(self):
        """If warmup_seconds > total duration, report shows 0 filtered but data preserved."""
        data = _make_data(10)  # 10 reqs at 0.5s apart = ~5s total
        filt = WarmupFilter(warmup_seconds=100.0)
        filtered, report = filt.filter(data)

        assert report.warmup_detected
        assert report.filtered_request_count == 0
        # Original data returned since empty BenchmarkData not allowed
        assert len(filtered.requests) == 10

    def test_filtered_metadata_preserved(self):
        """Filtered data should keep same instance config."""
        data = _make_data(100)
        filt = WarmupFilter(warmup_seconds=5.0)
        filtered, _ = filt.filter(data)

        assert filtered.metadata.num_prefill_instances == 2
        assert filtered.metadata.num_decode_instances == 6
        assert filtered.metadata.total_instances == 8

    def test_invalid_warmup_seconds(self):
        """Negative warmup_seconds should raise."""
        import pytest
        with pytest.raises(ValueError):
            WarmupFilter(warmup_seconds=-1.0)

    def test_invalid_warmup_factor(self):
        """Non-positive warmup_factor should raise."""
        import pytest
        with pytest.raises(ValueError):
            WarmupFilter(warmup_factor=0.0)

    def test_invalid_window_size(self):
        """Non-positive window_size should raise."""
        import pytest
        with pytest.raises(ValueError):
            WarmupFilter(window_size_s=0.0)

    def test_model_dump_serializable(self):
        """Report should be JSON-serializable."""
        data = _make_data(100, warmup_count=10, warmup_latency_factor=5.0)
        filt = WarmupFilter(warmup_seconds=5.0)
        _, report = filt.filter(data)

        dumped = report.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert "warmup_detected" in parsed

    def test_single_request(self):
        """Should handle single-request benchmark."""
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=1.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="req-0000",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=1000.0,
                ),
            ],
        )
        filt = WarmupFilter()
        filtered, report = filt.filter(data)
        assert report.filtered_request_count in (0, 1)

    def test_warmup_window_model(self):
        """WarmupWindow should have correct fields."""
        w = WarmupWindow(
            start_timestamp=1000.0,
            end_timestamp=1010.0,
            duration_s=10.0,
            requests_excluded=5,
        )
        assert w.duration_s == 10.0
        assert w.requests_excluded == 5

    def test_latency_comparison_model(self):
        """LatencyComparison should store values."""
        lc = LatencyComparison(
            before_p50_ms=100.0,
            before_p95_ms=200.0,
            before_p99_ms=300.0,
            after_p50_ms=80.0,
            after_p95_ms=150.0,
            after_p99_ms=200.0,
        )
        assert lc.before_p95_ms == 200.0
        assert lc.after_p95_ms == 150.0


class TestFilterWarmup:
    def test_programmatic_api(self, tmp_path):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = filter_warmup(str(path), warmup_seconds=5.0)

        assert isinstance(result, dict)
        assert "report" in result
        assert "filtered_data" in result
        assert result["report"]["warmup_detected"]

    def test_programmatic_api_auto_detect(self, tmp_path):
        data = _make_data(200, warmup_count=20, warmup_latency_factor=10.0)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = filter_warmup(str(path), window_size_s=5.0)
        assert isinstance(result, dict)


class TestCLIWarmupFilter:
    def test_cli_table_output(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "warmup-filter",
            "--benchmark", str(path),
            "--warmup-seconds", "5.0",
        ]):
            main()

        captured = capsys.readouterr()
        assert "Warm-up detected" in captured.out

    def test_cli_json_output(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "warmup-filter",
            "--benchmark", str(path),
            "--warmup-seconds", "5.0",
            "--output-format", "json",
        ]):
            main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "warmup_detected" in result

    def test_cli_output_file(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))
        out_path = tmp_path / "filtered.json"

        with patch("sys.argv", [
            "xpyd-plan", "warmup-filter",
            "--benchmark", str(path),
            "--warmup-seconds", "5.0",
            "--output", str(out_path),
        ]):
            main()

        assert out_path.exists()
        filtered = json.loads(out_path.read_text())
        assert "requests" in filtered
        assert "metadata" in filtered

    def test_cli_no_warmup(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", [
            "xpyd-plan", "warmup-filter",
            "--benchmark", str(path),
        ]):
            main()

        captured = capsys.readouterr()
        assert "No warm-up" in captured.out or "Warm-up" in captured.out
