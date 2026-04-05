"""Tests for latency jitter analysis (M77)."""

from __future__ import annotations

import json
import sys
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.jitter import (
    JitterAnalyzer,
    JitterClassification,
    JitterReport,
    _classify_jitter,
    _compute_jitter,
    _percentile,
    analyze_jitter,
)


def _make_data(
    n: int = 50,
    ttft_base: float = 100.0,
    ttft_jitter: float = 5.0,
    tpot_base: float = 20.0,
    tpot_jitter: float = 2.0,
) -> BenchmarkData:
    """Generate benchmark data with controlled jitter."""
    import random

    rng = random.Random(42)
    requests = []
    for i in range(n):
        ttft = ttft_base + rng.uniform(-ttft_jitter, ttft_jitter)
        tpot = tpot_base + rng.uniform(-tpot_jitter, tpot_jitter)
        total = ttft + tpot * 50  # approximate
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
            total_instances=4, measured_qps=100.0),
        requests=requests,
    )


def _make_high_jitter_data(n: int = 50) -> BenchmarkData:
    """Generate data with high jitter."""
    import random

    rng = random.Random(99)
    requests = []
    for i in range(n):
        # Alternating low and high values -> high CV
        if i % 2 == 0:
            ttft = 50.0 + rng.uniform(0, 10)
        else:
            ttft = 500.0 + rng.uniform(0, 100)
        tpot = 10.0 + rng.uniform(0, 2)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=round(ttft, 2),
                tpot_ms=round(tpot, 2),
                total_latency_ms=round(ttft + tpot * 50, 2),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4, measured_qps=100.0),
        requests=requests,
    )


class TestClassifyJitter:
    def test_stable(self):
        assert _classify_jitter(0.1) == JitterClassification.STABLE
        assert _classify_jitter(0.3) == JitterClassification.STABLE

    def test_moderate(self):
        assert _classify_jitter(0.31) == JitterClassification.MODERATE
        assert _classify_jitter(0.7) == JitterClassification.MODERATE

    def test_high(self):
        assert _classify_jitter(0.71) == JitterClassification.HIGH
        assert _classify_jitter(1.5) == JitterClassification.HIGH


class TestPercentile:
    def test_single(self):
        assert _percentile([5.0], 50.0) == 5.0

    def test_median(self):
        assert _percentile([1.0, 2.0, 3.0], 50.0) == 2.0

    def test_p95(self):
        vals = list(range(1, 101))
        p95 = _percentile([float(v) for v in vals], 95.0)
        assert 94.0 < p95 < 96.0

    def test_empty(self):
        assert _percentile([], 50.0) == 0.0


class TestComputeJitter:
    def test_constant_values(self):
        stats = _compute_jitter([10.0, 10.0, 10.0, 10.0])
        assert stats.consecutive_mean == 0.0
        assert stats.std == 0.0
        assert stats.cv == 0.0
        assert stats.classification == JitterClassification.STABLE

    def test_varying_values(self):
        stats = _compute_jitter([10.0, 20.0, 10.0, 20.0])
        assert stats.consecutive_mean == 10.0
        assert stats.std > 0
        assert stats.cv > 0


class TestJitterAnalyzer:
    def test_basic_analysis(self):
        data = _make_data()
        analyzer = JitterAnalyzer()
        report = analyzer.analyze(data)

        assert isinstance(report, JitterReport)
        assert len(report.metrics) == 3
        assert report.worst_metric in ("ttft_ms", "tpot_ms", "total_latency_ms")

    def test_metric_names(self):
        data = _make_data()
        report = JitterAnalyzer().analyze(data)
        names = [m.metric for m in report.metrics]
        assert names == ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def test_sample_size(self):
        data = _make_data(n=30)
        report = JitterAnalyzer().analyze(data)
        for m in report.metrics:
            assert m.sample_size == 30

    def test_stable_data(self):
        data = _make_data(ttft_jitter=1.0, tpot_jitter=0.5)
        report = JitterAnalyzer().analyze(data)
        for m in report.metrics:
            assert m.stats.classification == JitterClassification.STABLE

    def test_high_jitter_data(self):
        data = _make_high_jitter_data()
        report = JitterAnalyzer().analyze(data)
        # TTFT should have high jitter due to alternating pattern
        ttft_metric = next(m for m in report.metrics if m.metric == "ttft_ms")
        assert ttft_metric.stats.classification == JitterClassification.HIGH

    def test_recommendation_high(self):
        data = _make_high_jitter_data()
        report = JitterAnalyzer().analyze(data)
        rec = report.recommendation.lower()
        assert "jitter" in rec or "unpredictable" in rec

    def test_recommendation_stable(self):
        data = _make_data(ttft_jitter=1.0, tpot_jitter=0.5)
        report = JitterAnalyzer().analyze(data)
        rec = report.recommendation.lower()
        assert "low" in rec or "predictable" in rec

    def test_too_few_requests(self):
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2, measured_qps=10.0),
            requests=[
                BenchmarkRequest(
                    request_id="req-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=100.0,
                    tpot_ms=20.0,
                    total_latency_ms=1100.0,
                    timestamp=1000.0,
                )
            ],
        )
        with pytest.raises(ValueError, match="at least 2"):
            JitterAnalyzer().analyze(data)


class TestAnalyzeJitterAPI:
    def test_returns_dict(self):
        data = _make_data()
        result = analyze_jitter(data)
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "worst_metric" in result
        assert "recommendation" in result

    def test_json_serializable(self):
        data = _make_data()
        result = analyze_jitter(data)
        json_str = json.dumps(result)
        assert json_str


class TestJitterModels:
    def test_report_serializable(self):
        data = _make_data()
        report = JitterAnalyzer().analyze(data)
        d = report.model_dump()
        roundtrip = JitterReport.model_validate(d)
        assert roundtrip.worst_metric == report.worst_metric

    def test_json_roundtrip(self):
        data = _make_data()
        report = JitterAnalyzer().analyze(data)
        json_str = report.model_dump_json()
        loaded = JitterReport.model_validate_json(json_str)
        assert len(loaded.metrics) == 3


class TestJitterCLI:
    def _write_benchmark(self) -> str:
        data = _make_data()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write(data.model_dump_json())
        f.close()
        return f.name

    def test_cli_table(self):
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        bench = self._write_benchmark()
        sys.argv = ["xpyd-plan", "jitter", "--benchmark", bench]
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                main()
            except SystemExit:
                pass
        # Rich output goes to console, not stdout; just check no crash

    def test_cli_json(self):
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        bench = self._write_benchmark()
        sys.argv = ["xpyd-plan", "jitter", "--benchmark", bench, "--output-format", "json"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                main()
            except SystemExit:
                pass
        output = buf.getvalue()
        parsed = json.loads(output)
        assert "metrics" in parsed
        assert len(parsed["metrics"]) == 3

    def test_cli_json_parseable(self):
        import io
        from contextlib import redirect_stdout

        from xpyd_plan.cli._main import main

        bench = self._write_benchmark()
        sys.argv = ["xpyd-plan", "jitter", "--benchmark", bench, "--output-format", "json"]
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                main()
            except SystemExit:
                pass
        data = json.loads(buf.getvalue())
        assert data["worst_metric"] in ("ttft_ms", "tpot_ms", "total_latency_ms")
