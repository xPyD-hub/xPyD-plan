"""Tests for percentile convergence analysis."""

from __future__ import annotations

import json
import random
import tempfile

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.convergence import (
    ConvergenceAnalyzer,
    ConvergencePoint,
    ConvergenceReport,
    MetricConvergence,
    StabilityStatus,
    analyze_convergence,
)


def _make_requests(
    n: int = 200,
    *,
    seed: int = 42,
    stable: bool = True,
) -> list[BenchmarkRequest]:
    """Generate synthetic requests.

    If stable=True, latencies are drawn from a tight distribution.
    If stable=False, latencies shift significantly across the sequence.
    """
    rng = random.Random(seed)
    requests = []
    for i in range(n):
        prompt_tokens = rng.randint(50, 500)
        output_tokens = rng.randint(20, 200)
        if stable:
            ttft = 50.0 + rng.gauss(0, 5)
            tpot = 10.0 + rng.gauss(0, 1)
        else:
            # Latency drifts significantly — early requests are fast, late are slow
            drift = (i / n) * 200  # 0 to 200ms drift
            ttft = 50.0 + drift + rng.gauss(0, 5)
            tpot = 10.0 + drift / 20 + rng.gauss(0, 1)
        total = max(ttft + tpot * output_tokens, 1.0)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=round(max(ttft, 0.1), 2),
                tpot_ms=round(max(tpot, 0.1), 2),
                total_latency_ms=round(total, 2),
                timestamp=1700000000.0 + i,
            )
        )
    return requests


def _make_benchmark(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=4,
            total_instances=6,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer."""

    def test_stable_data_reports_stable(self):
        data = _make_benchmark(_make_requests(500, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        assert report.overall_status == StabilityStatus.STABLE
        assert report.total_requests == 500

    def test_unstable_data_reports_not_stable(self):
        data = _make_benchmark(_make_requests(500, stable=False))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        assert report.overall_status in (StabilityStatus.MARGINAL, StabilityStatus.UNSTABLE)

    def test_report_has_three_metrics(self):
        data = _make_benchmark(_make_requests(100, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        assert len(report.metrics) == 3
        fields = {m.field for m in report.metrics}
        assert fields == {"ttft_ms", "tpot_ms", "total_latency_ms"}

    def test_convergence_points_count(self):
        data = _make_benchmark(_make_requests(100, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=10)
        for m in report.metrics:
            assert len(m.points) == 10
            # Last point should cover all requests
            assert m.points[-1].sample_size == 100
            assert m.points[-1].sample_fraction == 1.0

    def test_custom_steps(self):
        data = _make_benchmark(_make_requests(100, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=5)
        for m in report.metrics:
            assert len(m.points) == 5

    def test_custom_threshold(self):
        data = _make_benchmark(_make_requests(500, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        # Very tight threshold
        report_tight = analyzer.analyze(threshold=0.001)
        # Very loose threshold
        report_loose = analyzer.analyze(threshold=0.5)
        # Loose should be at least as stable as tight
        order = [StabilityStatus.STABLE, StabilityStatus.MARGINAL, StabilityStatus.UNSTABLE]
        assert order.index(report_loose.overall_status) <= order.index(report_tight.overall_status)

    def test_cv_values_are_nonnegative(self):
        data = _make_benchmark(_make_requests(200, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        for m in report.metrics:
            assert m.cv_p95 >= 0
            assert m.cv_p99 >= 0

    def test_sample_fractions_monotonic(self):
        data = _make_benchmark(_make_requests(200, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=10)
        for m in report.metrics:
            fractions = [p.sample_fraction for p in m.points]
            assert fractions == sorted(fractions)

    def test_sample_sizes_monotonic(self):
        data = _make_benchmark(_make_requests(200, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=10)
        for m in report.metrics:
            sizes = [p.sample_size for p in m.points]
            assert sizes == sorted(sizes)

    def test_min_stable_sample_size_for_stable_data(self):
        data = _make_benchmark(_make_requests(500, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        for m in report.metrics:
            if m.status == StabilityStatus.STABLE:
                # Should have a min stable size
                assert m.min_stable_sample_size is not None
                assert m.min_stable_sample_size > 0
                assert m.min_stable_sample_size <= 500

    def test_recommended_min_requests_for_stable(self):
        data = _make_benchmark(_make_requests(500, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze()
        if report.overall_status == StabilityStatus.STABLE:
            assert report.recommended_min_requests is not None

    def test_small_dataset(self):
        data = _make_benchmark(_make_requests(5, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=5)
        assert report.total_requests == 5
        # Should still produce results without crashing

    def test_single_request(self):
        data = _make_benchmark(_make_requests(1, stable=True))
        analyzer = ConvergenceAnalyzer(data)
        report = analyzer.analyze(steps=5)
        assert report.total_requests == 1

    def test_convergence_point_model(self):
        p = ConvergencePoint(
            sample_size=100,
            sample_fraction=0.5,
            p50=50.0,
            p95=95.0,
            p99=99.0,
        )
        assert p.sample_size == 100
        assert p.p50 == 50.0

    def test_metric_convergence_model(self):
        mc = MetricConvergence(
            field="ttft_ms",
            points=[],
            cv_p95=0.03,
            cv_p99=0.04,
            status=StabilityStatus.STABLE,
            min_stable_sample_size=50,
        )
        assert mc.field == "ttft_ms"
        assert mc.status == StabilityStatus.STABLE

    def test_report_model(self):
        report = ConvergenceReport(
            metrics=[],
            overall_status=StabilityStatus.STABLE,
            total_requests=100,
            recommended_min_requests=50,
        )
        assert report.total_requests == 100

    def test_stability_status_enum(self):
        assert StabilityStatus.STABLE.value == "stable"
        assert StabilityStatus.MARGINAL.value == "marginal"
        assert StabilityStatus.UNSTABLE.value == "unstable"


class TestAnalyzeConvergenceAPI:
    """Tests for the programmatic API."""

    def test_returns_dict(self):
        data = _make_benchmark(_make_requests(100, stable=True))
        result = analyze_convergence(data)
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "overall_status" in result
        assert "total_requests" in result

    def test_custom_params(self):
        data = _make_benchmark(_make_requests(100, stable=True))
        result = analyze_convergence(data, steps=5, threshold=0.1)
        assert isinstance(result, dict)
        for m in result["metrics"]:
            assert len(m["points"]) == 5


class TestConvergenceCLI:
    """Tests for CLI integration."""

    def test_cli_json_output(self):
        import subprocess

        requests = _make_requests(100, stable=True)
        data = _make_benchmark(requests)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()
            result = subprocess.run(
                ["xpyd-plan", "convergence",
                 "--benchmark", f.name, "--output-format", "json"],
                capture_output=True, text=True,
            )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "metrics" in output
        assert "overall_status" in output

    def test_cli_table_output(self):
        import subprocess

        requests = _make_requests(100, stable=True)
        data = _make_benchmark(requests)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data.model_dump_json(indent=2))
            f.flush()
            result = subprocess.run(
                ["xpyd-plan", "convergence",
                 "--benchmark", f.name, "--output-format", "table"],
                capture_output=True, text=True,
            )
        assert result.returncode == 0
        out = result.stdout.lower()
        assert "convergence" in out or "stability" in out or "stable" in out
