"""Tests for throughput scaling analysis."""

from __future__ import annotations

import json
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.scaling import ScalingAnalyzer, analyze_scaling


def _make_benchmark(
    num_prefill: int,
    num_decode: int,
    qps: float,
    num_requests: int = 50,
) -> BenchmarkData:
    """Create a benchmark dataset with given configuration."""
    total = num_prefill + num_decode
    requests = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100,
            output_tokens=50,
            ttft_ms=20.0,
            tpot_ms=10.0,
            total_latency_ms=70.0,
            timestamp=1000.0 + i,
        )
        for i in range(num_requests)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=total,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestScalingAnalyzer:
    """Tests for ScalingAnalyzer."""

    def test_basic_linear_scaling(self) -> None:
        """Perfect linear scaling should have 100% efficiency everywhere."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),   # 2 instances, 10 QPS
            _make_benchmark(2, 2, 20.0),   # 4 instances, 20 QPS
            _make_benchmark(3, 3, 30.0),   # 6 instances, 30 QPS
        ]
        analyzer = ScalingAnalyzer()
        report = analyzer.analyze(benchmarks)

        assert len(report.curve.points) == 3
        for point in report.curve.points:
            assert point.scaling_efficiency == 1.0
        assert report.curve.knee_point is None

    def test_diminishing_returns(self) -> None:
        """Scaling that degrades should detect knee point."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),   # 2 inst, 5.0/inst
            _make_benchmark(2, 2, 18.0),   # 4 inst, 4.5/inst, eff=0.9
            _make_benchmark(3, 3, 21.0),   # 6 inst, 3.5/inst, eff=0.7
        ]
        analyzer = ScalingAnalyzer(knee_threshold=0.8)
        report = analyzer.analyze(benchmarks)

        assert report.curve.knee_point is not None
        assert report.curve.knee_point.total_instances == 6
        assert report.curve.optimal_point.total_instances == 4

    def test_all_below_threshold(self) -> None:
        """When all points are below threshold, pick best efficiency."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),   # baseline
            _make_benchmark(2, 2, 14.0),   # eff=0.7
            _make_benchmark(3, 3, 15.0),   # eff=0.5
        ]
        analyzer = ScalingAnalyzer(knee_threshold=0.9)
        report = analyzer.analyze(benchmarks)

        # Baseline itself has eff=1.0 which is above threshold
        # So optimal is baseline
        assert report.curve.optimal_point.total_instances == 2

    def test_minimum_two_benchmarks(self) -> None:
        """Need at least 2 benchmarks."""
        with pytest.raises(ValueError, match="at least 2"):
            ScalingAnalyzer().analyze([_make_benchmark(1, 1, 10.0)])

    def test_same_instance_count_error(self) -> None:
        """Same instance count in all benchmarks is an error."""
        with pytest.raises(ValueError, match="distinct"):
            ScalingAnalyzer().analyze([
                _make_benchmark(1, 1, 10.0),
                _make_benchmark(1, 1, 12.0),
            ])

    def test_invalid_knee_threshold(self) -> None:
        """Knee threshold must be between 0 and 1 exclusive."""
        with pytest.raises(ValueError):
            ScalingAnalyzer(knee_threshold=0.0)
        with pytest.raises(ValueError):
            ScalingAnalyzer(knee_threshold=1.0)
        with pytest.raises(ValueError):
            ScalingAnalyzer(knee_threshold=-0.1)

    def test_sorted_by_instances(self) -> None:
        """Points should be sorted by instance count regardless of input order."""
        benchmarks = [
            _make_benchmark(3, 3, 30.0),
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        instances = [p.total_instances for p in report.curve.points]
        assert instances == [2, 4, 6]

    def test_per_instance_qps(self) -> None:
        """Per-instance QPS should be correctly calculated."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 3, 25.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        assert report.curve.points[0].per_instance_qps == 5.0
        assert report.curve.points[1].per_instance_qps == 5.0

    def test_baseline_per_instance_qps(self) -> None:
        """Baseline per-instance QPS should come from smallest config."""
        benchmarks = [
            _make_benchmark(2, 2, 20.0),
            _make_benchmark(1, 1, 12.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        assert report.curve.baseline_per_instance_qps == 6.0

    def test_recommendation_no_knee(self) -> None:
        """Recommendation when no knee point exists."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        assert "efficient" in report.recommendation.lower()

    def test_recommendation_with_knee(self) -> None:
        """Recommendation when knee point exists."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 18.0),
            _make_benchmark(3, 3, 21.0),
        ]
        report = ScalingAnalyzer(knee_threshold=0.8).analyze(benchmarks)
        assert "diminishing" in report.recommendation.lower()

    def test_analyze_scaling_api(self) -> None:
        """Programmatic API returns dict."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        result = analyze_scaling(benchmarks)
        assert isinstance(result, dict)
        assert "curve" in result
        assert "recommendation" in result

    def test_asymmetric_pd_ratios(self) -> None:
        """Handle benchmarks with different P:D ratios."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(1, 3, 18.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        assert report.curve.points[0].num_prefill == 1
        assert report.curve.points[1].num_decode == 3

    def test_many_points(self) -> None:
        """Handle many scaling points."""
        benchmarks = []
        for i in range(1, 11):
            # Simulate sub-linear scaling
            qps = 10.0 * i * (1 - 0.02 * i)
            benchmarks.append(_make_benchmark(i, i, qps))
        report = ScalingAnalyzer(knee_threshold=0.8).analyze(benchmarks)
        assert len(report.curve.points) == 10

    def test_scaling_efficiency_clamped(self) -> None:
        """Efficiency should be clamped to [0, 1]."""
        # Second config has BETTER per-instance QPS than baseline (super-linear)
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 25.0),  # super-linear
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        for p in report.curve.points:
            assert 0.0 <= p.scaling_efficiency <= 1.0

    def test_model_dump(self) -> None:
        """ScalingReport should be serializable."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 18.0),
            _make_benchmark(3, 3, 21.0),
        ]
        report = ScalingAnalyzer(knee_threshold=0.8).analyze(benchmarks)
        data = report.model_dump()
        assert isinstance(data, dict)
        assert data["curve"]["knee_threshold"] == 0.8

    def test_two_benchmarks_minimum(self) -> None:
        """Exactly 2 benchmarks should work."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 18.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        assert len(report.curve.points) == 2

    def test_custom_knee_threshold(self) -> None:
        """Custom knee threshold changes knee detection."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 18.0),   # eff=0.9
            _make_benchmark(3, 3, 21.0),   # eff=0.7
        ]
        # With 0.95 threshold, knee should be at 4 instances (eff=0.9 < 0.95)
        report = ScalingAnalyzer(knee_threshold=0.95).analyze(benchmarks)
        assert report.curve.knee_point is not None
        assert report.curve.knee_point.total_instances == 4

    def test_json_roundtrip(self) -> None:
        """Report can be serialized and deserialized."""
        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]
        report = ScalingAnalyzer().analyze(benchmarks)
        data = json.loads(json.dumps(report.model_dump()))
        assert data["curve"]["points"][0]["total_instances"] == 2


class TestScalingCLI:
    """Tests for scaling CLI subcommand."""

    def test_cli_scaling_table(self) -> None:
        """CLI scaling command produces output."""
        from xpyd_plan.cli._main import main

        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]

        files = []
        for i, bench in enumerate(benchmarks):
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix=f"bench{i}_"
            )
            json.dump(bench.model_dump(), f)
            f.close()
            files.append(f.name)

        # Should not raise
        try:
            main(["scaling", "--benchmark", *files])
        except SystemExit:
            pass  # OK if it exits cleanly

    def test_cli_scaling_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI scaling JSON output is valid JSON."""
        from xpyd_plan.cli._main import main

        benchmarks = [
            _make_benchmark(1, 1, 10.0),
            _make_benchmark(2, 2, 20.0),
        ]

        files = []
        for i, bench in enumerate(benchmarks):
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix=f"bench{i}_"
            )
            json.dump(bench.model_dump(), f)
            f.close()
            files.append(f.name)

        main(["scaling", "--benchmark", *files, "--output-format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "curve" in data
