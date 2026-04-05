"""Tests for variance decomposition analysis."""

from __future__ import annotations

import json
import subprocess
import tempfile

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.variance_decomp import (
    ComponentSignificance,
    VarianceDecomposer,
    VarianceReport,
    decompose_variance,
)


def _make_data(requests: list[dict]) -> BenchmarkData:
    """Build BenchmarkData from simplified request dicts."""
    reqs = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=r["prompt_tokens"],
            output_tokens=r["output_tokens"],
            ttft_ms=r["ttft_ms"],
            tpot_ms=r["tpot_ms"],
            total_latency_ms=r["total_latency_ms"],
            timestamp=r.get("timestamp", 1000.0 + i),
        )
        for i, r in enumerate(requests)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=reqs,
    )


def _make_simple_data(n: int = 50) -> BenchmarkData:
    """Generate data where prompt_tokens strongly drives ttft_ms."""
    requests = []
    for i in range(n):
        pt = 100 + i * 20
        ot = 50 + (i % 10) * 5
        requests.append({
            "prompt_tokens": pt,
            "output_tokens": ot,
            "ttft_ms": pt * 0.5 + 10,  # strongly correlated with prompt
            "tpot_ms": ot * 0.3 + 5,   # correlated with output
            "total_latency_ms": pt * 0.5 + ot * 0.3 + 15,
            "timestamp": 1000.0 + i * 2.0,
        })
    return _make_data(requests)


class TestVarianceDecomposer:
    """Tests for VarianceDecomposer."""

    def test_basic_decomposition(self):
        data = _make_simple_data()
        decomposer = VarianceDecomposer()
        report = decomposer.decompose(data)

        assert isinstance(report, VarianceReport)
        assert report.sample_size == 50
        assert len(report.metrics) == 3  # ttft, tpot, total

    def test_metric_names(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        metric_names = [m.metric for m in report.metrics]
        assert metric_names == ["ttft_ms", "tpot_ms", "total_latency_ms"]

    def test_components_per_metric(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            names = [c.name for c in m.components]
            assert "prompt_tokens" in names
            assert "output_tokens" in names
            assert "temporal" in names
            assert "residual" in names
            assert len(m.components) == 4

    def test_contributions_sum_roughly_100(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            total_pct = sum(c.contribution_pct for c in m.components)
            assert 99.0 <= total_pct <= 101.0, f"{m.metric}: {total_pct}"

    def test_prompt_dominant_for_ttft(self):
        """TTFT is driven by prompt_tokens, so it should be dominant or significant."""
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        ttft = report.metrics[0]
        assert ttft.metric == "ttft_ms"
        prompt_comp = next(c for c in ttft.components if c.name == "prompt_tokens")
        assert prompt_comp.contribution_pct > 30  # should be high

    def test_r_squared_range(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            assert 0 <= m.r_squared <= 1

    def test_minimum_requests_raises(self):
        data = _make_data([
            {"prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 50, "tpot_ms": 15, "total_latency_ms": 65},
            {"prompt_tokens": 200, "output_tokens": 60,
             "ttft_ms": 100, "tpot_ms": 18, "total_latency_ms": 118},
        ])
        with pytest.raises(ValueError, match="at least 3"):
            VarianceDecomposer().decompose(data)

    def test_three_requests_works(self):
        data = _make_data([
            {"prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 50, "tpot_ms": 15, "total_latency_ms": 65},
            {"prompt_tokens": 200, "output_tokens": 60,
             "ttft_ms": 100, "tpot_ms": 18, "total_latency_ms": 118},
            {"prompt_tokens": 300, "output_tokens": 70,
             "ttft_ms": 150, "tpot_ms": 21, "total_latency_ms": 171},
        ])
        report = VarianceDecomposer().decompose(data)
        assert report.sample_size == 3

    def test_identical_values_zero_variance(self):
        """All same latency values should yield zero variance."""
        data = _make_data([
            {
                "prompt_tokens": 100 + i, "output_tokens": 50,
                "ttft_ms": 50, "tpot_ms": 15, "total_latency_ms": 65,
            }
            for i in range(10)
        ])
        report = VarianceDecomposer().decompose(data)
        tpot = next(m for m in report.metrics if m.metric == "tpot_ms")
        assert tpot.total_variance == 0.0

    def test_significance_classification(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            for c in m.components:
                assert isinstance(c.significance, ComponentSignificance)

    def test_dominant_factor_set_when_significant(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        # At least one metric should have a dominant factor
        has_dominant = any(m.dominant_factor is not None for m in report.metrics)
        assert has_dominant

    def test_recommendation_not_empty(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        assert len(report.recommendation) > 0

    def test_temporal_bins_parameter(self):
        data = _make_simple_data()
        report5 = VarianceDecomposer(n_temporal_bins=5).decompose(data)
        report20 = VarianceDecomposer(n_temporal_bins=20).decompose(data)
        # Both should produce valid reports
        assert report5.sample_size == report20.sample_size == 50

    def test_sum_of_squares_non_negative(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            for c in m.components:
                assert c.sum_of_squares >= 0

    def test_total_variance_non_negative(self):
        data = _make_simple_data()
        report = VarianceDecomposer().decompose(data)
        for m in report.metrics:
            assert m.total_variance >= 0

    def test_temporal_dominant_with_time_pattern(self):
        """When latency increases with time, temporal should explain variance."""
        requests = []
        for i in range(50):
            # Latency grows purely with time, not tokens
            requests.append({
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 50 + i * 5,
                "tpot_ms": 15 + i * 2,
                "total_latency_ms": 65 + i * 7,
                "timestamp": 1000.0 + i * 10.0,
            })
        data = _make_data(requests)
        report = VarianceDecomposer().decompose(data)
        ttft = report.metrics[0]
        temporal = next(c for c in ttft.components if c.name == "temporal")
        # Temporal should have meaningful contribution since latency grows with time
        assert temporal.contribution_pct > 10


class TestDecomposeVarianceAPI:
    """Tests for the programmatic decompose_variance() API."""

    def test_returns_dict(self):
        data = _make_simple_data()
        result = decompose_variance(data)
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "sample_size" in result
        assert "recommendation" in result

    def test_n_temporal_bins_kwarg(self):
        data = _make_simple_data()
        result = decompose_variance(data, n_temporal_bins=5)
        assert result["sample_size"] == 50

    def test_json_serializable(self):
        data = _make_simple_data()
        result = decompose_variance(data)
        # Should not raise
        json.dumps(result)


class TestVarianceCLI:
    """Tests for the CLI variance subcommand."""

    def _write_benchmark(self, tmp_dir: str, data: BenchmarkData) -> str:
        import os
        path = os.path.join(tmp_dir, "bench.json")
        with open(path, "w") as f:
            json.dump(data.model_dump(), f)
        return path

    def test_cli_json_output(self):
        data = _make_simple_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_benchmark(tmp, data)
            result = subprocess.run(
                ["xpyd-plan", "variance", "--benchmark", path, "--output-format", "json"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            output = json.loads(result.stdout)
            assert "metrics" in output
            assert len(output["metrics"]) == 3

    def test_cli_table_output(self):
        data = _make_simple_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_benchmark(tmp, data)
            result = subprocess.run(
                ["xpyd-plan", "variance", "--benchmark", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "Variance Decomposition" in result.stdout
