"""Tests for variance decomposition."""

from __future__ import annotations

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.variance_decomp import (
    MetricDecomposition,
    Significance,
    VarianceComponent,
    VarianceDecomposer,
    VarianceReport,
    decompose_variance,
)


def _make_data(requests: list[dict]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=[BenchmarkRequest(**r) for r in requests],
    )


def _simple_requests(n: int = 50) -> list[dict]:
    """Generate requests where TTFT correlates strongly with prompt_tokens."""
    reqs = []
    for i in range(n):
        pt = 100 + i * 10
        ot = 50 + (i % 5) * 20
        reqs.append({
            "request_id": f"req-{i}",
            "prompt_tokens": pt,
            "output_tokens": ot,
            "ttft_ms": pt * 0.5 + 10,  # strong correlation with prompt_tokens
            "tpot_ms": ot * 0.3 + 5,   # moderate correlation with output_tokens
            "total_latency_ms": pt * 0.5 + ot * 0.3 + 15,
            "timestamp": 1000.0 + i * 2.0,
        })
    return reqs


class TestVarianceDecomposer:
    def test_basic_analysis(self) -> None:
        data = _make_data(_simple_requests(50))
        decomposer = VarianceDecomposer()
        report = decomposer.analyze(data)

        assert report.total_requests == 50
        assert len(report.decompositions) == 3

    def test_metric_names(self) -> None:
        data = _make_data(_simple_requests(20))
        report = VarianceDecomposer().analyze(data)
        names = {d.metric for d in report.decompositions}
        assert names == {"ttft", "tpot", "total_latency"}

    def test_components_sum_to_100(self) -> None:
        data = _make_data(_simple_requests(50))
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            total_pct = sum(c.contribution_pct for c in decomp.components)
            assert abs(total_pct - 100.0) < 1.0, f"{decomp.metric}: components sum to {total_pct}"

    def test_four_components(self) -> None:
        data = _make_data(_simple_requests(30))
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            factors = [c.factor for c in decomp.components]
            assert factors == ["prompt_tokens", "output_tokens", "temporal", "residual"]

    def test_ttft_dominated_by_prompt(self) -> None:
        """TTFT = prompt_tokens * 0.5 + 10, so prompt_tokens should dominate."""
        data = _make_data(_simple_requests(50))
        report = VarianceDecomposer().analyze(data)
        ttft_decomp = next(d for d in report.decompositions if d.metric == "ttft")
        assert ttft_decomp.dominant_factor == "prompt_tokens"

    def test_r_squared_range(self) -> None:
        data = _make_data(_simple_requests(50))
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            assert 0.0 <= decomp.r_squared <= 1.0

    def test_significance_classification(self) -> None:
        assert Significance.HIGH.value == "HIGH"
        assert Significance.NEGLIGIBLE.value == "NEGLIGIBLE"

    def test_too_few_requests(self) -> None:
        data = _make_data([
            {"request_id": "r1", "prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 60, "tpot_ms": 20, "total_latency_ms": 80, "timestamp": 1000.0},
            {"request_id": "r2", "prompt_tokens": 200, "output_tokens": 100,
             "ttft_ms": 110, "tpot_ms": 35, "total_latency_ms": 145, "timestamp": 1002.0},
        ])
        with pytest.raises(ValueError, match="At least 3 requests"):
            VarianceDecomposer().analyze(data)

    def test_constant_latency(self) -> None:
        """All values identical — zero variance case."""
        reqs = [
            {"request_id": f"r{i}", "prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 50.0, "tpot_ms": 20.0, "total_latency_ms": 70.0, "timestamp": 1000.0 + i}
            for i in range(10)
        ]
        data = _make_data(reqs)
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            assert decomp.total_variance == 0
            assert decomp.r_squared == 0

    def test_temporal_bins_parameter(self) -> None:
        data = _make_data(_simple_requests(30))
        report = VarianceDecomposer(temporal_bins=5).analyze(data)
        assert report.total_requests == 30

    def test_recommendations_generated(self) -> None:
        data = _make_data(_simple_requests(50))
        report = VarianceDecomposer().analyze(data)
        assert len(report.recommendations) > 0

    def test_recommendations_mention_dominant(self) -> None:
        data = _make_data(_simple_requests(50))
        report = VarianceDecomposer().analyze(data)
        # TTFT dominated by prompt_tokens — should have a recommendation
        prompt_recs = [r for r in report.recommendations if "prompt" in r.lower()]
        assert len(prompt_recs) > 0


class TestVarianceModels:
    def test_variance_component_model(self) -> None:
        comp = VarianceComponent(
            factor="prompt_tokens",
            sum_of_squares=1234.5,
            contribution_pct=45.2,
            significance=Significance.HIGH,
        )
        assert comp.factor == "prompt_tokens"
        assert comp.contribution_pct == 45.2

    def test_metric_decomposition_model(self) -> None:
        decomp = MetricDecomposition(
            metric="ttft",
            total_variance=5000.0,
            components=[],
            dominant_factor="prompt_tokens",
            r_squared=0.85,
        )
        assert decomp.r_squared == 0.85

    def test_variance_report_model(self) -> None:
        report = VarianceReport(
            total_requests=100,
            decompositions=[],
            recommendations=["test recommendation"],
        )
        assert report.total_requests == 100

    def test_json_roundtrip(self) -> None:
        data = _make_data(_simple_requests(20))
        report = VarianceDecomposer().analyze(data)
        d = report.model_dump()
        restored = VarianceReport.model_validate(d)
        assert restored.total_requests == report.total_requests
        assert len(restored.decompositions) == len(report.decompositions)


class TestDecomposeVarianceAPI:
    def test_programmatic_api(self) -> None:
        data = _make_data(_simple_requests(30))
        result = decompose_variance(data)
        assert isinstance(result, dict)
        assert result["total_requests"] == 30
        assert "decompositions" in result
        assert "recommendations" in result

    def test_programmatic_api_temporal_bins(self) -> None:
        data = _make_data(_simple_requests(30))
        result = decompose_variance(data, temporal_bins=5)
        assert result["total_requests"] == 30


class TestEdgeCases:
    def test_same_timestamps(self) -> None:
        """All requests at same timestamp — temporal should contribute nothing."""
        reqs = [
            {"request_id": f"r{i}", "prompt_tokens": 100 + i * 10, "output_tokens": 50 + i * 5,
             "ttft_ms": 50.0 + i * 5, "tpot_ms": 20.0 + i * 2, "total_latency_ms": 70.0 + i * 7,
             "timestamp": 1000.0}
            for i in range(20)
        ]
        data = _make_data(reqs)
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            temporal_comp = next(c for c in decomp.components if c.factor == "temporal")
            assert temporal_comp.contribution_pct == 0.0

    def test_same_tokens_different_latency(self) -> None:
        """All requests have same tokens but different latency — residual should dominate."""
        reqs = [
            {"request_id": f"r{i}", "prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 50.0 + i * 3, "tpot_ms": 20.0 + i * 2, "total_latency_ms": 70.0 + i * 5,
             "timestamp": 1000.0 + i * 2}
            for i in range(20)
        ]
        data = _make_data(reqs)
        report = VarianceDecomposer().analyze(data)
        for decomp in report.decompositions:
            # prompt and output should explain nothing since they're constant
            prompt_comp = next(c for c in decomp.components if c.factor == "prompt_tokens")
            output_comp = next(c for c in decomp.components if c.factor == "output_tokens")
            assert prompt_comp.contribution_pct == 0.0
            assert output_comp.contribution_pct == 0.0

    def test_minimal_requests(self) -> None:
        """Exactly 3 requests — minimum viable."""
        reqs = [
            {"request_id": "r0", "prompt_tokens": 100, "output_tokens": 50,
             "ttft_ms": 50, "tpot_ms": 20, "total_latency_ms": 70, "timestamp": 1000.0},
            {"request_id": "r1", "prompt_tokens": 200, "output_tokens": 100,
             "ttft_ms": 100, "tpot_ms": 35, "total_latency_ms": 135, "timestamp": 1002.0},
            {"request_id": "r2", "prompt_tokens": 300, "output_tokens": 150,
             "ttft_ms": 150, "tpot_ms": 50, "total_latency_ms": 200, "timestamp": 1004.0},
        ]
        data = _make_data(reqs)
        report = VarianceDecomposer().analyze(data)
        assert report.total_requests == 3
