"""Tests for benchmark normalization."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.normalizer import (
    BenchmarkNormalizer,
    GPUType,
    NormalizationConfig,
    normalize_benchmark,
)


def _make_data(
    n: int = 100,
    ttft_base: float = 50.0,
    tpot_base: float = 30.0,
    total_base: float = 200.0,
    qps: float = 10.0,
) -> BenchmarkData:
    """Create synthetic benchmark data."""
    import random

    random.seed(42)
    requests = []
    for i in range(n):
        noise = 1 + random.gauss(0, 0.1)
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + random.randint(0, 200),
                output_tokens=50 + random.randint(0, 100),
                ttft_ms=ttft_base * max(0.1, noise),
                tpot_ms=tpot_base * max(0.1, noise),
                total_latency_ms=total_base * max(0.1, noise),
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        requests=requests,
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
    )


class TestBenchmarkNormalizer:
    """Tests for BenchmarkNormalizer."""

    def test_same_gpu_no_change(self) -> None:
        data = _make_data()
        config = NormalizationConfig(source_gpu="A100-80G", target_gpu="A100-80G")
        report = BenchmarkNormalizer(config).normalize(data)
        assert report.scaling_factor == pytest.approx(1.0)
        assert report.ttft_stats.original_p50 == pytest.approx(report.ttft_stats.normalized_p50)
        assert report.normalized_qps == pytest.approx(report.original_qps)

    def test_h100_to_a100_scales_up(self) -> None:
        """H100 is faster, so normalizing to A100 should increase latencies."""
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="H100-80G", target_gpu="A100-80G")
        # H100 factor=1.8, A100 factor=1.0, scaling=1.8
        assert report.scaling_factor == pytest.approx(1.8)
        assert report.ttft_stats.normalized_p50 > report.ttft_stats.original_p50
        assert report.normalized_qps < report.original_qps

    def test_a100_to_h100_scales_down(self) -> None:
        """A100 is slower, so normalizing to H100 should decrease latencies."""
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="A100-80G", target_gpu="H100-80G")
        assert report.scaling_factor == pytest.approx(1.0 / 1.8, rel=1e-3)
        assert report.ttft_stats.normalized_p50 < report.ttft_stats.original_p50
        assert report.normalized_qps > report.original_qps

    def test_scaling_preserves_percentile_order(self) -> None:
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="H100-80G")
        assert report.ttft_stats.normalized_p50 <= report.ttft_stats.normalized_p95
        assert report.ttft_stats.normalized_p95 <= report.ttft_stats.normalized_p99

    def test_unknown_gpu_raises(self) -> None:
        data = _make_data()
        with pytest.raises(ValueError, match="Unknown GPU type"):
            normalize_benchmark(data, source_gpu="UNKNOWN-GPU")

    def test_custom_factors(self) -> None:
        data = _make_data()
        report = normalize_benchmark(
            data,
            source_gpu="MY-GPU",
            target_gpu="A100-80G",
            custom_factors={"MY-GPU": 2.5},
        )
        assert report.scaling_factor == pytest.approx(2.5)

    def test_normalize_data_returns_benchmark(self) -> None:
        data = _make_data(n=10)
        config = NormalizationConfig(source_gpu="H100-80G", target_gpu="A100-80G")
        normalizer = BenchmarkNormalizer(config)
        result = normalizer.normalize_data(data)
        assert isinstance(result, BenchmarkData)
        assert len(result.requests) == 10
        # Latencies should be scaled
        scaling = 1.8
        for orig, norm in zip(data.requests, result.requests):
            assert norm.ttft_ms == pytest.approx(orig.ttft_ms * scaling, rel=1e-6)
            assert norm.tpot_ms == pytest.approx(orig.tpot_ms * scaling, rel=1e-6)

    def test_normalize_data_preserves_tokens(self) -> None:
        data = _make_data(n=5)
        config = NormalizationConfig(source_gpu="H100-80G", target_gpu="A100-80G")
        result = BenchmarkNormalizer(config).normalize_data(data)
        for orig, norm in zip(data.requests, result.requests):
            assert norm.prompt_tokens == orig.prompt_tokens
            assert norm.output_tokens == orig.output_tokens

    def test_empty_data_raises(self) -> None:
        """BenchmarkData enforces min_length=1, so empty data can't be created."""
        with pytest.raises(Exception):
            BenchmarkData(
                requests=[],
                metadata=BenchmarkMetadata(
                    num_prefill_instances=1,
                    num_decode_instances=1,
                    total_instances=2,
                    measured_qps=1.0,
                ),
            )

    def test_all_known_gpus_in_report(self) -> None:
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="A100-80G")
        gpu_names = {g.gpu_type for g in report.known_gpus}
        assert "A100-80G" in gpu_names
        assert "H100-80G" in gpu_names
        assert "H200-141G" in gpu_names

    def test_gpu_type_enum(self) -> None:
        assert GPUType.A100_80G.value == "A100-80G"
        assert GPUType.H100_80G.value == "H100-80G"

    def test_report_request_count(self) -> None:
        data = _make_data(n=50)
        report = normalize_benchmark(data, source_gpu="A100-80G")
        assert report.request_count == 50

    def test_a10g_to_a100(self) -> None:
        """A10G is slower, normalizing to A100 should decrease latencies."""
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="A10G-24G", target_gpu="A100-80G")
        assert report.scaling_factor == pytest.approx(0.45)
        assert report.ttft_stats.normalized_p50 < report.ttft_stats.original_p50

    def test_bidirectional_consistency(self) -> None:
        """Normalizing A→B then B→A should return to original."""
        data = _make_data(n=20)
        config_ab = NormalizationConfig(source_gpu="A100-80G", target_gpu="H100-80G")
        norm_ab = BenchmarkNormalizer(config_ab)
        data_ab = norm_ab.normalize_data(data)

        config_ba = NormalizationConfig(source_gpu="H100-80G", target_gpu="A100-80G")
        norm_ba = BenchmarkNormalizer(config_ba)
        data_round = norm_ba.normalize_data(data_ab)

        for orig, rnd in zip(data.requests, data_round.requests):
            assert rnd.ttft_ms == pytest.approx(orig.ttft_ms, rel=1e-6)

    def test_qps_scaling(self) -> None:
        data = _make_data(qps=100.0)
        report = normalize_benchmark(data, source_gpu="H100-80G", target_gpu="A100-80G")
        # H100→A100: scaling=1.8, QPS should decrease
        assert report.normalized_qps == pytest.approx(100.0 / 1.8, rel=1e-3)

    def test_normalize_data_qps(self) -> None:
        data = _make_data(qps=50.0)
        config = NormalizationConfig(source_gpu="H100-80G", target_gpu="A100-80G")
        result = BenchmarkNormalizer(config).normalize_data(data)
        assert result.metadata.measured_qps == pytest.approx(50.0 / 1.8, rel=1e-3)

    def test_custom_factors_override_defaults(self) -> None:
        data = _make_data()
        # Override A100 factor
        report = normalize_benchmark(
            data,
            source_gpu="A100-80G",
            target_gpu="A100-80G",
            custom_factors={"A100-80G": 3.0},
        )
        assert report.scaling_factor == pytest.approx(1.0)

    def test_tpot_and_total_stats(self) -> None:
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="H100-80G")
        assert report.tpot_stats.scaling_factor == pytest.approx(1.8)
        assert report.total_latency_stats.scaling_factor == pytest.approx(1.8)

    def test_h200_to_a100(self) -> None:
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="H200-141G", target_gpu="A100-80G")
        assert report.scaling_factor == pytest.approx(2.2)

    def test_l40s_factor(self) -> None:
        data = _make_data()
        report = normalize_benchmark(data, source_gpu="L40S-48G", target_gpu="A100-80G")
        assert report.scaling_factor == pytest.approx(0.7)


class TestCLI:
    """Tests for normalize CLI subcommand."""

    def test_cli_table_output(self, tmp_path) -> None:
        from xpyd_plan.cli._normalize import _cmd_normalize

        data = _make_data(n=50)
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(json.dumps(data.model_dump()))

        import argparse

        args = argparse.Namespace(
            benchmark=str(bench_file),
            source_gpu="H100-80G",
            target_gpu="A100-80G",
            output=None,
            output_format="table",
        )
        # Should not raise
        _cmd_normalize(args)

    def test_cli_json_output(self, tmp_path, capsys) -> None:
        from xpyd_plan.cli._normalize import _cmd_normalize

        data = _make_data(n=50)
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(json.dumps(data.model_dump()))

        import argparse

        args = argparse.Namespace(
            benchmark=str(bench_file),
            source_gpu="H100-80G",
            target_gpu="A100-80G",
            output=None,
            output_format="json",
        )
        _cmd_normalize(args)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["scaling_factor"] == pytest.approx(1.8)

    def test_cli_output_file(self, tmp_path) -> None:
        from xpyd_plan.cli._normalize import _cmd_normalize

        data = _make_data(n=20)
        bench_file = tmp_path / "bench.json"
        bench_file.write_text(json.dumps(data.model_dump()))
        out_file = tmp_path / "normalized.json"

        import argparse

        args = argparse.Namespace(
            benchmark=str(bench_file),
            source_gpu="H100-80G",
            target_gpu="A100-80G",
            output=str(out_file),
            output_format="table",
        )
        _cmd_normalize(args)
        assert out_file.exists()
        result = json.loads(out_file.read_text())
        assert "requests" in result
