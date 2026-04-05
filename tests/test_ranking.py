"""Tests for benchmark ranking."""

from __future__ import annotations

import json
import subprocess

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.ranking import (
    BenchmarkRanker,
    RankingReport,
    RankWeights,
    rank_benchmarks,
)


def _make_data(
    num_prefill: int,
    num_decode: int,
    qps: float,
    ttft_base: float,
    tpot_base: float,
    n: int = 50,
) -> BenchmarkData:
    """Build BenchmarkData with controlled latency."""
    reqs = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100 + i,
            output_tokens=50 + i,
            ttft_ms=ttft_base + (i % 10) * 2,
            tpot_ms=tpot_base + (i % 5) * 1,
            total_latency_ms=ttft_base + tpot_base + (i % 10) * 3,
            timestamp=1000.0 + i,
        )
        for i in range(n)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=reqs,
    )


def _write_benchmark(path: str, data: BenchmarkData) -> None:
    """Write benchmark data to JSON file."""
    import pathlib

    pathlib.Path(path).write_text(
        json.dumps(data.model_dump(), default=str),
        encoding="utf-8",
    )


class TestBenchmarkRanker:
    """Tests for BenchmarkRanker."""

    def test_empty_input(self):
        ranker = BenchmarkRanker()
        report = ranker.rank([])
        assert report.benchmark_count == 0
        assert report.best is None
        assert report.rankings == []

    def test_single_benchmark(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        ranker = BenchmarkRanker()
        report = ranker.rank([("bench1.json", data)])
        assert report.benchmark_count == 1
        assert report.best is not None
        assert report.best.rank == 1
        assert report.best.file == "bench1.json"

    def test_ranking_order(self):
        """Lower latency + higher QPS should rank higher."""
        fast = _make_data(2, 4, 15.0, 30.0, 10.0)
        slow = _make_data(2, 2, 8.0, 100.0, 50.0)
        ranker = BenchmarkRanker()
        report = ranker.rank([
            ("slow.json", slow), ("fast.json", fast),
        ])
        assert report.rankings[0].file == "fast.json"
        assert report.rankings[1].file == "slow.json"

    def test_sla_compliance_scoring(self):
        good = _make_data(2, 4, 10.0, 20.0, 5.0)
        bad = _make_data(2, 2, 10.0, 200.0, 80.0)
        ranker = BenchmarkRanker(
            sla_ttft_ms=100.0, sla_tpot_ms=50.0,
        )
        report = ranker.rank([
            ("good.json", good), ("bad.json", bad),
        ])
        assert report.rankings[0].file == "good.json"
        good_r = report.rankings[0]
        bad_r = report.rankings[1]
        assert good_r.sla_score > bad_r.sla_score

    def test_no_sla_gives_perfect_score(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        ranker = BenchmarkRanker()
        report = ranker.rank([("bench.json", data)])
        assert report.rankings[0].sla_score == 100.0

    def test_weights_affect_ranking(self):
        """High QPS high lat vs low QPS low lat."""
        high_qps = _make_data(2, 4, 50.0, 100.0, 40.0)
        low_lat = _make_data(2, 2, 5.0, 10.0, 5.0)

        # Throughput-heavy
        w1 = RankWeights(
            sla_compliance=0.0, latency=0.0, throughput=1.0,
        )
        r1 = BenchmarkRanker(weights=w1).rank([
            ("high_qps.json", high_qps),
            ("low_lat.json", low_lat),
        ])
        assert r1.rankings[0].file == "high_qps.json"

        # Latency-heavy
        w2 = RankWeights(
            sla_compliance=0.0, latency=1.0, throughput=0.0,
        )
        r2 = BenchmarkRanker(weights=w2).rank([
            ("high_qps.json", high_qps),
            ("low_lat.json", low_lat),
        ])
        assert r2.rankings[0].file == "low_lat.json"

    def test_composite_score_range(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        report = BenchmarkRanker().rank([("b.json", data)])
        score = report.rankings[0].composite_score
        assert 0.0 <= score <= 100.0

    def test_report_fields(self):
        data = _make_data(3, 5, 12.0, 40.0, 15.0)
        ranker = BenchmarkRanker(sla_ttft_ms=200.0)
        report = ranker.rank([("bench.json", data)])
        r = report.rankings[0]
        assert r.prefill_instances == 3
        assert r.decode_instances == 5
        assert r.total_instances == 8
        assert r.measured_qps == 12.0
        assert r.ttft_p95_ms > 0
        assert r.tpot_p95_ms > 0
        assert report.sla_ttft_ms == 200.0

    def test_three_benchmarks(self):
        b1 = _make_data(1, 3, 10.0, 80.0, 30.0)
        b2 = _make_data(2, 4, 20.0, 40.0, 15.0)
        b3 = _make_data(3, 5, 15.0, 60.0, 25.0)
        report = BenchmarkRanker().rank([
            ("b1.json", b1), ("b2.json", b2), ("b3.json", b3),
        ])
        assert report.benchmark_count == 3
        ranks = [r.rank for r in report.rankings]
        assert sorted(ranks) == [1, 2, 3]

    def test_identical_benchmarks_same_score(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        report = BenchmarkRanker().rank([
            ("a.json", data), ("b.json", data),
        ])
        s0 = report.rankings[0].composite_score
        s1 = report.rankings[1].composite_score
        assert s0 == s1


class TestRankWeights:
    """Tests for RankWeights model."""

    def test_default_weights(self):
        w = RankWeights()
        assert w.sla_compliance == 0.4
        assert w.latency == 0.4
        assert w.throughput == 0.2

    def test_custom_weights(self):
        w = RankWeights(
            sla_compliance=0.5, latency=0.3, throughput=0.2,
        )
        assert w.sla_compliance == 0.5


class TestRankingModels:
    """Tests for Pydantic models."""

    def test_report_serializable(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        report = BenchmarkRanker().rank([("b.json", data)])
        d = report.model_dump()
        assert "rankings" in d
        assert "best" in d
        assert "weights" in d

    def test_json_roundtrip(self):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        report = BenchmarkRanker().rank([("b.json", data)])
        j = report.model_dump_json()
        restored = RankingReport.model_validate_json(j)
        assert restored.benchmark_count == report.benchmark_count


class TestRankBenchmarksAPI:
    """Tests for the programmatic API."""

    def test_api_returns_dict(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = rank_benchmarks([f])
        assert isinstance(result, dict)
        assert result["benchmark_count"] == 1

    def test_api_with_sla(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = rank_benchmarks([f], sla_ttft_ms=200.0)
        assert result["sla_ttft_ms"] == 200.0


class TestRankingCLI:
    """Tests for the CLI subcommand."""

    def test_cli_table_output(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = subprocess.run(
            ["xpyd-plan", "ranking", "--benchmark", f],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_json_output(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = subprocess.run(
            [
                "xpyd-plan", "ranking",
                "--benchmark", f, "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["benchmark_count"] == 1

    def test_cli_with_sla_flags(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = subprocess.run(
            [
                "xpyd-plan", "ranking", "--benchmark", f,
                "--sla-ttft", "200", "--sla-tpot", "100",
                "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["sla_ttft_ms"] == 200.0

    def test_cli_multiple_benchmarks(self, tmp_path):
        d1 = _make_data(2, 2, 10.0, 50.0, 20.0)
        d2 = _make_data(3, 3, 15.0, 30.0, 10.0)
        f1 = str(tmp_path / "b1.json")
        f2 = str(tmp_path / "b2.json")
        _write_benchmark(f1, d1)
        _write_benchmark(f2, d2)
        result = subprocess.run(
            [
                "xpyd-plan", "ranking",
                "--benchmark", f1, f2,
                "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert parsed["benchmark_count"] == 2

    def test_cli_json_parseable(self, tmp_path):
        data = _make_data(2, 2, 10.0, 50.0, 20.0)
        f = str(tmp_path / "bench.json")
        _write_benchmark(f, data)
        result = subprocess.run(
            [
                "xpyd-plan", "ranking",
                "--benchmark", f, "--output-format", "json",
            ],
            capture_output=True, text=True,
        )
        parsed = json.loads(result.stdout)
        assert "rankings" in parsed
        assert "best" in parsed
