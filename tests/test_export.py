"""Tests for JSON/CSV export and programmatic API (M9)."""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import (
    AnalysisResult,
    MultiScenarioResult,
    RatioCandidate,
    SLACheck,
)
from xpyd_plan.export import (
    _candidate_to_row,
    analyze,
    export_batch,
    result_to_csv,
    result_to_json,
)
from xpyd_plan.models import SLAConfig

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_requests(
    n: int = 100,
    ttft_range: tuple[float, float] = (50, 500),
    tpot_range: tuple[float, float] = (10, 40),
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    requests = []
    base_ts = 1700000000.0
    for i in range(n):
        prompt_tokens = rng.randint(100, 2000)
        output_tokens = rng.randint(50, 500)
        ttft = rng.uniform(*ttft_range)
        tpot = rng.uniform(*tpot_range)
        total = ttft + tpot * output_tokens
        requests.append({
            "request_id": f"req-{i:04d}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 2),
            "total_latency_ms": round(total, 2),
            "timestamp": base_ts + i * 0.1,
        })
    return requests


def _make_benchmark_data(
    n: int = 100, num_p: int = 2, num_d: int = 6, qps: float = 10.0, **kwargs
) -> dict:
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(n=n, **kwargs),
    }


@pytest.fixture
def benchmark_file(tmp_path: Path) -> Path:
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(_make_benchmark_data()))
    return p


@pytest.fixture
def benchmark_dir(tmp_path: Path) -> Path:
    d = tmp_path / "benchmarks"
    d.mkdir()
    for qps in [5.0, 10.0, 15.0]:
        p = d / f"bench_qps{qps:.0f}.json"
        p.write_text(json.dumps(_make_benchmark_data(qps=qps, seed=int(qps))))
    return d


@pytest.fixture
def sla() -> SLAConfig:
    return SLAConfig(ttft_ms=600.0, tpot_ms=50.0)


@pytest.fixture
def analysis_result() -> AnalysisResult:
    sla_check = SLACheck(
        ttft_p95_ms=400.0, ttft_p99_ms=450.0,
        tpot_p95_ms=35.0, tpot_p99_ms=38.0,
        total_latency_p95_ms=5000.0, total_latency_p99_ms=5500.0,
        meets_ttft=True, meets_tpot=True, meets_total_latency=True, meets_all=True,
    )
    candidates = [
        RatioCandidate(
            num_prefill=2, num_decode=6,
            prefill_utilization=0.8, decode_utilization=0.9,
            waste_rate=0.1, meets_sla=True, sla_check=sla_check,
        ),
        RatioCandidate(
            num_prefill=3, num_decode=5,
            prefill_utilization=0.6, decode_utilization=0.95,
            waste_rate=0.35, meets_sla=True, sla_check=sla_check,
        ),
        RatioCandidate(
            num_prefill=1, num_decode=7,
            prefill_utilization=0.99, decode_utilization=0.7,
            waste_rate=0.29, meets_sla=False, sla_check=sla_check,
        ),
    ]
    return AnalysisResult(
        best=candidates[0], candidates=candidates, total_instances=8,
    )


# ── result_to_json ────────────────────────────────────────────────────────


class TestResultToJson:
    def test_valid_json(self, analysis_result: AnalysisResult) -> None:
        out = result_to_json(analysis_result)
        data = json.loads(out)
        assert "best" in data
        assert "candidates" in data

    def test_candidates_count(self, analysis_result: AnalysisResult) -> None:
        data = json.loads(result_to_json(analysis_result))
        assert len(data["candidates"]) == 3

    def test_indent_custom(self, analysis_result: AnalysisResult) -> None:
        out = result_to_json(analysis_result, indent=4)
        assert "    " in out

    def test_best_fields(self, analysis_result: AnalysisResult) -> None:
        data = json.loads(result_to_json(analysis_result))
        best = data["best"]
        assert best["num_prefill"] == 2
        assert best["num_decode"] == 6
        assert best["meets_sla"] is True


# ── result_to_csv ─────────────────────────────────────────────────────────


class TestResultToCsv:
    def test_csv_header(self, analysis_result: AnalysisResult) -> None:
        out = result_to_csv(analysis_result)
        reader = csv.DictReader(io.StringIO(out))
        assert "config" in reader.fieldnames
        assert "waste_rate" in reader.fieldnames

    def test_csv_row_count(self, analysis_result: AnalysisResult) -> None:
        out = result_to_csv(analysis_result)
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert len(rows) == 3

    def test_csv_values(self, analysis_result: AnalysisResult) -> None:
        out = result_to_csv(analysis_result)
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert rows[0]["config"] == "2P:6D"
        assert rows[0]["meets_sla"] == "True"

    def test_csv_empty_result(self) -> None:
        result = AnalysisResult(best=None, candidates=[], total_instances=8)
        out = result_to_csv(result)
        assert out == ""

    def test_csv_no_sla_check(self) -> None:
        c = RatioCandidate(
            num_prefill=2, num_decode=6,
            prefill_utilization=0.8, decode_utilization=0.9,
            waste_rate=0.1, meets_sla=True, sla_check=None,
        )
        result = AnalysisResult(best=c, candidates=[c], total_instances=8)
        out = result_to_csv(result)
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert rows[0]["ttft_p95_ms"] == ""


# ── _candidate_to_row ─────────────────────────────────────────────────────


class TestCandidateToRow:
    def test_basic_fields(self) -> None:
        c = RatioCandidate(
            num_prefill=3, num_decode=5,
            prefill_utilization=0.6, decode_utilization=0.95,
            waste_rate=0.35, meets_sla=True,
        )
        row = _candidate_to_row(c)
        assert row["config"] == "3P:5D"
        assert row["total_instances"] == 8
        assert "qps" not in row

    def test_with_qps(self) -> None:
        c = RatioCandidate(
            num_prefill=2, num_decode=6,
            prefill_utilization=0.8, decode_utilization=0.9,
            waste_rate=0.1, meets_sla=True,
        )
        row = _candidate_to_row(c, scenario_qps=10.0)
        assert row["qps"] == 10.0


# ── Multi-scenario CSV ────────────────────────────────────────────────────


class TestMultiScenarioCsv:
    def test_multi_csv_has_qps(self) -> None:
        from xpyd_plan.benchmark_models import ScenarioResult

        sla_check = SLACheck(
            ttft_p95_ms=400.0, ttft_p99_ms=450.0,
            tpot_p95_ms=35.0, tpot_p99_ms=38.0,
            total_latency_p95_ms=5000.0, total_latency_p99_ms=5500.0,
            meets_ttft=True, meets_tpot=True, meets_total_latency=True, meets_all=True,
        )
        c = RatioCandidate(
            num_prefill=2, num_decode=6,
            prefill_utilization=0.8, decode_utilization=0.9,
            waste_rate=0.1, meets_sla=True, sla_check=sla_check,
        )
        scenario = ScenarioResult(
            qps=10.0,
            analysis=AnalysisResult(best=c, candidates=[c], total_instances=8),
        )
        multi = MultiScenarioResult(
            scenarios=[scenario], total_instances=8, unified_best=c,
        )
        out = result_to_csv(multi)
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert rows[0]["qps"] == "10.0"


# ── Programmatic API (analyze) ────────────────────────────────────────────


class TestAnalyzeApi:
    def test_single_file(self, benchmark_file: Path, sla: SLAConfig) -> None:
        result = analyze(str(benchmark_file), sla_config=sla)
        assert result["mode"] == "single"
        assert "analysis" in result
        assert "candidates" in result["analysis"]

    def test_single_file_string(self, benchmark_file: Path) -> None:
        result = analyze(str(benchmark_file))
        assert result["mode"] == "single"

    def test_dict_sla(self, benchmark_file: Path) -> None:
        result = analyze(str(benchmark_file), sla_config={"ttft_ms": 600.0})
        assert result["mode"] == "single"

    def test_none_sla(self, benchmark_file: Path) -> None:
        result = analyze(str(benchmark_file), sla_config=None)
        assert result["mode"] == "single"


# ── Batch export ──────────────────────────────────────────────────────────


class TestExportBatch:
    def test_json_export(self, benchmark_dir: Path, sla: SLAConfig) -> None:
        out = export_batch(str(benchmark_dir), sla_config=sla, output_format="json")
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 3
        assert "file" in data[0]
        assert "analysis" in data[0]

    def test_csv_export(self, benchmark_dir: Path, sla: SLAConfig) -> None:
        out = export_batch(str(benchmark_dir), sla_config=sla, output_format="csv")
        reader = csv.DictReader(io.StringIO(out))
        rows = list(reader)
        assert len(rows) > 0
        assert "file" in reader.fieldnames

    def test_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            export_batch(str(empty))

    def test_invalid_format(self, benchmark_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            export_batch(str(benchmark_dir), output_format="xml")
