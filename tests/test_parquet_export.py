"""Tests for parquet_export module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.parquet_export import (
    ExportMode,
    ParquetConfig,
    ParquetExporter,
    _check_sla,
    _classify_workload,
    export_parquet,
)

# Check if pyarrow is available
try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

requires_pyarrow = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")


def _make_benchmark(
    num_requests: int = 10,
    qps: float = 100.0,
    prefill: int = 2,
    decode: int = 2,
) -> BenchmarkData:
    """Create a simple benchmark dataset for testing."""
    requests = []
    for i in range(num_requests):
        requests.append(
            BenchmarkRequest(
                request_id=f"req_{i:04d}",
                prompt_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                ttft_ms=20.0 + i * 2.0,
                tpot_ms=5.0 + i * 0.5,
                total_latency_ms=100.0 + i * 10.0,
                timestamp=1700000000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        requests=requests,
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
    )


def _write_benchmark(path: Path, bench: BenchmarkData) -> Path:
    """Write a benchmark to JSON file."""
    data = {
        "requests": [r.model_dump() for r in bench.requests],
        "metadata": bench.metadata.model_dump(),
    }
    path.write_text(json.dumps(data))
    return path


# --- Unit tests for helper functions ---


class TestClassifyWorkload:
    def test_short(self):
        assert _classify_workload(30, 30) == "SHORT"

    def test_long(self):
        assert _classify_workload(1500, 600) == "LONG"

    def test_prefill_heavy(self):
        assert _classify_workload(400, 100) == "PREFILL_HEAVY"

    def test_decode_heavy(self):
        assert _classify_workload(100, 400) == "DECODE_HEAVY"

    def test_balanced(self):
        assert _classify_workload(200, 200) == "BALANCED"


class TestCheckSLA:
    def test_all_pass_no_thresholds(self):
        assert _check_sla(100, 50, 200, None, None, None) is True

    def test_ttft_fail(self):
        assert _check_sla(100, 50, 200, 80, None, None) is False

    def test_tpot_fail(self):
        assert _check_sla(100, 50, 200, None, 40, None) is False

    def test_total_fail(self):
        assert _check_sla(100, 50, 200, None, None, 150) is False

    def test_all_pass(self):
        assert _check_sla(100, 50, 200, 200, 100, 300) is True


# --- Integration tests requiring pyarrow ---


@requires_pyarrow
class TestParquetExporterRequests:
    def test_single_benchmark(self, tmp_path):
        bench = _make_benchmark(num_requests=5)
        out = tmp_path / "output.parquet"
        exporter = ParquetExporter()
        result = exporter.export([bench], out)

        assert result.total_requests == 5
        assert result.total_benchmarks == 1
        assert result.mode == ExportMode.REQUESTS
        assert result.enriched is False
        assert "request_id" in result.columns
        assert "source" not in result.columns  # single benchmark
        assert out.exists()
        assert result.file_size_bytes > 0

        # Verify parquet content
        table = pq.read_table(str(out))
        assert len(table) == 5
        assert "request_id" in table.column_names

    def test_multi_benchmark_adds_source(self, tmp_path):
        b1 = _make_benchmark(num_requests=3, qps=50)
        b2 = _make_benchmark(num_requests=4, qps=100)
        out = tmp_path / "multi.parquet"
        exporter = ParquetExporter()
        result = exporter.export([b1, b2], out, source_tags=["low_qps", "high_qps"])

        assert result.total_requests == 7
        assert "source" in result.columns

        table = pq.read_table(str(out))
        sources = table.column("source").to_pylist()
        assert sources.count("low_qps") == 3
        assert sources.count("high_qps") == 4

    def test_enriched_export(self, tmp_path):
        bench = _make_benchmark(num_requests=5)
        out = tmp_path / "enriched.parquet"
        config = ParquetConfig(
            enrich=True,
            sla_ttft_ms=25.0,
            sla_tpot_ms=7.0,
            sla_total_ms=150.0,
        )
        exporter = ParquetExporter(config)
        result = exporter.export([bench], out)

        assert result.enriched is True
        assert "sla_pass" in result.columns
        assert "workload_category" in result.columns

        table = pq.read_table(str(out))
        assert "sla_pass" in table.column_names
        assert "workload_category" in table.column_names


@requires_pyarrow
class TestParquetExporterSummary:
    def test_summary_mode(self, tmp_path):
        bench = _make_benchmark(num_requests=20)
        out = tmp_path / "summary.parquet"
        config = ParquetConfig(mode=ExportMode.SUMMARY)
        exporter = ParquetExporter(config)
        result = exporter.export([bench], out)

        assert result.mode == ExportMode.SUMMARY
        assert result.total_requests == 1  # one summary row

        table = pq.read_table(str(out))
        assert len(table) == 1
        assert "ttft_p95_ms" in table.column_names
        assert "mean_prompt_tokens" in table.column_names

    def test_summary_multi_benchmark(self, tmp_path):
        b1 = _make_benchmark(num_requests=10, qps=50)
        b2 = _make_benchmark(num_requests=15, qps=100)
        out = tmp_path / "summary_multi.parquet"
        config = ParquetConfig(mode=ExportMode.SUMMARY)
        exporter = ParquetExporter(config)
        result = exporter.export([b1, b2], out, source_tags=["run1", "run2"])

        assert result.total_requests == 2  # two summary rows
        table = pq.read_table(str(out))
        assert len(table) == 2


@requires_pyarrow
class TestParquetExporterBoth:
    def test_both_mode(self, tmp_path):
        bench = _make_benchmark(num_requests=5)
        out = tmp_path / "both.parquet"
        config = ParquetConfig(mode=ExportMode.BOTH)
        exporter = ParquetExporter(config)
        result = exporter.export([bench], out)

        assert result.mode == ExportMode.BOTH
        assert result.total_requests == 5


@requires_pyarrow
class TestParquetExporterEdgeCases:
    def test_empty_benchmarks_raises(self, tmp_path):
        out = tmp_path / "empty.parquet"
        exporter = ParquetExporter()
        with pytest.raises(ValueError, match="At least one"):
            exporter.export([], out)

    def test_default_source_tags(self, tmp_path):
        b1 = _make_benchmark(num_requests=2)
        b2 = _make_benchmark(num_requests=3)
        out = tmp_path / "default_tags.parquet"
        exporter = ParquetExporter()
        exporter.export([b1, b2], out)

        table = pq.read_table(str(out))
        sources = table.column("source").to_pylist()
        assert "bench_0" in sources
        assert "bench_1" in sources


@requires_pyarrow
class TestExportParquetAPI:
    def test_programmatic_api(self, tmp_path):
        bench = _make_benchmark(num_requests=8)
        out = tmp_path / "api.parquet"
        result = export_parquet([bench], out)

        assert isinstance(result, dict)
        assert result["total_requests"] == 8
        assert result["mode"] == "requests"
        assert out.exists()

    def test_api_with_enrichment(self, tmp_path):
        bench = _make_benchmark(num_requests=5)
        out = tmp_path / "api_enriched.parquet"
        result = export_parquet(
            [bench],
            out,
            enrich=True,
            sla_ttft_ms=50.0,
        )
        assert result["enriched"] is True

    def test_api_summary_mode(self, tmp_path):
        bench = _make_benchmark(num_requests=10)
        out = tmp_path / "api_summary.parquet"
        result = export_parquet([bench], out, mode="summary")
        assert result["mode"] == "summary"
        assert result["total_requests"] == 1


class TestPyarrowImportError:
    def test_import_error_message(self, tmp_path, monkeypatch):
        """Test graceful error when pyarrow is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        bench = _make_benchmark(num_requests=5)
        out = tmp_path / "no_pyarrow.parquet"
        exporter = ParquetExporter()

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyarrow is required"):
            exporter.export([bench], out)
