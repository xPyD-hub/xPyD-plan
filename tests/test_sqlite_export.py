"""Tests for sqlite_export module."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.sqlite_export import (
    SQLiteExportConfig,
    SQLiteExporter,
    SQLiteExportReport,
    _percentile,
    export_to_sqlite,
)


def _make_benchmark(
    n: int = 50,
    prefill: int = 2,
    decode: int = 2,
    qps: float = 10.0,
) -> BenchmarkData:
    """Create a simple benchmark dataset."""
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + i * 5,
                output_tokens=50 + i * 2,
                ttft_ms=20.0 + i * 0.5,
                tpot_ms=10.0 + i * 0.3,
                total_latency_ms=80.0 + i * 1.0,
                timestamp=1000.0 + i * 0.1,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestPercentile:
    def test_empty_list(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([5.0], 50) == 5.0

    def test_p50(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_p95(self) -> None:
        vals = list(range(1, 101))
        result = _percentile([float(v) for v in vals], 95)
        assert result > 90


class TestSQLiteExportConfig:
    def test_defaults(self) -> None:
        config = SQLiteExportConfig()
        assert config.append is False
        assert config.benchmark_id is None
        assert config.include_summary is True
        assert config.create_indexes is True

    def test_custom(self) -> None:
        config = SQLiteExportConfig(append=True, benchmark_id="test-1")
        assert config.append is True
        assert config.benchmark_id == "test-1"


class TestSQLiteExportReport:
    def test_serializable(self) -> None:
        report = SQLiteExportReport(
            output_path="/tmp/test.db",
            total_requests=100,
            total_benchmarks=1,
            benchmark_ids=["b1"],
            tables_created=["requests", "metadata"],
            indexes_created=6,
            file_size_bytes=4096,
            appended=False,
        )
        data = report.model_dump()
        assert data["total_requests"] == 100
        assert json.loads(json.dumps(data)) == data


class TestSQLiteExporter:
    def test_basic_export(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=20)
        exporter = SQLiteExporter()
        result = exporter.export([bench], tmp_path / "test.db")

        assert result.total_requests == 20
        assert result.total_benchmarks == 1
        assert "requests" in result.tables_created
        assert "metadata" in result.tables_created
        assert "analysis_summary" in result.tables_created
        assert result.indexes_created == 6
        assert result.file_size_bytes > 0
        assert result.appended is False

    def test_data_integrity(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=10)
        exporter = SQLiteExporter()
        exporter.export([bench], tmp_path / "test.db")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        rows = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
        assert rows[0] == 10

        meta = conn.execute("SELECT * FROM metadata").fetchone()
        assert meta is not None
        conn.close()

    def test_empty_benchmarks_raises(self, tmp_path: Path) -> None:
        exporter = SQLiteExporter()
        with pytest.raises(ValueError, match="At least one benchmark"):
            exporter.export([], tmp_path / "test.db")

    def test_multiple_benchmarks(self, tmp_path: Path) -> None:
        b1 = _make_benchmark(n=10, prefill=1, decode=3)
        b2 = _make_benchmark(n=15, prefill=3, decode=1)
        exporter = SQLiteExporter()
        result = exporter.export([b1, b2], tmp_path / "test.db")

        assert result.total_requests == 25
        assert result.total_benchmarks == 2
        assert len(result.benchmark_ids) == 2

    def test_source_tags(self, tmp_path: Path) -> None:
        b1 = _make_benchmark(n=5)
        b2 = _make_benchmark(n=5)
        exporter = SQLiteExporter()
        result = exporter.export(
            [b1, b2], tmp_path / "test.db", source_tags=["run-a", "run-b"]
        )

        assert result.benchmark_ids == ["run-a", "run-b"]

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ids = conn.execute(
            "SELECT DISTINCT benchmark_id FROM requests ORDER BY benchmark_id"
        ).fetchall()
        assert [r[0] for r in ids] == ["run-a", "run-b"]
        conn.close()

    def test_custom_benchmark_id(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        config = SQLiteExportConfig(benchmark_id="custom-id")
        exporter = SQLiteExporter(config)
        result = exporter.export([bench], tmp_path / "test.db")

        assert result.benchmark_ids == ["custom-id"]

    def test_append_mode(self, tmp_path: Path) -> None:
        bench1 = _make_benchmark(n=5)
        exporter = SQLiteExporter()
        exporter.export([bench1], tmp_path / "test.db", source_tags=["first"])

        bench2 = _make_benchmark(n=8)
        config = SQLiteExportConfig(append=True)
        exporter2 = SQLiteExporter(config)
        result = exporter2.export([bench2], tmp_path / "test.db", source_tags=["second"])

        assert result.appended is True

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 13  # 5 + 8
        meta_count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
        assert meta_count == 2
        conn.close()

    def test_overwrite_mode(self, tmp_path: Path) -> None:
        bench1 = _make_benchmark(n=10)
        exporter = SQLiteExporter()
        exporter.export([bench1], tmp_path / "test.db")

        bench2 = _make_benchmark(n=3)
        result = exporter.export([bench2], tmp_path / "test.db")

        assert result.appended is False
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 3
        conn.close()

    def test_no_summary(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        config = SQLiteExportConfig(include_summary=False)
        exporter = SQLiteExporter(config)
        result = exporter.export([bench], tmp_path / "test.db")

        assert "analysis_summary" not in result.tables_created

    def test_no_indexes(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        config = SQLiteExportConfig(create_indexes=False)
        exporter = SQLiteExporter(config)
        result = exporter.export([bench], tmp_path / "test.db")

        assert result.indexes_created == 0

    def test_summary_table_values(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=100)
        exporter = SQLiteExporter()
        exporter.export([bench], tmp_path / "test.db")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        row = conn.execute("SELECT * FROM analysis_summary").fetchone()
        assert row is not None
        # benchmark_id, ttft_p50, ttft_p95, ttft_p99, tpot_p50, ...
        assert row[1] > 0  # ttft_p50
        assert row[2] > row[1]  # ttft_p95 > ttft_p50
        conn.close()

    def test_request_columns(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        exporter = SQLiteExporter()
        exporter.export([bench], tmp_path / "test.db")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute("PRAGMA table_info(requests)")
        cols = [row[1] for row in cursor.fetchall()]
        assert "benchmark_id" in cols
        assert "request_id" in cols
        assert "prompt_tokens" in cols
        assert "ttft_ms" in cols
        assert "tpot_ms" in cols
        assert "total_latency_ms" in cols
        assert "timestamp" in cols
        conn.close()

    def test_sql_query_works(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=50)
        exporter = SQLiteExporter()
        exporter.export([bench], tmp_path / "test.db")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        # Simulate a user query
        result = conn.execute(
            "SELECT AVG(ttft_ms), AVG(tpot_ms) FROM requests WHERE prompt_tokens > 200"
        ).fetchone()
        assert result[0] is not None
        assert result[1] is not None
        conn.close()


class TestExportToSqliteAPI:
    def test_basic(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=10)
        result = export_to_sqlite([bench], tmp_path / "test.db")

        assert result["total_requests"] == 10
        assert result["total_benchmarks"] == 1
        assert isinstance(result["benchmark_ids"], list)

    def test_with_options(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        result = export_to_sqlite(
            [bench],
            tmp_path / "test.db",
            append=False,
            benchmark_id="my-run",
            include_summary=True,
            create_indexes=True,
        )

        assert result["benchmark_ids"] == ["my-run"]

    def test_append(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        export_to_sqlite([bench], tmp_path / "test.db", source_tags=["a"])
        result = export_to_sqlite(
            [bench], tmp_path / "test.db", append=True, source_tags=["b"]
        )

        assert result["appended"] is True

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        bench = _make_benchmark(n=5)
        result = export_to_sqlite([bench], tmp_path / "test.db")
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized == result


class TestSQLiteExportCLI:
    def test_cli_table_output(self, tmp_path: Path) -> None:
        """Test CLI integration by importing and checking parser setup."""
        import argparse

        from xpyd_plan.cli._sqlite_export import add_sqlite_export_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_sqlite_export_parser(subparsers)

        args = parser.parse_args([
            "sqlite-export",
            "--benchmark", "test.json",
            "--output", "test.db",
            "--append",
            "--no-summary",
        ])
        assert args.command == "sqlite-export"
        assert args.append is True
        assert args.no_summary is True

    def test_cli_json_output_flag(self) -> None:
        import argparse

        from xpyd_plan.cli._sqlite_export import add_sqlite_export_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_sqlite_export_parser(subparsers)

        args = parser.parse_args([
            "sqlite-export",
            "--benchmark", "a.json",
            "--output", "a.db",
            "--output-format", "json",
        ])
        assert args.output_format == "json"
