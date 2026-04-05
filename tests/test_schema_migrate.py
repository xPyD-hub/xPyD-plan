"""Tests for schema_migrate module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.schema_migrate import (
    SchemaMigrator,
    SchemaVersion,
    detect_version,
    migrate_schema,
)

# ---------- SchemaVersion ----------


class TestSchemaVersion:
    def test_parse_single_digit(self) -> None:
        v = SchemaVersion.parse("1")
        assert v.major == 1
        assert v.minor == 0

    def test_parse_major_minor(self) -> None:
        v = SchemaVersion.parse("2.0")
        assert v.major == 2
        assert v.minor == 0

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError):
            SchemaVersion.parse("1.2.3")

    def test_str(self) -> None:
        assert str(SchemaVersion(major=2, minor=0)) == "2.0"

    def test_comparison(self) -> None:
        v1 = SchemaVersion(major=1, minor=0)
        v2 = SchemaVersion(major=2, minor=0)
        assert v1 < v2
        assert v2 > v1
        assert v1 <= v2
        assert v2 >= v1
        assert v1 == SchemaVersion(major=1, minor=0)
        assert v1 != v2

    def test_hash(self) -> None:
        v1a = SchemaVersion(major=1, minor=0)
        v1b = SchemaVersion(major=1, minor=0)
        assert hash(v1a) == hash(v1b)
        assert {v1a, v1b} == {v1a}


# ---------- detect_version ----------


def _make_v1_data() -> dict:
    return {
        "metadata": {
            "num_prefill_instances": 2,
            "num_decode_instances": 2,
            "total_instances": 4,
            "measured_qps": 10.0,
        },
        "requests": [
            {
                "request_id": "r1",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 20.0,
                "tpot_ms": 5.0,
                "total_latency_ms": 270.0,
                "timestamp": 1000.0,
            }
        ],
    }


def _make_v2_data() -> dict:
    d = _make_v1_data()
    d["schema_version"] = "2.0"
    d["metadata"]["run_id"] = "abc-123"
    d["metadata"]["schema_version"] = "2.0"
    return d


class TestDetectVersion:
    def test_detect_v1_native(self) -> None:
        v = detect_version(_make_v1_data())
        assert v == SchemaVersion(major=1, minor=0)

    def test_detect_v1_xpyd_bench(self) -> None:
        data = {"bench_config": {}, "results": []}
        v = detect_version(data)
        assert v == SchemaVersion(major=1, minor=0)

    def test_detect_explicit_version(self) -> None:
        data = _make_v1_data()
        data["schema_version"] = "2.0"
        v = detect_version(data)
        assert v == SchemaVersion(major=2, minor=0)

    def test_detect_unknown_format(self) -> None:
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_version({"random": "stuff"})


# ---------- SchemaMigrator ----------


class TestSchemaMigrator:
    def setup_method(self) -> None:
        self.migrator = SchemaMigrator()

    def test_detect(self) -> None:
        v = self.migrator.detect(_make_v1_data())
        assert v == SchemaVersion(major=1, minor=0)

    def test_needs_migration_v1(self) -> None:
        assert self.migrator.needs_migration(_make_v1_data()) is True

    def test_needs_migration_v2(self) -> None:
        assert self.migrator.needs_migration(_make_v2_data()) is False

    def test_migrate_v1_to_v2(self) -> None:
        result = self.migrator.migrate(_make_v1_data())
        assert result.migrated is True
        assert result.source_version == "1.0"
        assert result.target_version == "2.0"
        assert result.data["schema_version"] == "2.0"
        assert "run_id" in result.data["metadata"]
        assert result.data["metadata"]["schema_version"] == "2.0"
        assert len(result.changes) >= 2

    def test_migrate_already_latest(self) -> None:
        result = self.migrator.migrate(_make_v2_data())
        assert result.migrated is False
        assert result.changes == []

    def test_migrate_dry_run(self) -> None:
        data = _make_v1_data()
        result = self.migrator.migrate(data, dry_run=True)
        assert result.migrated is False
        assert len(result.changes) > 0
        # Original data should not have schema_version
        assert "schema_version" not in result.data

    def test_migrate_downgrade_error(self) -> None:
        v2 = _make_v2_data()
        with pytest.raises(ValueError, match="Cannot downgrade"):
            self.migrator.migrate(v2, target_version=SchemaVersion(major=1, minor=0))

    def test_migrate_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_v1_data()))

        result = self.migrator.migrate_file(str(f))
        assert result.migrated is True

        # File should be overwritten
        written = json.loads(f.read_text())
        assert written["schema_version"] == "2.0"

    def test_migrate_file_dry_run(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        original = _make_v1_data()
        f.write_text(json.dumps(original))

        result = self.migrator.migrate_file(str(f), dry_run=True)
        assert result.migrated is False

        # File should NOT be changed
        written = json.loads(f.read_text())
        assert "schema_version" not in written

    def test_migrate_file_output_path(self, tmp_path: Path) -> None:
        src = tmp_path / "input.json"
        dst = tmp_path / "output.json"
        src.write_text(json.dumps(_make_v1_data()))

        result = self.migrator.migrate_file(str(src), output_path=str(dst))
        assert result.migrated is True
        assert dst.exists()
        written = json.loads(dst.read_text())
        assert written["schema_version"] == "2.0"

        # Source should be unchanged
        original = json.loads(src.read_text())
        assert "schema_version" not in original


# ---------- Programmatic API ----------


class TestMigrateSchema:
    def test_migrate_schema_api(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_v1_data()))

        result = migrate_schema(str(f))
        assert result.migrated is True
        assert result.target_version == "2.0"

    def test_migrate_schema_specific_version(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_v1_data()))

        result = migrate_schema(str(f), target_version="2.0")
        assert result.migrated is True

    def test_migrate_schema_dry_run(self, tmp_path: Path) -> None:
        f = tmp_path / "bench.json"
        f.write_text(json.dumps(_make_v1_data()))

        result = migrate_schema(str(f), dry_run=True)
        assert result.migrated is False
        assert len(result.changes) > 0
