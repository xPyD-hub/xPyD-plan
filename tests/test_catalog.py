"""Tests for catalog module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.catalog import CatalogQuery, DatasetCatalog, manage_catalog


def _make_benchmark(tmp_path: Path, name: str = "bench.json", **overrides) -> Path:
    """Create a minimal benchmark JSON file."""
    data = {
        "config": {
            "num_prefill_instances": overrides.get("prefill", 2),
            "num_decode_instances": overrides.get("decode", 6),
            "total_instances": overrides.get("total", 8),
        },
        "metadata": {
            "gpu_type": overrides.get("gpu_type", "H100-80G"),
            "model_name": overrides.get("model_name", "llama-70b"),
        },
        "measured_qps": overrides.get("qps", 10.5),
        "requests": [
            {
                "request_id": f"r{i}",
                "prompt_tokens": 100,
                "output_tokens": 50,
                "ttft_ms": 20.0,
                "tpot_ms": 10.0,
                "total_latency_ms": 520.0,
                "timestamp": 1000.0 + i,
            }
            for i in range(overrides.get("n_requests", 5))
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def catalog(tmp_path):
    db_path = str(tmp_path / "test_catalog.db")
    cat = DatasetCatalog(db_path=db_path)
    yield cat
    cat.close()


class TestDatasetCatalog:
    def test_add_and_get(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path)
        entry = catalog.add(str(bench))
        assert entry.id > 0
        assert entry.gpu_type == "H100-80G"
        assert entry.model_name == "llama-70b"
        assert entry.prefill_instances == 2
        assert entry.decode_instances == 6
        assert entry.pd_ratio == "2:6"
        assert entry.measured_qps == 10.5
        assert entry.request_count == 5

        got = catalog.get(entry.id)
        assert got is not None
        assert got.file_hash == entry.file_hash

    def test_duplicate_detection(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path)
        catalog.add(str(bench))
        with pytest.raises(ValueError, match="Duplicate"):
            catalog.add(str(bench))

    def test_remove(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path)
        entry = catalog.add(str(bench))
        assert catalog.remove(entry.id) is True
        assert catalog.get(entry.id) is None

    def test_remove_nonexistent(self, catalog):
        assert catalog.remove(999) is False

    def test_list_all(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", qps=5.0)
        b2 = _make_benchmark(tmp_path, "b2.json", qps=15.0)
        catalog.add(str(b1))
        catalog.add(str(b2))
        report = catalog.list_all()
        assert report.total_count == 2
        assert len(report.entries) == 2

    def test_search_by_gpu(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", gpu_type="H100-80G")
        b2 = _make_benchmark(tmp_path, "b2.json", gpu_type="A100-80G")
        catalog.add(str(b1))
        catalog.add(str(b2))
        report = catalog.search(CatalogQuery(gpu_type="H100-80G"))
        assert report.total_count == 1
        assert report.entries[0].gpu_type == "H100-80G"

    def test_search_by_qps_range(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", qps=5.0)
        b2 = _make_benchmark(tmp_path, "b2.json", qps=15.0)
        b3 = _make_benchmark(tmp_path, "b3.json", qps=25.0)
        catalog.add(str(b1))
        catalog.add(str(b2))
        catalog.add(str(b3))
        report = catalog.search(CatalogQuery(min_qps=10.0, max_qps=20.0))
        assert report.total_count == 1
        assert report.entries[0].measured_qps == 15.0

    def test_search_by_pd_ratio(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", prefill=2, decode=6)
        b2 = _make_benchmark(tmp_path, "b2.json", prefill=4, decode=4)
        catalog.add(str(b1))
        catalog.add(str(b2))
        report = catalog.search(CatalogQuery(pd_ratio="4:4"))
        assert report.total_count == 1

    def test_search_by_model(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", model_name="llama-70b")
        b2 = _make_benchmark(tmp_path, "b2.json", model_name="mistral-7b")
        catalog.add(str(b1))
        catalog.add(str(b2))
        report = catalog.search(CatalogQuery(model_name="mistral-7b"))
        assert report.total_count == 1

    def test_search_by_instances(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json", total=4, prefill=1, decode=3)
        b2 = _make_benchmark(tmp_path, "b2.json", total=8, prefill=2, decode=6)
        b3 = _make_benchmark(tmp_path, "b3.json", total=16, prefill=4, decode=12)
        catalog.add(str(b1))
        catalog.add(str(b2))
        catalog.add(str(b3))
        report = catalog.search(CatalogQuery(min_instances=6, max_instances=10))
        assert report.total_count == 1
        assert report.entries[0].total_instances == 8

    def test_search_empty_result(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path)
        catalog.add(str(bench))
        report = catalog.search(CatalogQuery(gpu_type="NONEXISTENT"))
        assert report.total_count == 0

    def test_search_no_filters(self, catalog, tmp_path):
        b1 = _make_benchmark(tmp_path, "b1.json")
        b2 = _make_benchmark(tmp_path, "b2.json", qps=20.0)
        catalog.add(str(b1))
        catalog.add(str(b2))
        report = catalog.search(CatalogQuery())
        assert report.total_count == 2

    def test_file_not_found(self, catalog):
        with pytest.raises(FileNotFoundError):
            catalog.add("/nonexistent/file.json")

    def test_get_nonexistent(self, catalog):
        assert catalog.get(999) is None

    def test_add_with_notes(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path)
        entry = catalog.add(str(bench), notes="baseline run")
        assert entry.notes == "baseline run"
        got = catalog.get(entry.id)
        assert got.notes == "baseline run"

    def test_no_pd_ratio_when_zero_instances(self, catalog, tmp_path):
        bench = _make_benchmark(tmp_path, prefill=0, decode=0)
        entry = catalog.add(str(bench))
        assert entry.pd_ratio == ""

    def test_metadata_from_cluster_config(self, tmp_path):
        """Test extraction from cluster_config key (alternative format)."""
        data = {
            "cluster_config": {
                "num_prefill_instances": 3,
                "num_decode_instances": 5,
                "total_instances": 8,
            },
            "measured_qps": 8.0,
            "requests": [
                {
                    "request_id": "r0",
                    "prompt_tokens": 100,
                    "output_tokens": 50,
                    "ttft_ms": 20,
                    "tpot_ms": 10,
                    "total_latency_ms": 500,
                    "timestamp": 1000,
                }
            ],
        }
        path = tmp_path / "alt.json"
        path.write_text(json.dumps(data))

        db_path = str(tmp_path / "cat.db")
        cat = DatasetCatalog(db_path=db_path)
        entry = cat.add(str(path))
        assert entry.pd_ratio == "3:5"
        assert entry.request_count == 1
        cat.close()


class TestManageCatalogAPI:
    def test_add_action(self, tmp_path):
        bench = _make_benchmark(tmp_path)
        db = str(tmp_path / "api.db")
        report = manage_catalog("add", db_path=db, file_path=str(bench))
        assert report.total_count == 1
        assert report.entries[0].gpu_type == "H100-80G"

    def test_list_action(self, tmp_path):
        bench = _make_benchmark(tmp_path)
        db = str(tmp_path / "api.db")
        manage_catalog("add", db_path=db, file_path=str(bench))
        report = manage_catalog("list", db_path=db)
        assert report.total_count == 1

    def test_search_action(self, tmp_path):
        bench = _make_benchmark(tmp_path)
        db = str(tmp_path / "api.db")
        manage_catalog("add", db_path=db, file_path=str(bench))
        report = manage_catalog("search", db_path=db, query=CatalogQuery(gpu_type="H100-80G"))
        assert report.total_count == 1

    def test_show_action(self, tmp_path):
        bench = _make_benchmark(tmp_path)
        db = str(tmp_path / "api.db")
        add_report = manage_catalog("add", db_path=db, file_path=str(bench))
        eid = add_report.entries[0].id
        report = manage_catalog("show", db_path=db, entry_id=eid)
        assert report.total_count == 1

    def test_remove_action(self, tmp_path):
        bench = _make_benchmark(tmp_path)
        db = str(tmp_path / "api.db")
        add_report = manage_catalog("add", db_path=db, file_path=str(bench))
        eid = add_report.entries[0].id
        report = manage_catalog("remove", db_path=db, entry_id=eid)
        assert "Removed" in report.message

    def test_unknown_action(self, tmp_path):
        db = str(tmp_path / "api.db")
        with pytest.raises(ValueError, match="Unknown action"):
            manage_catalog("invalid", db_path=db)
