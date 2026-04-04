"""Tests for benchmark annotation and tagging system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xpyd_plan.annotation import (
    AnnotatedBenchmark,
    Annotation,
    AnnotationManager,
    FilterResult,
    annotate_benchmark,
)


@pytest.fixture()
def bench_dir(tmp_path: Path) -> Path:
    """Create a directory with sample benchmark files."""
    for i in range(3):
        p = tmp_path / f"bench_{i}.json"
        p.write_text(json.dumps({"id": i}))
    return tmp_path


@pytest.fixture()
def bench_file(bench_dir: Path) -> Path:
    """Return path to first benchmark file."""
    return bench_dir / "bench_0.json"


class TestAnnotation:
    """Test Annotation model."""

    def test_tags_path(self) -> None:
        a = Annotation(benchmark_path="/data/bench.json", tags={"model": "llama"})
        assert a.tags_path == Path("/data/bench.tags.yaml")

    def test_tags_path_nested(self) -> None:
        a = Annotation(benchmark_path="/data/sub/run.json", tags={})
        assert a.tags_path == Path("/data/sub/run.tags.yaml")

    def test_empty_tags(self) -> None:
        a = Annotation(benchmark_path="test.json")
        assert a.tags == {}


class TestAnnotationManager:
    """Test AnnotationManager operations."""

    def test_add_tags(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        result = mgr.add_tags(bench_file, {"model": "llama3-70b", "gpu": "H100"})
        assert result.tags == {"model": "llama3-70b", "gpu": "H100"}
        # Verify sidecar exists
        sidecar = bench_file.parent / "bench_0.tags.yaml"
        assert sidecar.exists()

    def test_add_tags_file_not_found(self, tmp_path: Path) -> None:
        mgr = AnnotationManager()
        with pytest.raises(FileNotFoundError):
            mgr.add_tags(tmp_path / "nonexistent.json", {"key": "val"})

    def test_add_tags_merge(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"model": "llama"})
        result = mgr.add_tags(bench_file, {"gpu": "H100"})
        assert result.tags == {"model": "llama", "gpu": "H100"}

    def test_add_tags_overwrite(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"model": "llama"})
        result = mgr.add_tags(bench_file, {"model": "mistral"})
        assert result.tags["model"] == "mistral"

    def test_get_tags_no_sidecar(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        result = mgr.get_tags(bench_file)
        assert result.tags == {}

    def test_get_tags_with_sidecar(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"env": "prod"})
        result = mgr.get_tags(bench_file)
        assert result.tags == {"env": "prod"}

    def test_remove_tags(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"a": "1", "b": "2", "c": "3"})
        result = mgr.remove_tags(bench_file, ["b"])
        assert result.tags == {"a": "1", "c": "3"}

    def test_remove_tags_nonexistent_key(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"a": "1"})
        result = mgr.remove_tags(bench_file, ["z"])
        assert result.tags == {"a": "1"}

    def test_remove_tags_file_not_found(self, tmp_path: Path) -> None:
        mgr = AnnotationManager()
        with pytest.raises(FileNotFoundError):
            mgr.remove_tags(tmp_path / "nope.json", ["key"])

    def test_clear_tags(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"a": "1"})
        sidecar = bench_file.parent / "bench_0.tags.yaml"
        assert sidecar.exists()
        result = mgr.clear_tags(bench_file)
        assert result.tags == {}
        assert not sidecar.exists()

    def test_clear_tags_no_sidecar(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        result = mgr.clear_tags(bench_file)
        assert result.tags == {}

    def test_filter_by_tags_match(self, bench_dir: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_dir / "bench_0.json", {"model": "llama", "gpu": "H100"})
        mgr.add_tags(bench_dir / "bench_1.json", {"model": "llama", "gpu": "A100"})
        mgr.add_tags(bench_dir / "bench_2.json", {"model": "mistral", "gpu": "H100"})

        result = mgr.filter_by_tags(bench_dir, {"model": "llama"})
        assert len(result.matched) == 2
        assert result.total_scanned == 3

    def test_filter_by_tags_multi_match(self, bench_dir: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_dir / "bench_0.json", {"model": "llama", "gpu": "H100"})
        mgr.add_tags(bench_dir / "bench_1.json", {"model": "llama", "gpu": "A100"})

        result = mgr.filter_by_tags(bench_dir, {"model": "llama", "gpu": "H100"})
        assert len(result.matched) == 1

    def test_filter_by_tags_no_match(self, bench_dir: Path) -> None:
        mgr = AnnotationManager()
        result = mgr.filter_by_tags(bench_dir, {"model": "gpt4"})
        assert len(result.matched) == 0
        assert result.total_scanned == 3

    def test_list_all_tags(self, bench_dir: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_dir / "bench_0.json", {"env": "prod"})
        results = mgr.list_all_tags(bench_dir)
        assert len(results) == 3
        tagged = [r for r in results if r.tags]
        assert len(tagged) == 1

    def test_sidecar_yaml_format(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"model": "llama"})
        sidecar = bench_file.parent / "bench_0.tags.yaml"
        data = yaml.safe_load(sidecar.read_text())
        assert data == {"tags": {"model": "llama"}}

    def test_remove_all_tags_cleans_sidecar(self, bench_file: Path) -> None:
        mgr = AnnotationManager()
        mgr.add_tags(bench_file, {"a": "1"})
        sidecar = bench_file.parent / "bench_0.tags.yaml"
        assert sidecar.exists()
        mgr.remove_tags(bench_file, ["a"])
        assert not sidecar.exists()


class TestAnnotateBenchmarkAPI:
    """Test programmatic annotate_benchmark() API."""

    def test_add_tags(self, bench_file: Path) -> None:
        result = annotate_benchmark(bench_file, tags={"model": "llama"})
        assert result.tags == {"model": "llama"}

    def test_remove_keys(self, bench_file: Path) -> None:
        annotate_benchmark(bench_file, tags={"a": "1", "b": "2"})
        result = annotate_benchmark(bench_file, remove_keys=["a"])
        assert result.tags == {"b": "2"}

    def test_add_and_remove(self, bench_file: Path) -> None:
        annotate_benchmark(bench_file, tags={"a": "1", "b": "2"})
        result = annotate_benchmark(bench_file, tags={"c": "3"}, remove_keys=["a"])
        assert "a" not in result.tags
        assert result.tags["c"] == "3"

    def test_get_only(self, bench_file: Path) -> None:
        result = annotate_benchmark(bench_file)
        assert result.tags == {}


class TestFilterResult:
    """Test FilterResult model."""

    def test_model_fields(self) -> None:
        fr = FilterResult(query_tags={"k": "v"}, total_scanned=5)
        assert fr.matched == []
        assert fr.total_scanned == 5


class TestAnnotatedBenchmark:
    """Test AnnotatedBenchmark model."""

    def test_defaults(self) -> None:
        ab = AnnotatedBenchmark(benchmark_path="test.json")
        assert ab.tags == {}
        assert ab.exists is True
