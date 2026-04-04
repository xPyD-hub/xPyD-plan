"""Benchmark annotation and tagging system.

Attach structured key-value metadata to benchmark files using sidecar
.tags.yaml files. Supports filtering and grouping benchmarks by tags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Annotation(BaseModel):
    """A set of key-value tags for a benchmark file."""

    benchmark_path: str = Field(..., description="Path to the benchmark file")
    tags: dict[str, str] = Field(default_factory=dict, description="Key-value tag pairs")

    @property
    def tags_path(self) -> Path:
        """Sidecar file path for storing tags."""
        p = Path(self.benchmark_path)
        return p.parent / f"{p.stem}.tags.yaml"


class AnnotatedBenchmark(BaseModel):
    """A benchmark file with its associated annotations."""

    benchmark_path: str
    tags: dict[str, str] = Field(default_factory=dict)
    exists: bool = True


class FilterResult(BaseModel):
    """Result of filtering benchmarks by tags."""

    query_tags: dict[str, str]
    matched: list[AnnotatedBenchmark] = Field(default_factory=list)
    total_scanned: int = 0


class AnnotationManager:
    """Manage tags on benchmark files via sidecar .tags.yaml files."""

    @staticmethod
    def _sidecar_path(benchmark_path: str | Path) -> Path:
        """Get the sidecar tags file path for a benchmark."""
        p = Path(benchmark_path)
        return p.parent / f"{p.stem}.tags.yaml"

    def add_tags(self, benchmark_path: str | Path, tags: dict[str, str]) -> Annotation:
        """Add or update tags on a benchmark file.

        Args:
            benchmark_path: Path to the benchmark JSON file.
            tags: Key-value pairs to add/update.

        Returns:
            Updated Annotation with all current tags.

        Raises:
            FileNotFoundError: If the benchmark file doesn't exist.
        """
        bp = Path(benchmark_path)
        if not bp.exists():
            msg = f"Benchmark file not found: {bp}"
            raise FileNotFoundError(msg)

        existing = self.get_tags(benchmark_path)
        existing.tags.update(tags)
        self._save(existing)
        return existing

    def remove_tags(self, benchmark_path: str | Path, keys: list[str]) -> Annotation:
        """Remove specific tag keys from a benchmark.

        Args:
            benchmark_path: Path to the benchmark JSON file.
            keys: Tag keys to remove.

        Returns:
            Updated Annotation after removal.

        Raises:
            FileNotFoundError: If the benchmark file doesn't exist.
        """
        bp = Path(benchmark_path)
        if not bp.exists():
            msg = f"Benchmark file not found: {bp}"
            raise FileNotFoundError(msg)

        annotation = self.get_tags(benchmark_path)
        for key in keys:
            annotation.tags.pop(key, None)
        self._save(annotation)
        return annotation

    def get_tags(self, benchmark_path: str | Path) -> Annotation:
        """Get all tags for a benchmark file.

        Args:
            benchmark_path: Path to the benchmark JSON file.

        Returns:
            Annotation with current tags (empty if no sidecar exists).
        """
        bp = Path(benchmark_path)
        sidecar = self._sidecar_path(bp)
        tags: dict[str, str] = {}
        if sidecar.exists():
            with open(sidecar) as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "tags" in data:
                    tags = {str(k): str(v) for k, v in data["tags"].items()}
        return Annotation(benchmark_path=str(bp), tags=tags)

    def clear_tags(self, benchmark_path: str | Path) -> Annotation:
        """Remove all tags from a benchmark file.

        Args:
            benchmark_path: Path to the benchmark JSON file.

        Returns:
            Empty Annotation.
        """
        bp = Path(benchmark_path)
        sidecar = self._sidecar_path(bp)
        if sidecar.exists():
            sidecar.unlink()
        return Annotation(benchmark_path=str(bp), tags={})

    def filter_by_tags(
        self,
        directory: str | Path,
        tags: dict[str, str],
        pattern: str = "*.json",
    ) -> FilterResult:
        """Find benchmark files matching all specified tags.

        Args:
            directory: Directory to scan for benchmark files.
            tags: Required key-value pairs (all must match).
            pattern: Glob pattern for benchmark files.

        Returns:
            FilterResult with matched benchmarks.
        """
        dirp = Path(directory)
        matched: list[AnnotatedBenchmark] = []
        scanned = 0

        for bp in sorted(dirp.glob(pattern)):
            if bp.name.endswith(".tags.yaml"):
                continue
            scanned += 1
            annotation = self.get_tags(bp)
            if all(annotation.tags.get(k) == v for k, v in tags.items()):
                matched.append(
                    AnnotatedBenchmark(
                        benchmark_path=str(bp),
                        tags=annotation.tags,
                    )
                )

        return FilterResult(query_tags=tags, matched=matched, total_scanned=scanned)

    def list_all_tags(
        self,
        directory: str | Path,
        pattern: str = "*.json",
    ) -> list[AnnotatedBenchmark]:
        """List all benchmark files with their tags in a directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern for benchmark files.

        Returns:
            List of AnnotatedBenchmark for each file found.
        """
        dirp = Path(directory)
        results: list[AnnotatedBenchmark] = []

        for bp in sorted(dirp.glob(pattern)):
            if bp.name.endswith(".tags.yaml"):
                continue
            annotation = self.get_tags(bp)
            results.append(
                AnnotatedBenchmark(
                    benchmark_path=str(bp),
                    tags=annotation.tags,
                )
            )

        return results

    def _save(self, annotation: Annotation) -> None:
        """Save annotation to sidecar file."""
        sidecar = self._sidecar_path(annotation.benchmark_path)
        if not annotation.tags:
            if sidecar.exists():
                sidecar.unlink()
            return
        with open(sidecar, "w") as f:
            yaml.dump({"tags": annotation.tags}, f, default_flow_style=False, sort_keys=True)


def annotate_benchmark(
    benchmark_path: str | Path,
    tags: Optional[dict[str, str]] = None,
    remove_keys: Optional[list[str]] = None,
) -> Annotation:
    """Programmatic API: add/remove tags on a benchmark file.

    Args:
        benchmark_path: Path to the benchmark JSON file.
        tags: Key-value pairs to add (optional).
        remove_keys: Tag keys to remove (optional).

    Returns:
        Updated Annotation.
    """
    manager = AnnotationManager()

    if remove_keys:
        manager.remove_tags(benchmark_path, remove_keys)

    if tags:
        return manager.add_tags(benchmark_path, tags)

    return manager.get_tags(benchmark_path)
