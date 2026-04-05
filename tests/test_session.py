"""Tests for benchmark session manager module."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.session import Session, SessionEntry, SessionManager, SessionReport, manage_session


def _make_benchmark(
    qps: float = 100.0,
    num_prefill: int = 2,
    num_decode: int = 2,
    num_requests: int = 50,
) -> BenchmarkData:
    """Generate synthetic benchmark data."""
    requests = []
    for i in range(num_requests):
        factor = 1.0 + (i % 10) * 0.05
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i}",
                prompt_tokens=100 + i,
                output_tokens=50 + i,
                ttft_ms=10.0 * factor,
                tpot_ms=5.0 * factor,
                total_latency_ms=100.0 * factor,
                timestamp=1000000.0 + i,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _write_benchmark(tmp_path: Path, name: str = "bench.json", **kwargs) -> Path:
    """Write a benchmark file and return its path."""
    data = _make_benchmark(**kwargs)
    p = tmp_path / name
    p.write_text(data.model_dump_json(indent=2))
    return p


class TestSessionManager:
    """Test SessionManager core functionality."""

    def test_create_session(self) -> None:
        mgr = SessionManager(":memory:")
        session = mgr.create("experiment-1", description="First experiment", tags=["h100"])
        assert session.id is not None
        assert session.name == "experiment-1"
        assert session.description == "First experiment"
        assert session.tags == ["h100"]
        mgr.close()

    def test_create_duplicate_raises(self) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("exp-1")
        mgr.close()

    def test_add_benchmark(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        bench_path = _write_benchmark(tmp_path)
        entry = mgr.add("exp-1", bench_path)
        assert entry.id is not None
        assert entry.num_requests == 50
        assert entry.measured_qps == 100.0
        assert entry.num_prefill == 2
        assert entry.num_decode == 2
        mgr.close()

    def test_add_duplicate_benchmark_raises(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        bench_path = _write_benchmark(tmp_path)
        mgr.add("exp-1", bench_path)
        with pytest.raises(ValueError, match="already in session"):
            mgr.add("exp-1", bench_path)
        mgr.close()

    def test_add_to_nonexistent_session_raises(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        bench_path = _write_benchmark(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            mgr.add("nonexistent", bench_path)
        mgr.close()

    def test_remove_benchmark(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        bench_path = _write_benchmark(tmp_path)
        mgr.add("exp-1", bench_path)
        mgr.remove("exp-1", bench_path)
        session = mgr.show("exp-1")
        assert len(session.entries) == 0
        mgr.close()

    def test_remove_nonexistent_raises(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        with pytest.raises(ValueError, match="not found"):
            mgr.remove("exp-1", "/fake/path.json")
        mgr.close()

    def test_list_sessions(self) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1", tags=["a"])
        mgr.create("exp-2", tags=["b"])
        sessions = mgr.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].name == "exp-1"
        assert sessions[1].name == "exp-2"
        mgr.close()

    def test_list_sessions_empty(self) -> None:
        mgr = SessionManager(":memory:")
        sessions = mgr.list_sessions()
        assert sessions == []
        mgr.close()

    def test_show_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1", description="Test")
        bench1 = _write_benchmark(tmp_path, "b1.json", qps=100)
        bench2 = _write_benchmark(tmp_path, "b2.json", qps=200)
        mgr.add("exp-1", bench1)
        mgr.add("exp-1", bench2)
        session = mgr.show("exp-1")
        assert session.name == "exp-1"
        assert session.description == "Test"
        assert len(session.entries) == 2
        mgr.close()

    def test_show_nonexistent_raises(self) -> None:
        mgr = SessionManager(":memory:")
        with pytest.raises(ValueError, match="not found"):
            mgr.show("nope")
        mgr.close()

    def test_delete_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("exp-1")
        bench_path = _write_benchmark(tmp_path)
        mgr.add("exp-1", bench_path)
        mgr.delete("exp-1")
        sessions = mgr.list_sessions()
        assert len(sessions) == 0
        mgr.close()

    def test_delete_nonexistent_raises(self) -> None:
        mgr = SessionManager(":memory:")
        with pytest.raises(ValueError, match="not found"):
            mgr.delete("nope")
        mgr.close()

    def test_multiple_benchmarks_in_session(self, tmp_path: Path) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("multi")
        for i in range(5):
            p = _write_benchmark(tmp_path, f"b{i}.json", qps=100 + i * 10)
            mgr.add("multi", p)
        session = mgr.show("multi")
        assert len(session.entries) == 5
        mgr.close()

    def test_session_tags_preserved(self) -> None:
        mgr = SessionManager(":memory:")
        mgr.create("tagged", tags=["h100", "nightly", "v2.0"])
        session = mgr.show("tagged")
        assert session.tags == ["h100", "nightly", "v2.0"]
        mgr.close()


class TestSessionModels:
    """Test Pydantic model validation."""

    def test_session_entry_model(self) -> None:
        entry = SessionEntry(
            session_id=1,
            benchmark_path="/tmp/bench.json",
            added_at=1000.0,
            num_requests=100,
            measured_qps=50.0,
            num_prefill=2,
            num_decode=3,
        )
        assert entry.session_id == 1
        assert entry.num_requests == 100

    def test_session_model(self) -> None:
        session = Session(
            name="test",
            created_at=1000.0,
            tags=["a", "b"],
        )
        assert session.name == "test"
        assert session.tags == ["a", "b"]
        assert session.entries == []

    def test_session_report_model(self) -> None:
        report = SessionReport(total_sessions=2, total_benchmarks=5)
        assert report.total_sessions == 2
        assert report.total_benchmarks == 5

    def test_session_json_roundtrip(self) -> None:
        session = Session(name="rt", created_at=1000.0, description="desc", tags=["x"])
        dumped = session.model_dump_json()
        loaded = Session.model_validate_json(dumped)
        assert loaded.name == "rt"
        assert loaded.tags == ["x"]


class TestManageSessionAPI:
    """Test the programmatic API."""

    def test_create_via_api(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        result = manage_session("create", db_path=db, name="api-test", tags=["tag1"])
        assert isinstance(result, Session)
        assert result.name == "api-test"

    def test_list_via_api(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        manage_session("create", db_path=db, name="s1")
        manage_session("create", db_path=db, name="s2")
        result = manage_session("list", db_path=db)
        assert isinstance(result, SessionReport)
        assert result.total_sessions == 2

    def test_add_and_show_via_api(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        bench = _write_benchmark(tmp_path, "b.json")
        manage_session("create", db_path=db, name="s1")
        result = manage_session("add", db_path=db, name="s1", benchmark_path=bench)
        assert isinstance(result, Session)
        assert len(result.entries) == 1

    def test_delete_via_api(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        manage_session("create", db_path=db, name="s1")
        result = manage_session("delete", db_path=db, name="s1")
        assert result is None
        report = manage_session("list", db_path=db)
        assert report.total_sessions == 0

    def test_remove_via_api(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        bench = _write_benchmark(tmp_path, "b.json")
        manage_session("create", db_path=db, name="s1")
        manage_session("add", db_path=db, name="s1", benchmark_path=bench)
        result = manage_session("remove", db_path=db, name="s1", benchmark_path=bench)
        assert result is None

    def test_unknown_action_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with pytest.raises(ValueError, match="Unknown action"):
            manage_session("invalid", db_path=db)

    def test_create_missing_name_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        with pytest.raises(ValueError, match="name is required"):
            manage_session("create", db_path=db)
