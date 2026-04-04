"""Tests for the interactive CLI dashboard (M17)."""

from __future__ import annotations

import io
from pathlib import Path

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.dashboard import (
    Dashboard,
    DashboardState,
    DashboardView,
    _build_latency_panel,
    _build_sla_panel,
    _build_utilization_panel,
    _read_streaming_input,
    build_dashboard_layout,
    run_dashboard,
)
from xpyd_plan.models import SLAConfig


def _make_metadata(
    prefill: int = 2, decode: int = 2, qps: float = 10.0
) -> BenchmarkMetadata:
    return BenchmarkMetadata(
        num_prefill_instances=prefill,
        num_decode_instances=decode,
        total_instances=prefill + decode,
        measured_qps=qps,
    )


def _make_requests(n: int = 20) -> list[BenchmarkRequest]:
    return [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=100 + i,
            output_tokens=50 + i,
            ttft_ms=30.0 + i * 2,
            tpot_ms=8.0 + i * 0.5,
            total_latency_ms=150.0 + i * 5,
            timestamp=1700000000.0 + i,
        )
        for i in range(n)
    ]


def _make_benchmark(n: int = 20) -> BenchmarkData:
    return BenchmarkData(
        metadata=_make_metadata(),
        requests=_make_requests(n),
    )


# --- DashboardState tests ---


class TestDashboardState:
    """Tests for DashboardState."""

    def test_initial_state(self) -> None:
        state = DashboardState()
        assert state.num_requests == 0
        assert state.running is True
        assert state.current_view == DashboardView.LATENCY
        assert state.metadata is None

    def test_update_from_data(self) -> None:
        state = DashboardState()
        data = _make_benchmark()
        state.update_from_data(data)
        assert state.num_requests == 20
        assert state.metadata is not None
        assert state.last_updated > 0

    def test_add_request(self) -> None:
        state = DashboardState()
        req = _make_requests(1)[0]
        state.add_request(req)
        assert state.num_requests == 1
        assert state.last_updated > 0

    def test_set_metadata(self) -> None:
        state = DashboardState()
        meta = _make_metadata()
        state.set_metadata(meta)
        assert state.metadata == meta
        assert state.total_instances == 4

    def test_switch_view(self) -> None:
        state = DashboardState()
        assert state.current_view == DashboardView.LATENCY
        state.switch_view(DashboardView.UTILIZATION)
        assert state.current_view == DashboardView.UTILIZATION
        state.switch_view(DashboardView.SLA)
        assert state.current_view == DashboardView.SLA

    def test_total_instances_override(self) -> None:
        state = DashboardState(total_instances=8)
        data = _make_benchmark()
        state.update_from_data(data)
        assert state.total_instances == 8  # Override preserved


# --- Panel rendering tests ---


class TestPanels:
    """Tests for panel rendering functions."""

    def test_latency_panel_empty(self) -> None:
        state = DashboardState()
        panel = _build_latency_panel(state)
        assert "No data yet" in str(panel.renderable)

    def test_latency_panel_with_data(self) -> None:
        state = DashboardState()
        state.update_from_data(_make_benchmark())
        panel = _build_latency_panel(state)
        assert "Latency Distribution" in panel.title

    def test_utilization_panel_empty(self) -> None:
        state = DashboardState()
        panel = _build_utilization_panel(state)
        assert "No data yet" in str(panel.renderable)

    def test_utilization_panel_with_data(self) -> None:
        state = DashboardState()
        state.update_from_data(_make_benchmark())
        panel = _build_utilization_panel(state)
        assert "Utilization" in panel.title

    def test_sla_panel_empty(self) -> None:
        state = DashboardState()
        panel = _build_sla_panel(state)
        assert "No data yet" in str(panel.renderable)

    def test_sla_panel_pass(self) -> None:
        sla = SLAConfig(ttft_ms=500.0, tpot_ms=100.0, max_latency_ms=1000.0)
        state = DashboardState(sla=sla)
        state.update_from_data(_make_benchmark())
        panel = _build_sla_panel(state)
        assert "SLA Status" in panel.title

    def test_sla_panel_fail(self) -> None:
        sla = SLAConfig(ttft_ms=1.0, tpot_ms=1.0, max_latency_ms=1.0)
        state = DashboardState(sla=sla)
        state.update_from_data(_make_benchmark())
        panel = _build_sla_panel(state)
        assert "SLA Status" in panel.title


# --- Layout tests ---


class TestLayout:
    """Tests for dashboard layout building."""

    def test_layout_renders(self) -> None:
        state = DashboardState()
        state.update_from_data(_make_benchmark())
        layout = build_dashboard_layout(state)
        assert layout is not None

    def test_layout_all_views(self) -> None:
        state = DashboardState()
        state.update_from_data(_make_benchmark())
        for view in DashboardView:
            state.switch_view(view)
            layout = build_dashboard_layout(state)
            assert layout is not None


# --- Dashboard class tests ---


class TestDashboard:
    """Tests for the Dashboard class."""

    def test_load_file(self, tmp_path: Path) -> None:
        data = _make_benchmark()
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        dashboard = Dashboard()
        dashboard.load_file(str(p))
        assert dashboard.state.num_requests == 20

    def test_load_data(self) -> None:
        data = _make_benchmark()
        dashboard = Dashboard()
        dashboard.load_data(data)
        assert dashboard.state.num_requests == 20

    def test_render(self) -> None:
        data = _make_benchmark()
        dashboard = Dashboard()
        dashboard.load_data(data)
        layout = dashboard.render()
        assert layout is not None

    def test_handle_key_quit(self) -> None:
        dashboard = Dashboard()
        result = dashboard.handle_key("q")
        assert result is False
        assert dashboard.state.running is False

    def test_handle_key_switch_views(self) -> None:
        dashboard = Dashboard()
        assert dashboard.handle_key("1") is True
        assert dashboard.state.current_view == DashboardView.LATENCY
        assert dashboard.handle_key("2") is True
        assert dashboard.state.current_view == DashboardView.UTILIZATION
        assert dashboard.handle_key("3") is True
        assert dashboard.state.current_view == DashboardView.SLA

    def test_handle_key_unknown(self) -> None:
        dashboard = Dashboard()
        assert dashboard.handle_key("x") is True

    def test_refresh_interval(self) -> None:
        dashboard = Dashboard(refresh_interval=5.0)
        assert dashboard.refresh_interval == 5.0

    def test_sla_config_passthrough(self) -> None:
        sla = SLAConfig(ttft_ms=100.0)
        dashboard = Dashboard(sla=sla)
        assert dashboard.state.sla.ttft_ms == 100.0


# --- Streaming input tests ---


class TestStreamingInput:
    """Tests for streaming JSONL input."""

    def test_read_requests(self) -> None:
        state = DashboardState()
        meta = _make_metadata()
        state.set_metadata(meta)

        reqs = _make_requests(5)
        lines = "\n".join(r.model_dump_json() for r in reqs) + "\n"
        stream = io.StringIO(lines)

        _read_streaming_input(state, stream)
        assert state.num_requests == 5

    def test_read_metadata_from_stream(self) -> None:
        state = DashboardState()
        meta = _make_metadata(prefill=3, decode=5, qps=20.0)
        lines = meta.model_dump_json() + "\n"
        stream = io.StringIO(lines)

        _read_streaming_input(state, stream)
        assert state.metadata is not None
        assert state.metadata.num_prefill_instances == 3

    def test_skip_malformed_lines(self) -> None:
        state = DashboardState()
        meta = _make_metadata()
        state.set_metadata(meta)

        req = _make_requests(1)[0]
        lines = f"not json\n{req.model_dump_json()}\n{{bad\n"
        stream = io.StringIO(lines)

        _read_streaming_input(state, stream)
        assert state.num_requests == 1

    def test_empty_lines_skipped(self) -> None:
        state = DashboardState()
        meta = _make_metadata()
        state.set_metadata(meta)

        req = _make_requests(1)[0]
        lines = f"\n\n{req.model_dump_json()}\n\n"
        stream = io.StringIO(lines)

        _read_streaming_input(state, stream)
        assert state.num_requests == 1


# --- Programmatic API tests ---


class TestRunDashboard:
    """Tests for the run_dashboard() API."""

    def test_from_data(self) -> None:
        data = _make_benchmark()
        dashboard = run_dashboard(data=data)
        assert dashboard.state.num_requests == 20

    def test_from_file(self, tmp_path: Path) -> None:
        data = _make_benchmark()
        p = tmp_path / "bench.json"
        p.write_text(data.model_dump_json())
        dashboard = run_dashboard(benchmark_path=str(p))
        assert dashboard.state.num_requests == 20

    def test_with_sla(self) -> None:
        data = _make_benchmark()
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=50.0)
        dashboard = run_dashboard(data=data, sla=sla)
        assert dashboard.state.sla.ttft_ms == 100.0

    def test_with_refresh_interval(self) -> None:
        dashboard = run_dashboard(data=_make_benchmark(), refresh_interval=0.5)
        assert dashboard.refresh_interval == 0.5

    def test_empty_dashboard(self) -> None:
        dashboard = run_dashboard()
        assert dashboard.state.num_requests == 0


# --- DashboardView enum tests ---


class TestDashboardView:
    """Tests for DashboardView enum."""

    def test_values(self) -> None:
        assert DashboardView.LATENCY.value == "latency"
        assert DashboardView.UTILIZATION.value == "utilization"
        assert DashboardView.SLA.value == "sla"

    def test_all_views(self) -> None:
        assert len(DashboardView) == 3
