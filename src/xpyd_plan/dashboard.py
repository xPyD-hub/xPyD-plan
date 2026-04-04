"""Interactive CLI dashboard for real-time benchmark monitoring.

Provides a Rich Live-based TUI with auto-refreshing panels showing
latency distributions, utilization metrics, and SLA compliance status.
Supports both file-based (static) and streaming (JSONL) input.
"""

from __future__ import annotations

import json
import threading
import time
from enum import Enum
from pathlib import Path
from typing import TextIO

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.models import SLAConfig


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


class DashboardView(str, Enum):
    """Available dashboard panel views."""

    LATENCY = "latency"
    UTILIZATION = "utilization"
    SLA = "sla"


class DashboardState:
    """Holds current dashboard data and view state."""

    def __init__(
        self,
        sla: SLAConfig | None = None,
        total_instances: int | None = None,
    ) -> None:
        self.requests: list[BenchmarkRequest] = []
        self.metadata: BenchmarkMetadata | None = None
        self.sla = sla or SLAConfig()
        self.total_instances = total_instances
        self.current_view = DashboardView.LATENCY
        self.running = True
        self.last_updated: float = 0.0

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    def update_from_data(self, data: BenchmarkData) -> None:
        """Load data from a BenchmarkData object."""
        self.requests = list(data.requests)
        self.metadata = data.metadata
        if self.total_instances is None:
            self.total_instances = data.metadata.total_instances
        self.last_updated = time.time()

    def add_request(self, request: BenchmarkRequest) -> None:
        """Add a single request (streaming mode)."""
        self.requests.append(request)
        self.last_updated = time.time()

    def set_metadata(self, metadata: BenchmarkMetadata) -> None:
        """Set or update metadata (streaming mode)."""
        self.metadata = metadata
        if self.total_instances is None:
            self.total_instances = metadata.total_instances

    def switch_view(self, view: DashboardView) -> None:
        """Switch the active dashboard view."""
        self.current_view = view


def _build_latency_panel(state: DashboardState) -> Panel:
    """Build the latency distribution panel."""
    if not state.requests:
        return Panel("[dim]No data yet[/dim]", title="📊 Latency Distribution")

    ttft = [r.ttft_ms for r in state.requests]
    tpot = [r.tpot_ms for r in state.requests]
    total = [r.total_latency_ms for r in state.requests]

    table = Table(show_header=True, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for name, values in [("TTFT (ms)", ttft), ("TPOT (ms)", tpot), ("Total (ms)", total)]:
        table.add_row(
            name,
            f"{_percentile(values, 50):.1f}",
            f"{_percentile(values, 95):.1f}",
            f"{_percentile(values, 99):.1f}",
            f"{min(values):.1f}",
            f"{max(values):.1f}",
        )

    return Panel(table, title="📊 Latency Distribution")


def _build_utilization_panel(state: DashboardState) -> Panel:
    """Build the utilization panel."""
    if not state.metadata or not state.requests:
        return Panel("[dim]No data yet[/dim]", title="⚡ Utilization")

    meta = state.metadata
    n_requests = len(state.requests)
    measured_qps = meta.measured_qps

    table = Table(show_header=True, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Prefill Instances", str(meta.num_prefill_instances))
    table.add_row("Decode Instances", str(meta.num_decode_instances))
    table.add_row("Total Instances", str(meta.total_instances))
    table.add_row("Measured QPS", f"{measured_qps:.2f}")
    table.add_row("Total Requests", str(n_requests))

    # Compute avg tokens per request
    avg_prompt = np.mean([r.prompt_tokens for r in state.requests])
    avg_output = np.mean([r.output_tokens for r in state.requests])
    table.add_row("Avg Prompt Tokens", f"{avg_prompt:.0f}")
    table.add_row("Avg Output Tokens", f"{avg_output:.0f}")

    # Prefill utilization: fraction of time prefill instances are busy
    avg_ttft = np.mean([r.ttft_ms for r in state.requests])
    prefill_busy_frac = (measured_qps * avg_ttft / 1000.0) / meta.num_prefill_instances
    prefill_util = min(prefill_busy_frac * 100.0, 100.0)

    # Decode utilization
    avg_decode_time = np.mean(
        [r.total_latency_ms - r.ttft_ms for r in state.requests]
    )
    decode_busy_frac = (measured_qps * avg_decode_time / 1000.0) / meta.num_decode_instances
    decode_util = min(decode_busy_frac * 100.0, 100.0)

    table.add_row("Prefill Utilization", f"{prefill_util:.1f}%")
    table.add_row("Decode Utilization", f"{decode_util:.1f}%")

    return Panel(table, title="⚡ Utilization")


def _build_sla_panel(state: DashboardState) -> Panel:
    """Build the SLA status panel."""
    if not state.requests:
        return Panel("[dim]No data yet[/dim]", title="✅ SLA Status")

    sla = state.sla
    ttft = [r.ttft_ms for r in state.requests]
    tpot = [r.tpot_ms for r in state.requests]
    total = [r.total_latency_ms for r in state.requests]

    table = Table(show_header=True, expand=True)
    table.add_column("SLA Metric", style="cyan")
    table.add_column("Threshold", justify="right")
    table.add_column("Measured P95", justify="right")
    table.add_column("Status", justify="center")

    checks = [
        ("TTFT", sla.ttft_ms, _percentile(ttft, 95)),
        ("TPOT", sla.tpot_ms, _percentile(tpot, 95)),
        ("Total Latency", sla.max_latency_ms, _percentile(total, 95)),
    ]

    for name, threshold, measured in checks:
        if threshold is None:
            status = "[dim]N/A[/dim]"
        elif measured <= threshold:
            status = "[green]✓ PASS[/green]"
        else:
            status = "[red]✗ FAIL[/red]"

        table.add_row(
            name,
            f"{threshold:.1f}" if threshold is not None else "—",
            f"{measured:.1f}",
            status,
        )

    return Panel(table, title="✅ SLA Status")


def build_dashboard_layout(state: DashboardState) -> Layout:
    """Build the full dashboard layout from current state."""
    layout = Layout()

    # Header
    header_text = Text()
    header_text.append("xPyD-plan Dashboard", style="bold cyan")
    header_text.append(f"  │  Requests: {state.num_requests}", style="dim")
    if state.last_updated > 0:
        elapsed = time.time() - state.last_updated
        header_text.append(f"  │  Updated: {elapsed:.0f}s ago", style="dim")
    header_text.append(
        f"  │  View: [{state.current_view.value}]  (1=latency 2=util 3=sla q=quit)",
        style="dim",
    )
    header = Panel(header_text, height=3)

    # Build panels based on current view
    if state.current_view == DashboardView.LATENCY:
        body = _build_latency_panel(state)
    elif state.current_view == DashboardView.UTILIZATION:
        body = _build_utilization_panel(state)
    else:
        body = _build_sla_panel(state)

    layout.split_column(
        Layout(header, size=3),
        Layout(body),
    )
    return layout


def _read_streaming_input(
    state: DashboardState,
    stream: TextIO,
    metadata: BenchmarkMetadata | None = None,
) -> None:
    """Read JSONL streaming input in a background thread."""
    if metadata is not None:
        state.set_metadata(metadata)

    for line in stream:
        if not state.running:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Check if it's metadata or a request
            if "measured_qps" in obj and "num_prefill_instances" in obj:
                state.set_metadata(BenchmarkMetadata(**obj))
            else:
                state.add_request(BenchmarkRequest(**obj))
        except (json.JSONDecodeError, Exception):
            continue  # Skip malformed lines


class Dashboard:
    """Interactive CLI dashboard for benchmark monitoring.

    Args:
        sla: SLA configuration for pass/fail checks.
        total_instances: Override total instance count.
        refresh_interval: Seconds between display refreshes.
    """

    def __init__(
        self,
        sla: SLAConfig | None = None,
        total_instances: int | None = None,
        refresh_interval: float = 2.0,
    ) -> None:
        self.state = DashboardState(sla=sla, total_instances=total_instances)
        self.refresh_interval = refresh_interval

    def load_file(self, path: str) -> None:
        """Load benchmark data from a JSON file."""
        text = Path(path).read_text()
        data = BenchmarkData.model_validate_json(text)
        self.state.update_from_data(data)

    def load_data(self, data: BenchmarkData) -> None:
        """Load benchmark data from a BenchmarkData object."""
        self.state.update_from_data(data)

    def render(self) -> Layout:
        """Render the current dashboard layout (for testing/programmatic use)."""
        return build_dashboard_layout(self.state)

    def run(
        self,
        stream: TextIO | None = None,
        metadata: BenchmarkMetadata | None = None,
        console: Console | None = None,
    ) -> None:
        """Run the interactive dashboard.

        Args:
            stream: Optional JSONL stream for live input.
            metadata: Metadata for streaming mode.
            console: Optional Rich Console instance.
        """
        console = console or Console()
        self.state.running = True

        reader_thread: threading.Thread | None = None
        if stream is not None:
            reader_thread = threading.Thread(
                target=_read_streaming_input,
                args=(self.state, stream, metadata),
                daemon=True,
            )
            reader_thread.start()

        try:
            with Live(
                build_dashboard_layout(self.state),
                console=console,
                refresh_per_second=1.0 / self.refresh_interval,
                screen=False,
            ) as live:
                while self.state.running:
                    live.update(build_dashboard_layout(self.state))
                    time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.state.running = False

    def handle_key(self, key: str) -> bool:
        """Handle a keyboard input. Returns False if dashboard should quit.

        Args:
            key: Single character key press.

        Returns:
            True to continue, False to quit.
        """
        if key == "q":
            self.state.running = False
            return False
        elif key == "1":
            self.state.switch_view(DashboardView.LATENCY)
        elif key == "2":
            self.state.switch_view(DashboardView.UTILIZATION)
        elif key == "3":
            self.state.switch_view(DashboardView.SLA)
        return True


def run_dashboard(
    benchmark_path: str | None = None,
    data: BenchmarkData | None = None,
    sla: SLAConfig | None = None,
    total_instances: int | None = None,
    refresh_interval: float = 2.0,
    stream: TextIO | None = None,
    metadata: BenchmarkMetadata | None = None,
) -> Dashboard:
    """Programmatic API to create and optionally run a dashboard.

    Args:
        benchmark_path: Path to benchmark JSON file.
        data: Pre-loaded BenchmarkData.
        sla: SLA configuration.
        total_instances: Override total instance count.
        refresh_interval: Seconds between refreshes.
        stream: JSONL stream for live mode.
        metadata: Metadata for streaming mode.

    Returns:
        Configured Dashboard instance (call .run() to start interactive mode).
    """
    dashboard = Dashboard(
        sla=sla,
        total_instances=total_instances,
        refresh_interval=refresh_interval,
    )

    if data is not None:
        dashboard.load_data(data)
    elif benchmark_path is not None:
        dashboard.load_file(benchmark_path)

    return dashboard
