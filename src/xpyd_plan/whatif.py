"""What-If Scenario Simulation for xpyd-plan (M11).

Lets users explore hypothetical scenarios — scaling QPS or changing instance
counts — without re-running benchmarks. All predictions are based on scaling
measured data, consistent with the analyzer's approach.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import (
    AnalysisResult,
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
    RatioCandidate,
)
from xpyd_plan.models import SLAConfig


class WhatIfScenario(BaseModel):
    """A single what-if scenario with its analysis results."""

    label: str = Field(..., description="Human-readable scenario label")
    total_instances: int
    scale_qps: float = Field(1.0, description="QPS multiplier applied")
    instance_delta: int = Field(0, description="Instance count change from baseline")
    best: RatioCandidate | None = None
    analysis: AnalysisResult | None = None


class WhatIfComparison(BaseModel):
    """Side-by-side comparison of multiple what-if scenarios."""

    baseline: WhatIfScenario
    scenarios: list[WhatIfScenario] = Field(default_factory=list)


class WhatIfSimulator:
    """Simulate what-if scenarios by scaling benchmark data.

    Usage:
        sim = WhatIfSimulator()
        sim.load(benchmark_data)
        result = sim.scale_qps(2.0, sla)          # double QPS
        result = sim.scale_instances(4, sla)       # add 4 instances
        comparison = sim.compare([...], sla)       # side-by-side
    """

    def __init__(self) -> None:
        self._data: BenchmarkData | None = None

    @property
    def data(self) -> BenchmarkData:
        if self._data is None:
            raise RuntimeError("No data loaded. Call load() or load_file() first.")
        return self._data

    def load(self, data: BenchmarkData) -> None:
        """Load benchmark data directly."""
        self._data = data

    def load_file(self, path: str | Path) -> BenchmarkData:
        """Load benchmark data from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        raw = json.loads(path.read_text())
        self._data = BenchmarkData(**raw)
        return self._data

    def load_from_dict(self, data: dict) -> BenchmarkData:
        """Load benchmark data from a dictionary."""
        self._data = BenchmarkData(**data)
        return self._data

    def _scale_data_for_qps(self, qps_multiplier: float) -> BenchmarkData:
        """Create a synthetic dataset with scaled QPS.

        When QPS scales by k, per-instance load increases by k, so latencies
        scale proportionally (more queuing, more contention).
        """
        data = self.data
        scaled_requests = []
        for r in data.requests:
            scaled_requests.append(
                BenchmarkRequest(
                    request_id=r.request_id,
                    prompt_tokens=r.prompt_tokens,
                    output_tokens=r.output_tokens,
                    ttft_ms=r.ttft_ms * qps_multiplier,
                    tpot_ms=r.tpot_ms * qps_multiplier,
                    total_latency_ms=r.total_latency_ms * qps_multiplier,
                    timestamp=r.timestamp,
                )
            )
        meta = data.metadata
        return BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=meta.num_prefill_instances,
                num_decode_instances=meta.num_decode_instances,
                total_instances=meta.total_instances,
                measured_qps=meta.measured_qps * qps_multiplier,
            ),
            requests=scaled_requests,
        )

    def scale_qps(
        self,
        qps_multiplier: float,
        sla: SLAConfig,
        total_instances: int | None = None,
    ) -> WhatIfScenario:
        """Predict impact of scaling QPS by the given multiplier.

        Args:
            qps_multiplier: Factor to scale QPS (e.g. 2.0 = double QPS).
            sla: SLA constraints.
            total_instances: Total instances to optimize for (default: same as data).

        Returns:
            WhatIfScenario with analysis at scaled QPS.
        """
        if qps_multiplier <= 0:
            raise ValueError("qps_multiplier must be positive")

        scaled_data = self._scale_data_for_qps(qps_multiplier)
        total = total_instances or scaled_data.metadata.total_instances

        analyzer = BenchmarkAnalyzer()
        analyzer._data = scaled_data
        analysis = analyzer.find_optimal_ratio(total, sla)

        label = f"{qps_multiplier:.1f}x QPS"
        return WhatIfScenario(
            label=label,
            total_instances=total,
            scale_qps=qps_multiplier,
            best=analysis.best,
            analysis=analysis,
        )

    def scale_instances(
        self,
        instance_delta: int,
        sla: SLAConfig,
    ) -> WhatIfScenario:
        """Predict impact of adding/removing instances.

        Args:
            instance_delta: Number of instances to add (positive) or remove (negative).
            sla: SLA constraints.

        Returns:
            WhatIfScenario with analysis at new instance count.
        """
        current_total = self.data.metadata.total_instances
        new_total = current_total + instance_delta
        if new_total < 2:
            raise ValueError(
                f"Cannot reduce to {new_total} instances (minimum 2). "
                f"Current: {current_total}, delta: {instance_delta}"
            )

        analyzer = BenchmarkAnalyzer()
        analyzer._data = self.data
        analysis = analyzer.find_optimal_ratio(new_total, sla)

        sign = "+" if instance_delta >= 0 else ""
        label = f"{sign}{instance_delta} instances ({new_total} total)"
        return WhatIfScenario(
            label=label,
            total_instances=new_total,
            instance_delta=instance_delta,
            best=analysis.best,
            analysis=analysis,
        )

    def baseline(self, sla: SLAConfig) -> WhatIfScenario:
        """Compute baseline scenario (current data as-is)."""
        total = self.data.metadata.total_instances
        analyzer = BenchmarkAnalyzer()
        analyzer._data = self.data
        analysis = analyzer.find_optimal_ratio(total, sla)

        return WhatIfScenario(
            label="Baseline",
            total_instances=total,
            best=analysis.best,
            analysis=analysis,
        )

    def compare(
        self,
        scenarios: list[dict],
        sla: SLAConfig,
    ) -> WhatIfComparison:
        """Compare multiple what-if scenarios side-by-side.

        Args:
            scenarios: List of scenario specs, each a dict with optional keys:
                - scale_qps: float (QPS multiplier)
                - add_instances: int (instance delta)
                - label: str (optional custom label)
            sla: SLA constraints.

        Returns:
            WhatIfComparison with baseline and all scenarios.
        """
        base = self.baseline(sla)
        results: list[WhatIfScenario] = []

        for spec in scenarios:
            qps_mult = spec.get("scale_qps", 1.0)
            inst_delta = spec.get("add_instances", 0)
            custom_label = spec.get("label")

            total = self.data.metadata.total_instances + inst_delta
            if total < 2:
                raise ValueError(
                    f"Scenario would result in {total} instances (minimum 2)"
                )

            # Scale QPS first, then analyze at new instance count
            if qps_mult != 1.0:
                scaled_data = self._scale_data_for_qps(qps_mult)
            else:
                scaled_data = self.data

            analyzer = BenchmarkAnalyzer()
            analyzer._data = scaled_data
            analysis = analyzer.find_optimal_ratio(total, sla)

            if custom_label:
                label = custom_label
            else:
                parts = []
                if qps_mult != 1.0:
                    parts.append(f"{qps_mult:.1f}x QPS")
                if inst_delta != 0:
                    sign = "+" if inst_delta >= 0 else ""
                    parts.append(f"{sign}{inst_delta} instances")
                label = ", ".join(parts) if parts else "Baseline"

            results.append(
                WhatIfScenario(
                    label=label,
                    total_instances=total,
                    scale_qps=qps_mult,
                    instance_delta=inst_delta,
                    best=analysis.best,
                    analysis=analysis,
                )
            )

        return WhatIfComparison(baseline=base, scenarios=results)


def what_if(
    benchmark_path: str,
    scenarios: list[dict],
    sla_config: SLAConfig | dict | None = None,
) -> dict[str, Any]:
    """Programmatic API for what-if simulation.

    Args:
        benchmark_path: Path to benchmark JSON file.
        scenarios: List of scenario specs (see WhatIfSimulator.compare).
        sla_config: SLA configuration.

    Returns:
        Dictionary with comparison results.
    """
    from xpyd_plan.bench_adapter import load_benchmark_auto

    if isinstance(sla_config, dict):
        sla_config = SLAConfig(**sla_config)
    elif sla_config is None:
        sla_config = SLAConfig()

    data = load_benchmark_auto(benchmark_path)
    sim = WhatIfSimulator()
    sim.load(data)
    comparison = sim.compare(scenarios, sla_config)
    return json.loads(comparison.model_dump_json())
