"""Batch pipeline runner for chaining analysis steps.

Define a multi-step analysis pipeline in YAML and execute it in one run.
Steps can reference outputs from earlier steps, enabling workflows like
validate → analyze → alert → report.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from xpyd_plan.benchmark_models import BenchmarkData


class StepType(str, Enum):
    """Supported pipeline step types."""

    VALIDATE = "validate"
    ANALYZE = "analyze"
    COMPARE = "compare"
    ALERT = "alert"
    EXPORT = "export"


class PipelineStep(BaseModel):
    """A single step in the pipeline."""

    name: str = Field(..., description="Step name (unique within pipeline)")
    type: StepType = Field(..., description="Step type")
    params: dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    continue_on_failure: bool = Field(
        False, description="Continue pipeline if this step fails"
    )


class PipelineConfig(BaseModel):
    """Pipeline configuration loaded from YAML."""

    name: str = Field("pipeline", description="Pipeline name")
    steps: list[PipelineStep] = Field(..., min_length=1, description="Ordered steps")

    @field_validator("steps")
    @classmethod
    def _unique_step_names(cls, v: list[PipelineStep]) -> list[PipelineStep]:
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            msg = f"Duplicate step names: {sorted(set(dupes))}"
            raise ValueError(msg)
        return v


class StepResult(BaseModel):
    """Result of a single pipeline step."""

    name: str
    type: StepType
    success: bool
    duration_ms: float = Field(..., ge=0)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""

    name: str
    total_steps: int = Field(..., ge=0)
    completed_steps: int = Field(..., ge=0)
    failed_steps: int = Field(..., ge=0)
    success: bool
    duration_ms: float = Field(..., ge=0)
    step_results: list[StepResult] = Field(default_factory=list)
    dry_run: bool = False


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML pipeline config file.

    Returns:
        Validated PipelineConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    p = Path(path)
    if not p.exists():
        msg = f"Pipeline config not found: {p}"
        raise FileNotFoundError(msg)
    with open(p) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = "Pipeline config must be a YAML mapping"
        raise ValueError(msg)
    return PipelineConfig(**raw)


class PipelineRunner:
    """Execute a pipeline of analysis steps sequentially.

    Each step type maps to a handler that receives benchmark data,
    step params, and outputs from prior steps.
    """

    def __init__(
        self,
        config: PipelineConfig,
        benchmark_paths: list[str | Path] | None = None,
    ) -> None:
        self._config = config
        self._benchmark_paths = [Path(p) for p in (benchmark_paths or [])]
        self._benchmarks: list[BenchmarkData] | None = None

    def _load_benchmarks(self) -> list[BenchmarkData]:
        """Load benchmark data from configured paths."""
        if self._benchmarks is not None:
            return self._benchmarks
        results: list[BenchmarkData] = []
        for p in self._benchmark_paths:
            with open(p) as f:
                raw = json.load(f)
            results.append(BenchmarkData(**raw))
        self._benchmarks = results
        return results

    def run(self, *, dry_run: bool = False) -> PipelineResult:
        """Execute all pipeline steps in order.

        Args:
            dry_run: If True, preview steps without executing.

        Returns:
            PipelineResult with per-step results.
        """
        start = time.monotonic()
        step_results: list[StepResult] = []
        outputs: dict[str, dict[str, Any]] = {}
        failed = 0
        completed = 0

        for step in self._config.steps:
            if dry_run:
                step_results.append(
                    StepResult(
                        name=step.name,
                        type=step.type,
                        success=True,
                        duration_ms=0.0,
                        output={"dry_run": True},
                    )
                )
                completed += 1
                continue

            step_start = time.monotonic()
            try:
                output = self._execute_step(step, outputs)
                elapsed = (time.monotonic() - step_start) * 1000
                step_results.append(
                    StepResult(
                        name=step.name,
                        type=step.type,
                        success=True,
                        duration_ms=round(elapsed, 2),
                        output=output,
                    )
                )
                outputs[step.name] = output
                completed += 1
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.monotonic() - step_start) * 1000
                step_results.append(
                    StepResult(
                        name=step.name,
                        type=step.type,
                        success=False,
                        duration_ms=round(elapsed, 2),
                        error=str(exc),
                    )
                )
                failed += 1
                completed += 1
                if not step.continue_on_failure:
                    break

        total_elapsed = (time.monotonic() - start) * 1000
        all_success = failed == 0

        return PipelineResult(
            name=self._config.name,
            total_steps=len(self._config.steps),
            completed_steps=completed,
            failed_steps=failed,
            success=all_success,
            duration_ms=round(total_elapsed, 2),
            step_results=step_results,
            dry_run=dry_run,
        )

    def _execute_step(
        self, step: PipelineStep, prior_outputs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Dispatch a step to its handler."""
        handler = _STEP_HANDLERS.get(step.type)
        if handler is None:
            msg = f"Unknown step type: {step.type}"
            raise ValueError(msg)
        return handler(self, step, prior_outputs)

    def _run_validate(
        self, step: PipelineStep, _prior: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute a validate step."""
        from xpyd_plan.validator import DataValidator

        benchmarks = self._load_benchmarks()
        if not benchmarks:
            msg = "No benchmark data loaded for validate step"
            raise ValueError(msg)

        method = step.params.get("method", "iqr")
        results = []
        for i, bm in enumerate(benchmarks):
            validator = DataValidator(method=method)
            result = validator.validate(bm)
            results.append({
                "benchmark_index": i,
                "quality_score": result.quality.overall,
                "outlier_count": result.outlier_count,
                "is_clean": result.outlier_count == 0,
            })
        return {"validations": results, "count": len(results)}

    def _run_analyze(
        self, step: PipelineStep, _prior: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute an analyze step."""
        from xpyd_plan.analyzer import BenchmarkAnalyzer
        from xpyd_plan.models import SLAConfig

        benchmarks = self._load_benchmarks()
        if not benchmarks:
            msg = "No benchmark data loaded for analyze step"
            raise ValueError(msg)

        sla_params = step.params.get("sla", {})
        sla = SLAConfig(**sla_params) if sla_params else SLAConfig(
            ttft_ms=100.0, tpot_ms=50.0, max_latency_ms=5000.0
        )

        results = []
        for i, bm in enumerate(benchmarks):
            analyzer = BenchmarkAnalyzer()
            analyzer._data = bm
            total = bm.metadata.total_instances
            analysis = analyzer.find_optimal_ratio(total, sla)
            best = None
            if analysis.best is not None:
                best = {
                    "ratio": f"{analysis.best.num_prefill}P:{analysis.best.num_decode}D",
                    "meets_sla": analysis.best.meets_sla,
                    "waste_rate": analysis.best.waste_rate,
                }
            results.append({
                "benchmark_index": i,
                "total_candidates": len(analysis.candidates),
                "best": best,
            })
        return {"analyses": results, "count": len(results)}

    def _run_compare(
        self, step: PipelineStep, _prior: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute a compare step."""
        from xpyd_plan.comparator import BenchmarkComparator

        benchmarks = self._load_benchmarks()
        if len(benchmarks) < 2:  # noqa: PLR2004
            msg = "Compare step requires at least 2 benchmark files"
            raise ValueError(msg)

        threshold = step.params.get("regression_threshold", 0.1)
        comparator = BenchmarkComparator(threshold=threshold)
        result = comparator.compare(benchmarks[0], benchmarks[1])
        return {
            "has_regression": result.has_regression,
            "regression_count": result.regression_count,
            "threshold": result.threshold,
        }

    def _run_alert(
        self, step: PipelineStep, _prior: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute an alert step."""
        from xpyd_plan.alerting import AlertEngine

        benchmarks = self._load_benchmarks()
        if not benchmarks:
            msg = "No benchmark data loaded for alert step"
            raise ValueError(msg)

        rules_path = step.params.get("rules")
        if not rules_path:
            msg = "Alert step requires 'rules' parameter (path to rules YAML)"
            raise ValueError(msg)

        engine = AlertEngine.from_yaml(rules_path)
        results = []
        has_critical = False
        for i, bm in enumerate(benchmarks):
            report = engine.evaluate(bm)
            if report.has_critical:
                has_critical = True
            results.append({
                "benchmark_index": i,
                "total_alerts": report.total_alerts,
                "triggered_alerts": report.triggered_count,
                "has_critical": report.has_critical,
            })
        return {
            "alerts": results,
            "has_critical": has_critical,
            "count": len(results),
        }

    def _run_export(
        self, step: PipelineStep, prior: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Execute an export step — serializes prior outputs to a file."""
        output_path = step.params.get("output")
        if not output_path:
            msg = "Export step requires 'output' parameter (file path)"
            raise ValueError(msg)

        fmt = step.params.get("format", "json")
        data = {
            "pipeline_outputs": {
                k: v for k, v in prior.items()
            },
        }

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(p, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            msg = f"Unsupported export format: {fmt}"
            raise ValueError(msg)

        return {"exported_to": str(p), "format": fmt}


_STEP_HANDLERS = {
    StepType.VALIDATE: PipelineRunner._run_validate,
    StepType.ANALYZE: PipelineRunner._run_analyze,
    StepType.COMPARE: PipelineRunner._run_compare,
    StepType.ALERT: PipelineRunner._run_alert,
    StepType.EXPORT: PipelineRunner._run_export,
}


def run_pipeline(
    config: PipelineConfig | str | Path,
    benchmark_paths: list[str | Path] | None = None,
    *,
    dry_run: bool = False,
) -> PipelineResult:
    """Programmatic API for pipeline execution.

    Args:
        config: PipelineConfig object or path to YAML config file.
        benchmark_paths: List of benchmark JSON file paths.
        dry_run: If True, preview steps without execution.

    Returns:
        PipelineResult with per-step outcomes.
    """
    if isinstance(config, (str, Path)):
        config = load_pipeline_config(config)
    runner = PipelineRunner(config, benchmark_paths=benchmark_paths)
    return runner.run(dry_run=dry_run)
