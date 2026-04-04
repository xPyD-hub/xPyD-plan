"""Alert rules engine for CI/CD threshold-based alerting.

Define alert rules (metric + threshold + comparator) and evaluate benchmark
data against them. Returns structured results with pass/fail for pipeline
integration.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from pydantic import BaseModel, Field

from xpyd_plan.benchmark_models import BenchmarkData


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Comparator(str, Enum):
    """Comparison operators for alert rules."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


_COMPARATOR_FNS: dict[Comparator, Callable[[float, float], bool]] = {
    Comparator.GT: lambda actual, threshold: actual > threshold,
    Comparator.GTE: lambda actual, threshold: actual >= threshold,
    Comparator.LT: lambda actual, threshold: actual < threshold,
    Comparator.LTE: lambda actual, threshold: actual <= threshold,
}

_COMPARATOR_SYMBOLS: dict[Comparator, str] = {
    Comparator.GT: ">",
    Comparator.GTE: ">=",
    Comparator.LT: "<",
    Comparator.LTE: "<=",
}

# Supported metric names and how to extract them from benchmark data.
# Each extractor takes BenchmarkData and returns a float.
_METRIC_EXTRACTORS: dict[str, Callable[[BenchmarkData], float]] = {
    "ttft_p50_ms": lambda d: float(np.percentile([r.ttft_ms for r in d.requests], 50)),
    "ttft_p95_ms": lambda d: float(np.percentile([r.ttft_ms for r in d.requests], 95)),
    "ttft_p99_ms": lambda d: float(np.percentile([r.ttft_ms for r in d.requests], 99)),
    "tpot_p50_ms": lambda d: float(np.percentile([r.tpot_ms for r in d.requests], 50)),
    "tpot_p95_ms": lambda d: float(np.percentile([r.tpot_ms for r in d.requests], 95)),
    "tpot_p99_ms": lambda d: float(np.percentile([r.tpot_ms for r in d.requests], 99)),
    "total_latency_p50_ms": lambda d: float(
        np.percentile([r.total_latency_ms for r in d.requests], 50)
    ),
    "total_latency_p95_ms": lambda d: float(
        np.percentile([r.total_latency_ms for r in d.requests], 95)
    ),
    "total_latency_p99_ms": lambda d: float(
        np.percentile([r.total_latency_ms for r in d.requests], 99)
    ),
    "measured_qps": lambda d: d.metadata.measured_qps,
    "request_count": lambda d: float(len(d.requests)),
}

SUPPORTED_METRICS = sorted(_METRIC_EXTRACTORS.keys())


class AlertRule(BaseModel):
    """A single alert rule definition."""

    name: str = Field(..., description="Human-readable rule name")
    metric: str = Field(..., description="Metric to evaluate (e.g., ttft_p99_ms)")
    comparator: Comparator = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value")
    severity: AlertSeverity = Field(
        AlertSeverity.CRITICAL, description="Alert severity"
    )
    message: str = Field(
        "", description="Custom message template (supports {metric}, {actual}, {threshold})"
    )

    def format_message(self, actual: float) -> str:
        """Format the alert message with actual values."""
        if self.message:
            return self.message.format(
                metric=self.metric,
                actual=f"{actual:.2f}",
                threshold=f"{self.threshold:.2f}",
                name=self.name,
            )
        symbol = _COMPARATOR_SYMBOLS[self.comparator]
        return f"{self.name}: {self.metric} = {actual:.2f} {symbol} {self.threshold:.2f}"


class AlertResult(BaseModel):
    """Result of evaluating a single alert rule."""

    rule_name: str
    metric: str
    actual_value: float
    threshold: float
    comparator: Comparator
    severity: AlertSeverity
    triggered: bool
    message: str


class AlertReport(BaseModel):
    """Aggregated alert evaluation report."""

    results: list[AlertResult] = Field(default_factory=list)
    total_rules: int = 0
    triggered_count: int = 0
    has_critical: bool = False
    has_warning: bool = False
    passed: bool = True

    @property
    def summary(self) -> str:
        """One-line summary of the alert report."""
        if self.passed:
            return f"PASSED — {self.total_rules} rules evaluated, 0 triggered"
        triggered = self.triggered_count
        critical = sum(
            1 for r in self.results if r.triggered and r.severity == AlertSeverity.CRITICAL
        )
        warning = sum(
            1 for r in self.results if r.triggered and r.severity == AlertSeverity.WARNING
        )
        parts = [f"{triggered} triggered"]
        if critical:
            parts.append(f"{critical} critical")
        if warning:
            parts.append(f"{warning} warning")
        return f"FAILED — {self.total_rules} rules evaluated, {', '.join(parts)}"


class AlertEngine:
    """Evaluate benchmark data against a set of alert rules."""

    def __init__(self, rules: list[AlertRule]) -> None:
        self._rules = rules
        self._validate_rules()

    def _validate_rules(self) -> None:
        """Validate that all rules reference supported metrics."""
        for rule in self._rules:
            if rule.metric not in _METRIC_EXTRACTORS:
                available = ", ".join(SUPPORTED_METRICS)
                raise ValueError(
                    f"Unknown metric '{rule.metric}' in rule '{rule.name}'. "
                    f"Supported: {available}"
                )

    @property
    def rules(self) -> list[AlertRule]:
        """Return the configured alert rules."""
        return list(self._rules)

    def evaluate(self, data: BenchmarkData) -> AlertReport:
        """Evaluate all rules against benchmark data."""
        results: list[AlertResult] = []
        for rule in self._rules:
            extractor = _METRIC_EXTRACTORS[rule.metric]
            actual = extractor(data)
            comparator_fn = _COMPARATOR_FNS[rule.comparator]
            triggered = comparator_fn(actual, rule.threshold)

            results.append(
                AlertResult(
                    rule_name=rule.name,
                    metric=rule.metric,
                    actual_value=round(actual, 4),
                    threshold=rule.threshold,
                    comparator=rule.comparator,
                    severity=rule.severity,
                    triggered=triggered,
                    message=rule.format_message(actual) if triggered else "",
                )
            )

        triggered_results = [r for r in results if r.triggered]
        has_critical = any(r.severity == AlertSeverity.CRITICAL for r in triggered_results)
        has_warning = any(r.severity == AlertSeverity.WARNING for r in triggered_results)

        return AlertReport(
            results=results,
            total_rules=len(results),
            triggered_count=len(triggered_results),
            has_critical=has_critical,
            has_warning=has_warning,
            passed=not has_critical,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlertEngine":
        """Load alert rules from a YAML file.

        Expected format:
        ```yaml
        rules:
          - name: "High P99 TTFT"
            metric: ttft_p99_ms
            comparator: gt
            threshold: 500.0
            severity: critical
        ```
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict) or "rules" not in raw:
            raise ValueError(f"Alert rules file must contain a 'rules' key: {path}")

        rules = [AlertRule(**r) for r in raw["rules"]]
        return cls(rules)


def evaluate_alerts(
    benchmark_path: str | Path,
    rules_path: str | Path,
) -> AlertReport:
    """Programmatic API: evaluate alerts for a benchmark file.

    Args:
        benchmark_path: Path to benchmark JSON file.
        rules_path: Path to alert rules YAML file.

    Returns:
        AlertReport with evaluation results.
    """
    with open(benchmark_path) as f:
        raw = json.load(f)
    data = BenchmarkData(**raw)
    engine = AlertEngine.from_yaml(rules_path)
    return engine.evaluate(data)
