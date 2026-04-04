"""Tests for the alert rules engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpyd_plan.alerting import (
    AlertEngine,
    AlertReport,
    AlertResult,
    AlertRule,
    AlertSeverity,
    Comparator,
    evaluate_alerts,
)
from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest


def _make_benchmark_data(
    ttft_values: list[float] | None = None,
    tpot_values: list[float] | None = None,
    total_latency_values: list[float] | None = None,
    qps: float = 100.0,
) -> BenchmarkData:
    """Create benchmark data with specified latency values."""
    n = len(ttft_values) if ttft_values else 100
    if ttft_values is None:
        ttft_values = [50.0 + i * 5 for i in range(n)]
    if tpot_values is None:
        tpot_values = [10.0 + i for i in range(n)]
    if total_latency_values is None:
        total_latency_values = [200.0 + i * 10 for i in range(n)]

    requests = [
        BenchmarkRequest(
            request_id=f"req-{i}",
            prompt_tokens=128,
            output_tokens=64,
            ttft_ms=ttft_values[i],
            tpot_ms=tpot_values[i],
            total_latency_ms=total_latency_values[i],
            timestamp=1000.0 + i,
        )
        for i in range(n)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
        requests=requests,
    )


def _make_rules_yaml(tmp_path: Path, rules: list[dict]) -> Path:
    """Write rules YAML and return path."""
    import yaml

    path = tmp_path / "rules.yaml"
    path.write_text(yaml.dump({"rules": rules}))
    return path


def _make_benchmark_json(tmp_path: Path, data: BenchmarkData) -> Path:
    """Write benchmark JSON and return path."""
    path = tmp_path / "bench.json"
    path.write_text(data.model_dump_json(indent=2))
    return path


class TestAlertRule:
    """Tests for AlertRule model."""

    def test_basic_creation(self) -> None:
        rule = AlertRule(
            name="High TTFT",
            metric="ttft_p99_ms",
            comparator=Comparator.GT,
            threshold=500.0,
        )
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.message == ""

    def test_format_message_default(self) -> None:
        rule = AlertRule(
            name="High TTFT",
            metric="ttft_p99_ms",
            comparator=Comparator.GT,
            threshold=500.0,
        )
        msg = rule.format_message(600.0)
        assert "High TTFT" in msg
        assert "600.00" in msg
        assert "500.00" in msg

    def test_format_message_custom(self) -> None:
        rule = AlertRule(
            name="High TTFT",
            metric="ttft_p99_ms",
            comparator=Comparator.GT,
            threshold=500.0,
            message="ALERT: {metric} is {actual}, limit {threshold}",
        )
        msg = rule.format_message(600.0)
        assert msg == "ALERT: ttft_p99_ms is 600.00, limit 500.00"


class TestAlertEngine:
    """Tests for AlertEngine."""

    def test_evaluate_no_alerts_triggered(self) -> None:
        data = _make_benchmark_data()
        rules = [
            AlertRule(
                name="TTFT check",
                metric="ttft_p99_ms",
                comparator=Comparator.GT,
                threshold=99999.0,
            ),
        ]
        engine = AlertEngine(rules)
        report = engine.evaluate(data)
        assert report.passed is True
        assert report.triggered_count == 0
        assert report.has_critical is False

    def test_evaluate_critical_triggered(self) -> None:
        data = _make_benchmark_data()
        rules = [
            AlertRule(
                name="TTFT check",
                metric="ttft_p99_ms",
                comparator=Comparator.GT,
                threshold=1.0,
                severity=AlertSeverity.CRITICAL,
            ),
        ]
        engine = AlertEngine(rules)
        report = engine.evaluate(data)
        assert report.passed is False
        assert report.triggered_count == 1
        assert report.has_critical is True

    def test_evaluate_warning_only_still_passes(self) -> None:
        data = _make_benchmark_data()
        rules = [
            AlertRule(
                name="QPS check",
                metric="measured_qps",
                comparator=Comparator.LT,
                threshold=200.0,
                severity=AlertSeverity.WARNING,
            ),
        ]
        engine = AlertEngine(rules)
        report = engine.evaluate(data)
        # Warning triggered but no critical → passed
        assert report.passed is True
        assert report.triggered_count == 1
        assert report.has_warning is True
        assert report.has_critical is False

    def test_evaluate_multiple_rules(self) -> None:
        data = _make_benchmark_data()
        rules = [
            AlertRule(
                name="TTFT OK",
                metric="ttft_p99_ms",
                comparator=Comparator.GT,
                threshold=99999.0,
            ),
            AlertRule(
                name="TPOT fail",
                metric="tpot_p99_ms",
                comparator=Comparator.GT,
                threshold=0.1,
            ),
        ]
        engine = AlertEngine(rules)
        report = engine.evaluate(data)
        assert report.total_rules == 2
        assert report.triggered_count == 1
        assert report.passed is False

    def test_comparators(self) -> None:
        data = _make_benchmark_data(qps=100.0)
        for comp, threshold, expected_trigger in [
            (Comparator.GT, 99.0, True),
            (Comparator.GT, 100.0, False),
            (Comparator.GTE, 100.0, True),
            (Comparator.LT, 101.0, True),
            (Comparator.LT, 100.0, False),
            (Comparator.LTE, 100.0, True),
        ]:
            rules = [
                AlertRule(
                    name="test",
                    metric="measured_qps",
                    comparator=comp,
                    threshold=threshold,
                )
            ]
            engine = AlertEngine(rules)
            report = engine.evaluate(data)
            assert report.results[0].triggered == expected_trigger, (
                f"Failed for {comp} {threshold}: expected {expected_trigger}"
            )

    def test_unknown_metric_raises(self) -> None:
        rules = [
            AlertRule(
                name="bad",
                metric="nonexistent_metric",
                comparator=Comparator.GT,
                threshold=1.0,
            ),
        ]
        with pytest.raises(ValueError, match="Unknown metric"):
            AlertEngine(rules)

    def test_rules_property(self) -> None:
        rules = [
            AlertRule(name="a", metric="measured_qps", comparator=Comparator.GT, threshold=1.0),
        ]
        engine = AlertEngine(rules)
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "a"

    def test_result_message_empty_when_not_triggered(self) -> None:
        data = _make_benchmark_data()
        rules = [
            AlertRule(
                name="ok",
                metric="ttft_p99_ms",
                comparator=Comparator.GT,
                threshold=99999.0,
            ),
        ]
        engine = AlertEngine(rules)
        report = engine.evaluate(data)
        assert report.results[0].message == ""

    def test_all_supported_metrics(self) -> None:
        """Verify all supported metrics can be extracted without error."""
        from xpyd_plan.alerting import SUPPORTED_METRICS

        data = _make_benchmark_data()
        for metric in SUPPORTED_METRICS:
            rules = [
                AlertRule(
                    name=f"check-{metric}",
                    metric=metric,
                    comparator=Comparator.GT,
                    threshold=-1.0,
                    severity=AlertSeverity.INFO,
                ),
            ]
            engine = AlertEngine(rules)
            report = engine.evaluate(data)
            assert report.results[0].actual_value >= 0


class TestAlertEngineYAML:
    """Tests for YAML loading."""

    def test_from_yaml(self, tmp_path: Path) -> None:
        rules_path = _make_rules_yaml(tmp_path, [
            {
                "name": "High TTFT",
                "metric": "ttft_p99_ms",
                "comparator": "gt",
                "threshold": 500.0,
                "severity": "critical",
            },
            {
                "name": "Low QPS",
                "metric": "measured_qps",
                "comparator": "lt",
                "threshold": 50.0,
                "severity": "warning",
            },
        ])
        engine = AlertEngine.from_yaml(rules_path)
        assert len(engine.rules) == 2

    def test_from_yaml_invalid_format(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("just a string\n")
        with pytest.raises(ValueError, match="'rules' key"):
            AlertEngine.from_yaml(path)


class TestAlertReport:
    """Tests for AlertReport."""

    def test_summary_passed(self) -> None:
        report = AlertReport(total_rules=3, triggered_count=0, passed=True)
        assert "PASSED" in report.summary

    def test_summary_failed(self) -> None:
        report = AlertReport(
            results=[
                AlertResult(
                    rule_name="test",
                    metric="ttft_p99_ms",
                    actual_value=600.0,
                    threshold=500.0,
                    comparator=Comparator.GT,
                    severity=AlertSeverity.CRITICAL,
                    triggered=True,
                    message="fail",
                ),
            ],
            total_rules=1,
            triggered_count=1,
            has_critical=True,
            passed=False,
        )
        assert "FAILED" in report.summary
        assert "1 critical" in report.summary


class TestEvaluateAlertsAPI:
    """Tests for the programmatic API."""

    def test_evaluate_alerts(self, tmp_path: Path) -> None:
        data = _make_benchmark_data()
        bench_path = _make_benchmark_json(tmp_path, data)
        rules_path = _make_rules_yaml(tmp_path, [
            {
                "name": "Always pass",
                "metric": "ttft_p99_ms",
                "comparator": "gt",
                "threshold": 99999.0,
            },
        ])
        report = evaluate_alerts(bench_path, rules_path)
        assert report.passed is True


class TestAlertCLI:
    """Tests for the alert CLI subcommand."""

    def test_alert_table_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from xpyd_plan.cli import main

        data = _make_benchmark_data()
        bench_path = _make_benchmark_json(tmp_path, data)
        rules_path = _make_rules_yaml(tmp_path, [
            {
                "name": "TTFT OK",
                "metric": "ttft_p99_ms",
                "comparator": "gt",
                "threshold": 99999.0,
            },
        ])
        main(["alert", "--benchmark", str(bench_path), "--rules", str(rules_path)])
        # No exception = exit 0 (passed)

    def test_alert_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from xpyd_plan.cli import main

        data = _make_benchmark_data()
        bench_path = _make_benchmark_json(tmp_path, data)
        rules_path = _make_rules_yaml(tmp_path, [
            {
                "name": "TTFT OK",
                "metric": "ttft_p99_ms",
                "comparator": "gt",
                "threshold": 99999.0,
            },
        ])
        main([
            "alert", "--benchmark", str(bench_path),
            "--rules", str(rules_path), "--output-format", "json",
        ])

    def test_alert_exits_nonzero_on_critical(self, tmp_path: Path) -> None:
        from xpyd_plan.cli import main

        data = _make_benchmark_data()
        bench_path = _make_benchmark_json(tmp_path, data)
        rules_path = _make_rules_yaml(tmp_path, [
            {
                "name": "Always fail",
                "metric": "ttft_p99_ms",
                "comparator": "gt",
                "threshold": 0.0,
                "severity": "critical",
            },
        ])
        with pytest.raises(SystemExit) as exc_info:
            main(["alert", "--benchmark", str(bench_path), "--rules", str(rules_path)])
        assert exc_info.value.code == 1
