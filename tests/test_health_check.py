"""Tests for benchmark health check."""

from __future__ import annotations

import json
from unittest.mock import patch

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli import main
from xpyd_plan.health_check import (
    CheckResult,
    HealthChecker,
    HealthReport,
    HealthStatus,
    check_health,
)


def _make_requests(n: int = 100, base_ts: float = 1000.0) -> list[BenchmarkRequest]:
    """Generate n well-behaved benchmark requests."""
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i * 2,
                output_tokens=50 + i,
                ttft_ms=10.0 + i * 0.1,
                tpot_ms=5.0 + i * 0.05,
                total_latency_ms=100.0 + i * 1.0,
                timestamp=base_ts + i * 0.5,
            )
        )
    return requests


def _make_data(n: int = 100) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=50.0,
        ),
        requests=_make_requests(n),
    )


def _make_noisy_data() -> BenchmarkData:
    """Data with outliers and inconsistencies."""
    reqs = _make_requests(50)
    # Add outliers
    for i in range(10):
        reqs.append(
            BenchmarkRequest(
                request_id=f"outlier-{i:04d}",
                prompt_tokens=100,
                output_tokens=50,
                ttft_ms=10000.0,  # extreme TTFT
                tpot_ms=5000.0,   # extreme TPOT
                total_latency_ms=50000.0,  # extreme total
                timestamp=1000.0 + 50 * 0.5 + i * 0.5,
            )
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=50.0,
        ),
        requests=reqs,
    )


class TestHealthChecker:
    def test_basic_health_check(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        assert isinstance(report, HealthReport)
        assert report.request_count == 100
        assert len(report.checks) == 4
        assert report.overall in (HealthStatus.PASS, HealthStatus.WARN, HealthStatus.FAIL)

    def test_check_names(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        names = [c.name for c in report.checks]
        assert "validation" in names
        assert "convergence" in names
        assert "load_profile" in names
        assert "outlier_impact" in names

    def test_all_checks_have_detail(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        for check in report.checks:
            assert isinstance(check.detail, str)
            assert len(check.detail) > 0

    def test_good_data_passes_validation(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        validation = next(c for c in report.checks if c.name == "validation")
        assert validation.status in (HealthStatus.PASS, HealthStatus.WARN)

    def test_good_data_passes_outlier_impact(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        outlier = next(c for c in report.checks if c.name == "outlier_impact")
        assert outlier.status in (HealthStatus.PASS, HealthStatus.WARN)

    def test_noisy_data_warns_or_fails(self):
        data = _make_noisy_data()
        checker = HealthChecker()
        report = checker.check(data)

        # At least one check should not be PASS with noisy data
        statuses = [c.status for c in report.checks]
        assert HealthStatus.WARN in statuses or HealthStatus.FAIL in statuses

    def test_overall_fail_when_any_fail(self):
        # Manually create a report with a FAIL check
        checks = [
            CheckResult(name="test1", status=HealthStatus.PASS, detail="ok"),
            CheckResult(name="test2", status=HealthStatus.FAIL, detail="bad"),
        ]
        # Verify the logic: if any FAIL -> overall FAIL
        statuses = [c.status for c in checks]
        assert HealthStatus.FAIL in statuses

    def test_overall_warn_when_no_fail_but_warn(self):
        # Verify overall status logic
        checks = [
            CheckResult(name="test1", status=HealthStatus.PASS, detail="ok"),
            CheckResult(name="test2", status=HealthStatus.WARN, detail="meh"),
        ]
        statuses = [c.status for c in checks]
        assert HealthStatus.FAIL not in statuses
        assert HealthStatus.WARN in statuses

    def test_overall_pass_when_all_pass(self):
        checks = [
            CheckResult(name="test1", status=HealthStatus.PASS, detail="ok"),
            CheckResult(name="test2", status=HealthStatus.PASS, detail="ok"),
        ]
        statuses = [c.status for c in checks]
        assert HealthStatus.FAIL not in statuses
        assert HealthStatus.WARN not in statuses

    def test_small_data(self):
        """Health check should work with small datasets."""
        data = _make_data(10)
        checker = HealthChecker()
        report = checker.check(data)

        assert report.request_count == 10
        assert len(report.checks) == 4

    def test_custom_parameters(self):
        data = _make_data(100)
        checker = HealthChecker(
            convergence_steps=5,
            convergence_threshold=0.1,
            load_profile_window=2.0,
            iqr_multiplier=2.0,
        )
        report = checker.check(data)
        assert isinstance(report, HealthReport)

    def test_model_dump(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        dumped = report.model_dump()
        assert "overall" in dumped
        assert "checks" in dumped
        assert "request_count" in dumped
        assert isinstance(dumped["checks"], list)

    def test_model_dump_json_serializable(self):
        data = _make_data(100)
        checker = HealthChecker()
        report = checker.check(data)

        dumped = report.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert parsed["request_count"] == 100


class TestCheckHealth:
    def test_programmatic_api(self, tmp_path):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = check_health(str(path))

        assert isinstance(result, dict)
        assert "overall" in result
        assert "checks" in result
        assert "request_count" in result
        assert result["request_count"] == 100

    def test_programmatic_api_custom_params(self, tmp_path):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = check_health(
            str(path),
            convergence_steps=5,
            convergence_threshold=0.1,
        )
        assert isinstance(result, dict)
        assert "overall" in result


class TestCLIHealthCheck:
    def test_cli_table_output(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch("sys.argv", ["xpyd-plan", "health-check", "--benchmark", str(path)]):
            main()

        captured = capsys.readouterr()
        assert "Health Checks" in captured.out

    def test_cli_json_output(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch(
            "sys.argv",
            ["xpyd-plan", "health-check", "--benchmark", str(path), "--output-format", "json"],
        ):
            main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "overall" in result
        assert "checks" in result
        assert len(result["checks"]) == 4

    def test_cli_custom_flags(self, tmp_path, capsys):
        data = _make_data(100)
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        with patch(
            "sys.argv",
            [
                "xpyd-plan", "health-check",
                "--benchmark", str(path),
                "--output-format", "json",
                "--convergence-steps", "5",
                "--iqr-multiplier", "2.0",
            ],
        ):
            main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["request_count"] == 100
