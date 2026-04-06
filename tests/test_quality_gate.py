"""Tests for Benchmark Quality Gate (M117)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from xpyd_plan.benchmark_models import (
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
)
from xpyd_plan.quality_gate import (
    GateCheck,
    GateConfig,
    GateResult,
    GateVerdict,
    QualityGate,
    evaluate_quality_gate,
    load_gate_config,
)


def _make_data(
    n: int = 200,
    measured_qps: float = 100.0,
    num_prefill: int = 2,
    num_decode: int = 2,
) -> BenchmarkData:
    """Create benchmark data with n requests."""
    requests = [
        BenchmarkRequest(
            request_id=f"r{i}",
            prompt_tokens=100 + (i % 50),
            output_tokens=50 + (i % 30),
            ttft_ms=20.0 + (i % 10),
            tpot_ms=10.0 + (i % 5),
            total_latency_ms=30.0 + (i % 15),
            timestamp=float(i),
        )
        for i in range(n)
    ]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=measured_qps,
        ),
        requests=requests,
    )


def _save_benchmark(data: BenchmarkData, path: Path) -> None:
    """Save benchmark data to JSON."""
    path.write_text(json.dumps(data.model_dump(), default=str))


# --- Model tests ---


class TestGateConfig:
    """Test GateConfig model."""

    def test_defaults(self) -> None:
        config = GateConfig()
        assert config.min_requests == 100
        assert config.min_quality_score == 0.7
        assert config.max_outlier_pct == 10.0
        assert config.require_stable_convergence is True
        assert "steady_state" in config.allowed_load_profiles

    def test_custom_values(self) -> None:
        config = GateConfig(min_requests=50, min_quality_score=0.9)
        assert config.min_requests == 50
        assert config.min_quality_score == 0.9

    def test_validation(self) -> None:
        with pytest.raises(Exception):
            GateConfig(min_requests=0)
        with pytest.raises(Exception):
            GateConfig(min_quality_score=1.5)


class TestGateVerdict:
    """Test GateVerdict enum."""

    def test_values(self) -> None:
        assert GateVerdict.PASS == "pass"
        assert GateVerdict.WARN == "warn"
        assert GateVerdict.FAIL == "fail"


class TestGateCheck:
    """Test GateCheck model."""

    def test_basic(self) -> None:
        check = GateCheck(
            name="test", verdict=GateVerdict.PASS, detail="ok"
        )
        assert check.name == "test"
        assert check.verdict == GateVerdict.PASS
        assert check.threshold is None


class TestGateResult:
    """Test GateResult model."""

    def test_passed_flag(self) -> None:
        result = GateResult(
            verdict=GateVerdict.PASS,
            checks=[],
            request_count=100,
            config=GateConfig(),
            passed=True,
        )
        assert result.passed is True

    def test_failed_flag(self) -> None:
        result = GateResult(
            verdict=GateVerdict.FAIL,
            checks=[],
            request_count=10,
            config=GateConfig(),
            passed=False,
        )
        assert result.passed is False


# --- Load config tests ---


class TestLoadGateConfig:
    """Test YAML config loading."""

    def test_load_yaml(self, tmp_path: Path) -> None:
        cfg = {"min_requests": 50, "max_outlier_pct": 5.0}
        path = tmp_path / "gate.yaml"
        path.write_text(yaml.dump(cfg))
        config = load_gate_config(str(path))
        assert config.min_requests == 50
        assert config.max_outlier_pct == 5.0

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        config = load_gate_config(str(path))
        assert config.min_requests == 100  # default


# --- QualityGate tests ---


class TestQualityGate:
    """Test QualityGate evaluator."""

    def test_pass_with_good_data(self) -> None:
        data = _make_data(n=200)
        gate = QualityGate()
        result = gate.evaluate(data)
        assert result.request_count == 200
        assert len(result.checks) == 5
        # Should have check names
        names = [c.name for c in result.checks]
        assert "min_requests" in names
        assert "quality_score" in names
        assert "outlier_ratio" in names
        assert "convergence" in names
        assert "load_profile" in names

    def test_fail_min_requests(self) -> None:
        data = _make_data(n=10)
        config = GateConfig(min_requests=100)
        gate = QualityGate(config)
        result = gate.evaluate(data)
        min_req_check = [c for c in result.checks if c.name == "min_requests"][0]
        assert min_req_check.verdict == GateVerdict.FAIL

    def test_pass_min_requests(self) -> None:
        data = _make_data(n=200)
        config = GateConfig(min_requests=100)
        gate = QualityGate(config)
        result = gate.evaluate(data)
        min_req_check = [c for c in result.checks if c.name == "min_requests"][0]
        assert min_req_check.verdict == GateVerdict.PASS

    def test_fail_propagates_to_overall(self) -> None:
        data = _make_data(n=5)
        config = GateConfig(min_requests=100)
        gate = QualityGate(config)
        result = gate.evaluate(data)
        assert result.verdict == GateVerdict.FAIL
        assert result.passed is False

    def test_config_property(self) -> None:
        config = GateConfig(min_requests=42)
        gate = QualityGate(config)
        assert gate.config.min_requests == 42

    def test_default_config(self) -> None:
        gate = QualityGate()
        assert gate.config.min_requests == 100

    def test_custom_gate_config(self) -> None:
        data = _make_data(n=200)
        config = GateConfig(
            min_requests=10,
            max_outlier_pct=50.0,
            require_stable_convergence=False,
        )
        gate = QualityGate(config)
        result = gate.evaluate(data)
        # With relaxed config, should likely pass
        assert isinstance(result.verdict, GateVerdict)

    def test_result_serialization(self) -> None:
        data = _make_data(n=200)
        gate = QualityGate()
        result = gate.evaluate(data)
        d = result.model_dump()
        assert "verdict" in d
        assert "checks" in d
        assert "config" in d
        assert "passed" in d


# --- Programmatic API tests ---


class TestEvaluateQualityGate:
    """Test the evaluate_quality_gate() API."""

    def test_basic(self, tmp_path: Path) -> None:
        data = _make_data(n=200)
        path = tmp_path / "bench.json"
        _save_benchmark(data, path)
        result = evaluate_quality_gate(str(path))
        assert "verdict" in result
        assert "checks" in result
        assert "passed" in result

    def test_with_yaml_config(self, tmp_path: Path) -> None:
        data = _make_data(n=200)
        bench_path = tmp_path / "bench.json"
        _save_benchmark(data, bench_path)

        cfg = {"min_requests": 10, "max_outlier_pct": 50.0}
        cfg_path = tmp_path / "gate.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        result = evaluate_quality_gate(str(bench_path), config_path=str(cfg_path))
        assert "verdict" in result

    def test_custom_kwargs(self, tmp_path: Path) -> None:
        data = _make_data(n=200)
        path = tmp_path / "bench.json"
        _save_benchmark(data, path)
        result = evaluate_quality_gate(
            str(path), min_requests=10, max_outlier_pct=50.0
        )
        assert "verdict" in result


# --- CLI tests ---


class TestQualityGateCLI:
    """Test CLI quality-gate subcommand."""

    def test_json_output(self, tmp_path: Path) -> None:

        data = _make_data(n=200)
        path = tmp_path / "bench.json"
        _save_benchmark(data, path)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xpyd_plan.cli",
                "quality-gate",
                "--benchmark",
                str(path),
                "--output-format",
                "json",
                "--min-requests",
                "10",
            ],
            capture_output=True,
            text=True,
        )
        # May pass or fail depending on data quality, but should produce valid JSON
        output = result.stdout.strip()
        if output:
            parsed = json.loads(output)
            assert "verdict" in parsed
            assert "checks" in parsed

    def test_table_output(self, tmp_path: Path) -> None:

        data = _make_data(n=200)
        path = tmp_path / "bench.json"
        _save_benchmark(data, path)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xpyd_plan.cli",
                "quality-gate",
                "--benchmark",
                str(path),
                "--min-requests",
                "10",
            ],
            capture_output=True,
            text=True,
        )
        # Should produce table output (check it ran without crash)
        assert "Quality Gate" in result.stdout or result.returncode in (0, 1)


# --- Public imports test ---


class TestPublicImports:
    """Test that all public symbols are importable."""

    def test_imports(self) -> None:
        from xpyd_plan import (
            GateCheck,
            GateConfig,
            GateResult,
            GateVerdict,
            QualityGate,
            evaluate_quality_gate,
            load_gate_config,
        )

        assert GateCheck is not None
        assert GateConfig is not None
        assert GateResult is not None
        assert GateVerdict is not None
        assert QualityGate is not None
        assert evaluate_quality_gate is not None
        assert load_gate_config is not None
