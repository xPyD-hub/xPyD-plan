"""Tests for the batch pipeline runner (M24)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xpyd_plan.pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
    PipelineStep,
    StepResult,
    StepType,
    load_pipeline_config,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_requests(n: int = 50, ttft: float = 20.0, tpot: float = 5.0) -> list[dict]:
    return [
        {
            "request_id": f"req-{i}",
            "prompt_tokens": 100,
            "output_tokens": 50,
            "ttft_ms": ttft + (i % 10) * 0.5,
            "tpot_ms": tpot + (i % 10) * 0.2,
            "total_latency_ms": ttft + tpot * 50 + (i % 10) * 2.0,
            "timestamp": 1700000000.0 + i,
        }
        for i in range(n)
    ]


def _make_benchmark_dict(
    num_p: int = 3, num_d: int = 5, qps: float = 100.0
) -> dict:
    return {
        "metadata": {
            "num_prefill_instances": num_p,
            "num_decode_instances": num_d,
            "total_instances": num_p + num_d,
            "measured_qps": qps,
        },
        "requests": _make_requests(),
    }


def _write_benchmark(tmp_path: Path, name: str = "bench.json", **kwargs) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(_make_benchmark_dict(**kwargs)))
    return p


def _write_pipeline_yaml(tmp_path: Path, config: dict, name: str = "pipeline.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(config))
    return p


# ---------------------------------------------------------------------------
# PipelineConfig model
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_basic_creation(self):
        cfg = PipelineConfig(
            name="test",
            steps=[PipelineStep(name="s1", type=StepType.VALIDATE)],
        )
        assert cfg.name == "test"
        assert len(cfg.steps) == 1

    def test_empty_steps_rejected(self):
        with pytest.raises(Exception):
            PipelineConfig(name="test", steps=[])

    def test_duplicate_step_names_rejected(self):
        with pytest.raises(Exception):
            PipelineConfig(
                name="test",
                steps=[
                    PipelineStep(name="dup", type=StepType.VALIDATE),
                    PipelineStep(name="dup", type=StepType.ANALYZE),
                ],
            )

    def test_default_name(self):
        cfg = PipelineConfig(
            steps=[PipelineStep(name="s1", type=StepType.VALIDATE)]
        )
        assert cfg.name == "pipeline"


# ---------------------------------------------------------------------------
# PipelineStep model
# ---------------------------------------------------------------------------


class TestPipelineStep:
    def test_default_params(self):
        step = PipelineStep(name="s1", type=StepType.ANALYZE)
        assert step.params == {}
        assert step.continue_on_failure is False

    def test_with_params(self):
        step = PipelineStep(
            name="s1",
            type=StepType.ALERT,
            params={"rules": "/tmp/rules.yaml"},
        )
        assert step.params["rules"] == "/tmp/rules.yaml"


# ---------------------------------------------------------------------------
# load_pipeline_config
# ---------------------------------------------------------------------------


class TestLoadPipelineConfig:
    def test_load_valid(self, tmp_path):
        cfg = {
            "name": "my-pipeline",
            "steps": [{"name": "v", "type": "validate"}],
        }
        p = _write_pipeline_yaml(tmp_path, cfg)
        loaded = load_pipeline_config(p)
        assert loaded.name == "my-pipeline"
        assert len(loaded.steps) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent/pipeline.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_pipeline_config(p)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_skips_execution(self, tmp_path):
        cfg = PipelineConfig(
            name="dry",
            steps=[
                PipelineStep(name="v", type=StepType.VALIDATE),
                PipelineStep(name="a", type=StepType.ANALYZE),
            ],
        )
        runner = PipelineRunner(cfg)
        result = runner.run(dry_run=True)
        assert result.dry_run is True
        assert result.success is True
        assert result.completed_steps == 2
        assert result.failed_steps == 0
        for sr in result.step_results:
            assert sr.output.get("dry_run") is True

    def test_dry_run_via_api(self, tmp_path):
        cfg = PipelineConfig(
            name="dry-api",
            steps=[PipelineStep(name="s1", type=StepType.VALIDATE)],
        )
        result = run_pipeline(cfg, dry_run=True)
        assert result.dry_run is True
        assert result.success is True


# ---------------------------------------------------------------------------
# Validate step
# ---------------------------------------------------------------------------


class TestValidateStep:
    def test_validate_runs(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="val",
            steps=[PipelineStep(name="v", type=StepType.VALIDATE)],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True
        assert result.step_results[0].output["count"] == 1

    def test_validate_no_data(self):
        cfg = PipelineConfig(
            name="val",
            steps=[PipelineStep(name="v", type=StepType.VALIDATE)],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[])
        result = runner.run()
        assert result.success is False
        assert "No benchmark data" in result.step_results[0].error


# ---------------------------------------------------------------------------
# Analyze step
# ---------------------------------------------------------------------------


class TestAnalyzeStep:
    def test_analyze_runs(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="ana",
            steps=[PipelineStep(name="a", type=StepType.ANALYZE)],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True
        assert result.step_results[0].output["count"] == 1

    def test_analyze_with_sla_params(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="ana-sla",
            steps=[
                PipelineStep(
                    name="a",
                    type=StepType.ANALYZE,
                    params={"sla": {"ttft_ms": 200.0, "tpot_ms": 100.0, "max_latency_ms": 10000.0}},
                )
            ],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True


# ---------------------------------------------------------------------------
# Compare step
# ---------------------------------------------------------------------------


class TestCompareStep:
    def test_compare_runs(self, tmp_path):
        b1 = _write_benchmark(tmp_path, "b1.json")
        b2 = _write_benchmark(tmp_path, "b2.json")
        cfg = PipelineConfig(
            name="cmp",
            steps=[PipelineStep(name="c", type=StepType.COMPARE)],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[b1, b2])
        result = runner.run()
        assert result.success is True
        assert "has_regression" in result.step_results[0].output

    def test_compare_needs_two_files(self, tmp_path):
        b1 = _write_benchmark(tmp_path, "b1.json")
        cfg = PipelineConfig(
            name="cmp",
            steps=[PipelineStep(name="c", type=StepType.COMPARE)],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[b1])
        result = runner.run()
        assert result.success is False
        assert "at least 2" in result.step_results[0].error


# ---------------------------------------------------------------------------
# Export step
# ---------------------------------------------------------------------------


class TestExportStep:
    def test_export_json(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        out = tmp_path / "output.json"
        cfg = PipelineConfig(
            name="exp",
            steps=[
                PipelineStep(name="v", type=StepType.VALIDATE),
                PipelineStep(
                    name="e",
                    type=StepType.EXPORT,
                    params={"output": str(out)},
                ),
            ],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True
        assert out.exists()
        data = json.loads(out.read_text())
        assert "pipeline_outputs" in data

    def test_export_no_output_path(self, tmp_path):
        cfg = PipelineConfig(
            name="exp",
            steps=[PipelineStep(name="e", type=StepType.EXPORT)],
        )
        runner = PipelineRunner(cfg)
        result = runner.run()
        assert result.success is False

    def test_export_unsupported_format(self, tmp_path):
        cfg = PipelineConfig(
            name="exp",
            steps=[
                PipelineStep(
                    name="e",
                    type=StepType.EXPORT,
                    params={"output": str(tmp_path / "out.xml"), "format": "xml"},
                )
            ],
        )
        runner = PipelineRunner(cfg)
        result = runner.run()
        assert result.success is False


# ---------------------------------------------------------------------------
# Multi-step pipeline
# ---------------------------------------------------------------------------


class TestMultiStep:
    def test_validate_then_analyze(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="multi",
            steps=[
                PipelineStep(name="validate", type=StepType.VALIDATE),
                PipelineStep(name="analyze", type=StepType.ANALYZE),
            ],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True
        assert result.completed_steps == 2

    def test_failure_halts_pipeline(self, tmp_path):
        cfg = PipelineConfig(
            name="halt",
            steps=[
                PipelineStep(name="v", type=StepType.VALIDATE),  # fails: no data
                PipelineStep(name="a", type=StepType.ANALYZE),
            ],
        )
        runner = PipelineRunner(cfg, benchmark_paths=[])
        result = runner.run()
        assert result.success is False
        assert result.completed_steps == 1  # halted after first failure
        assert result.failed_steps == 1

    def test_continue_on_failure(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="continue",
            steps=[
                PipelineStep(
                    name="v",
                    type=StepType.VALIDATE,
                    continue_on_failure=True,
                ),  # no data → fail but continue
                PipelineStep(name="a", type=StepType.ANALYZE),
            ],
        )
        # First step has no benchmark, but continue_on_failure=True
        # However we DO pass benchmarks, so both should succeed
        runner = PipelineRunner(cfg, benchmark_paths=[bench])
        result = runner.run()
        assert result.success is True
        assert result.completed_steps == 2


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


class TestProgrammaticAPI:
    def test_run_pipeline_with_config_object(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="api",
            steps=[PipelineStep(name="v", type=StepType.VALIDATE)],
        )
        result = run_pipeline(cfg, benchmark_paths=[bench])
        assert isinstance(result, PipelineResult)
        assert result.success is True

    def test_run_pipeline_with_yaml_path(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        yaml_cfg = {
            "name": "from-yaml",
            "steps": [{"name": "v", "type": "validate"}],
        }
        p = _write_pipeline_yaml(tmp_path, yaml_cfg)
        result = run_pipeline(p, benchmark_paths=[bench])
        assert result.success is True
        assert result.name == "from-yaml"


# ---------------------------------------------------------------------------
# StepResult model
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_serializable(self):
        sr = StepResult(
            name="s1",
            type=StepType.VALIDATE,
            success=True,
            duration_ms=10.5,
            output={"key": "value"},
        )
        data = sr.model_dump()
        assert isinstance(json.dumps(data), str)

    def test_error_field(self):
        sr = StepResult(
            name="s1",
            type=StepType.ANALYZE,
            success=False,
            duration_ms=5.0,
            error="something broke",
        )
        assert sr.error == "something broke"


# ---------------------------------------------------------------------------
# PipelineResult model
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_serializable(self, tmp_path):
        bench = _write_benchmark(tmp_path)
        cfg = PipelineConfig(
            name="ser",
            steps=[PipelineStep(name="v", type=StepType.VALIDATE)],
        )
        result = run_pipeline(cfg, benchmark_paths=[bench])
        data = result.model_dump()
        assert isinstance(json.dumps(data), str)
        assert data["name"] == "ser"
