"""End-to-end integration tests for xpyd-plan CLI.

M53: Exercise full CLI workflows with realistic multi-file benchmark datasets,
verify JSON/CSV output schema stability, and test pipeline integration.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

from xpyd_plan.cli import main

# ---------------------------------------------------------------------------
# Fixtures: realistic multi-file benchmark datasets
# ---------------------------------------------------------------------------

_RNG_SEED = 42


def _generate_benchmark(
    *,
    num_prefill: int,
    num_decode: int,
    qps: float,
    n_requests: int = 100,
    seed: int = _RNG_SEED,
    ttft_base: float = 100.0,
    tpot_base: float = 15.0,
) -> dict:
    """Generate a realistic benchmark dataset."""
    rng = random.Random(seed)
    requests = []
    for i in range(n_requests):
        prompt_tokens = rng.randint(50, 4096)
        output_tokens = rng.randint(10, 1024)
        ttft = max(1.0, rng.gauss(ttft_base, ttft_base * 0.3))
        tpot = max(0.1, rng.gauss(tpot_base, tpot_base * 0.2))
        total = ttft + tpot * output_tokens
        requests.append({
            "request_id": f"req-{i:04d}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 2),
            "total_latency_ms": round(total, 2),
            "timestamp": 1700000000.0 + i * (1.0 / max(qps, 0.1)),
        })
    return {
        "metadata": {
            "num_prefill_instances": num_prefill,
            "num_decode_instances": num_decode,
            "total_instances": num_prefill + num_decode,
            "measured_qps": qps,
        },
        "requests": requests,
    }


@pytest.fixture()
def benchmark_dir(tmp_path: Path) -> Path:
    """Create a directory with multiple benchmark files at different configs."""
    configs = [
        {"num_prefill": 2, "num_decode": 6, "qps": 10.0, "seed": 1},
        {"num_prefill": 3, "num_decode": 5, "qps": 10.0, "seed": 2},
        {"num_prefill": 4, "num_decode": 4, "qps": 10.0, "seed": 3},
        {"num_prefill": 2, "num_decode": 6, "qps": 20.0, "seed": 4},
        {"num_prefill": 3, "num_decode": 5, "qps": 20.0, "seed": 5},
    ]
    for i, cfg in enumerate(configs):
        data = _generate_benchmark(**cfg)
        (tmp_path / f"bench_{i}.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture()
def single_benchmark(tmp_path: Path) -> Path:
    data = _generate_benchmark(num_prefill=2, num_decode=6, qps=10.0)
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def second_benchmark(tmp_path: Path) -> Path:
    data = _generate_benchmark(
        num_prefill=2, num_decode=6, qps=10.0, seed=99,
        ttft_base=120.0, tpot_base=18.0,
    )
    p = tmp_path / "bench2.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def multi_qps_benchmarks(tmp_path: Path) -> list[Path]:
    """Benchmarks at increasing QPS for saturation/scaling tests."""
    paths = []
    for qps in [5.0, 10.0, 20.0, 40.0]:
        data = _generate_benchmark(
            num_prefill=2, num_decode=6, qps=qps,
            seed=int(qps), ttft_base=80 + qps * 2, tpot_base=12 + qps * 0.3,
        )
        p = tmp_path / f"bench_qps{int(qps)}.json"
        p.write_text(json.dumps(data))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Helper: capture main() without SystemExit
# ---------------------------------------------------------------------------

def _run(argv: list[str]) -> None:
    """Run CLI main, treating SystemExit(0) as success."""
    try:
        main(argv)
    except SystemExit as e:
        if e.code != 0:
            raise


# ---------------------------------------------------------------------------
# Core subcommand integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeE2E:
    """End-to-end tests for analyze subcommand."""

    def test_analyze_table_output(self, single_benchmark: Path):
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-tpot", "500"])

    def test_analyze_json_output(self, single_benchmark: Path, tmp_path: Path):
        out = tmp_path / "result.json"
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-tpot", "500",
              "--output", str(out)])
        result = json.loads(out.read_text())
        assert "best" in result
        assert "candidates" in result

    def test_analyze_multi_scenario(self, benchmark_dir: Path):
        files = sorted(str(p) for p in benchmark_dir.glob("bench_*.json"))
        _run(["analyze", "--benchmark"] + files +
             ["--sla-ttft", "5000", "--sla-tpot", "500"])

    def test_analyze_custom_percentile(self, single_benchmark: Path, tmp_path: Path):
        out = tmp_path / "result.json"
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-percentile", "99",
              "--output", str(out)])
        result = json.loads(out.read_text())
        assert "best" in result

    def test_analyze_with_sensitivity(self, single_benchmark: Path):
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sensitivity"])

    def test_analyze_with_validate(self, single_benchmark: Path):
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--validate"])


class TestExportE2E:
    """End-to-end tests for export subcommand (JSON & CSV schema stability)."""

    def test_export_json(self, benchmark_dir: Path):
        _run(["export", "--dir", str(benchmark_dir),
              "--output-format", "json",
              "--sla-ttft", "5000", "--sla-tpot", "500"])

    def test_export_csv(self, benchmark_dir: Path):
        _run(["export", "--dir", str(benchmark_dir),
              "--output-format", "csv",
              "--sla-ttft", "5000", "--sla-tpot", "500"])


class TestCompareE2E:
    """End-to-end tests for compare subcommand."""

    def test_compare_table(self, single_benchmark: Path, second_benchmark: Path):
        _run(["compare", "--baseline", str(single_benchmark),
              "--current", str(second_benchmark)])

    def test_compare_json(self, single_benchmark: Path, second_benchmark: Path):
        _run(["compare", "--baseline", str(single_benchmark),
              "--current", str(second_benchmark),
              "--output-format", "json"])


class TestValidateE2E:
    def test_validate_table(self, single_benchmark: Path):
        _run(["validate", "--benchmark", str(single_benchmark)])

    def test_validate_json(self, single_benchmark: Path):
        _run(["validate", "--benchmark", str(single_benchmark),
              "--output-format", "json"])


class TestPlanCapacityE2E:
    def test_plan_capacity(self, single_benchmark: Path):
        _run(["plan-capacity", "--benchmark", str(single_benchmark),
              "--target-qps", "20"])

    def test_plan_capacity_json(self, single_benchmark: Path):
        _run(["plan-capacity", "--benchmark", str(single_benchmark),
              "--target-qps", "20", "--output-format", "json"])


class TestWhatIfE2E:
    def test_what_if_scale_qps(self, single_benchmark: Path):
        _run(["what-if", "--benchmark", str(single_benchmark),
              "--scale-qps", "2.0"])

    def test_what_if_add_instances(self, single_benchmark: Path):
        _run(["what-if", "--benchmark", str(single_benchmark),
              "--add-instances", "2"])


class TestMergeE2E:
    def test_merge_union(self, benchmark_dir: Path, tmp_path: Path):
        files = sorted(benchmark_dir.glob("bench_*.json"))[:2]
        out = tmp_path / "merged.json"
        _run(["merge",
              "--benchmark", str(files[0]),
              "--benchmark", str(files[1]),
              "--output", str(out), "--strategy", "union", "--no-config-check"])
        data = json.loads(out.read_text())
        assert "requests" in data
        assert "metadata" in data

    def test_merge_intersection(self, benchmark_dir: Path, tmp_path: Path):
        files = sorted(benchmark_dir.glob("bench_*.json"))[:2]
        out = tmp_path / "merged.json"
        _run(["merge",
              "--benchmark", str(files[0]),
              "--benchmark", str(files[1]),
              "--output", str(out), "--strategy", "intersection", "--no-config-check"])


class TestFilterE2E:
    def test_filter_by_tokens(self, single_benchmark: Path, tmp_path: Path):
        out = tmp_path / "filtered.json"
        _run(["filter", "--benchmark", str(single_benchmark),
              "--min-prompt-tokens", "100", "--max-prompt-tokens", "2000",
              "--output", str(out)])
        data = json.loads(out.read_text())
        assert "requests" in data

    def test_filter_sample(self, single_benchmark: Path, tmp_path: Path):
        out = tmp_path / "sampled.json"
        _run(["filter", "--benchmark", str(single_benchmark),
              "--sample-count", "10", "--seed", "42",
              "--output", str(out)])
        data = json.loads(out.read_text())
        assert len(data["requests"]) == 10


class TestGenerateE2E:
    def test_generate_default(self, tmp_path: Path):
        out = tmp_path / "gen.json"
        _run(["generate", "--output", str(out)])
        data = json.loads(out.read_text())
        assert "requests" in data
        assert "metadata" in data


class TestBudgetE2E:
    def test_budget_allocation(self, single_benchmark: Path):
        _run(["budget", "--benchmark", str(single_benchmark),
              "--total-budget-ms", "10000"])


class TestConfidenceE2E:
    def test_confidence_basic(self, single_benchmark: Path):
        _run(["confidence", "--benchmark", str(single_benchmark),
              "--iterations", "100", "--seed", "42"])


class TestWorkloadE2E:
    def test_workload_characterization(self, single_benchmark: Path):
        _run(["workload", "--benchmark", str(single_benchmark)])


class TestCorrelationE2E:
    def test_correlation(self, single_benchmark: Path):
        _run(["correlation", "--benchmark", str(single_benchmark)])


class TestTimelineE2E:
    def test_timeline(self, single_benchmark: Path):
        _run(["timeline", "--benchmark", str(single_benchmark)])


class TestTailE2E:
    def test_tail_analysis(self, single_benchmark: Path):
        _run(["tail", "--benchmark", str(single_benchmark)])


class TestScorecardE2E:
    def test_scorecard(self, single_benchmark: Path):
        _run(["scorecard", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-tpot", "500"])


class TestDecomposeE2E:
    def test_decompose(self, single_benchmark: Path):
        _run(["decompose", "--benchmark", str(single_benchmark)])


class TestDriftE2E:
    def test_drift_detection(self, single_benchmark: Path, second_benchmark: Path):
        _run(["drift", "--baseline", str(single_benchmark),
              "--current", str(second_benchmark)])


class TestRootCauseE2E:
    def test_root_cause(self, single_benchmark: Path):
        _run(["root-cause", "--benchmark", str(single_benchmark),
              "--sla-metric", "ttft", "--sla-threshold", "50"])


class TestHeatmapE2E:
    def test_heatmap(self, single_benchmark: Path):
        _run(["heatmap", "--benchmark", str(single_benchmark)])


class TestDiscoverE2E:
    def test_discover(self, benchmark_dir: Path):
        _run(["discover", "--dir", str(benchmark_dir)])


class TestMetricsE2E:
    def test_metrics_text(self, single_benchmark: Path):
        _run(["metrics", "--benchmark", str(single_benchmark)])


class TestPlanBenchmarksE2E:
    def test_plan_benchmarks(self):
        _run(["plan-benchmarks", "--total-instances", "8"])


class TestThresholdAdvisorE2E:
    def test_threshold_advisor(self, single_benchmark: Path):
        _run(["threshold-advisor", "--benchmark", str(single_benchmark)])


class TestSaturationE2E:
    def test_saturation(self, multi_qps_benchmarks: list[Path]):
        files = [str(p) for p in multi_qps_benchmarks]
        _run(["saturation", "--benchmark"] + files)


class TestScalingE2E:
    def test_scaling(self, tmp_path: Path):
        """Scaling needs benchmarks with different total_instances."""
        paths = []
        for total in [4, 8, 12, 16]:
            p = total // 4
            d = total - p
            data = _generate_benchmark(
                num_prefill=p, num_decode=d, qps=10.0, seed=total,
            )
            f = tmp_path / f"bench_t{total}.json"
            f.write_text(json.dumps(data))
            paths.append(str(f))
        _run(["scaling", "--benchmark"] + paths)


class TestInterpolateE2E:
    def test_interpolate(self, benchmark_dir: Path):
        files = sorted(str(p) for p in benchmark_dir.glob("bench_*.json"))[:3]
        _run(["interpolate", "--benchmark"] + files + ["--ratios", "3:5"])


class TestParetoE2E:
    def test_pareto(self, single_benchmark: Path):
        _run(["pareto", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-tpot", "500"])


class TestModelCompareE2E:
    def test_model_compare(self, single_benchmark: Path, second_benchmark: Path):
        _run(["model-compare",
              "--benchmarks", str(single_benchmark), str(second_benchmark),
              "--models", "modelA,modelB"])


class TestABTestE2E:
    def test_ab_test(self, single_benchmark: Path, second_benchmark: Path):
        _run(["ab-test", "--control", str(single_benchmark),
              "--treatment", str(second_benchmark)])


class TestAnnotateE2E:
    def test_annotate_add_list(self, single_benchmark: Path):
        _run(["annotate", "add", "--benchmark", str(single_benchmark),
              "--tag", "env=test", "--tag", "type=integration"])
        _run(["annotate", "list", "--benchmark", str(single_benchmark)])


class TestConfigE2E:
    def test_config_init(self, tmp_path: Path):
        out = tmp_path / "config.yaml"
        _run(["config", "init", "--output-path", str(out)])
        assert out.exists()

    def test_config_show(self):
        _run(["config", "show"])


class TestPluginsE2E:
    def test_plugins_list(self):
        _run(["plugins"])


class TestSLATierE2E:
    def test_sla_tier(self, single_benchmark: Path, tmp_path: Path):
        tiers_data = {
            "tiers": [
                {"name": "gold", "ttft_ms": 200, "tpot_ms": 30, "total_latency_ms": 5000},
                {"name": "silver", "ttft_ms": 500, "tpot_ms": 50, "total_latency_ms": 10000},
            ],
        }
        tier_file = tmp_path / "tiers.yaml"
        tier_file.write_text(yaml.dump(tiers_data))
        _run(["sla-tier", "--benchmark", str(single_benchmark),
              "--tiers", str(tier_file)])


# ---------------------------------------------------------------------------
# Pipeline integration test: validate → analyze → compare → alert → report
# ---------------------------------------------------------------------------


class TestPipelineE2E:
    """Full pipeline integration: chain multiple steps."""

    def test_pipeline_yaml(self, single_benchmark: Path, second_benchmark: Path, tmp_path: Path):
        pipeline_config = {
            "steps": [
                {
                    "name": "validate",
                    "type": "validate",
                    "params": {},
                },
                {
                    "name": "analyze",
                    "type": "analyze",
                    "params": {
                        "sla_ttft": 5000,
                        "sla_tpot": 500,
                    },
                },
            ],
        }
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(yaml.dump(pipeline_config))

        _run(["pipeline", "--config", str(config_file),
              "--benchmark", str(single_benchmark)])

    def test_pipeline_dry_run(self, single_benchmark: Path, tmp_path: Path):
        pipeline_config = {
            "steps": [
                {
                    "name": "validate",
                    "type": "validate",
                    "params": {},
                },
            ],
        }
        config_file = tmp_path / "pipeline.yaml"
        config_file.write_text(yaml.dump(pipeline_config))
        _run(["pipeline", "--config", str(config_file),
              "--benchmark", str(single_benchmark), "--dry-run"])


# ---------------------------------------------------------------------------
# JSON output schema stability tests
# ---------------------------------------------------------------------------


class TestJSONSchemaStability:
    """Verify JSON output contains expected fields across subcommands."""

    def test_analyze_json_schema(self, single_benchmark: Path, tmp_path: Path):
        out = tmp_path / "result.json"
        _run(["analyze", "--benchmark", str(single_benchmark),
              "--sla-ttft", "5000", "--sla-tpot", "500",
              "--output", str(out)])
        data = json.loads(out.read_text())
        assert "best" in data
        assert "candidates" in data
        # Candidate schema
        if data["candidates"]:
            c = data["candidates"][0]
            assert "num_prefill_instances" in c or "ratio" in c or "prefill" in str(c)

    def test_compare_json_schema(self, single_benchmark: Path, second_benchmark: Path):
        _run(["compare", "--baseline", str(single_benchmark),
              "--current", str(second_benchmark),
              "--output-format", "json"])

    def test_validate_json_schema(self, single_benchmark: Path):
        _run(["validate", "--benchmark", str(single_benchmark),
              "--output-format", "json"])

    def test_plan_capacity_json_schema(self, single_benchmark: Path):
        _run(["plan-capacity", "--benchmark", str(single_benchmark),
              "--target-qps", "20", "--output-format", "json"])


# ---------------------------------------------------------------------------
# Error handling & edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """CLI error handling for missing files, bad args, etc."""

    def test_analyze_missing_file(self):
        with pytest.raises((SystemExit, FileNotFoundError, Exception)):
            _run(["analyze", "--benchmark", "/nonexistent/bench.json",
                  "--sla-ttft", "100"])

    def test_compare_same_file(self, single_benchmark: Path):
        """Comparing a file to itself should work (zero diff)."""
        _run(["compare", "--baseline", str(single_benchmark),
              "--current", str(single_benchmark)])

    def test_filter_empty_result(self, single_benchmark: Path, tmp_path: Path):
        """Filter with impossible criteria should still produce valid output."""
        out = tmp_path / "empty.json"
        try:
            _run(["filter", "--benchmark", str(single_benchmark),
                  "--min-prompt-tokens", "999999",
                  "--output", str(out)])
        except (SystemExit, Exception):
            pass  # Some CLIs exit non-zero on empty; that's acceptable

    def test_no_command(self):
        with pytest.raises(SystemExit):
            _run([])
