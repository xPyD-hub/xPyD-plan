"""Tests for the new CLI 'analyze' subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpyd_plan.cli import main


def _make_benchmark_file(tmp_path: Path, n: int = 50) -> Path:
    import random

    rng = random.Random(123)
    requests = []
    for i in range(n):
        ttft = rng.uniform(50, 500)
        tpot = rng.uniform(10, 40)
        out_tokens = rng.randint(50, 500)
        requests.append({
            "request_id": f"req-{i}",
            "prompt_tokens": rng.randint(100, 2000),
            "output_tokens": out_tokens,
            "ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 2),
            "total_latency_ms": round(ttft + tpot * out_tokens, 2),
            "timestamp": 1700000000.0 + i * 0.1,
        })
    data = {
        "metadata": {
            "num_prefill_instances": 2,
            "num_decode_instances": 6,
            "total_instances": 8,
            "measured_qps": 10.0,
        },
        "requests": requests,
    }
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data))
    return p


def test_cli_analyze_basic(tmp_path: Path):
    bench = _make_benchmark_file(tmp_path)
    main(["analyze", "--benchmark", str(bench), "--sla-ttft", "10000", "--sla-tpot", "10000"])


def test_cli_analyze_with_output(tmp_path: Path):
    bench = _make_benchmark_file(tmp_path)
    out = tmp_path / "result.json"
    main([
        "analyze",
        "--benchmark", str(bench),
        "--sla-ttft", "10000",
        "--output", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert "best" in result
    assert "candidates" in result


def test_cli_analyze_custom_total(tmp_path: Path):
    bench = _make_benchmark_file(tmp_path)
    main([
        "analyze",
        "--benchmark", str(bench),
        "--total-instances", "4",
    ])


def test_cli_no_command():
    with pytest.raises(SystemExit):
        main([])


def test_cli_plan_deprecated(tmp_path: Path, capsys):
    """Legacy plan command should still work but warn."""
    cfg = {
        "sla": {"ttft_ms": 1000},
        "gpu": {
            "name": "test",
            "prefill_tokens_per_sec": 50000,
            "decode_tokens_per_sec": 2000,
        },
        "budget": 4,
        "dataset": {
            "prompt_len_mean": 500,
            "prompt_len_p95": 1000,
            "output_len_mean": 200,
            "output_len_p95": 400,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    import yaml
    cfg_path.write_text(yaml.dump(cfg))

    with pytest.warns(DeprecationWarning, match="deprecated"):
        main(["plan", "--config", str(cfg_path)])
