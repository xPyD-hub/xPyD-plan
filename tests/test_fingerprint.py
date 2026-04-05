"""Tests for environment fingerprint module."""

from __future__ import annotations

import json
from pathlib import Path

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.cli import main
from xpyd_plan.fingerprint import (
    Compatibility,
    EnvironmentFingerprint,
    EnvironmentFingerprinter,
    FingerprintComparison,
    FingerprintDiff,
    fingerprint_benchmark,
)


def _make_requests(n: int = 50, base_ts: float = 1000.0) -> list[BenchmarkRequest]:
    requests = []
    for i in range(n):
        requests.append(
            BenchmarkRequest(
                request_id=f"req-{i:04d}",
                prompt_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                ttft_ms=10.0 + i * 0.5,
                tpot_ms=5.0 + i * 0.2,
                total_latency_ms=100.0 + i * 2.0,
                timestamp=base_ts + i * 0.1,
            )
        )
    return requests


def _make_data(
    n: int = 50,
    prefill: int = 2,
    decode: int = 6,
    qps: float = 100.0,
) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=prefill,
            num_decode_instances=decode,
            total_instances=prefill + decode,
            measured_qps=qps,
        ),
        requests=_make_requests(n),
    )


class TestEnvironmentFingerprinter:
    def test_extract_basic(self):
        data = _make_data(50)
        fp = EnvironmentFingerprinter()
        result = fp.extract(data)

        assert result.num_prefill_instances == 2
        assert result.num_decode_instances == 6
        assert result.total_instances == 8
        assert result.measured_qps == 100.0
        assert result.num_requests == 50
        assert result.prompt_tokens_min == 100
        assert result.prompt_tokens_max == 590
        assert result.output_tokens_min == 50
        assert result.output_tokens_max == 295
        assert len(result.hash) == 16

    def test_hash_deterministic(self):
        data = _make_data(50)
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(data)
        r2 = fp.extract(data)
        assert r1.hash == r2.hash

    def test_hash_changes_with_instances(self):
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50, prefill=2, decode=6))
        r2 = fp.extract(_make_data(50, prefill=3, decode=5))
        assert r1.hash != r2.hash

    def test_hash_stable_across_qps(self):
        """Hash is based on major fields only, so QPS change doesn't affect it."""
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50, qps=100.0))
        r2 = fp.extract(_make_data(50, qps=200.0))
        assert r1.hash == r2.hash

    def test_extract_single_request(self):
        data = BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=1,
                num_decode_instances=1,
                total_instances=2,
                measured_qps=10.0,
            ),
            requests=[
                BenchmarkRequest(
                    request_id="req-0",
                    prompt_tokens=100,
                    output_tokens=50,
                    ttft_ms=10.0,
                    tpot_ms=5.0,
                    total_latency_ms=100.0,
                    timestamp=1000.0,
                )
            ],
        )
        fp = EnvironmentFingerprinter()
        result = fp.extract(data)
        assert result.num_requests == 1
        assert result.prompt_tokens_min == 100
        assert result.prompt_tokens_max == 100


class TestFingerprintComparison:
    def test_identical(self):
        fp = EnvironmentFingerprinter()
        data = _make_data(50)
        r1 = fp.extract(data)
        r2 = fp.extract(data)
        comp = fp.compare(r1, r2)

        assert comp.identical is True
        assert comp.compatibility == Compatibility.IDENTICAL
        assert len(comp.differences) == 0

    def test_compatible_qps_diff(self):
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50, qps=100.0))
        r2 = fp.extract(_make_data(50, qps=200.0))
        comp = fp.compare(r1, r2)

        assert comp.identical is False
        assert comp.compatibility == Compatibility.COMPATIBLE
        assert any(d.field == "measured_qps" for d in comp.differences)

    def test_incompatible_instance_diff(self):
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50, prefill=2, decode=6))
        r2 = fp.extract(_make_data(50, prefill=4, decode=4))
        comp = fp.compare(r1, r2)

        assert comp.identical is False
        assert comp.compatibility == Compatibility.INCOMPATIBLE
        field_names = {d.field for d in comp.differences}
        assert "num_prefill_instances" in field_names
        assert "num_decode_instances" in field_names

    def test_compatible_request_count_diff(self):
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50))
        r2 = fp.extract(_make_data(100))
        comp = fp.compare(r1, r2)

        assert comp.compatibility == Compatibility.COMPATIBLE
        assert any(d.field == "num_requests" for d in comp.differences)

    def test_diff_values_correct(self):
        fp = EnvironmentFingerprinter()
        r1 = fp.extract(_make_data(50, prefill=2, decode=6))
        r2 = fp.extract(_make_data(50, prefill=3, decode=5))
        comp = fp.compare(r1, r2)

        prefill_diff = next(d for d in comp.differences if d.field == "num_prefill_instances")
        assert prefill_diff.baseline_value == "2"
        assert prefill_diff.current_value == "3"


class TestFingerprintBenchmarkAPI:
    def test_single_file(self, tmp_path: Path):
        data = _make_data(50)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        result = fingerprint_benchmark(str(f))
        assert isinstance(result, EnvironmentFingerprint)
        assert result.num_requests == 50

    def test_compare_two_files(self, tmp_path: Path):
        d1 = _make_data(50, prefill=2, decode=6)
        d2 = _make_data(50, prefill=4, decode=4)
        f1 = tmp_path / "bench1.json"
        f2 = tmp_path / "bench2.json"
        f1.write_text(d1.model_dump_json())
        f2.write_text(d2.model_dump_json())

        result = fingerprint_benchmark(str(f1), str(f2))
        assert isinstance(result, FingerprintComparison)
        assert result.compatibility == Compatibility.INCOMPATIBLE


class TestFingerprintCLI:
    def test_cli_single(self, tmp_path: Path):
        data = _make_data(50)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main(["fingerprint", "--benchmark", str(f), "--output-format", "json"])

    def test_cli_compare(self, tmp_path: Path):
        d1 = _make_data(50, prefill=2, decode=6)
        d2 = _make_data(50, prefill=4, decode=4)
        f1 = tmp_path / "bench1.json"
        f2 = tmp_path / "bench2.json"
        f1.write_text(d1.model_dump_json())
        f2.write_text(d2.model_dump_json())

        main([
            "fingerprint", "--benchmark", str(f1),
            "--compare", str(f2), "--output-format", "json",
        ])

    def test_cli_table_output(self, tmp_path: Path):
        data = _make_data(50)
        f = tmp_path / "bench.json"
        f.write_text(data.model_dump_json())

        main(["fingerprint", "--benchmark", str(f)])

    def test_cli_compare_table(self, tmp_path: Path):
        d1 = _make_data(50, qps=100.0)
        d2 = _make_data(50, qps=200.0)
        f1 = tmp_path / "bench1.json"
        f2 = tmp_path / "bench2.json"
        f1.write_text(d1.model_dump_json())
        f2.write_text(d2.model_dump_json())

        main(["fingerprint", "--benchmark", str(f1), "--compare", str(f2)])


class TestModels:
    def test_fingerprint_serialization(self):
        fp = EnvironmentFingerprint(
            num_prefill_instances=2,
            num_decode_instances=6,
            total_instances=8,
            measured_qps=100.0,
            num_requests=50,
            prompt_tokens_min=100,
            prompt_tokens_max=500,
            output_tokens_min=50,
            output_tokens_max=250,
            hash="abcdef1234567890",
        )
        data = json.loads(fp.model_dump_json())
        assert data["hash"] == "abcdef1234567890"

    def test_comparison_serialization(self):
        comp = FingerprintComparison(
            baseline_hash="aaa",
            current_hash="bbb",
            compatibility=Compatibility.COMPATIBLE,
            differences=[
                FingerprintDiff(field="measured_qps", baseline_value="100.0", current_value="200.0")
            ],
            identical=False,
        )
        data = json.loads(comp.model_dump_json())
        assert data["compatibility"] == "COMPATIBLE"
        assert len(data["differences"]) == 1
