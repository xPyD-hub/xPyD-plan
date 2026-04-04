"""Tests for workload characterization and clustering."""

from __future__ import annotations

import json

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.models import SLAConfig
from xpyd_plan.workload import (
    WorkloadCategory,
    WorkloadClassifier,
    WorkloadReport,
    _classify_request,
    classify_workload,
)


def _make_request(
    prompt_tokens: int = 100,
    output_tokens: int = 100,
    ttft_ms: float = 50.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 200.0,
    rid: str = "r1",
    ts: float = 1000.0,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=rid,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=ts,
    )


def _make_data(requests: list[BenchmarkRequest]) -> BenchmarkData:
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=10.0,
        ),
        requests=requests,
    )


# --- Classification tests ---


class TestClassifyRequest:
    def test_short(self):
        req = _make_request(prompt_tokens=50, output_tokens=50)
        assert _classify_request(req) == WorkloadCategory.SHORT

    def test_long(self):
        req = _make_request(prompt_tokens=1500, output_tokens=1000)
        assert _classify_request(req) == WorkloadCategory.LONG

    def test_prefill_heavy(self):
        # ratio = 600/100 = 6.0 >= 3.0, total = 700 (between 256 and 2048)
        req = _make_request(prompt_tokens=600, output_tokens=100)
        assert _classify_request(req) == WorkloadCategory.PREFILL_HEAVY

    def test_decode_heavy(self):
        # ratio = 100/600 = 0.167 <= 0.33, total = 700
        req = _make_request(prompt_tokens=100, output_tokens=600)
        assert _classify_request(req) == WorkloadCategory.DECODE_HEAVY

    def test_balanced(self):
        # ratio = 300/300 = 1.0, total = 600
        req = _make_request(prompt_tokens=300, output_tokens=300)
        assert _classify_request(req) == WorkloadCategory.BALANCED


# --- Classifier tests ---


class TestWorkloadClassifier:
    def test_basic_classification(self):
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid="s1"),  # SHORT
            _make_request(prompt_tokens=300, output_tokens=300, rid="b1"),  # BALANCED
            _make_request(prompt_tokens=600, output_tokens=100, rid="p1"),  # PREFILL_HEAVY
        ]
        data = _make_data(requests)
        report = WorkloadClassifier().classify(data)
        assert isinstance(report, WorkloadReport)
        assert report.profile.total_requests == 3
        non_empty = [c for c in report.classes if c.request_count > 0]
        assert len(non_empty) == 3

    def test_dominant_class(self):
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"s{i}") for i in range(5)
        ] + [
            _make_request(prompt_tokens=300, output_tokens=300, rid="b1"),
        ]
        data = _make_data(requests)
        report = WorkloadClassifier().classify(data)
        assert report.profile.dominant_class == WorkloadCategory.SHORT

    def test_all_five_categories(self):
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid="short"),
            _make_request(prompt_tokens=1500, output_tokens=1000, rid="long"),
            _make_request(prompt_tokens=600, output_tokens=100, rid="prefill"),
            _make_request(prompt_tokens=100, output_tokens=600, rid="decode"),
            _make_request(prompt_tokens=300, output_tokens=300, rid="balanced"),
        ]
        data = _make_data(requests)
        report = WorkloadClassifier().classify(data)
        non_empty = [c for c in report.classes if c.request_count > 0]
        assert len(non_empty) == 5

    def test_sla_compliance(self):
        # All short requests with low latency
        requests = [
            _make_request(
                prompt_tokens=50, output_tokens=50, ttft_ms=20.0,
                tpot_ms=5.0, total_latency_ms=100.0, rid=f"r{i}",
            )
            for i in range(20)
        ]
        data = _make_data(requests)
        sla = SLAConfig(ttft_ms=50.0, tpot_ms=20.0, max_latency_ms=200.0)
        report = WorkloadClassifier(sla=sla).classify(data)
        short_class = next(c for c in report.classes if c.category == WorkloadCategory.SHORT)
        assert short_class.meets_sla is True
        assert short_class.sla_margin_ms is not None
        assert short_class.sla_margin_ms >= 0

    def test_sla_violation(self):
        requests = [
            _make_request(
                prompt_tokens=50, output_tokens=50, ttft_ms=100.0,
                tpot_ms=30.0, total_latency_ms=500.0, rid=f"r{i}",
            )
            for i in range(20)
        ]
        data = _make_data(requests)
        sla = SLAConfig(ttft_ms=50.0, tpot_ms=20.0, max_latency_ms=200.0)
        report = WorkloadClassifier(sla=sla).classify(data)
        short_class = next(c for c in report.classes if c.category == WorkloadCategory.SHORT)
        assert short_class.meets_sla is False
        assert short_class.sla_margin_ms < 0

    def test_bottleneck_identification(self):
        requests = (
            # Short requests: low latency
            [
                _make_request(
                    prompt_tokens=50, output_tokens=50, ttft_ms=20.0,
                    tpot_ms=5.0, total_latency_ms=100.0, rid=f"s{i}",
                )
                for i in range(10)
            ]
            # Balanced requests: high latency (bottleneck)
            + [
                _make_request(
                    prompt_tokens=300, output_tokens=300, ttft_ms=150.0,
                    tpot_ms=40.0, total_latency_ms=600.0, rid=f"b{i}",
                )
                for i in range(10)
            ]
        )
        data = _make_data(requests)
        sla = SLAConfig(ttft_ms=100.0, tpot_ms=20.0, max_latency_ms=400.0)
        report = WorkloadClassifier(sla=sla).classify(data)
        assert report.bottleneck is not None
        assert report.bottleneck.category == WorkloadCategory.BALANCED
        assert report.bottleneck.meets_sla is False

    def test_no_sla_no_bottleneck(self):
        requests = [_make_request(rid=f"r{i}") for i in range(5)]
        data = _make_data(requests)
        report = WorkloadClassifier().classify(data)
        assert report.bottleneck is None

    def test_percentile_stats(self):
        requests = [
            _make_request(
                prompt_tokens=300, output_tokens=300,
                ttft_ms=10.0 + i * 5, tpot_ms=5.0 + i, total_latency_ms=100.0 + i * 10,
                rid=f"r{i}",
            )
            for i in range(100)
        ]
        data = _make_data(requests)
        report = classify_workload(data)
        balanced = next(c for c in report.classes if c.category == WorkloadCategory.BALANCED)
        assert balanced.ttft_p95_ms > balanced.ttft_p50_ms
        assert balanced.tpot_p99_ms >= balanced.tpot_p95_ms

    def test_empty_class_stats(self):
        # Only short requests → other classes empty
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"r{i}")
            for i in range(5)
        ]
        data = _make_data(requests)
        report = classify_workload(data)
        empty = [c for c in report.classes if c.request_count == 0]
        assert len(empty) >= 1
        for c in empty:
            assert c.fraction == 0.0
            assert c.meets_sla is None

    def test_classification_thresholds_in_profile(self):
        requests = [_make_request(rid="r1")]
        data = _make_data(requests)
        report = classify_workload(data)
        t = report.profile.classification_thresholds
        assert "ratio_high" in t
        assert "ratio_low" in t
        assert "total_short" in t
        assert "total_long" in t

    def test_fractions_sum_to_one(self):
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid="s1"),
            _make_request(prompt_tokens=1500, output_tokens=1000, rid="l1"),
            _make_request(prompt_tokens=600, output_tokens=100, rid="p1"),
            _make_request(prompt_tokens=100, output_tokens=600, rid="d1"),
            _make_request(prompt_tokens=300, output_tokens=300, rid="b1"),
        ]
        data = _make_data(requests)
        report = classify_workload(data)
        total_frac = sum(c.fraction for c in report.classes)
        assert abs(total_frac - 1.0) < 1e-9


# --- Programmatic API tests ---


class TestClassifyWorkloadAPI:
    def test_returns_report(self):
        requests = [_make_request(rid=f"r{i}") for i in range(10)]
        data = _make_data(requests)
        report = classify_workload(data)
        assert isinstance(report, WorkloadReport)

    def test_with_sla(self):
        requests = [_make_request(rid=f"r{i}") for i in range(10)]
        data = _make_data(requests)
        sla = SLAConfig(ttft_ms=100.0)
        report = classify_workload(data, sla=sla)
        non_empty = [c for c in report.classes if c.request_count > 0]
        for c in non_empty:
            assert c.meets_sla is not None

    def test_sla_percentile_respected(self):
        # Create requests with varied latency
        requests = [
            _make_request(
                prompt_tokens=300, output_tokens=300,
                ttft_ms=10.0 + i * 2, tpot_ms=5.0, total_latency_ms=100.0,
                rid=f"r{i}",
            )
            for i in range(100)
        ]
        data = _make_data(requests)
        sla_p95 = SLAConfig(ttft_ms=200.0, sla_percentile=95.0)
        sla_p99 = SLAConfig(ttft_ms=200.0, sla_percentile=99.0)
        r95 = classify_workload(data, sla=sla_p95)
        r99 = classify_workload(data, sla=sla_p99)
        b95 = next(c for c in r95.classes if c.category == WorkloadCategory.BALANCED)
        b99 = next(c for c in r99.classes if c.category == WorkloadCategory.BALANCED)
        # P99 margin should be <= P95 margin (stricter)
        assert b99.sla_margin_ms <= b95.sla_margin_ms


# --- CLI tests ---


class TestWorkloadCLI:
    def _make_benchmark_file(self, tmp_path, requests):
        data = {
            "metadata": {
                "num_prefill_instances": 2,
                "num_decode_instances": 2,
                "total_instances": 4,
                "measured_qps": 10.0,
            },
            "requests": [
                {
                    "request_id": r.request_id,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "ttft_ms": r.ttft_ms,
                    "tpot_ms": r.tpot_ms,
                    "total_latency_ms": r.total_latency_ms,
                    "timestamp": r.timestamp,
                }
                for r in requests
            ],
        }
        fp = tmp_path / "bench.json"
        fp.write_text(json.dumps(data))
        return str(fp)

    def test_cli_table_output(self, tmp_path):
        from xpyd_plan.cli._main import main

        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"r{i}")
            for i in range(10)
        ]
        bench = self._make_benchmark_file(tmp_path, requests)
        # Should not raise
        main(["workload", "--benchmark", bench])

    def test_cli_json_output(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"r{i}")
            for i in range(10)
        ]
        bench = self._make_benchmark_file(tmp_path, requests)
        main(["workload", "--benchmark", bench, "--output-format", "json"])

    def test_cli_with_sla(self, tmp_path):
        from xpyd_plan.cli._main import main

        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"r{i}")
            for i in range(10)
        ]
        bench = self._make_benchmark_file(tmp_path, requests)
        main([
            "workload", "--benchmark", bench,
            "--sla-ttft", "100", "--sla-tpot", "50",
            "--output-format", "json",
        ])

    def test_cli_json_parseable(self, tmp_path, capsys):
        from xpyd_plan.cli._main import main

        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid=f"r{i}")
            for i in range(10)
        ]
        bench = self._make_benchmark_file(tmp_path, requests)
        main(["workload", "--benchmark", bench, "--output-format", "json"])


# --- Model serialization tests ---


class TestWorkloadModels:
    def test_report_serializable(self):
        requests = [
            _make_request(prompt_tokens=50, output_tokens=50, rid="s1"),
            _make_request(prompt_tokens=600, output_tokens=100, rid="p1"),
        ]
        data = _make_data(requests)
        report = classify_workload(data)
        d = report.model_dump()
        assert "profile" in d
        assert "classes" in d
        assert isinstance(d["classes"], list)

    def test_json_roundtrip(self):
        requests = [_make_request(rid=f"r{i}") for i in range(5)]
        data = _make_data(requests)
        sla = SLAConfig(ttft_ms=100.0)
        report = classify_workload(data, sla=sla)
        j = report.model_dump_json()
        restored = WorkloadReport.model_validate_json(j)
        assert restored.profile.total_requests == report.profile.total_requests
