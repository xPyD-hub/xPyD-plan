"""Tests for token efficiency analysis (M61)."""

from __future__ import annotations

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.token_efficiency import (
    EfficiencyGrade,
    PerRequestEfficiency,
    TokenEfficiencyAnalyzer,
    TokenEfficiencyReport,
    analyze_token_efficiency,
)


def _make_request(
    request_id: str,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 50.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 550.0,
    timestamp: float = 1000.0,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        timestamp=timestamp,
    )


def _make_data(
    requests: list[BenchmarkRequest] | None = None,
    num_prefill: int = 2,
    num_decode: int = 6,
) -> BenchmarkData:
    if requests is None:
        requests = [_make_request(f"req-{i}", timestamp=1000.0 + i) for i in range(20)]
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=num_prefill,
            num_decode_instances=num_decode,
            total_instances=num_prefill + num_decode,
            measured_qps=10.0,
        ),
        requests=requests,
    )


class TestTokenEfficiencyAnalyzer:
    """Tests for TokenEfficiencyAnalyzer."""

    def test_basic_analysis(self):
        data = _make_data()
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert isinstance(report, TokenEfficiencyReport)
        assert report.num_requests == 20
        assert report.per_request == []  # details not requested
        assert report.aggregate.total_output_tokens == 20 * 50
        assert report.aggregate.total_prompt_tokens == 20 * 100
        assert report.aggregate.total_tokens == 20 * 150

    def test_include_details(self):
        data = _make_data()
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze(include_details=True)

        assert len(report.per_request) == 20
        for pr in report.per_request:
            assert isinstance(pr, PerRequestEfficiency)
            assert pr.output_tokens_per_sec > 0
            assert pr.total_tokens_per_sec > 0
            assert pr.total_tokens == pr.prompt_tokens + pr.output_tokens

    def test_output_tps_calculation(self):
        """Output TPS = output_tokens / (total_latency_ms / 1000)."""
        req = _make_request("r1", output_tokens=100, total_latency_ms=1000.0)
        data = _make_data(requests=[req])
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze(include_details=True)

        # 100 tokens / 1.0 sec = 100 TPS
        assert report.per_request[0].output_tokens_per_sec == 100.0

    def test_decode_tps_calculation(self):
        """Decode TPS = output_tokens / ((total_latency - ttft) / 1000)."""
        req = _make_request(
            "r1", output_tokens=200, ttft_ms=200.0, total_latency_ms=1200.0
        )
        data = _make_data(requests=[req])
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze(include_details=True)

        # 200 tokens / 1.0 sec = 200
        assert report.per_request[0].decode_tokens_per_sec == 200.0

    def test_zero_decode_time(self):
        """When ttft == total_latency, decode_tps should be 0."""
        req = _make_request("r1", ttft_ms=500.0, total_latency_ms=500.0)
        data = _make_data(requests=[req])
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze(include_details=True)

        assert report.per_request[0].decode_tokens_per_sec == 0.0

    def test_instance_efficiency(self):
        reqs = [_make_request(f"r{i}", output_tokens=100, prompt_tokens=200) for i in range(10)]
        data = _make_data(requests=reqs, num_prefill=2, num_decode=3)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        ie = report.instance_efficiency
        assert ie.output_tokens_per_instance == 1000 / 5  # 200
        assert ie.output_tokens_per_decode_instance == round(1000 / 3, 2)
        assert ie.prompt_tokens_per_prefill_instance == 2000 / 2  # 1000

    def test_grade_excellent(self):
        """Consistent requests -> EXCELLENT grade."""
        reqs = [
            _make_request(f"r{i}", output_tokens=50, total_latency_ms=500.0)
            for i in range(50)
        ]
        data = _make_data(requests=reqs)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        # All identical -> P95/P50 = 1.0
        assert report.grade == EfficiencyGrade.EXCELLENT

    def test_grade_poor_with_variable_data(self):
        """Highly variable requests -> worse grade."""
        reqs = []
        for i in range(50):
            latency = 100.0 if i < 25 else 5000.0  # 50% outliers
            reqs.append(
                _make_request(f"r{i}", output_tokens=50, total_latency_ms=latency)
            )
        data = _make_data(requests=reqs)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        # With half requests at very different latencies, grade should not be EXCELLENT
        assert report.grade != EfficiencyGrade.EXCELLENT

    def test_aggregate_percentiles(self):
        data = _make_data()
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        agg = report.aggregate
        assert agg.output_tps_min <= agg.output_tps_p50 <= agg.output_tps_p95 <= agg.output_tps_max
        assert agg.decode_tps_p50 <= agg.decode_tps_p95 <= agg.decode_tps_p99
        assert agg.total_tps_p50 <= agg.total_tps_p95

    def test_single_request(self):
        req = _make_request("r1", prompt_tokens=50, output_tokens=25, total_latency_ms=250.0)
        data = _make_data(requests=[req])
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert report.num_requests == 1
        assert report.aggregate.total_tokens == 75

    def test_programmatic_api(self):
        data = _make_data()
        result = analyze_token_efficiency(data)

        assert isinstance(result, dict)
        assert "aggregate" in result
        assert "instance_efficiency" in result
        assert "grade" in result
        assert "num_requests" in result

    def test_programmatic_api_with_details(self):
        data = _make_data()
        result = analyze_token_efficiency(data, include_details=True)

        assert len(result["per_request"]) == 20

    def test_model_serialization(self):
        data = _make_data()
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        d = report.model_dump()
        assert isinstance(d["grade"], str)
        assert d["grade"] in ("EXCELLENT", "GOOD", "FAIR", "POOR")

    def test_varied_requests(self):
        """Requests with different token counts produce valid output."""
        reqs = [
            _make_request("r1", prompt_tokens=10, output_tokens=5, total_latency_ms=100.0),
            _make_request("r2", prompt_tokens=500, output_tokens=200, total_latency_ms=2000.0),
            _make_request("r3", prompt_tokens=100, output_tokens=50, total_latency_ms=500.0),
        ]
        data = _make_data(requests=reqs)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert report.aggregate.total_output_tokens == 255
        assert report.aggregate.total_prompt_tokens == 610

    def test_large_batch(self):
        reqs = [
            _make_request(
                f"r{i}",
                output_tokens=50 + (i % 20),
                total_latency_ms=400.0 + (i % 30) * 10,
                timestamp=1000.0 + i * 0.1,
            )
            for i in range(500)
        ]
        data = _make_data(requests=reqs)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert report.num_requests == 500
        assert report.aggregate.output_tps_mean > 0

    def test_total_tps_includes_prompt(self):
        req = _make_request(
            "r1", prompt_tokens=200, output_tokens=100, total_latency_ms=1000.0
        )
        data = _make_data(requests=[req])
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze(include_details=True)

        pr = report.per_request[0]
        # total_tps = (200+100) / 1.0 = 300
        assert pr.total_tokens_per_sec == 300.0
        # output_tps = 100 / 1.0 = 100
        assert pr.output_tokens_per_sec == 100.0

    def test_grade_good(self):
        """Moderate variability -> GOOD grade."""
        reqs = []
        for i in range(100):
            # Moderate spread: 400-700ms
            latency = 400.0 + (i % 10) * 30
            reqs.append(_make_request(f"r{i}", output_tokens=50, total_latency_ms=latency))
        data = _make_data(requests=reqs)
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert report.grade in (EfficiencyGrade.EXCELLENT, EfficiencyGrade.GOOD)

    def test_report_fields_complete(self):
        data = _make_data()
        analyzer = TokenEfficiencyAnalyzer(data)
        report = analyzer.analyze()

        assert report.grade_reason != ""
        assert isinstance(report.grade, EfficiencyGrade)
        assert report.instance_efficiency.output_tokens_per_instance > 0
        assert report.aggregate.output_tps_mean > 0
