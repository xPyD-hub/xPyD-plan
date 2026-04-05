"""Tests for deduplication analysis."""

from __future__ import annotations

import json

import pytest

from xpyd_plan.benchmark_models import BenchmarkData, BenchmarkMetadata, BenchmarkRequest
from xpyd_plan.dedup import DeduplicationAnalyzer, DeduplicationReport, DuplicateGroup


def _make_request(
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    ttft_ms: float = 20.0,
    tpot_ms: float = 10.0,
    total_latency_ms: float = 520.0,
    timestamp: float = 1000.0,
    request_id: str = "r1",
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


def _make_data(requests: list[BenchmarkRequest], qps: float = 10.0) -> BenchmarkData:
    if not requests:
        # BenchmarkData requires min_length=1, use a dummy for empty test
        return BenchmarkData(
            metadata=BenchmarkMetadata(
                num_prefill_instances=2,
                num_decode_instances=2,
                total_instances=4,
                measured_qps=qps,
            ),
            requests=[_make_request()],
        )
    return BenchmarkData(
        metadata=BenchmarkMetadata(
            num_prefill_instances=2,
            num_decode_instances=2,
            total_instances=4,
            measured_qps=qps,
        ),
        requests=requests,
    )


class TestDeduplicationAnalyzerSingleRequest:
    def test_single_request(self):
        data = _make_data([_make_request()])
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 1
        assert report.duplicate_count == 0
        assert report.group_count == 0


class TestDeduplicationAnalyzerNoDuplicates:
    def test_unique_requests(self):
        requests = [
            _make_request(prompt_tokens=100, ttft_ms=20.0, request_id="r1"),
            _make_request(prompt_tokens=200, ttft_ms=30.0, request_id="r2"),
            _make_request(prompt_tokens=300, ttft_ms=40.0, request_id="r3"),
        ]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 3
        assert report.unique_requests == 3
        assert report.duplicate_count == 0
        assert report.duplication_rate == 0.0
        assert report.group_count == 0


class TestDeduplicationAnalyzerExactDuplicates:
    def test_exact_duplicates_detected(self):
        r = _make_request(request_id="r1")
        requests = [r, r.model_copy(update={"request_id": "r2"})]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 2
        assert report.unique_requests == 1
        assert report.duplicate_count == 1
        assert report.group_count == 1
        assert report.groups[0].is_exact is True
        assert report.groups[0].count == 2

    def test_multiple_exact_groups(self):
        r1 = _make_request(prompt_tokens=100, request_id="r1")
        r2 = _make_request(prompt_tokens=200, request_id="r2")
        requests = [
            r1,
            r1.model_copy(update={"request_id": "r1b"}),
            r2,
            r2.model_copy(update={"request_id": "r2b"}),
            r2.model_copy(update={"request_id": "r2c"}),
        ]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert report.total_requests == 5
        assert report.duplicate_count == 3  # 1 from group1, 2 from group2
        assert report.unique_requests == 2
        assert report.group_count == 2

    def test_large_group(self):
        r = _make_request()
        requests = [r.model_copy(update={"request_id": f"r{i}"}) for i in range(10)]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert report.largest_group_size == 10
        assert report.duplicate_count == 9


class TestDeduplicationAnalyzerNearDuplicates:
    def test_near_duplicates_with_tolerance(self):
        r1 = _make_request(ttft_ms=20.0, tpot_ms=10.0, request_id="r1")
        r2 = _make_request(ttft_ms=20.5, tpot_ms=10.3, request_id="r2")
        data = _make_data([r1, r2])
        analyzer = DeduplicationAnalyzer()

        # Exact: no match
        report_exact = analyzer.analyze(data, tolerance_ms=0.0)
        assert report_exact.duplicate_count == 0

        # Tolerance 1.0: match
        report_near = analyzer.analyze(data, tolerance_ms=1.0)
        assert report_near.duplicate_count == 1
        assert report_near.groups[0].is_exact is False

    def test_near_duplicates_different_tokens_not_grouped(self):
        r1 = _make_request(prompt_tokens=100, ttft_ms=20.0, request_id="r1")
        r2 = _make_request(prompt_tokens=200, ttft_ms=20.0, request_id="r2")
        data = _make_data([r1, r2])
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data, tolerance_ms=1.0)
        assert report.duplicate_count == 0

    def test_tolerance_boundary(self):
        r1 = _make_request(ttft_ms=20.0, tpot_ms=10.0, request_id="r1")
        r2 = _make_request(ttft_ms=21.0, tpot_ms=10.0, request_id="r2")
        data = _make_data([r1, r2])
        analyzer = DeduplicationAnalyzer()

        # tolerance exactly 1.0 should match
        report = analyzer.analyze(data, tolerance_ms=1.0)
        assert report.duplicate_count == 1

        # tolerance 0.5 should not match
        report = analyzer.analyze(data, tolerance_ms=0.5)
        assert report.duplicate_count == 0


class TestDeduplicationAnalyzerRecommendation:
    def test_high_duplication_recommendation(self):
        r = _make_request()
        # 20 copies = 19/20 = 95% duplication
        requests = [r.model_copy(update={"request_id": f"r{i}"}) for i in range(20)]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert "High duplication" in report.recommendation

    def test_moderate_duplication_recommendation(self):
        r = _make_request()
        # 2 dups out of 50 = 2% duplication
        requests = [_make_request(prompt_tokens=i, request_id=f"r{i}") for i in range(1, 49)]
        requests.append(r.model_copy(update={"request_id": "dup1"}))
        requests.append(r.model_copy(update={"request_id": "dup2"}))
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert "Moderate duplication" in report.recommendation

    def test_clean_data_recommendation(self):
        requests = [_make_request(prompt_tokens=i, request_id=f"r{i}") for i in range(1, 101)]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        report = analyzer.analyze(data)
        assert "clean" in report.recommendation.lower()


class TestDeduplicate:
    def test_deduplicate_removes_extras(self):
        r = _make_request()
        requests = [
            r.model_copy(update={"request_id": "r1"}),
            r.model_copy(update={"request_id": "r2"}),
            r.model_copy(update={"request_id": "r3"}),
        ]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        deduped = analyzer.deduplicate(data)
        assert len(deduped.requests) == 1
        assert deduped.requests[0].request_id == "r1"

    def test_deduplicate_adjusts_qps(self):
        r = _make_request()
        requests = [r.model_copy(update={"request_id": f"r{i}"}) for i in range(4)]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        deduped = analyzer.deduplicate(data)
        assert deduped.metadata.measured_qps == pytest.approx(2.5, abs=0.01)  # 10.0 * 1/4

    def test_deduplicate_preserves_unique(self):
        requests = [_make_request(prompt_tokens=i, request_id=f"r{i}") for i in range(1, 6)]
        data = _make_data(requests)
        analyzer = DeduplicationAnalyzer()
        deduped = analyzer.deduplicate(data)
        assert len(deduped.requests) == 5

    def test_deduplicate_single_request(self):
        data = _make_data([_make_request()])
        analyzer = DeduplicationAnalyzer()
        deduped = analyzer.deduplicate(data)
        assert len(deduped.requests) == 1

    def test_deduplicate_with_tolerance(self):
        r1 = _make_request(ttft_ms=20.0, tpot_ms=10.0, request_id="r1")
        r2 = _make_request(ttft_ms=20.2, tpot_ms=10.1, request_id="r2")
        r3 = _make_request(prompt_tokens=500, request_id="r3")
        data = _make_data([r1, r2, r3])
        analyzer = DeduplicationAnalyzer()
        deduped = analyzer.deduplicate(data, tolerance_ms=1.0)
        assert len(deduped.requests) == 2


class TestAnalyzeDedupAPI:
    def test_programmatic_api(self, tmp_path):
        from xpyd_plan.dedup import analyze_dedup

        r = _make_request()
        data = _make_data([
            r.model_copy(update={"request_id": "r1"}),
            r.model_copy(update={"request_id": "r2"}),
            _make_request(prompt_tokens=999, request_id="r3"),
        ])
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(data.model_dump()))

        result = analyze_dedup(str(path))
        assert result["total_requests"] == 3
        assert result["duplicate_count"] == 1
        assert result["unique_requests"] == 2


class TestDuplicateGroupModel:
    def test_group_validation(self):
        g = DuplicateGroup(
            group_id=0,
            count=3,
            representative_prompt_tokens=100,
            representative_output_tokens=50,
            request_indices=[0, 1, 2],
            is_exact=True,
        )
        assert g.count == 3

    def test_group_min_count(self):
        with pytest.raises(Exception):
            DuplicateGroup(
                group_id=0,
                count=1,  # min 2
                representative_prompt_tokens=100,
                representative_output_tokens=50,
                request_indices=[0],
                is_exact=True,
            )


class TestDeduplicationReportModel:
    def test_report_serialization(self):
        report = DeduplicationReport(
            total_requests=10,
            unique_requests=8,
            duplicate_count=2,
            duplication_rate=0.2,
            group_count=1,
            largest_group_size=3,
            groups=[],
            recommendation="test",
        )
        d = report.model_dump()
        assert d["total_requests"] == 10
        assert d["duplication_rate"] == 0.2
