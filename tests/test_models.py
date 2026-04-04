"""Tests for core data models."""

from __future__ import annotations

import pytest

from xpyd_plan.models import (
    DatasetStats,
    PDConfig,
    SLAConfig,
)


class TestSLAConfig:
    def test_no_constraints(self):
        sla = SLAConfig()
        assert not sla.has_constraints()

    def test_with_constraints(self):
        sla = SLAConfig(ttft_ms=1000, tpot_ms=50)
        assert sla.has_constraints()


class TestDatasetStats:
    def test_from_records(self):
        records = [
            {"prompt_len": 100, "output_len": 50},
            {"prompt_len": 200, "output_len": 100},
            {"prompt_len": 300, "output_len": 150},
            {"prompt_len": 1000, "output_len": 500},
        ]
        stats = DatasetStats.from_records(records, num_requests=100)
        assert stats.prompt_len_mean == 400.0
        assert stats.output_len_mean == 200.0
        assert stats.num_requests == 100

    def test_from_records_empty(self):
        with pytest.raises(ValueError, match="records must not be empty"):
            DatasetStats.from_records([])


class TestPDConfig:
    def test_total_and_ratio(self):
        c = PDConfig(num_prefill=2, num_decode=6)
        assert c.total == 8
        assert c.ratio_str == "2P:6D"
