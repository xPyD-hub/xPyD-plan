"""Tests for GPU profile library."""

from __future__ import annotations

import pytest

from xpyd_plan.gpu_profiles import get_gpu_profile, list_gpu_profiles


class TestGPUProfiles:
    def test_list_profiles(self):
        profiles = list_gpu_profiles()
        assert "A100-80G" in profiles
        assert "H100-80G" in profiles
        assert "L40S-48G" in profiles
        assert "B200-192G" in profiles
        assert "H200-141G" in profiles
        assert "A10G-24G" in profiles
        assert "A100-40G" in profiles
        assert len(profiles) == 7

    def test_get_a100(self):
        gpu = get_gpu_profile("A100-80G")
        assert gpu.name == "A100-80G"
        assert gpu.prefill_tokens_per_sec > 0
        assert gpu.decode_tokens_per_sec > 0
        assert gpu.memory_gb == 80.0

    def test_get_h100(self):
        gpu = get_gpu_profile("H100-80G")
        assert gpu.name == "H100-80G"
        # H100 should be faster than A100
        a100 = get_gpu_profile("A100-80G")
        assert gpu.prefill_tokens_per_sec > a100.prefill_tokens_per_sec
        assert gpu.decode_tokens_per_sec > a100.decode_tokens_per_sec

    def test_get_l40s(self):
        gpu = get_gpu_profile("L40S-48G")
        assert gpu.name == "L40S-48G"
        assert gpu.memory_gb == 48.0
        assert gpu.cost_per_hour == 1.50

    def test_get_b200(self):
        gpu = get_gpu_profile("B200-192G")
        assert gpu.name == "B200-192G"
        assert gpu.memory_gb == 192.0
        # B200 should be fastest
        h100 = get_gpu_profile("H100-80G")
        assert gpu.prefill_tokens_per_sec > h100.prefill_tokens_per_sec
        assert gpu.decode_tokens_per_sec > h100.decode_tokens_per_sec

    def test_get_h200(self):
        gpu = get_gpu_profile("H200-141G")
        assert gpu.name == "H200-141G"
        assert gpu.memory_gb == 141.0
        # H200 faster than H100 but slower than B200
        h100 = get_gpu_profile("H100-80G")
        b200 = get_gpu_profile("B200-192G")
        assert h100.prefill_tokens_per_sec < gpu.prefill_tokens_per_sec
        assert gpu.prefill_tokens_per_sec < b200.prefill_tokens_per_sec

    def test_get_a10g(self):
        gpu = get_gpu_profile("A10G-24G")
        assert gpu.name == "A10G-24G"
        assert gpu.memory_gb == 24.0
        # A10G should be cheapest
        for name in list_gpu_profiles():
            other = get_gpu_profile(name)
            assert gpu.cost_per_hour <= other.cost_per_hour

    def test_a100_40g_vs_80g(self):
        a40 = get_gpu_profile("A100-40G")
        a80 = get_gpu_profile("A100-80G")
        assert a40.memory_gb < a80.memory_gb
        assert a40.cost_per_hour < a80.cost_per_hour

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown GPU profile"):
            get_gpu_profile("RTX-4090")

    def test_returned_profile_is_copy(self):
        """Modifying returned profile should not affect the library."""
        gpu1 = get_gpu_profile("A100-80G")
        gpu1.cost_per_hour = 999.0
        gpu2 = get_gpu_profile("A100-80G")
        assert gpu2.cost_per_hour != 999.0
