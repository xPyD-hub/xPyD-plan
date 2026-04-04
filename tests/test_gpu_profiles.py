"""Tests for GPU profile library."""

from __future__ import annotations

import pytest

from xpyd_plan.gpu_profiles import get_gpu_profile, list_gpu_profiles


class TestGPUProfiles:
    def test_list_profiles(self):
        profiles = list_gpu_profiles()
        assert "A100-80G" in profiles
        assert "H100-80G" in profiles

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

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown GPU profile"):
            get_gpu_profile("RTX-4090")

    def test_returned_profile_is_copy(self):
        """Modifying returned profile should not affect the library."""
        gpu1 = get_gpu_profile("A100-80G")
        gpu1.cost_per_hour = 999.0
        gpu2 = get_gpu_profile("A100-80G")
        assert gpu2.cost_per_hour != 999.0
