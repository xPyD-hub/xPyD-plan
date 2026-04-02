"""Smoke tests for xpyd-plan CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_plan_help():
    """xpyd-plan --help should exit 0."""
    result = subprocess.run(
        ["xpyd-plan", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "xpyd-plan" in result.stdout
