"""Request replay schedule generator.

Extract request arrival patterns from benchmark data and generate
reproducible replay schedules for benchmark reproduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReplayEntry(BaseModel):
    """A single entry in a replay schedule."""

    offset_ms: float = Field(..., ge=0, description="Time offset from start in milliseconds")
    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens")


class ReplayConfig(BaseModel):
    """Configuration for replay schedule generation."""

    time_scale: float = Field(1.0, gt=0, description="Time scaling factor (2.0 = 2x faster)")
    target_qps: float | None = Field(None, gt=0, description="Override QPS (uniform distribution)")


class ReplaySchedule(BaseModel):
    """Generated replay schedule."""

    entries: list[ReplayEntry] = Field(default_factory=list, description="Ordered replay entries")
    total_duration_ms: float = Field(..., ge=0, description="Total schedule duration in ms")
    request_count: int = Field(..., ge=0, description="Number of requests in the schedule")
    effective_qps: float = Field(..., ge=0, description="Effective QPS of the schedule")
    config: ReplayConfig = Field(..., description="Configuration used to generate this schedule")


class ReplayGenerator:
    """Generate replay schedules from benchmark data."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self._config = config or ReplayConfig()

    def generate(self, data: dict[str, Any]) -> ReplaySchedule:
        """Generate a replay schedule from benchmark data.

        Args:
            data: Parsed benchmark JSON data.

        Returns:
            ReplaySchedule with ordered entries.

        Raises:
            ValueError: If data is missing required fields.
        """
        requests = self._extract_requests(data)
        if not requests:
            return ReplaySchedule(
                entries=[],
                total_duration_ms=0.0,
                request_count=0,
                effective_qps=0.0,
                config=self._config,
            )

        # Sort by timestamp
        requests.sort(key=lambda r: r["timestamp"])

        base_timestamp = requests[0]["timestamp"]

        if self._config.target_qps is not None:
            entries = self._generate_uniform(requests, self._config.target_qps)
        else:
            entries = self._generate_from_timestamps(requests, base_timestamp)

        # Apply time scaling
        if self._config.time_scale != 1.0:
            scale = self._config.time_scale
            entries = [
                ReplayEntry(
                    offset_ms=e.offset_ms / scale,
                    prompt_tokens=e.prompt_tokens,
                    output_tokens=e.output_tokens,
                )
                for e in entries
            ]

        total_duration_ms = entries[-1].offset_ms if entries else 0.0
        effective_qps = (
            (len(entries) / (total_duration_ms / 1000.0)) if total_duration_ms > 0 else 0.0
        )

        return ReplaySchedule(
            entries=entries,
            total_duration_ms=total_duration_ms,
            request_count=len(entries),
            effective_qps=round(effective_qps, 2),
            config=self._config,
        )

    def _extract_requests(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract request list from benchmark data (native or xpyd-bench format)."""
        if "requests" in data:
            reqs = data["requests"]
        elif "results" in data:
            reqs = data["results"]
        else:
            raise ValueError(
                "Benchmark data must have 'requests' or 'results' field."
            )

        validated = []
        for r in reqs:
            if "timestamp" not in r:
                raise ValueError(
                    "Each request must have a 'timestamp' field for replay generation."
                )
            validated.append(
                {
                    "timestamp": float(r["timestamp"]),
                    "prompt_tokens": int(r.get("prompt_tokens", 0)),
                    "output_tokens": int(r.get("output_tokens", 0)),
                }
            )
        return validated

    def _generate_from_timestamps(
        self,
        requests: list[dict[str, Any]],
        base_timestamp: float,
    ) -> list[ReplayEntry]:
        """Generate entries preserving original inter-arrival times."""
        return [
            ReplayEntry(
                offset_ms=(r["timestamp"] - base_timestamp) * 1000.0,
                prompt_tokens=r["prompt_tokens"],
                output_tokens=r["output_tokens"],
            )
            for r in requests
        ]

    def _generate_uniform(
        self,
        requests: list[dict[str, Any]],
        target_qps: float,
    ) -> list[ReplayEntry]:
        """Redistribute arrivals uniformly at target QPS."""
        interval_ms = 1000.0 / target_qps
        return [
            ReplayEntry(
                offset_ms=i * interval_ms,
                prompt_tokens=r["prompt_tokens"],
                output_tokens=r["output_tokens"],
            )
            for i, r in enumerate(requests)
        ]

    def generate_from_file(self, path: str | Path) -> ReplaySchedule:
        """Generate replay schedule from a benchmark JSON file.

        Args:
            path: Path to benchmark JSON file.

        Returns:
            ReplaySchedule.
        """
        import json

        with open(path) as f:
            data = json.load(f)
        return self.generate(data)


def generate_replay(
    path: str | Path,
    time_scale: float = 1.0,
    target_qps: float | None = None,
) -> ReplaySchedule:
    """Programmatic API: generate a replay schedule from a benchmark file.

    Args:
        path: Path to benchmark JSON file.
        time_scale: Time scaling factor (default 1.0).
        target_qps: Override QPS with uniform distribution (optional).

    Returns:
        ReplaySchedule.
    """
    config = ReplayConfig(time_scale=time_scale, target_qps=target_qps)
    generator = ReplayGenerator(config)
    return generator.generate_from_file(path)
