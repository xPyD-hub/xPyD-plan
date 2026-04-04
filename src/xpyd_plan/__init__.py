"""xPyD-plan: PD ratio planner for xPyD proxy."""

__version__ = "0.2.0"

from xpyd_plan.analyzer import BenchmarkAnalyzer
from xpyd_plan.benchmark_models import (
    AnalysisResult,
    BenchmarkData,
    BenchmarkMetadata,
    BenchmarkRequest,
    RatioCandidate,
    SLACheck,
    UtilizationResult,
)
from xpyd_plan.comparator import (
    BenchmarkComparator,
    ComparisonResult,
    MetricDelta,
    compare_benchmarks,
)
from xpyd_plan.config import ConfigProfile, load_config
from xpyd_plan.gpu_profiles import get_gpu_profile, list_gpu_profiles
from xpyd_plan.models import (
    DatasetStats,
    GPUProfile,
    PDConfig,
    SLAConfig,
)
from xpyd_plan.trend import TrendEntry, TrendReport, TrendTracker, track_trend

__all__ = [
    "AnalysisResult",
    "BenchmarkAnalyzer",
    "BenchmarkComparator",
    "ComparisonResult",
    "MetricDelta",
    "compare_benchmarks",
    "BenchmarkData",
    "BenchmarkMetadata",
    "BenchmarkRequest",
    "ConfigProfile",
    "DatasetStats",
    "GPUProfile",
    "PDConfig",
    "RatioCandidate",
    "SLACheck",
    "SLAConfig",
    "UtilizationResult",
    "get_gpu_profile",
    "list_gpu_profiles",
    "load_config",
    "track_trend",
    "TrendEntry",
    "TrendReport",
    "TrendTracker",
]
