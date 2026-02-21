"""Pars-FET: VLSI Process Technology Feature Extraction"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("parsfet")
except (ImportError, PackageNotFoundError):
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.0"

from .data import (
    Dataset,
    load_files,
    load_from_pattern,
)
from .path_delay import (
    # User-facing input types (Pydantic v2)
    PathSpec,
    AnalysisConfig,
    WireLoadModel,
    # Resolution result
    ManualResolution,
    # Output types (dataclass)
    TimingPoint,
    TimingPath,
    # Functions
    estimate_path_delay,
    resolve_manual,
    propagate,
    compute_slack,
)

__all__ = [
    # Dataset
    "Dataset",
    "load_files",
    "load_from_pattern",
    # Path delay — input
    "PathSpec",
    "AnalysisConfig",
    "WireLoadModel",
    # Path delay — output
    "TimingPoint",
    "TimingPath",
    # Path delay — functions
    "estimate_path_delay",
    "resolve_manual",
    "propagate",
    "compute_slack",
]
