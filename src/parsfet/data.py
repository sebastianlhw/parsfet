"""Data API for batch loading and analysis of technology libraries.

This module provides a high-level API for loading multiple Liberty files
and converting them to Pandas DataFrames or NumPy arrays for plotting
and machine learning workflows.

Example:
    >>> from parsfet.data import load_from_pattern
    >>> df = load_from_pattern("testdata/**/*.lib").to_dataframe()
    >>> df.groupby("cell_type")["d0_ratio"].mean()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .parsers.liberty import LibertyParser
from .normalizers.invd1 import INVD1Normalizer, NormalizedMetrics

if TYPE_CHECKING:
    from .models.liberty import LibertyLibrary


# Feature columns used by to_numpy() - matches NormalizedMetrics.to_feature_vector()
FEATURE_COLUMNS = [
    "area_ratio",
    "d0_ratio",
    "k_ratio",
    "leakage_ratio",
    "input_cap_ratio",
    "num_inputs",
    "num_outputs",
    "is_sequential",
]


@dataclass
class LibraryEntry:
    """Container for a loaded and normalized library.

    Attributes:
        library: The parsed LibertyLibrary object.
        normalizer: The INVD1Normalizer for this library.
        metrics: Dictionary of cell name to NormalizedMetrics.
    """

    library: LibertyLibrary
    normalizer: INVD1Normalizer
    metrics: dict[str, NormalizedMetrics] = field(default_factory=dict)


class Dataset:
    """A dataset of normalized cell metrics from multiple libraries.

    Use the module-level functions `load_files()` or `load_from_pattern()`
    to create a Dataset instance.

    Attributes:
        entries: List of LibraryEntry objects, one per loaded library.

    Example:
        >>> ds = load_files(["lib1.lib", "lib2.lib"])
        >>> df = ds.to_dataframe()
        >>> X, y, encoder = ds.to_numpy()
    """

    def __init__(self) -> None:
        """Initialize an empty Dataset."""
        self.entries: list[LibraryEntry] = []
        self._parser = LibertyParser()

    def load_files(self, paths: list[Path | str]) -> "Dataset":
        """Load and normalize a list of Liberty files.

        Args:
            paths: List of file paths (Path or str) to Liberty (.lib) files.

        Returns:
            self, for method chaining.

        Raises:
            FileNotFoundError: If a file does not exist.
            ValueError: If no baseline cell is found in a library.
        """
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            lib = self._parser.parse(path)

            try:
                normalizer = INVD1Normalizer(lib)
                metrics = normalizer.normalize_all()
            except ValueError:
                # No baseline cell - skip normalization
                normalizer = None
                metrics = {}

            self.entries.append(
                LibraryEntry(library=lib, normalizer=normalizer, metrics=metrics)
            )

        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all loaded data to a flat Pandas DataFrame.

        Each row represents a single cell. Columns include:
        - library: Source library name
        - cell: Cell name
        - cell_type: Classified cell type (inverter, nand, etc.)
        - Ratio columns: area_ratio, d0_ratio, k_ratio, etc.
        - Raw metric columns: raw_area_um2, raw_d0_ns, etc.
        - Context: voltage, temperature, process_node

        Returns:
            A DataFrame with one row per cell across all libraries.
        """
        rows = []

        for entry in self.entries:
            lib = entry.library

            # Baseline info (for cross-library normalization context)
            baseline_d0 = 0.0
            baseline_area = 0.0
            if entry.normalizer:
                baseline_d0 = entry.normalizer.baseline.d0
                baseline_area = entry.normalizer.baseline.area

            for cell_name, m in entry.metrics.items():
                rows.append(
                    {
                        # Identity
                        "library": lib.name,
                        "cell": cell_name,
                        "cell_type": m.cell_type,
                        # Ratios
                        "area_ratio": m.area_ratio,
                        "d0_ratio": m.d0_ratio,
                        "k_ratio": m.k_ratio,
                        "leakage_ratio": m.leakage_ratio,
                        "input_cap_ratio": m.input_cap_ratio,
                        "drive_strength": m.drive_strength,
                        # Raw metrics
                        "raw_area_um2": m.raw_area,
                        "raw_d0_ns": m.raw_d0_ns,
                        "raw_k_ns_per_pf": m.raw_k_ns_per_pf,
                        "raw_leakage": m.raw_leakage,
                        "raw_input_cap_pf": m.raw_input_cap,
                        # Cell properties
                        "num_inputs": m.num_inputs,
                        "num_outputs": m.num_outputs,
                        "is_sequential": m.is_sequential,
                        # Library context
                        "voltage": lib.nom_voltage,
                        "temperature": lib.nom_temperature,
                        "process_node": lib.process_node,
                        # Baseline for reconstruction
                        "baseline_d0_ns": baseline_d0,
                        "baseline_area_um2": baseline_area,
                    }
                )

        df = pd.DataFrame(rows)

        # Use categorical for cell_type for efficient groupby
        if not df.empty:
            df["cell_type"] = df["cell_type"].astype("category")

        return df

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """Convert dataset to NumPy arrays for ML.

        Returns:
            A tuple of (X, y, label_map) where:
            - X: Feature matrix (n_samples, n_features). Features are defined
              by FEATURE_COLUMNS.
            - y: Label array (n_samples,) with encoded cell_type.
            - label_map: Dictionary mapping encoded integers back to cell_type strings.
        """
        df = self.to_dataframe()

        if df.empty:
            return np.array([]), np.array([]), {}

        # Features
        X = df[FEATURE_COLUMNS].to_numpy()

        # Labels (cell_type encoded)
        cell_types = df["cell_type"].cat.categories.tolist()
        label_map = {i: t for i, t in enumerate(cell_types)}
        y = df["cell_type"].cat.codes.to_numpy()

        return X, y, label_map


# Module-level convenience functions


def load_files(paths: list[Path | str]) -> Dataset:
    """Load and normalize a list of Liberty files.

    Args:
        paths: List of file paths to Liberty (.lib) files.

    Returns:
        A Dataset containing the loaded libraries.

    Example:
        >>> ds = load_files(["lib1.lib", "lib2.lib"])
        >>> df = ds.to_dataframe()
    """
    return Dataset().load_files(paths)


def load_from_pattern(pattern: str) -> Dataset:
    """Load Liberty files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "testdata/**/*.lib").

    Returns:
        A Dataset containing all matched libraries.

    Example:
        >>> ds = load_from_pattern("testdata/**/*.lib")
        >>> df = ds.to_dataframe()
    """
    paths = glob(pattern, recursive=True)
    return Dataset().load_files(paths)
