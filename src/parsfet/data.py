"""Data API for batch loading and analysis of technology libraries.

This module provides a high-level API for loading multiple Liberty files,
optionally with LEF/TechLEF physical data, and converting them to Pandas
DataFrames or NumPy arrays for plotting and machine learning workflows.

Example:
    >>> from parsfet.data import load_from_pattern
    >>> df = load_from_pattern("testdata/**/*.lib").to_dataframe()
    >>> df.groupby("cell_type")["d0_ratio"].mean()

    # With LEF/TechLEF data:
    >>> ds = Dataset()
    >>> ds.load_files(["lib.lib"]).load_tech_lef("tech.lef")
    >>> df = ds.to_dataframe()  # Includes lef_width, lef_height, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

from .parsers.liberty import LibertyParser
from .parsers.lef import LEFParser, TechLEFParser
from .normalizers.invd1 import INVD1Normalizer, NormalizedMetrics
from .models.physical import CellPhysical, TechInfo
from .exceptions import DuplicateCellError

if TYPE_CHECKING:
    from .models.liberty import LibertyLibrary
    from .models.lef import LEFLibrary, TechLEF


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
        lef_cells: Dictionary of cell name to CellPhysical (from LEF).
        tech_info: TechInfo from TechLEF (if loaded).
        source_file: Path to the source Liberty file.
    """

    library: LibertyLibrary
    normalizer: Optional[INVD1Normalizer]
    metrics: dict[str, NormalizedMetrics] = field(default_factory=dict)
    lef_cells: dict[str, CellPhysical] = field(default_factory=dict)
    tech_info: Optional[TechInfo] = None
    source_file: Optional[Path] = None


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
        self._lef_parser = LEFParser()
        self._tech_lef_parser = TechLEFParser()

    def load_files(self, paths: list[Path | str], normalize: bool = True) -> "Dataset":
        """Load a list of Liberty files.

        Args:
            paths: List of file paths (Path or str) to Liberty (.lib) files.
            normalize: If True, normalize cells immediately (original behavior).
                If False, defer normalization for later combine() call.

        Returns:
            self, for method chaining.

        Raises:
            FileNotFoundError: If a file does not exist.
            ValueError: If no baseline cell is found and normalize=True.
        """
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            lib = self._parser.parse(path)

            if normalize:
                try:
                    normalizer = INVD1Normalizer(lib)
                    metrics = normalizer.normalize_all()
                except ValueError:
                    # No baseline cell - skip normalization
                    normalizer = None
                    metrics = {}
            else:
                normalizer = None
                metrics = {}

            self.entries.append(
                LibraryEntry(
                    library=lib,
                    normalizer=normalizer,
                    metrics=metrics,
                    source_file=path,
                )
            )

        return self

    def load_lef(
        self,
        paths: list[Path | str],
        name_mapper: Optional[Callable[[str], str]] = None,
    ) -> "Dataset":
        """Load LEF files and match macros to Liberty cells by name.

        Args:
            paths: List of file paths to LEF files.
            name_mapper: Optional function to transform Liberty cell names
                before looking up in LEF macros. Useful when naming conventions
                differ (e.g., `lambda n: n.replace("_X", "D")`).

        Returns:
            self, for method chaining.
        """
        # Parse all LEF files and collect macros
        all_macros: dict[str, CellPhysical] = {}
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"LEF file not found: {path}")

            lef_lib = self._lef_parser.parse(path)
            for macro_name, macro in lef_lib.macros.items():
                all_macros[macro_name] = CellPhysical.from_macro(macro)

        # Match to Liberty cells in each entry
        for entry in self.entries:
            for cell_name in entry.metrics.keys():
                lookup_name = name_mapper(cell_name) if name_mapper else cell_name
                if lookup_name in all_macros:
                    entry.lef_cells[cell_name] = all_macros[lookup_name]

        return self

    def load_tech_lef(self, path: Path | str) -> "Dataset":
        """Load a TechLEF file for technology context.

        Args:
            path: Path to the TechLEF file.

        Returns:
            self, for method chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TechLEF file not found: {path}")

        tech_lef = self._tech_lef_parser.parse(path)
        tech_info = TechInfo.from_tech_lef(tech_lef)

        # Apply to all entries
        for entry in self.entries:
            entry.tech_info = tech_info

        return self

    def find_duplicates(self) -> dict[str, list[tuple[int, Path]]]:
        """Detect cells that appear in multiple loaded entries.

        Returns:
            A dictionary mapping cell name to list of (entry_index, source_file).
            Empty dict means no duplicates found.

        Example:
            >>> ds = Dataset()
            >>> ds.load_files(["lib1.lib", "lib2.lib"], normalize=False)
            >>> dups = ds.find_duplicates()
            >>> if dups:
            ...     print(f"Found {len(dups)} duplicate cells")
        """
        cell_sources: dict[str, list[tuple[int, Path]]] = {}

        for idx, entry in enumerate(self.entries):
            source = entry.source_file or Path(f"<entry_{idx}>")
            for cell_name in entry.library.cells.keys():
                if cell_name not in cell_sources:
                    cell_sources[cell_name] = []
                cell_sources[cell_name].append((idx, source))

        # Filter to only cells appearing in multiple entries
        return {
            name: sources
            for name, sources in cell_sources.items()
            if len(sources) > 1
        }

    def combine(self, allow_duplicates: bool = False) -> "Dataset":
        """Combine all entries into ONE grand dataset with unified normalization.

        This method merges all cells from all loaded libraries into a single
        unified dataset. A baseline inverter is found from the combined cell
        pool, and ALL cells are re-normalized against this single baseline.

        Args:
            allow_duplicates: If False (default), raise DuplicateCellError when
                cells with the same name appear in multiple files. If True,
                first occurrence wins and later occurrences are ignored.

        Returns:
            A new Dataset with one unified LibraryEntry containing all cells.

        Raises:
            DuplicateCellError: If duplicates found and allow_duplicates=False.
            ValueError: If no entries loaded or no baseline cell found.

        Example:
            >>> ds = Dataset()
            >>> ds.load_files(["lib1.lib", "lib2.lib"], normalize=False)
            >>> combined = ds.combine()
            >>> df = combined.to_dataframe()  # Unified normalization
        """
        if not self.entries:
            raise ValueError("No entries loaded. Call load_files() first.")

        # Check for duplicates
        duplicates = self.find_duplicates()
        if duplicates and not allow_duplicates:
            raise DuplicateCellError(duplicates)

        # Merge all cells into a combined library
        # Use first entry as base and merge cells from others
        from .models.liberty import LibertyLibrary, Cell

        base_entry = self.entries[0]
        combined_cells: dict[str, Cell] = {}
        cell_sources: dict[str, Path] = {}  # Track origin for each cell

        for entry in self.entries:
            source = entry.source_file or Path("<unknown>")
            for cell_name, cell in entry.library.cells.items():
                if cell_name not in combined_cells:
                    combined_cells[cell_name] = cell
                    cell_sources[cell_name] = source
                # else: duplicate, skip (first wins)

        # Create a combined library using the first library's metadata
        combined_lib = LibertyLibrary(
            name=f"combined_{len(self.entries)}_libs",
            technology=base_entry.library.technology,
            delay_model=base_entry.library.delay_model,
            time_unit=base_entry.library.time_unit,
            capacitive_load_unit=base_entry.library.capacitive_load_unit,
            voltage_unit=base_entry.library.voltage_unit,
            current_unit=base_entry.library.current_unit,
            leakage_power_unit=base_entry.library.leakage_power_unit,
            pulling_resistance_unit=base_entry.library.pulling_resistance_unit,
            nom_voltage=base_entry.library.nom_voltage,
            nom_temperature=base_entry.library.nom_temperature,
            nom_process=base_entry.library.nom_process,
            operating_conditions=base_entry.library.operating_conditions,
            vt_flavor=base_entry.library.vt_flavor,
            process_node=base_entry.library.process_node,
            foundry=base_entry.library.foundry,
            lu_table_templates=base_entry.library.lu_table_templates,
            cells=combined_cells,
            attributes=base_entry.library.attributes,
        )

        # Create normalizer from combined library (finds baseline from all cells)
        try:
            normalizer = INVD1Normalizer(combined_lib)
            metrics = normalizer.normalize_all()
        except ValueError as e:
            raise ValueError(f"No baseline cell found in combined dataset: {e}") from e

        # Merge LEF data
        combined_lef: dict[str, CellPhysical] = {}
        for entry in self.entries:
            combined_lef.update(entry.lef_cells)

        # Use first available tech_info
        tech_info = None
        for entry in self.entries:
            if entry.tech_info:
                tech_info = entry.tech_info
                break

        # Create combined entry
        combined_entry = LibraryEntry(
            library=combined_lib,
            normalizer=normalizer,
            metrics=metrics,
            lef_cells=combined_lef,
            tech_info=tech_info,
            source_file=None,  # Combined from multiple files
        )

        # Store cell sources for provenance tracking
        combined_entry._cell_sources = cell_sources  # type: ignore

        # Create new dataset with combined entry
        result = Dataset()
        result.entries = [combined_entry]
        result._cell_sources = cell_sources  # Store at dataset level too

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all loaded data to a flat Pandas DataFrame.

        Each row represents a single cell. Columns include:
        - library: Source library name
        - cell: Cell name
        - cell_type: Classified cell type (inverter, nand, etc.)
        - Ratio columns: area_ratio, d0_ratio, k_ratio, etc.
        - Raw metric columns: raw_area_um2, raw_d0_ns, etc.
        - Context: voltage, temperature, process_node
        - LEF columns (if loaded): lef_width, lef_height, lef_area, pin_layers_json
        - TechLEF columns (if loaded): metal_stack_height
        - source_file: Path to the source .lib file (for provenance tracking)

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
                row = {
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

                # Add LEF physical data if available
                lef_cell = entry.lef_cells.get(cell_name)
                if lef_cell:
                    row["lef_width"] = lef_cell.width
                    row["lef_height"] = lef_cell.height
                    row["lef_area"] = lef_cell.area
                    # Pin info as JSON for easy analysis (includes use type)
                    pin_info = {
                        name: {
                            "direction": p.direction,
                            "use": p.use,
                            "layers": p.layers,
                        }
                        for name, p in lef_cell.pins.items()
                    }
                    row["pin_layers_json"] = json.dumps(pin_info)
                else:
                    row["lef_width"] = None
                    row["lef_height"] = None
                    row["lef_area"] = None
                    row["pin_layers_json"] = None

                # Add TechLEF data if available
                if entry.tech_info:
                    row["metal_stack_height"] = entry.tech_info.metal_stack_height
                else:
                    row["metal_stack_height"] = None

                # Add source file provenance
                # For combined datasets, check _cell_sources; otherwise use entry.source_file
                if hasattr(self, "_cell_sources") and cell_name in self._cell_sources:
                    row["source_file"] = str(self._cell_sources[cell_name])
                elif entry.source_file:
                    row["source_file"] = str(entry.source_file)
                else:
                    row["source_file"] = None

                rows.append(row)

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

    def export_to_json(self, entry_index: int = 0) -> dict:
        """Export combined Liberty + LEF/TechLEF data to a JSON-serializable dict.

        This produces a comprehensive JSON structure that includes:
        - Liberty timing data (normalized metrics, FO4 operating point)
        - LEF cell physical data (dimensions, pin layers, pin use types)
        - TechLEF technology data (metal stack, layer min sizes)

        Args:
            entry_index: Index of the library entry to export (default 0).

        Returns:
            A dictionary suitable for JSON serialization.

        Example:
            >>> ds = Dataset()
            >>> ds.load_files(["lib.lib"]).load_lef(["cell.lef"]).load_tech_lef("tech.lef")
            >>> data = ds.export_to_json()
            >>> with open("output.json", "w") as f:
            ...     json.dump(data, f, indent=2)
        """
        if not self.entries:
            return {"error": "No libraries loaded"}

        if entry_index >= len(self.entries):
            return {"error": f"Entry index {entry_index} out of range"}

        entry = self.entries[entry_index]

        # Get base normalized JSON from normalizer
        if entry.normalizer:
            result = entry.normalizer.export_to_json()
        else:
            result = {
                "library": entry.library.name,
                "cells": {},
                "error": "No normalizer available (missing baseline cell)",
            }

        # Add technology info from TechLEF
        if entry.tech_info:
            result["technology"] = {
                "metal_stack_height": entry.tech_info.metal_stack_height,
                "units_database": entry.tech_info.units_database,
                "manufacturing_grid": entry.tech_info.manufacturing_grid,
                "layers": {
                    name: {
                        "type": layer.layer_type,
                        "min_size_um": layer.min_size,
                        "direction": layer.direction,
                        "pitch_um": layer.pitch,
                        "spacing_um": layer.spacing,
                    }
                    for name, layer in entry.tech_info.layers.items()
                    if layer.layer_type == "routing"
                },
            }

        # Add physical data to each cell
        for cell_name in result.get("cells", {}):
            if cell_name in entry.lef_cells:
                lef_cell = entry.lef_cells[cell_name]
                result["cells"][cell_name]["physical"] = {
                    "width_um": lef_cell.width,
                    "height_um": lef_cell.height,
                    "area_um2": lef_cell.area,
                    "pins": {
                        pin_name: {
                            "direction": pin.direction,
                            "use": pin.use,  # power, ground, clock, signal
                            "layers": pin.layers,
                        }
                        for pin_name, pin in lef_cell.pins.items()
                    },
                }

        return result

    def save_json(self, path: Path | str, entry_index: int = 0, indent: int = 2) -> None:
        """Save combined Liberty + LEF/TechLEF data to a JSON file.

        Args:
            path: Output file path.
            entry_index: Index of the library entry to export (default 0).
            indent: JSON indentation (default 2).
        """
        data = self.export_to_json(entry_index)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)


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
