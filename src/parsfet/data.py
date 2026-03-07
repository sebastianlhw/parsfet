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

from .exceptions import DuplicateCellError
from .models.physical import CellPhysical, TechInfo
from .normalizers.invd1 import INVD1Normalizer, NormalizedMetrics
from .parsers.lef import LEFParser, TechLEFParser
from .parsers.liberty import LibertyParser

# Re-export path-delay public types so ``from parsfet.data import ...`` still works.
from .path_delay import (
    WireLoadModel,
    PathSpec,
    AnalysisConfig,
    TimingPath,
    TimingPoint,
    estimate_path_delay as _estimate_path_delay_impl,
)

if TYPE_CHECKING:
    from .models.lef import LEFLibrary, TechLEF
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
        lef_cells: Dictionary of cell name to CellPhysical (from LEF).
        tech_info: TechInfo from TechLEF (if loaded).
        source_file: Path to the source Liberty file.
        from_json: True if this entry was loaded from a JSON export.
        raw_metrics_cache: Raw metrics for cells loaded from JSON (for re-normalization).
    """

    library: LibertyLibrary
    normalizer: Optional[INVD1Normalizer]
    metrics: dict[str, NormalizedMetrics] = field(default_factory=dict)
    lef_cells: dict[str, CellPhysical] = field(default_factory=dict)
    tech_info: Optional[TechInfo] = None
    source_file: Optional[Path] = None
    # For JSON imports
    from_json: bool = False
    raw_metrics_cache: dict[str, dict] = field(default_factory=dict)


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
        # Staging queues — populated by load_* calls, flushed by _ensure_combined()
        self._pending_libs: list[tuple] = []   # (LibertyLibrary | json_entry, Path | None)
        self._pending_lefs: list[tuple] = []   # (paths, name_mapper)
        self._pending_tech_lef: Optional[Path] = None
        self._combined: bool = False            # True once _do_combine() has run
        self._combine_opts: dict = {}           # baseline / allow_duplicates overrides

    def load_files(self, paths: list[Path | str]) -> "Dataset":
        """Stage a list of Liberty or JSON files for loading.

        Files are parsed immediately but normalization and combining happen
        lazily on first access (``to_dataframe()``, ``query_cell_at()``, etc.).
        To combine multiple VT libraries into one Dataset, simply call
        ``load_files()`` multiple times or pass all paths at once.

        Args:
            paths: List of file paths (Path or str) to Liberty (.lib) or
                previously-exported JSON (.json) files.

        Returns:
            self, for method chaining.

        Raises:
            FileNotFoundError: If a file does not exist.
        """
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() == ".json":
                self._stage_json_file(path)
            else:
                self._stage_lib_file(path)
        self._combined = False  # Invalidate — new content pending
        return self

    def _stage_lib_file(self, path: Path) -> None:
        """Parse a Liberty file and add it to the staging queue."""
        lib = self._parser.parse(path)
        self._pending_libs.append(("lib", lib, path))

    def _stage_json_file(self, path: Path) -> None:
        """Parse a JSON export and add it to the staging queue."""
        from .models.export import ExportedLibrary
        from .models.liberty import Cell, LibertyLibrary

        exported = ExportedLibrary.from_json_file(str(path))

        # Reconstruct minimal Cell objects
        cells: dict[str, Cell] = {}
        for cell_name, exported_cell in exported.cells.items():
            raw = exported_cell.raw
            if raw:
                cells[cell_name] = Cell(
                    name=cell_name,
                    area=raw.area_um2,
                    cell_leakage_power=raw.leakage,
                    is_sequential=exported_cell.is_sequential,
                )
            else:
                cells[cell_name] = Cell(
                    name=cell_name,
                    area=exported_cell.area_ratio * exported.baseline.area_um2,
                    cell_leakage_power=exported_cell.leakage_ratio * exported.baseline.leakage,
                    is_sequential=exported_cell.is_sequential,
                )

        lib = LibertyLibrary(name=exported.library, cells=cells)

        raw_metrics_cache = {}
        for cell_name, exported_cell in exported.cells.items():
            if exported_cell.raw:
                raw_metrics_cache[cell_name] = {
                    "area": exported_cell.raw.area_um2,
                    "d0_ns": exported_cell.raw.d0_ns,
                    "k_ns_per_pf": exported_cell.raw.k_ns_per_pf,
                    "leakage": exported_cell.raw.leakage,
                    "input_cap": exported_cell.raw.input_cap_pf,
                    "cell_type": exported_cell.cell_type,
                    "num_inputs": exported_cell.num_inputs,
                    "num_outputs": exported_cell.num_outputs,
                    "is_sequential": exported_cell.is_sequential,
                }

        entry = LibraryEntry(
            library=lib,
            normalizer=None,
            metrics={},
            source_file=path,
            from_json=True,
            raw_metrics_cache=raw_metrics_cache,
        )
        self._pending_libs.append(("json_entry", entry, path))

    def load_lef(
        self,
        paths: list[Path | str],
        name_mapper: Optional[Callable[[str], str]] = None,
    ) -> "Dataset":
        """Stage LEF files to be matched to Liberty cells on first data access.

        Args:
            paths: List of file paths to LEF files.
            name_mapper: Optional function to transform Liberty cell names
                before looking up in LEF macros.

        Returns:
            self, for method chaining.
        """
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"LEF file not found: {path}")
        self._pending_lefs.append(([Path(p) for p in paths], name_mapper))
        self._combined = False
        return self

    def load_tech_lef(self, path: Path | str) -> "Dataset":
        """Stage a TechLEF file for technology context.

        Args:
            path: Path to the TechLEF file.

        Returns:
            self, for method chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TechLEF file not found: {path}")
        self._pending_tech_lef = path
        self._combined = False
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

        for idx, (kind, obj, path) in enumerate(self._pending_libs):
            lib = obj.library if kind == "json_entry" else obj
            source = path or Path(f"<entry_{idx}>")
            for cell_name in lib.cells.keys():
                if cell_name not in cell_sources:
                    cell_sources[cell_name] = []
                cell_sources[cell_name].append((idx, source))

        # Filter to only cells appearing in multiple entries
        return {name: sources for name, sources in cell_sources.items() if len(sources) > 1}

    def _ensure_combined(self) -> None:
        """Trigger lazy combine if not already done. Called internally by all read methods."""
        if not self._combined:
            self._do_combine(**self._combine_opts)

    def _do_combine(
        self,
        allow_duplicates: bool = False,
        baseline: Optional[str] = None,
    ) -> None:
        """Internal: flush all staged libs/LEFs/TechLEF into a single combined entry."""
        from .models.liberty import Cell, LibertyLibrary

        # Build entry list from staging queue (in order)
        stage_entries: list[LibraryEntry] = []
        for kind, obj, path in self._pending_libs:
            if kind == "lib":
                stage_entries.append(LibraryEntry(
                    library=obj,
                    normalizer=None,
                    metrics={},
                    source_file=path,
                ))
            else:  # json_entry
                stage_entries.append(obj)

        if not stage_entries:
            self._combined = True
            return

        # Duplicate detection
        cell_sources: dict[str, list[tuple[int, Path]]] = {}
        for idx, e in enumerate(stage_entries):
            src = e.source_file or Path(f"<entry_{idx}>")
            for cell_name in e.library.cells:
                cell_sources.setdefault(cell_name, []).append((idx, src))
        duplicates = {n: s for n, s in cell_sources.items() if len(s) > 1}
        if duplicates and not allow_duplicates:
            raise DuplicateCellError(duplicates)

        # Merge cells (first wins on duplicate)
        combined_cells: dict[str, Cell] = {}
        cell_origin: dict[str, int] = {}  # cell_name -> entry index
        for idx, e in enumerate(stage_entries):
            for cell_name, cell in e.library.cells.items():
                if cell_name not in combined_cells:
                    combined_cells[cell_name] = cell
                    cell_origin[cell_name] = idx

        base = stage_entries[0]
        combined_lib = LibertyLibrary(
            name=f"combined_{len(stage_entries)}_libs" if len(stage_entries) > 1 else base.library.name,
            technology=base.library.technology,
            delay_model=base.library.delay_model,
            time_unit=base.library.time_unit,
            capacitive_load_unit=base.library.capacitive_load_unit,
            voltage_unit=base.library.voltage_unit,
            current_unit=base.library.current_unit,
            leakage_power_unit=base.library.leakage_power_unit,
            pulling_resistance_unit=base.library.pulling_resistance_unit,
            nom_voltage=base.library.nom_voltage,
            nom_temperature=base.library.nom_temperature,
            nom_process=base.library.nom_process,
            operating_conditions=base.library.operating_conditions,
            vt_flavor=base.library.vt_flavor,
            process_node=base.library.process_node,
            foundry=base.library.foundry,
            lu_table_templates=base.library.lu_table_templates,
            cells=combined_cells,
            attributes=base.library.attributes,
        )

        try:
            normalizer = INVD1Normalizer(combined_lib, baseline_name=baseline)
        except ValueError as e:
            raise ValueError(f"No baseline cell found in combined dataset: {e}") from e

        metrics: dict[str, NormalizedMetrics] = {}
        cell_file_sources: dict[str, Path] = {}
        for cell_name, cell in combined_lib.cells.items():
            src_idx = cell_origin[cell_name]
            src_entry = stage_entries[src_idx]
            cell_file_sources[cell_name] = src_entry.source_file or Path("<unknown>")
            if src_entry.from_json and cell_name in src_entry.raw_metrics_cache:
                raw = src_entry.raw_metrics_cache[cell_name]
                metrics[cell_name] = normalizer.normalize_from_raw(
                    cell_name=cell_name,
                    raw_area=raw["area"],
                    raw_d0_ns=raw["d0_ns"],
                    raw_k_ns_per_pf=raw["k_ns_per_pf"],
                    raw_leakage=raw["leakage"],
                    raw_input_cap=raw["input_cap"],
                    cell_type_str=raw.get("cell_type", "unknown"),
                    num_inputs=raw.get("num_inputs", 1),
                    num_outputs=raw.get("num_outputs", 1),
                    is_sequential=raw.get("is_sequential", False),
                )
            else:
                metrics[cell_name] = normalizer.normalize(cell)

        # Apply staged LEF data
        combined_lef: dict[str, CellPhysical] = {}
        for lef_paths, name_mapper in self._pending_lefs:
            all_macros: dict[str, CellPhysical] = {}
            for p in lef_paths:
                lef_lib = self._lef_parser.parse(p)
                for macro_name, macro in lef_lib.macros.items():
                    all_macros[macro_name] = CellPhysical.from_macro(macro)
            for cell_name in combined_cells:
                lookup = name_mapper(cell_name) if name_mapper else cell_name
                if lookup in all_macros:
                    combined_lef[cell_name] = all_macros[lookup]

        # Apply staged TechLEF
        tech_info = None
        if self._pending_tech_lef:
            tech_lef = self._tech_lef_parser.parse(self._pending_tech_lef)
            tech_info = TechInfo.from_tech_lef(tech_lef)

        combined_entry = LibraryEntry(
            library=combined_lib,
            normalizer=normalizer,
            metrics=metrics,
            lef_cells=combined_lef,
            tech_info=tech_info,
            source_file=None,
        )
        combined_entry._cell_sources = cell_file_sources  # type: ignore

        self.entries = [combined_entry]
        self._cell_sources = cell_file_sources
        self._combined = True

    def combine(self, allow_duplicates: bool = False, baseline: Optional[str] = None) -> "Dataset":
        """Combine all staged libraries into ONE dataset with unified normalization.

        Calling this method is optional — all read methods (``to_dataframe()``,
        ``query_cell_at()``, etc.) trigger combining automatically on first access.
        Call ``combine()`` explicitly to control ``allow_duplicates`` or ``baseline``
        before the first read, or to force a re-combine after adding more files.

        Args:
            allow_duplicates: If False (default), raise DuplicateCellError when
                cells with the same name appear in multiple files. If True,
                first occurrence wins.
            baseline: Name of the baseline inverter cell to use for normalization.
                Auto-detected if not provided.

        Returns:
            self, for method chaining.

        Raises:
            DuplicateCellError: If duplicates found and allow_duplicates=False.
            ValueError: If no baseline cell found.

        Example:
            >>> ds = Dataset()
            >>> ds.load_files(["svt.lib", "lvt.lib"])  # auto-combined on first access
            >>> df = ds.to_dataframe()                  # triggers combine here

            >>> # Or explicitly control options:
            >>> ds.combine(allow_duplicates=True).to_dataframe()
        """
        self._combine_opts = {"allow_duplicates": allow_duplicates, "baseline": baseline}
        self._combined = False  # Force re-combine with new opts
        self._do_combine(allow_duplicates=allow_duplicates, baseline=baseline)
        return self

    # ------------------------------------------------------------------
    # Public resolve + property accessors (P0/P1 API improvements)
    # ------------------------------------------------------------------

    def resolve(self) -> "Dataset":
        """Trigger the lazy combine immediately and return self.

        Calling this is equivalent to touching any read method, but makes
        the intent explicit in scripts that inspect ``entries`` directly.
        Idempotent — safe to call multiple times.

        Returns:
            self, for method chaining.

        Example::

            >>> ds = Dataset().load_files(["svt.lib", "lvt.lib"]).resolve()
            >>> lib = ds.library   # always populated after resolve()
        """
        self._ensure_combined()
        return self

    @property
    def library(self):
        """The combined :class:`~parsfet.models.liberty.LibertyLibrary`.

        Triggers lazy combine on first access.

        Raises:
            ValueError: If no libraries have been loaded.

        Example::

            >>> ds = Dataset().load_files(["lib.lib"])
            >>> print(ds.library.name)
        """
        self._ensure_combined()
        if not self.entries:
            raise ValueError("No libraries loaded.")
        return self.entries[0].library

    @property
    def normalizer(self):
        """The :class:`~parsfet.normalizers.invd1.INVD1Normalizer` for the combined library.

        Triggers lazy combine on first access.

        Returns:
            The normalizer, or ``None`` if no baseline inverter was found.

        Example::

            >>> print(ds.normalizer.baseline_cell.name)
        """
        self._ensure_combined()
        if not self.entries:
            return None
        return self.entries[0].normalizer

    @property
    def baseline(self):
        """The baseline :class:`~parsfet.models.liberty.Cell` used for normalisation.

        Triggers lazy combine on first access.

        Returns:
            The baseline ``Cell`` object, or ``None`` if unavailable.

        Example::

            >>> print(ds.baseline.name, ds.baseline.area)
        """
        norm = self.normalizer
        return norm.baseline_cell if norm else None

    @property
    def cell_names(self) -> list[str]:
        """Sorted list of all cell names in the combined library.

        Triggers lazy combine on first access.

        Returns:
            An empty list if no libraries are loaded.

        Example::

            >>> print(ds.cell_names[:5])
        """
        self._ensure_combined()
        if not self.entries:
            return []
        return sorted(self.entries[0].library.cells.keys())

    @property
    def lef_cells(self) -> dict:
        """Dict of LEF macro data matched to Liberty cells.

        Populated after :meth:`load_lef` and lazy combine.
        Returns an empty dict if no LEF files were loaded.

        Example::

            >>> matched = len(ds.lef_cells)
        """
        self._ensure_combined()
        if not self.entries:
            return {}
        return self.entries[0].lef_cells or {}

    @property
    def tech_info(self):
        """:class:`~parsfet.models.physical.TechInfo` from the loaded TechLEF.

        ``None`` if no TechLEF file was loaded.

        Example::

            >>> print(ds.tech_info.metal_stack_height)
        """
        self._ensure_combined()
        if not self.entries:
            return None
        return self.entries[0].tech_info

    def cell(self, name: str):
        """Look up a single :class:`~parsfet.models.liberty.Cell` by name.

        Triggers lazy combine on first access.

        Args:
            name: Exact cell name as it appears in the Liberty file.

        Returns:
            The ``Cell`` object.

        Raises:
            KeyError: If the cell is not found in the combined library.
            ValueError: If no libraries have been loaded.

        Example::

            >>> inv = ds.cell("INV_X1")
            >>> print(inv.area)
        """
        return self.library.cells[name]

    def __getitem__(self, name: str):
        """Shorthand for :meth:`cell`.

        Example::

            >>> inv = ds["INV_X1"]
        """
        return self.cell(name)

    def to_dataframe(self) -> pd.DataFrame:

        """Convert all loaded data to a flat Pandas DataFrame.

        Each row represents a single cell. Columns include:
        - library: Source library name
        - cell: Cell name
        - cell_type: Classified cell type (inverter, nand, etc.)
        - Ratio columns: area_ratio, d0_ratio, k_ratio, etc.
        - Raw metric columns: raw_area_um2, raw_d0_ns, raw_k_ns_per_pf,
          raw_leakage, raw_input_cap_pf
        - Fit quality: fit_r_squared (R² of linear model), fit_residual_pct
        - Context: voltage, temperature, process_node
        - LEF columns (if loaded): lef_width, lef_height, lef_area, pin_layers_json
        - TechLEF columns (if loaded): metal_stack_height
        - source_file: Path to the source .lib file (for provenance tracking)

        Returns:
            A DataFrame with one row per cell across all libraries.
        """
        rows = []

        self._ensure_combined()

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
                    "cell_type": m.cell_type.name.lower(),  # Convert enum to lowercase string
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
                    "raw_power_fo4": m.raw_power_fo4,  # None = no power tables; 0.0 = valid zero-energy result
                    # Cell properties
                    "num_inputs": m.num_inputs,
                    "num_outputs": m.num_outputs,
                    "is_sequential": m.is_sequential,
                    # Fit quality — reliability of the linear model for this cell
                    "fit_r_squared": m.fit_r_squared,
                    "fit_residual_pct": m.fit_residual_pct,
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
                    # Pin info as JSON for easy analysis (includes use type and per-layer area)
                    pin_info = {
                        name: {
                            "direction": p.direction,
                            "use": p.use,
                            "layers": p.layers,
                            "area_by_layer": p.area_by_layer,
                        }
                        for name, p in lef_cell.pins.items()
                    }
                    row["pin_layers_json"] = json.dumps(pin_info)
                    # OBS area per layer as JSON. Serializes '{}' when this cell has no
                    # obstruction shapes (vs. None which means no LEF was loaded at all).
                    row["obs_area_json"] = json.dumps(lef_cell.obs_area_by_layer)
                else:
                    row["lef_width"] = None
                    row["lef_height"] = None
                    row["lef_area"] = None
                    row["pin_layers_json"] = None
                    row["obs_area_json"] = None

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

    def to_vector(self) -> list[float]:
        """Extract library-level fingerprint feature vector for ML.

        This returns a compact 15-element vector representing the technology
        library's characteristics. Unlike to_numpy() which returns per-cell
        features, this returns aggregate library statistics.

        Returns:
            A 15-element feature vector:
            - [0-3]: Mean ratios (area, d0, k, leakage)
            - [4-7]: Std ratios (area, d0, k, leakage)
            - [8]: Total cells / 1000 (normalized)
            - [9-10]: Combinational ratio, sequential ratio
            - [11-13]: Inverter ratio, NAND ratio, DFF ratio
            - [14]: Drive diversity (max_area / min_area)

        Example:
            >>> ds = Dataset().load_files(["lib.lib"])
            >>> vector = ds.to_vector()
            >>> print(len(vector))  # 15
        """
        self._ensure_combined()

        if not self.entries:
            return [0.0] * 15

        entry = self.entries[0]

        if not entry.normalizer:
            # No normalization available
            return [0.0] * 15

        summary = entry.normalizer.get_summary()

        # Extract statistics
        def get_stat(category: str, stat: str) -> float:
            return summary.get(category, {}).get(stat, 0.0)

        mean_area = get_stat("area_ratio_stats", "mean")
        mean_d0 = get_stat("d0_ratio_stats", "mean")
        mean_k = get_stat("k_ratio_stats", "mean")
        mean_leakage = get_stat("leakage_ratio_stats", "mean")

        std_area = get_stat("area_ratio_stats", "std")
        std_d0 = get_stat("d0_ratio_stats", "std")
        std_k = get_stat("k_ratio_stats", "std")
        std_leakage = get_stat("leakage_ratio_stats", "std")

        total_cells = summary.get("total_cells", 0)

        # Cell type distribution
        # Note: Summary uses lowercase strings from CellType enum
        type_counts = summary.get("cell_type_counts", {})
        
        sequential_count = type_counts.get("flip_flop", 0) + type_counts.get("latch", 0)
        combinational_count = total_cells - sequential_count

        # Functional type ratios
        # Using specific CellType keys
        inverter_count = type_counts.get("inverter", 0)
        nand_count = type_counts.get("nand", 0)
        dff_count = type_counts.get("flip_flop", 0)

        # Drive diversity
        min_area = get_stat("area_ratio_stats", "min")
        max_area = get_stat("area_ratio_stats", "max")
        drive_diversity = max_area / max(1.0, min_area) if min_area > 0 else 0.0

        return [
            mean_area,
            mean_d0,
            mean_k,
            mean_leakage,
            std_area,
            std_d0,
            std_k,
            std_leakage,
            float(total_cells) / 1000.0,  # Normalized cell count
            float(combinational_count) / max(1, total_cells),
            float(sequential_count) / max(1, total_cells),
            float(inverter_count) / max(1, total_cells),
            float(nand_count) / max(1, total_cells),
            float(dff_count) / max(1, total_cells),
            drive_diversity,
        ]

    def estimate_path_delay(
        self,
        path: list["PathSpec"],
        config: "AnalysisConfig | None" = None,
    ) -> "TimingPath":
        """Convenience shim — delegates to :func:`parsfet.path_delay.estimate_path_delay`.

        See that function for full documentation and examples.
        """
        return _estimate_path_delay_impl(self, path, config)

    def list_cell_arcs(
        self,
        cell_name: str,
        entry_index: int = 0,
    ) -> list[dict]:
        """List all timing arcs defined for a cell.

        Use this as a discovery step before calling :meth:`query_cell_at` to find
        the exact ``from_pin``, ``to_pin``, and ``timing_type`` values available.

        Args:
            cell_name: Name of the cell.
            entry_index: Index of the library entry (default 0).

        Returns:
            A list of dicts, one per arc, each with:
            - ``from_pin`` (str): Related (input) pin driving the arc.
            - ``to_pin`` (str): Output pin where the arc terminates.
            - ``timing_type`` (str | None): Arc semantics (e.g. ``"combinational"``,
              ``"rising_edge"``, ``"setup_rising"``).
            - ``timing_sense`` (str): Unateness (e.g. ``"positive_unate"``).

        Raises:
            KeyError: If ``cell_name`` is not found in the entry.
            IndexError: If ``entry_index`` is out of range.

        Example:
            >>> ds = Dataset().load_files(["lib.lib"])
            >>> arcs = ds.list_cell_arcs("NAND2_X1")
            >>> for arc in arcs:
            ...     print(arc["from_pin"], arc["to_pin"], arc["timing_type"])
            A ZN combinational
            B ZN combinational
        """
        self._ensure_combined()

        if entry_index >= len(self.entries):
            raise IndexError(f"Entry index {entry_index} out of range")
        lib = self.entries[0].library
        if cell_name not in lib.cells:
            raise KeyError(f"Cell '{cell_name}' not found in library entry {entry_index}")
        cell = lib.cells[cell_name]

        # NOTE: The Liberty parser flattens all timing{} blocks into Cell.timing_arcs
        # without retaining which output pin each arc belongs to. We therefore cannot
        # reliably determine to_pin per arc — it is always returned as None.
        # TODO: store the owning output-pin name on TimingArc during parsing so
        #       to_pin can be populated correctly.
        return [
            {
                "from_pin": arc.related_pin,
                "to_pin": None,  # Not available without parser-side attribution
                "timing_type": arc.timing_type,
                "timing_sense": arc.timing_sense,
            }
            for arc in cell.timing_arcs
        ]

    def query_cell_at(
        self,
        cell_name: str,
        slew_ns: float,
        load_pf: float,
        *,
        from_pin: "Optional[str]" = None,
        to_pin: "Optional[str]" = None,
        timing_type: "Optional[str]" = None,
        entry_index: int = 0,
    ) -> list[dict]:
        """Query delay, output slew, and switching energy at a specific operating point.

        Uses the raw NLDM lookup tables for interpolation (more accurate than the linear
        D0+k approximation away from the FO4 point). Accepts canonical units (ns, pF)
        and converts to library raw units internally.

        Args:
            cell_name: Name of the cell to query.
            slew_ns: Input transition time in **nanoseconds** (canonical units).
            load_pf: Output load capacitance in **picofarads** (canonical units).
            from_pin: Filter to arcs driven by this input pin (e.g. ``"A"``). ``None``
                      includes all input pins.
            to_pin: Accepted for forward-compatibility but **does not currently filter**
                    results. The Liberty parser flattens ``timing{}`` blocks into a single
                    ``Cell.timing_arcs`` list without retaining which output pin each arc
                    belongs to, so per-output-pin filtering is not yet possible.
                    Pass ``None`` (default) to avoid confusion.
            timing_type: Filter by arc type (e.g. ``"combinational"``, ``"rising_edge"``).
                         ``None`` includes all types.
            entry_index: Index of the library entry (default 0).

        Returns:
            A list of dicts, one per matching arc, each containing:

            - ``from_pin`` (str): Related input pin.
            - ``timing_type`` (str | None): Arc timing type.
            - ``timing_sense`` (str): Unateness.
            - ``delay_ns`` (float): Average rise+fall delay in nanoseconds.
            - ``output_slew_ns`` (float): Average rise+fall output transition in nanoseconds.
            - ``energy_fj`` (float): Switching energy in library energy units at the
              given operating point. ``0.0`` when no power tables exist for this arc.

        Raises:
            KeyError: If ``cell_name`` is not found in the library entry.
            IndexError: If ``entry_index`` is out of range.
            ValueError: If no timing arcs match the given filters.

        Example:
            >>> ds = Dataset().load_files(["lib.lib"])
            >>> # All arcs for INV_X1
            >>> arcs = ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.01)
            >>> print(arcs[0]["delay_ns"])
            >>> # Specific arc for NAND2_X1
            >>> arcs = ds.query_cell_at("NAND2_X1", 0.05, 0.01, from_pin="A")
            >>> print(arcs[0]["delay_ns"], arcs[0]["energy_fj"])
        """
        self._ensure_combined()

        if entry_index >= len(self.entries):
            raise IndexError(f"Entry index {entry_index} out of range")
        entry = self.entries[0]
        lib = entry.library
        if cell_name not in lib.cells:
            raise KeyError(f"Cell '{cell_name}' not found in library entry {entry_index}")
        cell = lib.cells[cell_name]

        # Convert canonical ns/pF to raw library units for interpolation
        un = lib.unit_normalizer
        raw_slew = slew_ns / un.time_multiplier if un.time_multiplier > 0 else slew_ns
        raw_load = load_pf / un.cap_multiplier if un.cap_multiplier > 0 else load_pf

        # Build mapping from related_pin -> list[PowerArc] for fast energy lookup
        power_by_pin: dict[str, list] = {}
        for parc in cell.power_arcs:
            key = parc.related_pin or ""
            power_by_pin.setdefault(key, []).append(parc)

        results: list[dict] = []
        for arc in cell.timing_arcs:
            # Apply filters
            if from_pin is not None and arc.related_pin != from_pin:
                continue
            if timing_type is not None and arc.timing_type != timing_type:
                continue
            # TODO: to_pin filtering requires storing the owning output-pin name on
            # TimingArc during parsing. For now, to_pin is silently ignored.

            # Delay and output-slew from timing arc tables
            delay = arc.delay_at(raw_slew, raw_load) * un.time_multiplier
            out_slew = arc.output_transition_at(raw_slew, raw_load) * un.time_multiplier

            # Energy from matching power arcs (same related_pin, or global arcs)
            energy = 0.0
            matching_parcs = power_by_pin.get(arc.related_pin or "", []) + power_by_pin.get("", [])
            energy_vals: list[float] = []
            for parc in matching_parcs:
                if parc.rise_power:
                    energy_vals.append(parc.rise_power.interpolate(raw_slew, raw_load))
                if parc.fall_power:
                    energy_vals.append(parc.fall_power.interpolate(raw_slew, raw_load))
            if energy_vals:
                energy = sum(energy_vals) / len(energy_vals)

            results.append({
                "from_pin": arc.related_pin,
                "timing_type": arc.timing_type,
                "timing_sense": arc.timing_sense,
                "delay_ns": delay,
                "output_slew_ns": out_slew,
                "energy_fj": energy,
            })

        if not results:
            active_filters = {
                k: v for k, v in {
                    "from_pin": from_pin,
                    "timing_type": timing_type,
                }.items() if v is not None
            }
            if active_filters:
                raise ValueError(
                    f"No timing arcs match filters {active_filters} for cell '{cell_name}'. "
                    f"Use list_cell_arcs('{cell_name}') to see available arcs."
                )
            # No arcs at all (cell has no timing data)
            return []

        return results

    def to_summary_dict(self) -> dict:
        """Generate a fingerprint-like summary dictionary.

        Returns a dictionary similar to TechnologyFingerprint.to_dict()
        for JSON export and human-readable summaries.

        Returns:
            A dictionary with baseline, statistics, and cell counts.
        """
        self._ensure_combined()

        if not self.entries:
            return {"error": "No entries loaded"}

        entry = self.entries[0]
        lib = entry.library

        if not entry.normalizer:
            return {
                "library": lib.name,
                "error": "No baseline inverter found",
            }

        summary = entry.normalizer.get_summary()

        # Extract cell counts
        total_cells = summary["total_cells"]
        type_counts = summary.get("cell_type_counts", {})
        
        sequential = type_counts.get("flip_flop", 0) + type_counts.get("latch", 0)
        combinational = total_cells - sequential

        # Functional type counts
        # Map from CellType enum names (lowercase) to output schema
        function_types = {
            "inverter": type_counts.get("inverter", 0),
            "buffer": type_counts.get("buffer", 0),
            "and": type_counts.get("and", 0),
            "or": type_counts.get("or", 0),
            "nand": type_counts.get("nand", 0),
            "nor": type_counts.get("nor", 0),
            "xor": type_counts.get("xor", 0),
            "xnor": type_counts.get("xnor", 0),
            "dff": type_counts.get("flip_flop", 0),
            "latch": type_counts.get("latch", 0),
        }

        return {
            "library": lib.name,
            "baseline": {
                "cell": summary["baseline_cell"],
                "area_um2": summary["baseline_raw"]["area"],
                "d0_ns": summary["baseline_raw"]["d0_ns"],
                "k_ns_per_pf": summary["baseline_raw"]["k_ns_per_pf"],
                "leakage": summary["baseline_raw"]["leakage"],
            },
            "normalized_stats": {
                "area": {
                    "mean": summary["area_ratio_stats"].get("mean", 0.0),
                    "std": summary["area_ratio_stats"].get("std", 0.0),
                    "min": summary["area_ratio_stats"].get("min", 0.0),
                    "max": summary["area_ratio_stats"].get("max", 0.0),
                },
                "d0": {
                    "mean": summary["d0_ratio_stats"].get("mean", 0.0),
                    "std": summary["d0_ratio_stats"].get("std", 0.0),
                },
                "k": {
                    "mean": summary["k_ratio_stats"].get("mean", 0.0),
                    "std": summary["k_ratio_stats"].get("std", 0.0),
                },
                "leakage": {
                    "mean": summary["leakage_ratio_stats"].get("mean", 0.0),
                    "std": summary["leakage_ratio_stats"].get("std", 0.0),
                },
            },
            "cell_counts": {
                "total": total_cells,
                "combinational": combinational,
                "sequential": sequential,
            },
            "function_types": function_types,
            "metadata": {
                "process_node": lib.process_node,
                "foundry": lib.foundry,
                "vt_flavor": lib.vt_flavor.value if lib.vt_flavor else None,
            },
        }

    def export_to_json(self, include_port_geometry: bool = False) -> dict:
        """Export combined Liberty + LEF/TechLEF data to a JSON-serializable dict.

        This produces a comprehensive JSON structure that includes:
        - Liberty timing data (normalized metrics, FO4 operating point)
        - LEF cell physical data (dimensions, pin layers, pin use types)
        - TechLEF technology data (metal stack, layer min sizes)

        Args:
            include_port_geometry: If True, each pin in the output will include
                a 'ports' list of raw rectangles (x1, y1, x2, y2, layer)
                from the LEF port geometry. Disabled by default to keep the
                output compact; enable for pin-access analysis or GNN use cases.

        Returns:
            A dictionary suitable for JSON serialization.

        Example:
            >>> ds = Dataset()
            >>> ds.load_files(["lib.lib"]).load_lef(["cell.lef"]).load_tech_lef("tech.lef")
            >>> data = ds.export_to_json()
            >>> with open("output.json", "w") as f:
            ...     json.dump(data, f, indent=2)
        """
        self._ensure_combined()

        if not self.entries:
            return {"error": "No libraries loaded"}

        entry = self.entries[0]

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
        def _serialize_pin(pin):
            d = {
                "direction": pin.direction,
                "use": pin.use,  # power, ground, clock, signal
                "layers": pin.layers,
                "area_by_layer": pin.area_by_layer,  # um² per layer (sum of rects)
            }
            if include_port_geometry:
                # Raw port rectangles for pin-access / GNN workflows.
                d["ports"] = pin.ports
            return d

        for cell_name in result.get("cells", {}):
            if cell_name in entry.lef_cells:
                lef_cell = entry.lef_cells[cell_name]
                result["cells"][cell_name]["physical"] = {
                    "width_um": lef_cell.width,
                    "height_um": lef_cell.height,
                    "area_um2": lef_cell.area,
                    "pins": {
                        pin_name: _serialize_pin(pin)
                        for pin_name, pin in lef_cell.pins.items()
                    },
                    # Sum of OBS rectangle areas per layer (um²). May overestimate
                    # if obstruction rectangles overlap within the same layer.
                    "obstructions_area": lef_cell.obs_area_by_layer,
                }

        return result

    def save_json(
        self,
        path: Path | str,
        indent: int = 2,
        include_port_geometry: bool = False,
    ) -> None:
        """Save combined Liberty + LEF/TechLEF data to a JSON file.

        Args:
            path: Output file path.
            indent: JSON indentation (default 2).
            include_port_geometry: If True, includes raw port rectangle coordinates
                for each pin. See export_to_json() for details.
        """
        data = self.export_to_json(include_port_geometry=include_port_geometry)
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
