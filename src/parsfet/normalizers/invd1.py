"""INVD1 Baseline Normalizer.

This module implements the normalization of technology metrics to a baseline inverter
(typically INVD1). This approach allows for process-independent comparisons by
expressing performance characteristics (area, delay, leakage) as multiples of
the fundamental atomic unit of the technology.

The normalization uses a linear delay model (D = D0 + k * Load) extracted at
the FO4 (Fanout-of-4) operating point.
"""

from dataclasses import dataclass
import statistics
from typing import Optional

from ..models.liberty import Cell, LibertyLibrary
from .classifier import CellType, classify_cell


@dataclass
class NormalizedMetrics:
    """Cell metrics normalized to the INVD1 baseline.

    Attributes:
        cell_name: Name of the cell.
        cell_type: Classified type of the cell (CellType enum).
        area_ratio: Cell area / Baseline area.
        d0_ratio: Intrinsic delay ratio (Cell D0 / Baseline D0).
        k_ratio: Load slope ratio (Cell k / Baseline k).
        leakage_ratio: Cell leakage / Baseline leakage.
        input_cap_ratio: Cell input capacitance / Baseline input capacitance.
        drive_strength: Estimated drive strength relative to baseline.
        num_inputs: Number of input pins.
        num_outputs: Number of output pins.
        is_sequential: True if the cell is sequential.
        raw_area: Raw area in um².
        raw_d0_ns: Raw intrinsic delay in ns.
        raw_k_ns_per_pf: Raw load slope in ns/pF.
        raw_leakage: Raw leakage power.
        raw_input_cap: Raw input capacitance in pF.
        fit_r_squared: R² of the linear fit (1.0 = perfect).
        fit_residual_pct: Signed residual at FO4 (%, positive = pessimistic).
    """

    cell_name: str
    cell_type: CellType = CellType.UNKNOWN  # From classifier

    # Core ratios
    area_ratio: float = 1.0  # cell_area / invd1_area
    d0_ratio: float = 1.0  # cell.D₀ / invd1.D₀ (intrinsic delay ratio)
    k_ratio: float = 1.0  # cell.k / invd1.k (load slope ratio)
    leakage_ratio: float = 1.0  # cell_leakage / invd1_leakage
    input_cap_ratio: float = 1.0  # input_cap / invd1_input_cap

    # Additional metrics
    drive_strength: float = 1.0  # Relative drive strength
    num_inputs: int = 1
    num_outputs: int = 1
    is_sequential: bool = False

    # Raw values for reference (in canonical units: ns, pF)
    raw_area: float = 0.0
    raw_d0_ns: float = 0.0  # Intrinsic delay (zero-load)
    raw_k_ns_per_pf: float = 0.0  # Load slope
    raw_leakage: float = 0.0
    raw_input_cap: float = 0.0

    # Fit quality metrics
    fit_r_squared: float = 1.0  # R² of linear fit (1.0 = perfect)
    fit_residual_pct: float = 0.0  # Signed residual at FO4 (%, positive = pessimistic)

    def to_dict(self) -> dict:
        """Converts the normalized metrics to a dictionary for JSON serialization."""
        return {
            "cell_name": self.cell_name,
            "cell_type": self.cell_type.name.lower(),  # Serialize enum to lowercase string
            "area_ratio": self.area_ratio,
            "d0_ratio": self.d0_ratio,
            "k_ratio": self.k_ratio,
            "leakage_ratio": self.leakage_ratio,
            "input_cap_ratio": self.input_cap_ratio,
            "delay_model": {
                "d0_ns": self.raw_d0_ns,
                "k_ns_per_pf": self.raw_k_ns_per_pf,
            },
            "fit_quality": {
                "r_squared": self.fit_r_squared,
                "fo4_residual_pct": self.fit_residual_pct,
            },
            "drive_strength": self.drive_strength,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "is_sequential": self.is_sequential,
            "raw": {
                "area_um2": self.raw_area,
                "d0_ns": self.raw_d0_ns,
                "k_ns_per_pf": self.raw_k_ns_per_pf,
                "leakage": self.raw_leakage,
                "input_cap_pf": self.raw_input_cap,
            },
        }

    def to_feature_vector(self) -> list[float]:
        """Converts metrics to a feature vector for machine learning tasks.

        Features include ratios for area, D0, k, leakage, input cap, as well as
        input/output counts and sequential flag.
        """
        return [
            self.area_ratio,
            self.d0_ratio,
            self.k_ratio,
            self.leakage_ratio,
            self.input_cap_ratio,
            float(self.num_inputs),
            float(self.num_outputs),
            1.0 if self.is_sequential else 0.0,
        ]

    def to_exported_cell(self) -> "ExportedCell":
        """Converts to Pydantic ExportedCell for JSON serialization.

        This provides a typed conversion to the export schema, useful when
        building ExportedLibrary objects programmatically.

        Returns:
            An ExportedCell instance with all fields populated.
        """
        from ..models.export import (
            ExportedCell,
            ExportedDelayModel,
            ExportedFitQuality,
            ExportedRawMetrics,
        )

        return ExportedCell(
            cell_name=self.cell_name,
            cell_type=self.cell_type.name.lower(),
            area_ratio=self.area_ratio,
            d0_ratio=self.d0_ratio,
            k_ratio=self.k_ratio,
            leakage_ratio=self.leakage_ratio,
            input_cap_ratio=self.input_cap_ratio,
            drive_strength=self.drive_strength,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            is_sequential=self.is_sequential,
            delay_model=ExportedDelayModel(
                d0_ns=self.raw_d0_ns,
                k_ns_per_pf=self.raw_k_ns_per_pf,
            ),
            fit_quality=ExportedFitQuality(
                r_squared=self.fit_r_squared,
                fo4_residual_pct=self.fit_residual_pct,
            ),
            raw=ExportedRawMetrics(
                area_um2=self.raw_area,
                d0_ns=self.raw_d0_ns,
                k_ns_per_pf=self.raw_k_ns_per_pf,
                leakage=self.raw_leakage,
                input_cap_pf=self.raw_input_cap,
            ),
        )


@dataclass
class BaselineMetrics:
    """Metrics extracted from the baseline cell (typically INVD1).

    Includes the linear delay model parameters (D0, k) and the FO4 operating point
    used for extraction.
    """

    cell_name: str
    area: float
    d0: float  # D₀: intrinsic delay (zero-load) in ns
    k: float  # k: load slope in ns/pF
    leakage: float
    input_cap: float  # Total input capacitance
    # FO4 operating point
    fo4_slew: float = 0.0
    fo4_load: float = 0.0


class INVD1Normalizer:
    """Normalizes library cells to an INVD1 baseline.

    This class handles the identification of the baseline cell, extraction of its
    parameters at the FO4 operating point, and the calculation of normalized ratios
    for all other cells in the library.
    """

    def __init__(self, library: LibertyLibrary, baseline_name: Optional[str] = None):
        """Initializes the normalizer with a library.

        Args:
            library: The parsed Liberty library.
            baseline_name: Optional name of the baseline cell. If not provided,
                automatic detection is attempted.

        Raises:
            ValueError: If the baseline cell cannot be found or identified.
        """
        self.library = library

        # Find baseline cell
        if baseline_name:
            if baseline_name not in library.cells:
                raise ValueError(f"Baseline cell '{baseline_name}' not found in library")
            self.baseline_cell = library.cells[baseline_name]
        else:
            self.baseline_cell = library.baseline_cell

        if not self.baseline_cell:
            raise ValueError(
                f"No baseline inverter found in library '{library.name}'. "
                f"Available cells: {list(library.cells.keys())[:10]}..."
            )

        # Get FO4 operating point (already in library units)
        self.fo4_slew, self.fo4_load = library.fo4_operating_point()

        # Get unit normalizer for converting to canonical units
        self.unit_normalizer = library.unit_normalizer

        # Extract baseline metrics at FO4 operating point (normalized to canonical units)
        self.baseline = self._extract_baseline_metrics(self.baseline_cell)

    def _extract_baseline_metrics(self, cell: Cell) -> BaselineMetrics:
        """Extracts key metrics from the baseline cell using the D0+k linear model.

        Converts all values to canonical units (ns, pF).
        """
        # Area (typically in um², no conversion needed)
        area = cell.area if cell.area > 0 else 1.0

        # Extract linear delay model: D = D₀ + k × Load
        # Use FO4 slew as the operating point for model extraction
        d0_raw, k_raw, _ = cell.linear_delay_model(self.fo4_slew)  # Ignore R² for baseline

        # Convert to canonical units
        if d0_raw <= 0:
            # Fallback: use FO4 delay as D₀ estimate (conservative)
            d0_raw = cell.delay_at(self.fo4_slew, self.fo4_load)
            if d0_raw <= 0:
                d0_raw = cell.representative_delay
            if d0_raw <= 0:
                d0_raw = 0.01  # 10ps default
            k_raw = 0.0  # Unknown slope

        d0 = self.unit_normalizer.normalize_time(d0_raw)
        # k is time/capacitance, so normalize both dimensions
        # k_canonical = k_raw * (time_scale / cap_scale)
        k = (
            self.unit_normalizer.normalize_time(k_raw)
            / self.unit_normalizer.normalize_capacitance(1.0)
            if k_raw > 0
            else 0.0
        )

        # Leakage power (no conversion for now)
        leakage = cell.cell_leakage_power if cell.cell_leakage_power else 1.0

        # Input capacitance, converted to pF
        input_cap = cell.total_input_capacitance
        if input_cap <= 0:
            input_cap = 0.001
        input_cap = self.unit_normalizer.normalize_capacitance(input_cap)

        # FO4 operating point in canonical units
        fo4_slew_ns = self.unit_normalizer.normalize_time(self.fo4_slew)
        fo4_load_pf = self.unit_normalizer.normalize_capacitance(self.fo4_load)

        return BaselineMetrics(
            cell_name=cell.name,
            area=area,
            d0=d0,
            k=k,
            leakage=leakage,
            input_cap=input_cap,
            fo4_slew=fo4_slew_ns,
            fo4_load=fo4_load_pf,
        )

    def normalize(self, cell: Cell) -> NormalizedMetrics:
        """Normalizes a single cell to the baseline parameters.

        Calculates ratios for area, delay (D0, k), leakage, and capacitance.

        Args:
            cell: The cell to normalize.

        Returns:
            A NormalizedMetrics object containing the calculated ratios and raw values.
        """
        # Classify cell by logic function
        cell_type = classify_cell(cell)

        # Extract raw values and convert to canonical units
        raw_area = cell.area if cell.area > 0 else 0.0

        # Extract linear delay model: D = D₀ + k × Load (using slowest arc)
        d0_raw, k_raw, fit_r2 = cell.linear_delay_model(self.fo4_slew)

        # Convert to canonical units
        if d0_raw <= 0:
            # Fallback: use delay at FO4 as D₀ estimate
            d0_raw = cell.delay_at(self.fo4_slew, self.fo4_load)
            if d0_raw <= 0:
                d0_raw = cell.representative_delay
            k_raw = 0.0
            fit_r2 = 1.0  # No fit quality data

        raw_d0_ns = self.unit_normalizer.normalize_time(d0_raw) if d0_raw > 0 else 0.0
        raw_k_ns_per_pf = (
            (
                self.unit_normalizer.normalize_time(k_raw)
                / self.unit_normalizer.normalize_capacitance(1.0)
            )
            if k_raw > 0
            else 0.0
        )

        raw_leakage = cell.cell_leakage_power if cell.cell_leakage_power else 0.0

        # Convert capacitance to pF
        raw_input_cap = cell.total_input_capacitance
        if raw_input_cap > 0:
            raw_input_cap = self.unit_normalizer.normalize_capacitance(raw_input_cap)

        # Compute FO4 residual: (model - actual) / actual * 100
        # Positive = pessimistic (model predicts slower)
        actual_fo4_delay = cell.delay_at(self.fo4_slew, self.fo4_load)
        if actual_fo4_delay > 0 and d0_raw > 0:
            model_fo4_delay = d0_raw + k_raw * self.fo4_load
            fit_residual_pct = ((model_fo4_delay - actual_fo4_delay) / actual_fo4_delay) * 100
        else:
            fit_residual_pct = 0.0

        # Compute ratios (both are now in canonical units)
        area_ratio = raw_area / self.baseline.area if self.baseline.area > 0 else 0.0
        d0_ratio = raw_d0_ns / self.baseline.d0 if self.baseline.d0 > 0 and raw_d0_ns > 0 else 1.0
        k_ratio = (
            raw_k_ns_per_pf / self.baseline.k
            if self.baseline.k > 0 and raw_k_ns_per_pf > 0
            else 1.0
        )
        leakage_ratio = raw_leakage / self.baseline.leakage if self.baseline.leakage > 0 else 0.0
        input_cap_ratio = (
            raw_input_cap / self.baseline.input_cap if self.baseline.input_cap > 0 else 0.0
        )

        # Estimate drive strength from area (larger cell = more drive)
        drive_strength = area_ratio if area_ratio > 0 else 1.0

        return NormalizedMetrics(
            cell_name=cell.name,
            cell_type=cell_type,
            area_ratio=area_ratio,
            d0_ratio=d0_ratio,
            k_ratio=k_ratio,
            leakage_ratio=leakage_ratio,
            input_cap_ratio=input_cap_ratio,
            drive_strength=drive_strength,
            num_inputs=len(cell.input_pins),
            num_outputs=len(cell.output_pins),
            is_sequential=cell.is_sequential,
            raw_area=raw_area,
            raw_d0_ns=raw_d0_ns,
            raw_k_ns_per_pf=raw_k_ns_per_pf,
            raw_leakage=raw_leakage,
            raw_input_cap=raw_input_cap,
            fit_r_squared=fit_r2,
            fit_residual_pct=fit_residual_pct,
        )

    def normalize_all(self) -> dict[str, NormalizedMetrics]:
        """Normalizes all cells in the library.

        Returns:
            A dictionary mapping cell names to their NormalizedMetrics.
        """
        return {name: self.normalize(cell) for name, cell in self.library.cells.items()}

    def normalize_from_raw(
        self,
        cell_name: str,
        raw_area: float,
        raw_d0_ns: float,
        raw_k_ns_per_pf: float,
        raw_leakage: float,
        raw_input_cap: float,
        cell_type_str: str = "unknown",
        num_inputs: int = 1,
        num_outputs: int = 1,
        is_sequential: bool = False,
    ) -> NormalizedMetrics:
        """Normalize a cell using only raw metrics (no Cell object needed).

        This method is used when re-normalizing cells imported from a JSON export.
        The raw metrics should already be in canonical units (ns, pF, um²).

        Args:
            cell_name: Name of the cell.
            raw_area: Cell area in um².
            raw_d0_ns: Intrinsic delay in ns.
            raw_k_ns_per_pf: Load slope in ns/pF.
            raw_leakage: Leakage power.
            raw_input_cap: Input capacitance in pF.
            cell_type_str: Cell type as lowercase string (e.g., "nand", "inverter").
            num_inputs: Number of input pins.
            num_outputs: Number of output pins.
            is_sequential: True if sequential cell.

        Returns:
            NormalizedMetrics with ratios computed against current baseline.
        """
        # Convert cell_type string to enum
        try:
            cell_type = CellType[cell_type_str.upper()]
        except KeyError:
            cell_type = CellType.UNKNOWN

        # Compute ratios against baseline
        area_ratio = raw_area / self.baseline.area if self.baseline.area > 0 else 0.0
        d0_ratio = raw_d0_ns / self.baseline.d0 if self.baseline.d0 > 0 and raw_d0_ns > 0 else 1.0
        k_ratio = (
            raw_k_ns_per_pf / self.baseline.k
            if self.baseline.k > 0 and raw_k_ns_per_pf > 0
            else 1.0
        )
        leakage_ratio = raw_leakage / self.baseline.leakage if self.baseline.leakage > 0 else 0.0
        input_cap_ratio = (
            raw_input_cap / self.baseline.input_cap if self.baseline.input_cap > 0 else 0.0
        )

        drive_strength = area_ratio if area_ratio > 0 else 1.0

        return NormalizedMetrics(
            cell_name=cell_name,
            cell_type=cell_type,
            area_ratio=area_ratio,
            d0_ratio=d0_ratio,
            k_ratio=k_ratio,
            leakage_ratio=leakage_ratio,
            input_cap_ratio=input_cap_ratio,
            drive_strength=drive_strength,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            is_sequential=is_sequential,
            raw_area=raw_area,
            raw_d0_ns=raw_d0_ns,
            raw_k_ns_per_pf=raw_k_ns_per_pf,
            raw_leakage=raw_leakage,
            raw_input_cap=raw_input_cap,
            fit_r_squared=1.0,  # No fit data available from raw
            fit_residual_pct=0.0,
        )


    def get_summary(self) -> dict:
        """Generates summary statistics for the normalized library.

        Includes statistical distributions (mean, min, max) of ratios and counts
        of cell types.

        Returns:
            A dictionary containing the summary statistics.
        """
        metrics = self.normalize_all()

        if not metrics:
            return {"error": "No cells to summarize"}

        areas = [m.area_ratio for m in metrics.values() if m.area_ratio > 0]
        d0s = [m.d0_ratio for m in metrics.values() if m.d0_ratio > 0]
        ks = [m.k_ratio for m in metrics.values() if m.k_ratio > 0]
        leakages = [m.leakage_ratio for m in metrics.values() if m.leakage_ratio > 0]

        # Cell type distribution
        type_counts: dict[str, int] = {}
        for m in metrics.values():
            type_name = m.cell_type.name.lower()
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        def stats(values: list[float]) -> dict:
            if not values:
                return {"count": 0}
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        return {
            "library_name": self.library.name,
            "baseline_cell": self.baseline.cell_name,
            "total_cells": len(self.library.cells),
            "cell_type_counts": type_counts,
            "area_ratio_stats": stats(areas),
            "d0_ratio_stats": stats(d0s),
            "k_ratio_stats": stats(ks),
            "leakage_ratio_stats": stats(leakages),
            "baseline_raw": {
                "area": self.baseline.area,
                "d0_ns": self.baseline.d0,
                "k_ns_per_pf": self.baseline.k,
                "leakage": self.baseline.leakage,
                "input_cap": self.baseline.input_cap,
            },
        }

    def export_to_json(self) -> dict:
        """Exports all normalized data to a JSON-serializable dictionary.

        Includes units, FO4 operating point, baseline metrics, and per-cell normalized data.
        """
        return {
            "library": self.library.name,
            "process_node": self.library.process_node,
            "foundry": self.library.foundry,
            "vt_flavor": self.library.vt_flavor.value if self.library.vt_flavor else None,
            "units": {
                "time": "ns",
                "capacitance": "pF",
                "area": "um^2",
                "source_time_unit": self.library.time_unit,
                "source_cap_unit": f"{self.library.capacitive_load_unit[0]},{self.library.capacitive_load_unit[1]}",
            },
            "fo4_operating_point": {
                "slew_ns": self.baseline.fo4_slew,
                "load_pf": self.baseline.fo4_load,
                "description": "FO4 load = 4x inverter cap; slew = inverter output slew at FO4",
            },
            "baseline": {
                "cell": self.baseline.cell_name,
                "area_um2": self.baseline.area,
                "d0_ns": self.baseline.d0,
                "k_ns_per_pf": self.baseline.k,
                "leakage": self.baseline.leakage,
                "input_cap_pf": self.baseline.input_cap,
            },
            "cells": {name: m.to_dict() for name, m in self.normalize_all().items()},
            "summary": self.get_summary(),
        }
