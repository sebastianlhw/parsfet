"""
INVD1 Baseline Normalizer

The fundamental insight (per Elon's first principles approach):
The inverter is the atomic unit of digital logic. By normalizing all
metrics to a baseline inverter (typically INVD1), we can:

1. Compare cells across different process nodes
2. Understand the "cost" of complex gates in inverter-equivalents
3. Enable ML models to learn process-independent patterns

Example:
    A NAND2 with normalized_delay = 1.5 means it's 1.5x slower than INVD1.
    If INVD1 in 7nm has delay = 10ps and in 130nm has delay = 100ps,
    the NAND2 would be ~15ps in 7nm and ~150ps in 130nm.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..models.liberty import Cell, LibertyLibrary
from .classifier import classify_cell


@dataclass
class NormalizedMetrics:
    """
    Cell metrics normalized to INVD1 baseline.

    All ratios are (cell_value / baseline_value), so:
    - ratio = 1.0 means same as baseline
    - ratio = 2.0 means 2x the baseline
    
    Delay is represented as linear model: D = D₀ + k × Load
    - d0_ratio: intrinsic delay ratio (cell.D₀ / baseline.D₀)
    - k_ratio: load slope ratio (cell.k / baseline.k)
    """

    cell_name: str
    cell_type: str = "unknown"  # From classifier: inverter, buffer, nand, etc.

    # Core ratios
    area_ratio: float = 1.0  # cell_area / invd1_area
    d0_ratio: float = 1.0    # cell.D₀ / invd1.D₀ (intrinsic delay ratio)
    k_ratio: float = 1.0     # cell.k / invd1.k (load slope ratio)
    leakage_ratio: float = 1.0  # cell_leakage / invd1_leakage
    input_cap_ratio: float = 1.0  # input_cap / invd1_input_cap

    # Additional metrics
    drive_strength: float = 1.0  # Relative drive strength
    num_inputs: int = 1
    num_outputs: int = 1
    is_sequential: bool = False

    # Raw values for reference (in canonical units: ns, pF)
    raw_area: float = 0.0
    raw_d0_ns: float = 0.0      # Intrinsic delay (zero-load)
    raw_k_ns_per_pf: float = 0.0  # Load slope
    raw_leakage: float = 0.0
    raw_input_cap: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "cell_name": self.cell_name,
            "cell_type": self.cell_type,
            "area_ratio": self.area_ratio,
            "d0_ratio": self.d0_ratio,
            "k_ratio": self.k_ratio,
            "leakage_ratio": self.leakage_ratio,
            "input_cap_ratio": self.input_cap_ratio,
            "delay_model": {
                "d0_ns": self.raw_d0_ns,
                "k_ns_per_pf": self.raw_k_ns_per_pf,
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
        """Convert to feature vector for ML
        
        Features: [area_ratio, d0_ratio, k_ratio, leakage_ratio, 
                   input_cap_ratio, num_inputs, num_outputs, is_sequential]
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


@dataclass
class BaselineMetrics:
    """Extracted metrics from the baseline cell
    
    Includes D₀+k linear delay model parameters.
    """

    cell_name: str
    area: float
    d0: float  # D₀: intrinsic delay (zero-load) in ns
    k: float   # k: load slope in ns/pF
    leakage: float
    input_cap: float  # Total input capacitance
    # FO4 operating point
    fo4_slew: float = 0.0
    fo4_load: float = 0.0


class INVD1Normalizer:
    """
    Normalize all cell metrics to INVD1 baseline.

    This enables cross-library comparison by expressing
    everything as multiples of the fundamental inverter.

    Delay is computed at the FO4 (fanout-of-4) operating point:
    - Load = 4× baseline inverter input capacitance
    - Slew = baseline inverter output transition at FO4 load

    Example usage:
        >>> parser = LibertyParser()
        >>> lib = parser.parse(Path("my_library.lib"))
        >>> normalizer = INVD1Normalizer(lib)
        >>> metrics = normalizer.normalize_all()
        >>> print(metrics["NAND2D1"].delay_ratio)  # e.g., 1.4
    """

    def __init__(self, library: LibertyLibrary, baseline_name: Optional[str] = None):
        """
        Initialize normalizer with a library.

        Args:
            library: Parsed Liberty library
            baseline_name: Optional specific baseline cell name.
                          If None, auto-detects INVD1 or equivalent.
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
        self.fo4_slew, self.fo4_load = library.fo4_operating_point

        # Get unit normalizer for converting to canonical units
        self.unit_normalizer = library.unit_normalizer

        # Extract baseline metrics at FO4 operating point (normalized to canonical units)
        self.baseline = self._extract_baseline_metrics(self.baseline_cell)

    def _extract_baseline_metrics(self, cell: Cell) -> BaselineMetrics:
        """Extract key metrics from baseline cell using D₀+k linear model.

        All values are converted to canonical units:
        - Time: nanoseconds (ns)
        - Capacitance: picofarads (pF)
        
        The linear delay model D = D₀ + k × Load is extracted using
        least-squares fitting on the cell's timing LUT.
        """
        # Area (typically in um², no conversion needed)
        area = cell.area if cell.area > 0 else 1.0

        # Extract linear delay model: D = D₀ + k × Load
        # Use FO4 slew as the operating point for model extraction
        d0_raw, k_raw = cell.linear_delay_model(self.fo4_slew)
        
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
        k = self.unit_normalizer.normalize_time(k_raw) / self.unit_normalizer.normalize_capacitance(1.0) if k_raw > 0 else 0.0

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
        """
        Normalize a single cell to baseline using D₀+k linear delay model.

        All raw values are converted to canonical units (ns, pF) for comparison.

        Args:
            cell: Cell to normalize

        Returns:
            NormalizedMetrics with D₀ ratio, k ratio, and cell type
        """
        # Classify cell by logic function
        cell_type = classify_cell(cell)
        
        # Extract raw values and convert to canonical units
        raw_area = cell.area if cell.area > 0 else 0.0

        # Extract linear delay model: D = D₀ + k × Load
        d0_raw, k_raw = cell.linear_delay_model(self.fo4_slew)
        
        # Convert to canonical units
        if d0_raw <= 0:
            # Fallback: use delay at FO4 as D₀ estimate
            d0_raw = cell.delay_at(self.fo4_slew, self.fo4_load)
            if d0_raw <= 0:
                d0_raw = cell.representative_delay
            k_raw = 0.0
        
        raw_d0_ns = self.unit_normalizer.normalize_time(d0_raw) if d0_raw > 0 else 0.0
        raw_k_ns_per_pf = (
            self.unit_normalizer.normalize_time(k_raw) / 
            self.unit_normalizer.normalize_capacitance(1.0)
        ) if k_raw > 0 else 0.0

        raw_leakage = cell.cell_leakage_power if cell.cell_leakage_power else 0.0

        # Convert capacitance to pF
        raw_input_cap = cell.total_input_capacitance
        if raw_input_cap > 0:
            raw_input_cap = self.unit_normalizer.normalize_capacitance(raw_input_cap)

        # Compute ratios (both are now in canonical units)
        area_ratio = raw_area / self.baseline.area if self.baseline.area > 0 else 0.0
        d0_ratio = raw_d0_ns / self.baseline.d0 if self.baseline.d0 > 0 and raw_d0_ns > 0 else 1.0
        k_ratio = raw_k_ns_per_pf / self.baseline.k if self.baseline.k > 0 and raw_k_ns_per_pf > 0 else 1.0
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
        )

    def normalize_all(self) -> dict[str, NormalizedMetrics]:
        """
        Normalize all cells in library.

        Returns:
            Dictionary mapping cell names to their normalized metrics
        """
        return {name: self.normalize(cell) for name, cell in self.library.cells.items()}

    def get_summary(self) -> dict:
        """
        Get summary statistics for the normalized library.

        Returns:
            Dictionary with statistical summaries
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
            type_counts[m.cell_type] = type_counts.get(m.cell_type, 0) + 1

        def stats(values: list[float]) -> dict:
            if not values:
                return {"count": 0}
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
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
        """Export all normalized data as JSON-serializable dict.

        All values are in canonical units:
        - Time: nanoseconds (ns)
        - Capacitance: picofarads (pF)
        - Area: um²
        """
        return {
            "library": self.library.name,
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
