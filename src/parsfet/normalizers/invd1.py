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


@dataclass
class NormalizedMetrics:
    """
    Cell metrics normalized to INVD1 baseline.

    All ratios are (cell_value / baseline_value), so:
    - ratio = 1.0 means same as baseline
    - ratio = 2.0 means 2x the baseline
    """

    cell_name: str

    # Core ratios
    area_ratio: float = 1.0  # cell_area / invd1_area
    delay_ratio: float = 1.0  # cell_delay / invd1_delay
    leakage_ratio: float = 1.0  # cell_leakage / invd1_leakage
    input_cap_ratio: float = 1.0  # input_cap / invd1_input_cap

    # Linear delay model: D = D₀ + k × Load
    intrinsic_delay_ns: float = 0.0  # D₀: zero-load delay
    load_slope_ns_per_pf: float = 0.0  # k: delay per unit load

    # Additional metrics
    drive_strength: float = 1.0  # Relative drive strength
    num_inputs: int = 1
    num_outputs: int = 1
    is_sequential: bool = False

    # Raw values for reference (in canonical units: ns, pF)
    raw_area: float = 0.0
    raw_delay: float = 0.0
    raw_leakage: float = 0.0
    raw_input_cap: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "cell_name": self.cell_name,
            "area_ratio": self.area_ratio,
            "delay_ratio": self.delay_ratio,
            "leakage_ratio": self.leakage_ratio,
            "input_cap_ratio": self.input_cap_ratio,
            "delay_model": {
                "intrinsic_delay_ns": self.intrinsic_delay_ns,
                "load_slope_ns_per_pf": self.load_slope_ns_per_pf,
            },
            "drive_strength": self.drive_strength,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "is_sequential": self.is_sequential,
            "raw": {
                "area_um2": self.raw_area,
                "delay_ns": self.raw_delay,
                "leakage": self.raw_leakage,
                "input_cap_pf": self.raw_input_cap,
            },
        }

    def to_feature_vector(self) -> list[float]:
        """Convert to feature vector for ML"""
        return [
            self.area_ratio,
            self.delay_ratio,
            self.leakage_ratio,
            self.input_cap_ratio,
            float(self.num_inputs),
            float(self.num_outputs),
            1.0 if self.is_sequential else 0.0,
        ]


@dataclass
class BaselineMetrics:
    """Extracted metrics from the baseline cell"""

    cell_name: str
    area: float
    delay: float  # Delay at FO4 operating point
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
        """Extract key metrics from baseline cell at FO4 operating point.

        All values are converted to canonical units:
        - Time: nanoseconds (ns)
        - Capacitance: picofarads (pF)
        """
        # Area (typically in um², no conversion needed)
        area = cell.area if cell.area > 0 else 1.0

        # Delay at FO4 operating point, converted to ns
        delay = cell.delay_at(self.fo4_slew, self.fo4_load)
        if delay <= 0:
            delay = cell.representative_delay
        if delay <= 0:
            delay = 0.01
        delay = self.unit_normalizer.normalize_time(delay)

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
            delay=delay,
            leakage=leakage,
            input_cap=input_cap,
            fo4_slew=fo4_slew_ns,
            fo4_load=fo4_load_pf,
        )

    def normalize(self, cell: Cell) -> NormalizedMetrics:
        """
        Normalize a single cell to baseline.

        All raw values are converted to canonical units (ns, pF) for comparison.

        Args:
            cell: Cell to normalize

        Returns:
            NormalizedMetrics with all ratios computed
        """
        # Extract raw values and convert to canonical units
        raw_area = cell.area if cell.area > 0 else 0.0

        # Get delay at FO4 operating point (with fallback), convert to ns
        raw_delay = cell.delay_at(self.fo4_slew, self.fo4_load)
        if raw_delay <= 0:
            raw_delay = cell.representative_delay
        if raw_delay > 0:
            raw_delay = self.unit_normalizer.normalize_time(raw_delay)

        raw_leakage = cell.cell_leakage_power if cell.cell_leakage_power else 0.0

        # Convert capacitance to pF
        raw_input_cap = cell.total_input_capacitance
        if raw_input_cap > 0:
            raw_input_cap = self.unit_normalizer.normalize_capacitance(raw_input_cap)

        # Compute ratios (both are now in canonical units, so ratio is unchanged)
        area_ratio = raw_area / self.baseline.area if self.baseline.area > 0 else 0.0
        delay_ratio = (
            raw_delay / self.baseline.delay if self.baseline.delay > 0 and raw_delay > 0 else 1.0
        )
        leakage_ratio = raw_leakage / self.baseline.leakage if self.baseline.leakage > 0 else 0.0
        input_cap_ratio = (
            raw_input_cap / self.baseline.input_cap if self.baseline.input_cap > 0 else 0.0
        )

        # Estimate drive strength from area (larger cell = more drive)
        drive_strength = area_ratio if area_ratio > 0 else 1.0

        return NormalizedMetrics(
            cell_name=cell.name,
            area_ratio=area_ratio,
            delay_ratio=delay_ratio,
            leakage_ratio=leakage_ratio,
            input_cap_ratio=input_cap_ratio,
            drive_strength=drive_strength,
            num_inputs=len(cell.input_pins),
            num_outputs=len(cell.output_pins),
            is_sequential=cell.is_sequential,
            raw_area=raw_area,
            raw_delay=raw_delay,
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
        delays = [m.delay_ratio for m in metrics.values() if m.delay_ratio > 0]
        leakages = [m.leakage_ratio for m in metrics.values() if m.leakage_ratio > 0]

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
            "area_ratio_stats": stats(areas),
            "delay_ratio_stats": stats(delays),
            "leakage_ratio_stats": stats(leakages),
            "baseline_raw": {
                "area": self.baseline.area,
                "delay": self.baseline.delay,
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
                "delay_ns": self.baseline.delay,
                "leakage": self.baseline.leakage,
                "input_cap_pf": self.baseline.input_cap,
            },
            "cells": {name: m.to_dict() for name, m in self.normalize_all().items()},
            "summary": self.get_summary(),
        }
