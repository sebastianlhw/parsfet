"""Liberty (.lib) file data models"""

from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field

from .common import OperatingCondition, UnitNormalizer, VtFlavor


class LookupTable(BaseModel):
    """Lookup table for timing/power data (1D or 2D)"""

    index_1: list[float] = Field(default_factory=list, description="Input slew values")
    index_2: list[float] = Field(default_factory=list, description="Output load values")
    values: list[Any] = Field(default_factory=list, description="1D or 2D table values")

    @property
    def is_2d(self) -> bool:
        """Check if this is a 2D table"""
        return bool(self.values and isinstance(self.values[0], list))

    @property
    def center_value(self) -> float:
        """Get center point of the LUT (typical operating point)"""
        if not self.values:
            return 0.0

        if self.is_2d:
            # 2D table
            mid_i = len(self.values) // 2
            row = self.values[mid_i]
            if isinstance(row, list) and row:
                mid_j = len(row) // 2
                return float(row[mid_j])
            return 0.0
        else:
            # 1D table
            mid = len(self.values) // 2
            val = self.values[mid]
            return float(val) if not isinstance(val, list) else 0.0

    def interpolate(self, slew: float, load: Optional[float] = None) -> float:
        """
        Interpolate the LUT at the given operating point.

        For 2D tables: bilinear interpolation on (slew, load)
        For 1D tables: linear interpolation on slew (load ignored)

        Out-of-range values are clamped to the nearest valid point.

        Args:
            slew: Input transition time (index_1)
            load: Output load capacitance (index_2), only for 2D tables

        Returns:
            Interpolated value at the operating point
        """
        if not self.values:
            return 0.0

        if self.is_2d:
            return self._interpolate_2d(slew, load if load is not None else 0.0)
        else:
            return self._interpolate_1d(slew)

    def _interpolate_1d(self, x: float) -> float:
        """Linear interpolation on 1D table with clamping"""
        if not self.index_1 or not self.values:
            return self.center_value

        # Clamp to valid range
        x = max(min(x, max(self.index_1)), min(self.index_1))

        # Find bracketing indices
        idx = self._find_bracket(self.index_1, x)
        if idx >= len(self.index_1) - 1:
            return float(self.values[-1])

        # Linear interpolation
        x0, x1 = self.index_1[idx], self.index_1[idx + 1]
        y0, y1 = float(self.values[idx]), float(self.values[idx + 1])

        if x1 == x0:
            return y0

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def _interpolate_2d(self, slew: float, load: float) -> float:
        """Bilinear interpolation on 2D table with clamping"""
        if not self.index_1 or not self.index_2 or not self.values:
            return self.center_value

        # Clamp to valid ranges
        slew = max(min(slew, max(self.index_1)), min(self.index_1))
        load = max(min(load, max(self.index_2)), min(self.index_2))

        # Find bracketing indices
        i = self._find_bracket(self.index_1, slew)
        j = self._find_bracket(self.index_2, load)

        # Clamp indices
        i = min(i, len(self.index_1) - 2)
        j = min(j, len(self.index_2) - 2)
        i = max(i, 0)
        j = max(j, 0)

        # Get corner values
        try:
            v00 = float(self.values[i][j])
            v01 = float(self.values[i][j + 1]) if j + 1 < len(self.values[i]) else v00
            v10 = float(self.values[i + 1][j]) if i + 1 < len(self.values) else v00
            v11 = (
                float(self.values[i + 1][j + 1])
                if (i + 1 < len(self.values) and j + 1 < len(self.values[i + 1]))
                else v00
            )
        except (IndexError, TypeError):
            return self.center_value

        # Compute interpolation weights
        x0, x1 = self.index_1[i], self.index_1[min(i + 1, len(self.index_1) - 1)]
        y0, y1 = self.index_2[j], self.index_2[min(j + 1, len(self.index_2) - 1)]

        tx = (slew - x0) / (x1 - x0) if x1 != x0 else 0.0
        ty = (load - y0) / (y1 - y0) if y1 != y0 else 0.0

        # Bilinear interpolation
        v0 = v00 + tx * (v10 - v00)
        v1 = v01 + tx * (v11 - v01)
        return v0 + ty * (v1 - v0)

    @staticmethod
    def _find_bracket(arr: list[float], val: float) -> int:
        """Find index i such that arr[i] <= val < arr[i+1]"""
        for i in range(len(arr) - 1):
            if arr[i] <= val < arr[i + 1]:
                return i
        return len(arr) - 2 if arr else 0

    def fit_linear_model(self, slew: float) -> tuple[float, float]:
        """
        Fit linear delay model: D = D₀ + k × Load

        At a fixed input slew, extracts the delay vs load relationship.

        Args:
            slew: Input slew to use for fitting

        Returns:
            (intrinsic_delay, load_slope) tuple where:
            - intrinsic_delay: D₀ (delay at zero load)
            - load_slope: k (delay increase per unit load)
        """
        if not self.is_2d or not self.index_2 or len(self.index_2) < 2:
            # 1D table or insufficient data
            return (self.center_value, 0.0)

        # Sample delay at multiple load points along the fixed slew
        loads = self.index_2
        delays = [self.interpolate(slew, load) for load in loads]

        # Simple linear regression: D = D₀ + k × Load
        # Using least squares: k = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
        n = len(loads)
        mean_load = sum(loads) / n
        mean_delay = sum(delays) / n

        numerator = sum((loads[i] - mean_load) * (delays[i] - mean_delay) for i in range(n))
        denominator = sum((loads[i] - mean_load) ** 2 for i in range(n))

        if denominator == 0:
            return (mean_delay, 0.0)

        slope = numerator / denominator  # k: delay per unit load
        intercept = mean_delay - slope * mean_load  # D₀: intrinsic delay

        return (max(intercept, 0.0), slope)  # Ensure non-negative intrinsic delay


class Pin(BaseModel):
    """Cell pin definition"""

    name: str
    direction: str = "input"  # "input", "output", "inout"
    capacitance: Optional[float] = Field(default=None, description="Input capacitance in pF")
    max_capacitance: Optional[float] = None
    min_capacitance: Optional[float] = None
    function: Optional[str] = Field(default=None, description="Boolean function expression")
    clock: bool = False

    # Power attributes
    rise_capacitance: Optional[float] = None
    fall_capacitance: Optional[float] = None

    model_config = {"extra": "allow"}


class TimingArc(BaseModel):
    """Timing arc from related_pin to output"""

    related_pin: str
    timing_sense: str = "positive_unate"  # positive_unate, negative_unate, non_unate
    timing_type: Optional[str] = None  # combinational, rising_edge, falling_edge, etc.

    # Delay tables
    cell_rise: Optional[LookupTable] = None
    cell_fall: Optional[LookupTable] = None

    # Transition tables
    rise_transition: Optional[LookupTable] = None
    fall_transition: Optional[LookupTable] = None

    # Constraint tables (for sequential cells)
    rise_constraint: Optional[LookupTable] = None
    fall_constraint: Optional[LookupTable] = None

    @property
    def representative_delay(self) -> float:
        """Get a representative delay value (average of rise/fall center points)"""
        delays = []
        if self.cell_rise:
            delays.append(self.cell_rise.center_value)
        if self.cell_fall:
            delays.append(self.cell_fall.center_value)
        return sum(delays) / len(delays) if delays else 0.0

    def delay_at(self, slew: float, load: float) -> float:
        """
        Get delay at specific operating point (slew, load).

        Returns average of rise and fall delays interpolated at the given point.
        """
        delays = []
        if self.cell_rise:
            delays.append(self.cell_rise.interpolate(slew, load))
        if self.cell_fall:
            delays.append(self.cell_fall.interpolate(slew, load))
        return sum(delays) / len(delays) if delays else 0.0

    def output_transition_at(self, slew: float, load: float) -> float:
        """
        Get output transition at specific operating point.

        Returns average of rise and fall transitions interpolated at the given point.
        """
        trans = []
        if self.rise_transition:
            trans.append(self.rise_transition.interpolate(slew, load))
        if self.fall_transition:
            trans.append(self.fall_transition.interpolate(slew, load))
        return sum(trans) / len(trans) if trans else slew  # Fallback to input slew

    def linear_delay_model(self, slew: float) -> tuple[float, float]:
        """
        Extract linear delay model: D = D₀ + k × Load

        Averages rise and fall linear models.

        Args:
            slew: Input slew for model extraction

        Returns:
            (intrinsic_delay, load_slope) tuple
        """
        models = []
        if self.cell_rise:
            models.append(self.cell_rise.fit_linear_model(slew))
        if self.cell_fall:
            models.append(self.cell_fall.fit_linear_model(slew))

        if not models:
            return (0.0, 0.0)

        avg_d0 = sum(m[0] for m in models) / len(models)
        avg_k = sum(m[1] for m in models) / len(models)
        return (avg_d0, avg_k)

    model_config = {"extra": "allow"}


class PowerArc(BaseModel):
    """Power consumption arc"""

    related_pin: Optional[str] = None
    when: Optional[str] = None  # Condition expression

    rise_power: Optional[LookupTable] = None
    fall_power: Optional[LookupTable] = None

    model_config = {"extra": "allow"}


class Cell(BaseModel):
    """Standard cell definition"""

    name: str
    area: float = Field(default=0.0, description="Cell area in um²")
    cell_leakage_power: Optional[float] = Field(default=None, description="Leakage power in nW")

    # Alternative leakage representations
    leakage_power_values: list[dict[str, Any]] = Field(
        default_factory=list, description="Conditional leakage values"
    )

    pins: dict[str, Pin] = Field(default_factory=dict)
    timing_arcs: list[TimingArc] = Field(default_factory=list)
    power_arcs: list[PowerArc] = Field(default_factory=list)

    # Cell attributes
    dont_use: bool = False
    dont_touch: bool = False
    is_sequential: bool = False

    # Raw attributes for completeness
    attributes: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def input_pins(self) -> list[str]:
        """List of input pin names"""
        return [name for name, pin in self.pins.items() if pin.direction == "input"]

    @computed_field
    @property
    def output_pins(self) -> list[str]:
        """List of output pin names"""
        return [name for name, pin in self.pins.items() if pin.direction == "output"]

    @property
    def total_input_capacitance(self) -> float:
        """Sum of all input pin capacitances"""
        return sum(pin.capacitance or 0.0 for pin in self.pins.values() if pin.direction == "input")

    @property
    def representative_delay(self) -> float:
        """Get representative delay across all timing arcs"""
        delays = [
            arc.representative_delay for arc in self.timing_arcs if arc.representative_delay > 0
        ]
        return sum(delays) / len(delays) if delays else 0.0

    def delay_at(self, slew: float, load: float) -> float:
        """Get average delay across all timing arcs at specific operating point"""
        delays = [arc.delay_at(slew, load) for arc in self.timing_arcs]
        delays = [d for d in delays if d > 0]
        return sum(delays) / len(delays) if delays else 0.0

    def output_transition_at(self, slew: float, load: float) -> float:
        """Get average output transition across all timing arcs at specific operating point"""
        trans = [arc.output_transition_at(slew, load) for arc in self.timing_arcs]
        trans = [t for t in trans if t > 0]
        return sum(trans) / len(trans) if trans else slew

    def linear_delay_model(self, slew: float) -> tuple[float, float]:
        """
        Extract linear delay model: D = D₀ + k × Load

        Averages linear models from all timing arcs.

        Args:
            slew: Input slew for model extraction

        Returns:
            (intrinsic_delay, load_slope) tuple where:
            - intrinsic_delay: D₀ (delay at zero load)
            - load_slope: k (delay increase per unit load)
        """
        models = [arc.linear_delay_model(slew) for arc in self.timing_arcs]
        models = [(d0, k) for d0, k in models if d0 > 0 or k > 0]

        if not models:
            return (0.0, 0.0)

        avg_d0 = sum(m[0] for m in models) / len(models)
        avg_k = sum(m[1] for m in models) / len(models)
        return (avg_d0, avg_k)

    model_config = {"extra": "allow"}


class LibertyLibrary(BaseModel):
    """Complete Liberty library"""

    name: str

    # Library-level attributes
    technology: Optional[str] = None
    delay_model: str = "table_lookup"

    # Units
    time_unit: str = "1ns"
    capacitive_load_unit: tuple[float, str] = (1.0, "pf")
    voltage_unit: str = "1V"
    current_unit: str = "1mA"
    leakage_power_unit: str = "1nW"
    pulling_resistance_unit: str = "1kohm"

    # Operating conditions
    nom_voltage: Optional[float] = None
    nom_temperature: Optional[float] = None
    nom_process: Optional[float] = None
    operating_conditions: Optional[OperatingCondition] = None

    # Inferred metadata
    vt_flavor: Optional[VtFlavor] = None
    process_node: Optional[str] = None  # e.g., "130nm", "7nm"
    foundry: Optional[str] = None

    # Lookup table templates
    lu_table_templates: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Cells
    cells: dict[str, Cell] = Field(default_factory=dict)

    # Raw attributes
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def unit_normalizer(self) -> UnitNormalizer:
        """
        Get a UnitNormalizer configured for this library's units.

        Use this to convert values to canonical units (ns, pF) for cross-library comparison.
        """
        return UnitNormalizer(time_unit=self.time_unit, cap_unit=self.capacitive_load_unit)

    @property
    def baseline_cell(self) -> Optional[Cell]:
        """
        Return INVD1 or equivalent baseline inverter.
        This is the fundamental reference cell for normalization.
        """
        # Try common inverter names in order of preference
        inv_names = [
            "INVD1",
            "INV_X1",
            "INVX1",
            "inv_1",
            "INVD1BWP",
            "sky130_fd_sc_hd__inv_1",
            "INV_X1_LVT",
            "INV_X1_HVT",
            "INVD0",
            "INV_X0P5",  # Smaller variants
        ]

        for name in inv_names:
            if name in self.cells:
                return self.cells[name]

        # Fallback: find smallest INV by name pattern (fast path)
        inverters = [
            (name, cell)
            for name, cell in self.cells.items()
            if "INV" in name.upper() and cell.area > 0
        ]
        if inverters:
            return min(inverters, key=lambda x: x[1].area)[1]

        # Final fallback: use classifier to find inverters by logic function
        try:
            from ..normalizers.classifier import classify_cell
            classified_inverters = [
                cell for cell in self.cells.values()
                if classify_cell(cell) == "inverter" and cell.area > 0
            ]
            if classified_inverters:
                return min(classified_inverters, key=lambda c: c.area)
        except ImportError:
            pass  # Classifier not available

        return None

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def get_cells_by_function(self, pattern: str) -> list[Cell]:
        """Get cells matching a function pattern (e.g., 'NAND', 'NOR', 'DFF')"""
        pattern_upper = pattern.upper()
        return [cell for name, cell in self.cells.items() if pattern_upper in name.upper()]

    @property
    def fo4_operating_point(self) -> tuple[float, float]:
        """
        Compute FO4 (fanout-of-4) operating point for this library.

        Returns:
            (typical_slew, typical_load) tuple where:
            - typical_load = 4× baseline inverter input capacitance
            - typical_slew = baseline inverter output transition at FO4 load

        This provides a self-consistent, physically meaningful operating point.
        """
        baseline = self.baseline_cell
        if not baseline:
            # Fallback to center of first timing arc's LUT indices
            return self._fallback_operating_point()

        # FO4 load = 4× inverter input capacitance
        inv_cap = baseline.total_input_capacitance
        if inv_cap <= 0:
            inv_cap = 0.001  # Default 1fF
        fo4_load = 4.0 * inv_cap

        # Get baseline's output transition at FO4 load
        # Use median slew from the LUT as initial estimate
        initial_slew = self._get_median_slew(baseline)

        # Get output slew at (initial_slew, fo4_load)
        # This makes the operating point self-consistent
        typical_slew = baseline.output_transition_at(initial_slew, fo4_load)
        if typical_slew <= 0:
            typical_slew = initial_slew

        return (typical_slew, fo4_load)

    def _get_median_slew(self, cell: Cell) -> float:
        """Get median slew value from cell's timing arc LUTs"""
        slews = []
        for arc in cell.timing_arcs:
            if arc.cell_rise and arc.cell_rise.index_1:
                slews.extend(arc.cell_rise.index_1)
            if arc.cell_fall and arc.cell_fall.index_1:
                slews.extend(arc.cell_fall.index_1)

        if slews:
            slews = sorted(set(slews))
            return slews[len(slews) // 2]
        return 0.01  # Default 10ps

    def _fallback_operating_point(self) -> tuple[float, float]:
        """Fallback operating point when no baseline cell exists"""
        # Find any timing arc and use its center
        for cell in self.cells.values():
            for arc in cell.timing_arcs:
                if arc.cell_rise and arc.cell_rise.index_1 and arc.cell_rise.index_2:
                    idx1 = arc.cell_rise.index_1
                    idx2 = arc.cell_rise.index_2
                    return (idx1[len(idx1) // 2], idx2[len(idx2) // 2])
        return (0.01, 0.001)  # Fallback: 10ps slew, 1fF load

    model_config = {"extra": "allow"}
