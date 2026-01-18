"""Liberty (.lib) file data models.

This module defines the Pydantic models for the Liberty format, covering libraries,
cells, pins, timing arcs, power arcs, and lookup tables. It includes logic for
interpolation, delay calculation, and linear model fitting.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field

from .common import OperatingCondition, UnitNormalizer, VtFlavor


class LookupTable(BaseModel):
    """Represents a 1D or 2D lookup table for timing or power data.

    Liberty uses Non-Linear Delay Models (NLDM) stored as tables indexed by input slew
    and/or output load.

    Attributes:
        index_1: List of index values for the first dimension (typically input slew).
        index_2: List of index values for the second dimension (typically output load).
        values: The table values. Can be a 1D list or a 2D list (list of lists).
    """

    index_1: list[float] = Field(default_factory=list, description="Input slew values")
    index_2: list[float] = Field(default_factory=list, description="Output load values")
    values: list[Any] = Field(default_factory=list, description="1D or 2D table values")

    @property
    def is_2d(self) -> bool:
        """Returns True if the table is 2D, False otherwise."""
        return bool(self.values and isinstance(self.values[0], list))

    @property
    def center_value(self) -> float:
        """Returns the value at the center of the table (typical operating point)."""
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
        """Interpolates the table at the given operating point.

        Uses bilinear interpolation for 2D tables and linear interpolation for 1D tables.
        Values outside the table indices are clamped to the nearest edge.

        Args:
            slew: The input transition time (index_1).
            load: The output load capacitance (index_2). Required for 2D tables.

        Returns:
            The interpolated value.
        """
        if not self.values:
            return 0.0

        if self.is_2d:
            return self._interpolate_2d(slew, load if load is not None else 0.0)
        else:
            return self._interpolate_1d(slew)

    def _interpolate_1d(self, x: float) -> float:
        """Performs linear interpolation on a 1D table with clamping."""
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
        """Performs bilinear interpolation on a 2D table with clamping."""
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
        """Finds index i such that arr[i] <= val < arr[i+1]."""
        for i in range(len(arr) - 1):
            if arr[i] <= val < arr[i + 1]:
                return i
        return len(arr) - 2 if arr else 0

    def fit_linear_model(self, slew: float) -> tuple[float, float, float]:
        """Fits a linear delay model: D = D0 + k * Load.

        At a fixed input slew, this method extracts the delay vs. load relationship
        using linear regression on the table data.

        Args:
            slew: The input slew value to use for fitting.

        Returns:
            A tuple (intrinsic_delay, load_slope, r_squared), where:
            - intrinsic_delay (D0): The delay at zero load.
            - load_slope (k): The delay increase per unit load (logical effort factor).
            - r_squared: Coefficient of determination (fit quality, 1.0 = perfect).
        """
        if not self.is_2d or not self.index_2 or len(self.index_2) < 2:
            # 1D table or insufficient data
            return (self.center_value, 0.0, 1.0)

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
            return (mean_delay, 0.0, 1.0)

        slope = numerator / denominator  # k: delay per unit load
        intercept = mean_delay - slope * mean_load  # D₀: intrinsic delay

        # Compute R² (coefficient of determination)
        d0 = max(intercept, 0.0)
        ss_res = sum((delays[i] - (d0 + slope * loads[i])) ** 2 for i in range(n))
        ss_tot = sum((delays[i] - mean_delay) ** 2 for i in range(n))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return (d0, slope, r_squared)  # Ensure non-negative intrinsic delay


class Pin(BaseModel):
    """Represents a pin definition for a cell.

    Attributes:
        name: The pin name.
        direction: Direction of signal flow (input, output, inout).
        capacitance: Input capacitance in pF.
        max_capacitance: Maximum load capacitance allowed.
        min_capacitance: Minimum load capacitance.
        function: Boolean function expression (for output pins).
        clock: True if the pin is a clock pin.
        rise_capacitance: Capacitance for rising transitions.
        fall_capacitance: Capacitance for falling transitions.
    """

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
    """Represents a timing arc from a related pin to an output pin.

    Defines the timing relationship (delay, transition time) between an input
    transition and the resulting output transition.

    Attributes:
        related_pin: The name of the input pin causing the transition.
        timing_sense: The logic sense (positive_unate, negative_unate, non_unate).
        timing_type: The type of timing check (combinational, setup, hold, etc.).
        cell_rise: Lookup table for propagation delay (output rising).
        cell_fall: Lookup table for propagation delay (output falling).
        rise_transition: Lookup table for output transition time (rising).
        fall_transition: Lookup table for output transition time (falling).
        rise_constraint: Lookup table for constraint time (rising).
        fall_constraint: Lookup table for constraint time (falling).
    """

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
        """Calculates a single representative delay value for the arc.

        Useful for high-level sorting or comparison. It averages the center values
        of the rise and fall delay tables.
        """
        delays = []
        if self.cell_rise:
            delays.append(self.cell_rise.center_value)
        if self.cell_fall:
            delays.append(self.cell_fall.center_value)
        return sum(delays) / len(delays) if delays else 0.0

    def delay_at(self, slew: float, load: float) -> float:
        """Calculates the delay at a specific operating point (slew, load).

        Returns the average of rise and fall delays interpolated at the given point.
        """
        delays = []
        if self.cell_rise:
            delays.append(self.cell_rise.interpolate(slew, load))
        if self.cell_fall:
            delays.append(self.cell_fall.interpolate(slew, load))
        return sum(delays) / len(delays) if delays else 0.0

    def output_transition_at(self, slew: float, load: float) -> float:
        """Calculates the output transition time at a specific operating point.

        Returns the average of rise and fall output transitions.
        """
        trans = []
        if self.rise_transition:
            trans.append(self.rise_transition.interpolate(slew, load))
        if self.fall_transition:
            trans.append(self.fall_transition.interpolate(slew, load))
        return sum(trans) / len(trans) if trans else slew  # Fallback to input slew

    def linear_delay_model(self, slew: float) -> tuple[float, float, float]:
        """Extracts a linear delay model: D = D0 + k * Load.

        Averages the linear models derived from the cell_rise and cell_fall tables.

        Args:
            slew: The input slew value for model extraction.

        Returns:
            A tuple (intrinsic_delay, load_slope, r_squared).
        """
        models = []
        if self.cell_rise:
            models.append(self.cell_rise.fit_linear_model(slew))
        if self.cell_fall:
            models.append(self.cell_fall.fit_linear_model(slew))

        if not models:
            return (0.0, 0.0, 1.0)

        avg_d0 = sum(m[0] for m in models) / len(models)
        avg_k = sum(m[1] for m in models) / len(models)
        avg_r2 = sum(m[2] for m in models) / len(models)
        return (avg_d0, avg_k, avg_r2)

    model_config = {"extra": "allow"}


class PowerArc(BaseModel):
    """Represents a power consumption arc.

    Attributes:
        related_pin: The pin related to the power event.
        when: Condition expression for when this power applies.
        rise_power: Lookup table for power during rising transition.
        fall_power: Lookup table for power during falling transition.
    """

    related_pin: Optional[str] = None
    when: Optional[str] = None  # Condition expression

    rise_power: Optional[LookupTable] = None
    fall_power: Optional[LookupTable] = None

    model_config = {"extra": "allow"}


class Cell(BaseModel):
    """Represents a standard cell definition.

    Attributes:
        name: The name of the cell (e.g., "NAND2_X1").
        area: The area of the cell in square micrometers.
        cell_leakage_power: The leakage power in nanowatts.
        leakage_power_values: List of conditional leakage power values.
        pins: Dictionary of pins keyed by name.
        timing_arcs: List of timing arcs.
        power_arcs: List of power arcs.
        dont_use: Flag indicating the cell should not be used for synthesis.
        dont_touch: Flag indicating the cell should not be modified.
        is_sequential: True if the cell contains state elements (FF, Latch).
        attributes: Additional unparsed attributes.
    """

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
        """Returns a list of names of all input pins."""
        return [name for name, pin in self.pins.items() if pin.direction == "input"]

    @computed_field
    @property
    def output_pins(self) -> list[str]:
        """Returns a list of names of all output pins."""
        return [name for name, pin in self.pins.items() if pin.direction == "output"]

    @property
    def total_input_capacitance(self) -> float:
        """Sum of all input pin capacitances."""
        return sum(pin.capacitance or 0.0 for pin in self.pins.values() if pin.direction == "input")

    @property
    def representative_delay(self) -> float:
        """Calculates a representative delay across all timing arcs.

        Useful for sorting or rough comparisons.
        """
        delays = [
            arc.representative_delay for arc in self.timing_arcs if arc.representative_delay > 0
        ]
        return sum(delays) / len(delays) if delays else 0.0

    def delay_at(self, slew: float, load: float) -> float:
        """Calculates average delay across all timing arcs at specific operating point."""
        delays = [arc.delay_at(slew, load) for arc in self.timing_arcs]
        delays = [d for d in delays if d > 0]
        return sum(delays) / len(delays) if delays else 0.0

    def output_transition_at(self, slew: float, load: float) -> float:
        """Calculates average output transition across all arcs at specific operating point."""
        trans = [arc.output_transition_at(slew, load) for arc in self.timing_arcs]
        trans = [t for t in trans if t > 0]
        return sum(trans) / len(trans) if trans else slew

    def linear_delay_model(self, slew: float) -> tuple[float, float, float]:
        """Extracts a linear delay model from the slowest timing arc.

        Uses the slowest arc (max delay) for conservative estimation.

        Args:
            slew: Input slew for model extraction.

        Returns:
            A tuple (intrinsic_delay, load_slope, r_squared).
        """
        models = [arc.linear_delay_model(slew) for arc in self.timing_arcs]
        models = [(d0, k, r2) for d0, k, r2 in models if d0 > 0 or k > 0]

        if not models:
            return (0.0, 0.0, 1.0)

        # Use slowest arc (max D₀) for conservative estimate
        slowest = max(models, key=lambda m: m[0])
        return slowest

    model_config = {"extra": "allow"}


class LibertyLibrary(BaseModel):
    """Represents a complete Liberty library.

    Contains library-level attributes, operating conditions, lookup table templates,
    and the collection of cells.

    Attributes:
        name: Library name.
        technology: Technology name (e.g., "cmos").
        delay_model: Delay model used (e.g., "table_lookup").
        time_unit: Time unit string (e.g., "1ns").
        capacitive_load_unit: Capacitance unit tuple (multiplier, unit).
        nom_voltage: Nominal voltage.
        nom_temperature: Nominal temperature.
        nom_process: Nominal process scaling.
        operating_conditions: Defined operating conditions.
        vt_flavor: Threshold voltage flavor (SVT, LVT, etc.).
        process_node: Process node string.
        foundry: Foundry name.
        lu_table_templates: Dictionary of LUT templates.
        cells: Dictionary of cells keyed by name.
        attributes: Additional unparsed attributes.
    """

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
        """Returns a UnitNormalizer configured for this library's units.

        Use this to convert values to canonical units (ns, pF) for cross-library comparison.
        """
        return UnitNormalizer(time_unit=self.time_unit, cap_unit=self.capacitive_load_unit)

    @property
    def baseline_cell(self) -> Optional[Cell]:
        """Identifies and returns the baseline inverter (INVD1) for normalization.

        The baseline inverter is the fundamental reference for logical effort calculations.
        It searches for standard names (INVD1, INV_X1) or infers based on area/function.

        Returns:
            The Cell object for the baseline inverter, or None if not found.
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
            from ..normalizers.classifier import classify_cell, CellType

            classified_inverters = [
                cell
                for cell in self.cells.values()
                if classify_cell(cell) == CellType.INVERTER and cell.area > 0
            ]
            if classified_inverters:
                return min(classified_inverters, key=lambda c: c.area)
        except ImportError:
            pass  # Classifier not available

        return None

    @property
    def cell_count(self) -> int:
        """Returns the total number of cells in the library."""
        return len(self.cells)

    def get_cells_by_function(self, pattern: str) -> list[Cell]:
        """Returns a list of cells whose names match a pattern (e.g., 'NAND')."""
        pattern_upper = pattern.upper()
        return [cell for name, cell in self.cells.items() if pattern_upper in name.upper()]

    def fo4_operating_point(
        self,
        tolerance: Optional[float] = None,
        max_iterations: int = 10,
        initial_slew_guess: Optional[float] = None,
        dampening_factor: float = 0.714,
        trace: bool = False,
    ) -> tuple[float, float]:
        """Computes the FO4 (fanout-of-4) operating point for this library.

        Uses iterative fixed-point convergence to find the "natural" slew where
        input slew equals output slew (the eigenmode of an inverter chain).

        Args:
            tolerance: Convergence tolerance. If None, uses 1% of min characterized slew.
            max_iterations: Maximum iterations before returning (default 10).
            initial_slew_guess: Starting slew. If None, uses median from LUT.
            dampening_factor: Relaxation factor (0-1). Default 0.714 for stability.
            trace: If True, prints convergence trace for debugging.

        Returns:
            A tuple (typical_slew, typical_load) where:
            - typical_load = 4 * baseline inverter input capacitance
            - typical_slew = consistent slew (input = output) at FO4 load
        """
        baseline = self.baseline_cell
        if not baseline:
            return self._fallback_operating_point()

        # FO4 load = 4× inverter input capacitance
        inv_cap = baseline.total_input_capacitance
        if inv_cap <= 0:
            inv_cap = 0.001  # Default 1fF
        fo4_load = 4.0 * inv_cap

        # Determine start point (Smart Start)
        if initial_slew_guess is not None:
            start_slew = initial_slew_guess
        else:
            start_slew = self._get_median_slew(baseline)

        # Determine dynamic tolerance if not specified
        if tolerance is None:
            min_slew = self._get_min_slew(baseline)
            tolerance = max(min_slew * 0.01, 1e-4)  # 1% of min slew or 0.1ps

        typical_slew = self._find_consistent_slew(
            baseline, fo4_load, start_slew, tolerance, max_iterations, dampening_factor, trace
        )

        return (typical_slew, fo4_load)

    def _find_consistent_slew(
        self,
        cell: Cell,
        load: float,
        start_slew: float,
        tolerance: float,
        max_iter: int = 10,
        alpha: float = 0.714,
        trace: bool = False,
    ) -> float:
        """Finds the consistent slew using damped fixed-point iteration.

        Iterates until input_slew ≈ output_slew (within tolerance).

        Args:
            cell: The baseline cell to use for transition lookup.
            load: The FO4 load capacitance.
            start_slew: Initial slew guess.
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations.
            alpha: Dampening factor (0-1). Higher = faster but less stable.
            trace: If True, prints each iteration for debugging.

        Returns:
            The converged slew value.
        """
        slew = start_slew
        for i in range(max_iter):
            target_slew = cell.output_transition_at(slew, load)

            # Apply Dampening (Successive Over-Relaxation)
            next_slew = (1 - alpha) * slew + alpha * target_slew

            delta = abs(next_slew - slew)
            if trace:
                print(f"  iter {i}: slew={slew:.6f} -> target={target_slew:.6f} -> next={next_slew:.6f} (delta={delta:.6f})")

            if delta < tolerance:
                if trace:
                    print(f"  Converged at iteration {i} with delta={delta:.6f} < tol={tolerance:.6f}")
                return next_slew

            slew = next_slew

        if trace:
            print(f"  Did not converge after {max_iter} iterations, returning {slew:.6f}")
        return slew

    def _get_median_slew(self, cell: Cell) -> float:
        """Get median slew value from cell's timing arc LUTs."""
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

    def _get_min_slew(self, cell: Cell) -> float:
        """Get minimum slew value from cell's timing arc LUTs for tolerance calculation."""
        slews = []
        for arc in cell.timing_arcs:
            if arc.cell_rise and arc.cell_rise.index_1:
                slews.extend(arc.cell_rise.index_1)
            if arc.cell_fall and arc.cell_fall.index_1:
                slews.extend(arc.cell_fall.index_1)

        if slews:
            return min(slews)
        return 0.001  # Default 1ps

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
