"""Pydantic models for Pars-FET JSON export format.

These models define the schema for exported JSON files, enabling:
- Validation when loading previously-exported data
- Type safety when working with exported structures
- Re-importing exports for combine operations
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExportedRawMetrics(BaseModel):
    """Raw cell metrics in canonical units."""

    area_um2: float = Field(default=0.0, description="Cell area in um²")
    d0_ns: float = Field(default=0.0, description="Intrinsic delay in ns")
    k_ns_per_pf: float = Field(default=0.0, description="Load slope in ns/pF")
    leakage: float = Field(default=0.0, description="Leakage power")
    input_cap_pf: float = Field(default=0.0, description="Input capacitance in pF")
    e0_unit: float = Field(default=0.0, description="Intrinsic energy (unit)")
    k_unit_per_pf: float = Field(default=0.0, description="Switching energy slope (unit/pF)")


class ExportedDelayModel(BaseModel):
    """Linear delay model parameters."""

    d0_ns: float = Field(default=0.0, description="Intrinsic delay in ns")
    k_ns_per_pf: float = Field(default=0.0, description="Load slope in ns/pF")


class ExportedPowerModel(BaseModel):
    """Linear power (energy) model parameters derived from rise_power/fall_power tables.

    Energy = internal_energy + switching_energy_slope * Load
    """

    e0_unit: float = Field(
        default=0.0, description="Intrinsic energy (E0) - 'Internal Power' component"
    )
    k_unit_per_pf: float = Field(
        default=0.0,
        description="Energy slope per load (k) - 'Switching Power' component",
    )


class ExportedFitQuality(BaseModel):
    """Fit quality metrics for the linear model."""

    r_squared: float = Field(default=1.0, description="R² of linear fit")
    fo4_residual_pct: float = Field(default=0.0, description="Residual at FO4 point")


class ExportedCell(BaseModel):
    """Exported cell with normalized and raw metrics."""

    cell_name: str
    cell_type: str = "unknown"

    # Normalized ratios
    area_ratio: float = 1.0
    d0_ratio: float = 1.0
    k_ratio: float = 1.0
    leakage_ratio: float = 1.0
    input_cap_ratio: float = 1.0

    # Additional metrics
    drive_strength: float = 1.0
    num_inputs: int = 1
    num_outputs: int = 1
    is_sequential: bool = False

    # Nested structures
    delay_model: Optional[ExportedDelayModel] = None
    delay_fit_quality: Optional[ExportedFitQuality] = None
    power_model: Optional[ExportedPowerModel] = None
    power_fit_quality: Optional[ExportedFitQuality] = None
    raw: Optional[ExportedRawMetrics] = None

    # Physical data (from LEF, if present)
    physical: Optional[dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ExportedBaseline(BaseModel):
    """Baseline cell metrics."""

    cell: str = Field(..., description="Baseline cell name")
    area_um2: float = Field(default=0.0, description="Baseline area in um²")
    d0_ns: float = Field(default=0.0, description="Baseline D0 in ns")
    k_ns_per_pf: float = Field(default=0.0, description="Baseline k in ns/pF")
    leakage: float = Field(default=0.0, description="Baseline leakage power")
    input_cap_pf: float = Field(default=0.0, description="Baseline input cap in pF")
    e0_unit: float = Field(default=0.0, description="Baseline E0 (unit)")
    k_unit_per_pf: float = Field(default=0.0, description="Baseline k (unit/pF)")


class ExportedFO4(BaseModel):
    """FO4 operating point."""

    slew_ns: float = 0.0
    load_pf: float = 0.0
    description: str = ""


class ExportedUnits(BaseModel):
    """Unit information for the export."""

    time: str = "ns"
    capacitance: str = "pF"
    area: str = "um^2"
    source_time_unit: Optional[str] = None
    source_cap_unit: Optional[str] = None


class ExportedTechnology(BaseModel):
    """Technology information from TechLEF."""

    metal_stack_height: Optional[int] = None
    units_database: Optional[float] = None
    manufacturing_grid: Optional[float] = None
    layers: dict[str, Any] = Field(default_factory=dict)


class ExportedLibrary(BaseModel):
    """Complete exported library structure.

    This is the top-level model for a Pars-FET JSON export file.
    """

    library: str = Field(..., description="Library name")
    units: Optional[ExportedUnits] = None
    fo4_operating_point: Optional[ExportedFO4] = None
    baseline: ExportedBaseline
    cells: dict[str, ExportedCell] = Field(default_factory=dict)
    summary: Optional[dict[str, Any]] = None
    technology: Optional[ExportedTechnology] = None

    model_config = {"extra": "allow"}

    @classmethod
    def from_json_file(cls, path: str) -> "ExportedLibrary":
        """Load and validate an exported JSON file."""
        import json
        from pathlib import Path

        with open(Path(path), "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_raw_cells(self) -> dict[str, ExportedRawMetrics]:
        """Extract raw metrics for all cells.

        Returns:
            Dict mapping cell name to ExportedRawMetrics.
        """
        result = {}
        for name, cell in self.cells.items():
            if cell.raw:
                result[name] = cell.raw
            else:
                # Fall back to delay_model if raw not present
                result[name] = ExportedRawMetrics(
                    area_um2=cell.area_ratio * self.baseline.area_um2,
                    d0_ns=cell.delay_model.d0_ns if cell.delay_model else 0.0,
                    k_ns_per_pf=cell.delay_model.k_ns_per_pf if cell.delay_model else 0.0,
                    leakage=cell.leakage_ratio * self.baseline.leakage,
                    input_cap_pf=cell.input_cap_ratio * self.baseline.input_cap_pf,
                )
        return result
