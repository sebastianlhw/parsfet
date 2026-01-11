"""Simplified physical data structures for LEF/TechLEF integration.

This module provides lightweight dataclasses that summarize the key physical
information from LEF and TechLEF files for easy integration with Liberty data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lef import Macro, MacroPin, MetalLayer, TechLEF


@dataclass
class PinPhysical:
    """Physical pin info from cell LEF.

    Attributes:
        name: Pin name (e.g., "A", "Y", "CLK").
        direction: Signal direction ("input", "output", "inout").
        layers: Layers used by pin ports (e.g., ["M1", "M2"]).
        use: Pin use type ("signal", "power", "ground", "clock", etc.).
    """

    name: str
    direction: str  # "input", "output", "inout"
    layers: list[str] = field(default_factory=list)
    use: str | None = None  # "signal", "power", "ground", "clock"

    @classmethod
    def from_macro_pin(cls, pin: MacroPin) -> PinPhysical:
        """Create from a MacroPin model."""
        return cls(
            name=pin.name,
            direction=pin.direction,
            layers=pin.layers_used,
            use=pin.use,
        )


@dataclass
class CellPhysical:
    """Physical cell info from LEF macro.

    Attributes:
        name: Cell name (e.g., "INV_X1", "NAND2_X2").
        width: Cell width in micrometers.
        height: Cell height in micrometers.
        area: Cell area in square micrometers.
        pins: Dictionary of pin name to PinPhysical.
    """

    name: str
    width: float
    height: float
    area: float
    pins: dict[str, PinPhysical] = field(default_factory=dict)

    @classmethod
    def from_macro(cls, macro: Macro) -> CellPhysical:
        """Create from a LEF Macro model."""
        pins = {
            name: PinPhysical.from_macro_pin(pin)
            for name, pin in macro.pins.items()
        }
        return cls(
            name=macro.name,
            width=macro.width,
            height=macro.height,
            area=macro.area,
            pins=pins,
        )


@dataclass
class LayerInfo:
    """Technology layer info from TechLEF.

    Attributes:
        name: Layer name (e.g., "M1", "V1").
        layer_type: Layer type ("routing", "cut", "masterslice", etc.).
        min_size: Minimum dimension (min_width or width fallback).
        direction: Preferred routing direction ("horizontal", "vertical").
        pitch: Routing pitch in micrometers.
        spacing: Minimum spacing in micrometers.
    """

    name: str
    layer_type: str
    min_size: float | None = None
    direction: str | None = None
    pitch: float | None = None
    spacing: float | None = None

    @classmethod
    def from_metal_layer(cls, layer: MetalLayer) -> LayerInfo:
        """Create from a MetalLayer model."""
        return cls(
            name=layer.name,
            layer_type=layer.layer_type.value if layer.layer_type else "unknown",
            min_size=layer.min_size,
            direction=layer.direction.value if layer.direction else None,
            pitch=layer.pitch,
            spacing=layer.spacing,
        )


@dataclass
class TechInfo:
    """Summary of technology from TechLEF.

    Attributes:
        layers: Dictionary of layer name to LayerInfo.
        units_database: Database units per micron.
        manufacturing_grid: Manufacturing grid resolution.
        metal_stack_height: Number of routing metal layers.
    """

    layers: dict[str, LayerInfo] = field(default_factory=dict)
    units_database: int = 1000
    manufacturing_grid: float | None = None
    metal_stack_height: int = 0

    @classmethod
    def from_tech_lef(cls, tech_lef: TechLEF) -> TechInfo:
        """Create from a TechLEF model."""
        layers = {
            name: LayerInfo.from_metal_layer(layer)
            for name, layer in tech_lef.layers.items()
        }
        return cls(
            layers=layers,
            units_database=tech_lef.units_database,
            manufacturing_grid=tech_lef.manufacturing_grid,
            metal_stack_height=tech_lef.metal_stack_height,
        )
