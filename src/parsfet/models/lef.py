"""LEF and TechLEF data models"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    """LEF layer types"""

    ROUTING = "routing"
    CUT = "cut"
    MASTERSLICE = "masterslice"
    OVERLAP = "overlap"
    IMPLANT = "implant"


class LayerDirection(str, Enum):
    """Preferred routing direction"""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class MetalLayer(BaseModel):
    """Metal or via layer definition"""

    name: str
    layer_type: LayerType = LayerType.ROUTING

    # Routing properties
    direction: Optional[LayerDirection] = None
    pitch: Optional[float] = Field(default=None, description="Routing pitch in um")
    width: Optional[float] = Field(default=None, description="Default width in um")
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    spacing: Optional[float] = Field(default=None, description="Minimum spacing in um")

    # Electrical properties
    resistance: Optional[float] = Field(default=None, description="Sheet resistance in ohm/sq")
    capacitance: Optional[float] = Field(default=None, description="Capacitance in pF/um")
    edge_capacitance: Optional[float] = None

    # Physical properties
    thickness: Optional[float] = Field(default=None, description="Layer thickness in um")
    height: Optional[float] = Field(default=None, description="Height above substrate in um")

    # Additional attributes
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Via(BaseModel):
    """Via definition connecting two layers"""

    name: str
    layers: list[str] = Field(default_factory=list, description="[bottom, cut, top] layers")

    # Geometry
    width: Optional[float] = None
    height: Optional[float] = None

    # Electrical
    resistance: Optional[float] = Field(default=None, description="Via resistance in ohm")

    # Enclosure rules
    enclosure: dict[str, float] = Field(default_factory=dict)

    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Site(BaseModel):
    """Placement site definition"""

    name: str
    class_type: str = "core"  # "core", "pad", "io"

    # Dimensions
    width: float = Field(description="Site width in um")
    height: float = Field(description="Site height in um")

    symmetry: list[str] = Field(default_factory=list, description="Allowed symmetries: X, Y, R90")

    model_config = {"extra": "allow"}


class Rect(BaseModel):
    """Rectangle geometry"""

    layer: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class MacroPin(BaseModel):
    """Pin definition within a macro"""

    name: str
    direction: str = "input"  # input, output, inout
    use: Optional[str] = None  # signal, power, ground, clock
    shape: Optional[str] = None  # abutment, feedthru

    # Port geometry
    ports: list[Rect] = Field(default_factory=list)

    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Macro(BaseModel):
    """Cell physical (macro) definition"""

    name: str
    class_type: str = "core"  # core, block, pad, endcap, cover

    # Origin and size
    origin: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = Field(description="(width, height) in um")

    # Placement info
    symmetry: list[str] = Field(default_factory=list)
    site: Optional[str] = None

    # Pins
    pins: dict[str, MacroPin] = Field(default_factory=dict)

    # Obstructions
    obstructions: list[Rect] = Field(default_factory=list)

    # Foreign reference
    foreign: Optional[str] = None

    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def height(self) -> float:
        return self.size[1]

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]

    model_config = {"extra": "allow"}


class LEFLibrary(BaseModel):
    """Complete LEF library (physical cell definitions)"""

    version: Optional[str] = None
    bus_bit_chars: str = "[]"
    divider_char: str = "/"

    # Units
    units_database: int = Field(default=1000, description="Database units per micron")
    manufacturing_grid: Optional[float] = None

    # Definitions
    sites: dict[str, Site] = Field(default_factory=dict)
    layers: dict[str, MetalLayer] = Field(default_factory=dict)
    vias: dict[str, Via] = Field(default_factory=dict)
    macros: dict[str, Macro] = Field(default_factory=dict)

    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def routing_layers(self) -> list[MetalLayer]:
        """Get only routing metal layers (ordered by layer name)"""
        routing = [l for l in self.layers.values() if l.layer_type == LayerType.ROUTING]
        return sorted(routing, key=lambda x: x.name)

    @property
    def cut_layers(self) -> list[MetalLayer]:
        """Get only via/cut layers"""
        return [l for l in self.layers.values() if l.layer_type == LayerType.CUT]

    model_config = {"extra": "allow"}


class TechLEF(BaseModel):
    """
    Technology LEF - contains only layer/via definitions, no macros.
    Typically provided by foundry.
    """

    version: Optional[str] = None

    # Units
    units_database: int = 1000
    manufacturing_grid: Optional[float] = None

    # Technology definitions only
    layers: dict[str, MetalLayer] = Field(default_factory=dict)
    vias: dict[str, Via] = Field(default_factory=dict)
    sites: dict[str, Site] = Field(default_factory=dict)

    # Process info
    process_node: Optional[str] = None
    foundry: Optional[str] = None

    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def metal_stack_height(self) -> int:
        """Count of routing metal layers"""
        return len([l for l in self.layers.values() if l.layer_type == LayerType.ROUTING])

    model_config = {"extra": "allow"}
