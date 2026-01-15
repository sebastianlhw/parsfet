"""LEF (Library Exchange Format) and TechLEF data models.

This module defines Pydantic models for representing the physical design data found
in LEF files. This includes technology definitions (layers, vias, sites) and
macro definitions (cell geometry, pins, obstructions).

These models support both standard LEF (macro definitions) and TechLEF (technology
rules provided by foundries).
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class LayerType(str, Enum):
    """Enumeration of LEF layer types."""

    ROUTING = "routing"
    CUT = "cut"
    MASTERSLICE = "masterslice"
    OVERLAP = "overlap"
    IMPLANT = "implant"


class LayerDirection(str, Enum):
    """Enumeration of preferred routing directions for metal layers."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class MetalLayer(BaseModel):
    """Represents a metal or via layer definition in LEF.

    Attributes:
        name: Layer name (e.g., "M1", "M2").
        layer_type: Type of the layer (routing, cut, etc.).
        direction: Preferred routing direction (horizontal/vertical).
        pitch: Routing pitch in micrometers.
        width: Default wire width in micrometers.
        min_width: Minimum width constraint.
        max_width: Maximum width constraint.
        spacing: Minimum spacing constraint.
        resistance: Sheet resistance in Ohm/square.
        capacitance: Area capacitance in pF/um^2.
        edge_capacitance: Edge capacitance in pF/um.
        thickness: Physical thickness of the layer.
        height: Height above substrate.
        attributes: Additional unparsed attributes.
    """

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

    @property
    def min_size(self) -> float | None:
        """Returns minimum dimension (min_width or width as fallback)."""
        return self.min_width if self.min_width is not None else self.width


class Via(BaseModel):
    """Represents a via definition connecting two layers.

    Attributes:
        name: Via name.
        layers: List of layers involved [bottom, cut, top].
        width: Via width.
        height: Via height.
        resistance: Via resistance in Ohms.
        enclosure: Dictionary of enclosure rules per layer.
        attributes: Additional unparsed attributes.
    """

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
    """Represents a placement site definition.

    Sites define the grid where standard cells can be placed.

    Attributes:
        name: Site name.
        class_type: Class of the site (e.g., "core", "pad").
        width: Width of the site in micrometers.
        height: Height of the site in micrometers.
        symmetry: List of allowed symmetries (X, Y, R90).
    """

    name: str
    class_type: str = "core"  # "core", "pad", "io"

    # Dimensions
    width: float = Field(description="Site width in um")
    height: float = Field(description="Site height in um")

    symmetry: list[str] = Field(default_factory=list, description="Allowed symmetries: X, Y, R90")

    model_config = {"extra": "allow"}


class Rect(BaseModel):
    """Represents a rectangular geometry.

    Attributes:
        layer: Layer name associated with the rectangle.
        x1, y1: Coordinates of the bottom-left corner.
        x2, y2: Coordinates of the top-right corner.
    """

    layer: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculates the width of the rectangle."""
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Calculates the height of the rectangle."""
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Calculates the area of the rectangle."""
        return self.width * self.height


class MacroPin(BaseModel):
    """Represents a pin definition within a macro (cell).

    Attributes:
        name: Pin name.
        direction: Signal direction (input, output, inout).
        use: Usage type (signal, power, ground, clock).
        shape: Pin shape type (abutment, feedthru).
        ports: List of rectangular geometries defining the pin's physical ports.
        attributes: Additional unparsed attributes.
    """

    name: str
    direction: str = "input"  # input, output, inout
    use: Optional[str] = None  # signal, power, ground, clock
    shape: Optional[str] = None  # abutment, feedthru

    # Port geometry
    ports: list[Rect] = Field(default_factory=list)

    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @property
    def layers_used(self) -> list[str]:
        """Returns list of unique layer names used by this pin's ports."""
        return list({port.layer for port in self.ports})


class Macro(BaseModel):
    """Represents a physical macro (cell) definition.

    Attributes:
        name: Macro name.
        class_type: Macro class (core, block, pad, etc.).
        origin: (x, y) coordinates of the origin.
        size: (width, height) tuple in micrometers.
        symmetry: Allowed symmetries.
        site: Name of the site pattern used.
        pins: Dictionary of pins keyed by pin name.
        obstructions: List of obstruction rectangles.
        foreign: Name of the foreign reference (if any).
        attributes: Additional unparsed attributes.
    """

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
        """Returns the width of the macro."""
        return self.size[0]

    @property
    def height(self) -> float:
        """Returns the height of the macro."""
        return self.size[1]

    @property
    def area(self) -> float:
        """Returns the area of the macro."""
        return self.size[0] * self.size[1]

    model_config = {"extra": "allow"}


class LEFLibrary(BaseModel):
    """Represents a complete LEF library containing physical cell definitions.

    Attributes:
        version: LEF file version.
        bus_bit_chars: Characters used for bus bit indexing.
        divider_char: Hierarchy divider character.
        units_database: Database units per micron (typically 1000 or 2000).
        manufacturing_grid: Manufacturing grid resolution.
        sites: Dictionary of site definitions.
        layers: Dictionary of layer definitions.
        vias: Dictionary of via definitions.
        macros: Dictionary of macro (cell) definitions.
        attributes: Additional unparsed attributes.
    """

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
        """Returns a list of routing metal layers, sorted by name."""
        routing = [l for l in self.layers.values() if l.layer_type == LayerType.ROUTING]
        return sorted(routing, key=lambda x: x.name)

    @property
    def cut_layers(self) -> list[MetalLayer]:
        """Returns a list of cut/via layers."""
        return [l for l in self.layers.values() if l.layer_type == LayerType.CUT]

    model_config = {"extra": "allow"}


class TechLEF(BaseModel):
    """Represents a Technology LEF.

    TechLEF files contain only layer, via, and site definitions, but typically
    no macros. They define the design rules and technology parameters provided
    by the foundry.

    Attributes:
        version: LEF version.
        units_database: Database units per micron.
        manufacturing_grid: Manufacturing grid.
        layers: Dictionary of layer definitions.
        vias: Dictionary of via definitions.
        sites: Dictionary of site definitions.
        process_node: Process node identifier (e.g., "7nm").
        foundry: Foundry name.
        attributes: Additional unparsed attributes.
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
        """Returns the total number of routing metal layers."""
        return len([l for l in self.layers.values() if l.layer_type == LayerType.ROUTING])

    model_config = {"extra": "allow"}
