from pathlib import Path

import pytest

from parsfet.models.lef import LayerDirection, LayerType, LEFLibrary
from parsfet.parsers.lef import LEFParser, TechLEFParser


def test_parse_lef_header_and_units(sample_lef_content):
    """Verifies parsing of LEF header and units.

    Checks:
        - Version string.
        - Database units (DBU per micron).
        - Manufacturing grid.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert isinstance(lib, LEFLibrary)
    assert lib.version == "5.8"
    assert lib.units_database == 1000
    assert lib.manufacturing_grid == 0.005


def test_parse_layers(sample_lef_content):
    """Verifies parsing of layer definitions.

    Checks:
        - Layer type (routing, cut).
        - Routing direction (horizontal/vertical).
        - Physical attributes (width, pitch, spacing).
        - Electrical attributes (resistance, capacitance).
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "M1" in lib.layers
    m1 = lib.layers["M1"]
    assert m1.layer_type == LayerType.ROUTING
    assert m1.direction == LayerDirection.HORIZONTAL
    assert m1.pitch == 0.2
    assert m1.width == 0.1
    assert m1.spacing == 0.1
    assert m1.resistance == 0.1
    assert m1.capacitance == 0.2

    assert "M2" in lib.layers
    m2 = lib.layers["M2"]
    assert m2.layer_type == LayerType.ROUTING
    assert m2.direction == LayerDirection.VERTICAL


def test_parse_vias(sample_lef_content):
    """Verifies parsing of via definitions.

    Checks:
        - Via name.
        - Layer connectivity list.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "M1_M2" in lib.vias
    via = lib.vias["M1_M2"]
    assert "M1" in via.layers
    assert "M2" in via.layers


def test_parse_sites(sample_lef_content):
    """Verifies parsing of site definitions.

    Checks:
        - Site class type.
        - Dimensions (width, height).
        - Symmetry options.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "core" in lib.sites
    site = lib.sites["core"]
    assert site.class_type == "core"
    assert site.width == 0.2
    assert site.height == 2.0
    assert site.symmetry == ["Y"]


def test_parse_macros(sample_lef_content):
    """Verifies parsing of macro (cell) definitions.

    Checks:
        - Macro class, origin, size, symmetry.
        - Site reference.
        - Pin definitions (direction, layer ports).
        - Obstruction geometries.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "INV_X1" in lib.macros
    macro = lib.macros["INV_X1"]
    assert macro.class_type == "core"
    assert macro.origin == (0.0, 0.0)
    assert macro.size == (1.0, 2.0)
    assert set(macro.symmetry) == {"X", "Y"}
    assert macro.site == "core"

    # Check Pins
    assert "A" in macro.pins
    pin_a = macro.pins["A"]
    assert pin_a.direction == "input"
    assert len(pin_a.ports) == 1
    assert pin_a.ports[0].layer == "M1"

    assert "Y" in macro.pins
    pin_y = macro.pins["Y"]
    assert pin_y.direction == "output"

    # Check Obstructions
    assert len(macro.obstructions) == 1
    obs = macro.obstructions[0]
    assert obs.layer == "M1"
    assert obs.x1 == 0.0
    assert obs.y2 == 2.0


def test_tech_lef_parser(sample_lef_content):
    """Verifies the TechLEF parser specialized behavior.

    Checks that the TechLEF parser correctly extracts technology information
    (units, layers, vias) while ignoring macro definitions, resulting in a
    lightweight TechLEF object.
    """
    # TechLEF parser should only parse layers/vias/sites, not macros
    parser = TechLEFParser()
    # TechLEF parser actually uses LEFParser internally and copies fields,
    # but the model TechLEF doesn't have 'macros' field.

    # Since sample_lef_content has macros, LEFParser will parse them,
    # but TechLEFParser constructs TechLEF which ignores them.

    tech_lib = parser.parse_string(sample_lef_content)

    assert tech_lib.units_database == 1000
    assert "M1" in tech_lib.layers
    assert "M1_M2" in tech_lib.vias

    # Verify it's a TechLEF object
    assert not hasattr(tech_lib, "macros")


def test_lef_validation(sample_lef_content):
    """Verifies LEF validation logic.

    Checks:
        - No warnings for valid content.
        - Warning generation for missing layer definitions.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    warnings = parser.validate(lib)
    assert len(warnings) == 0

    # Create invalid state
    lib.layers = {}
    warnings = parser.validate(lib)
    assert "No layers defined" in warnings


def test_parse_file(sample_lef_file):
    """Verifies parsing directly from a LEF file."""
    parser = LEFParser()
    lib = parser.parse(sample_lef_file)
    assert lib.units_database == 1000
    assert "M1" in lib.layers


# --- New tests for helper methods and physical dataclasses ---


def test_macro_pin_layers_used(sample_lef_content):
    """Verifies extraction of used layers from MacroPin."""
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    macro = lib.macros["INV_X1"]
    pin_a = macro.pins["A"]

    # Pin A uses M1 layer
    assert pin_a.layers_used == ["M1"]


def test_metal_layer_min_size(sample_lef_content):
    """Verifies calculation of min_size property for MetalLayer.

    Should fallback to 'width' if 'min_width' is not specified.
    """
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    m1 = lib.layers["M1"]
    # min_width not set, should fallback to width
    assert m1.min_size == 0.1  # width is 0.1


def test_cell_physical_from_macro(sample_lef_content):
    """Verifies conversion from LEF Macro to simplified CellPhysical model."""
    from parsfet.models.physical import CellPhysical

    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    macro = lib.macros["INV_X1"]
    cell = CellPhysical.from_macro(macro)

    assert cell.name == "INV_X1"
    assert cell.width == 1.0
    assert cell.height == 2.0
    assert cell.area == 2.0

    # Check pins
    assert "A" in cell.pins
    assert "Y" in cell.pins
    assert cell.pins["A"].direction == "input"
    assert cell.pins["Y"].direction == "output"


def test_tech_info_from_tech_lef(sample_lef_content):
    """Verifies conversion from TechLEF to simplified TechInfo model."""
    from parsfet.models.physical import TechInfo

    parser = TechLEFParser()
    tech = parser.parse_string(sample_lef_content)

    info = TechInfo.from_tech_lef(tech)

    assert info.units_database == 1000
    assert info.manufacturing_grid == 0.005
    assert "M1" in info.layers
    assert "M2" in info.layers

    # Check layer info
    m1 = info.layers["M1"]
    assert m1.layer_type == "routing"
    assert m1.direction == "horizontal"
    assert m1.min_size == 0.1  # Falls back to width
