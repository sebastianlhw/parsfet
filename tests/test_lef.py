import pytest
from pathlib import Path
from parsfet.parsers.lef import LEFParser, TechLEFParser
from parsfet.models.lef import LEFLibrary, LayerType, LayerDirection

def test_parse_lef_header_and_units(sample_lef_content):
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert isinstance(lib, LEFLibrary)
    assert lib.version == "5.8"
    assert lib.units_database == 1000
    assert lib.manufacturing_grid == 0.005

def test_parse_layers(sample_lef_content):
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
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "M1_M2" in lib.vias
    via = lib.vias["M1_M2"]
    assert "M1" in via.layers
    assert "M2" in via.layers

def test_parse_sites(sample_lef_content):
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    assert "core" in lib.sites
    site = lib.sites["core"]
    assert site.class_type == "core"
    assert site.width == 0.2
    assert site.height == 2.0
    assert site.symmetry == ["Y"]

def test_parse_macros(sample_lef_content):
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
    parser = LEFParser()
    lib = parser.parse_string(sample_lef_content)

    warnings = parser.validate(lib)
    assert len(warnings) == 0

    # Create invalid state
    lib.layers = {}
    warnings = parser.validate(lib)
    assert "No layers defined" in warnings

def test_parse_file(sample_lef_file):
    parser = LEFParser()
    lib = parser.parse(sample_lef_file)
    assert lib.units_database == 1000
    assert "M1" in lib.layers
