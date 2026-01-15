import pytest
from pathlib import Path
from parsfet.parsers.liberty import LibertyParser
from parsfet.models.liberty import LibertyLibrary, Cell, LookupTable

def test_parse_liberty_library_attributes(sample_liberty_content):
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    assert isinstance(lib, LibertyLibrary)
    assert lib.name == "test_lib"
    assert lib.technology == "cmos"
    assert lib.delay_model == "table_lookup"
    assert lib.time_unit == "1ns"
    assert lib.voltage_unit == "1V"
    assert lib.capacitive_load_unit == (1.0, "pf")
    assert lib.nom_process == 1.0
    assert lib.nom_temperature == 25.0
    assert lib.nom_voltage == 1.2

def test_parse_cells_and_pins(sample_liberty_content):
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    assert "INV_X1" in lib.cells
    inv = lib.cells["INV_X1"]
    assert inv.area == 1.5
    assert inv.cell_leakage_power == 0.05
    assert not inv.is_sequential

    assert "A" in inv.pins
    assert inv.pins["A"].direction == "input"
    assert inv.pins["A"].capacitance == 0.002

    assert "Y" in inv.pins
    assert inv.pins["Y"].direction == "output"
    assert inv.pins["Y"].function == "!A"

    assert "DFF_X1" in lib.cells
    dff = lib.cells["DFF_X1"]
    assert dff.is_sequential
    assert dff.pins["CLK"].clock is True

def test_timing_arcs_and_lut(sample_liberty_content):
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    inv = lib.cells["INV_X1"]
    # There should be at least one timing arc for Y
    arcs = inv.timing_arcs
    assert len(arcs) > 0
    arc = arcs[0]

    assert arc.related_pin == "A"
    assert arc.timing_sense == "negative_unate"

    # Check LUT
    assert arc.cell_rise is not None
    assert arc.cell_rise.is_2d
    assert len(arc.cell_rise.index_1) == 5
    assert len(arc.cell_rise.index_2) == 5
    assert len(arc.cell_rise.values) == 5
    assert len(arc.cell_rise.values[0]) == 5

def test_lut_interpolation_1d():
    lut = LookupTable(
        index_1=[0.1, 0.5, 1.0],
        values=[0.1, 0.5, 1.0]
    )

    # Exact match
    assert lut.interpolate(0.5) == 0.5

    # Interpolation
    assert pytest.approx(lut.interpolate(0.3)) == 0.3

    # Extrapolation (clamping)
    assert lut.interpolate(0.0) == 0.1
    assert lut.interpolate(2.0) == 1.0

def test_lut_interpolation_2d():
    # Simple bilinear surface: z = x + y
    lut = LookupTable(
        index_1=[0.0, 1.0],
        index_2=[0.0, 1.0],
        values=[
            [0.0, 1.0],  # x=0: z=0, z=1
            [1.0, 2.0]   # x=1: z=1, z=2
        ]
    )

    # Corners
    assert lut.interpolate(0.0, 0.0) == 0.0
    assert lut.interpolate(1.0, 1.0) == 2.0

    # Center
    assert pytest.approx(lut.interpolate(0.5, 0.5)) == 1.0

    # Edges
    assert pytest.approx(lut.interpolate(0.5, 0.0)) == 0.5
    assert pytest.approx(lut.interpolate(0.0, 0.5)) == 0.5

def test_parse_from_file(sample_liberty_file):
    parser = LibertyParser()
    lib = parser.parse(sample_liberty_file)
    assert lib.name == "test_lib" # Name comes from content parsing, but if parse_string is called, it might use file stem if name not in file?
    # Actually parser.parse passes name=path.stem. But inside parse_string, if _qualifier is found, it overrides.
    # In our sample content: library(test_lib). So name should be test_lib.

    assert "INV_X1" in lib.cells

def test_unit_normalizer(sample_liberty_content):
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    normalizer = lib.unit_normalizer
    # defined units: time 1ns, cap 1pf
    # Target canonical: ns, pF

    # 1ns -> 1ns (factor 1)
    assert normalizer.time_multiplier == 1.0
    # 1pf -> 1pf (factor 1)
    assert normalizer.cap_multiplier == 1.0

    # Test with different units
    content_ps_ff = sample_liberty_content.replace('time_unit : "1ns"', 'time_unit : "1ps"') \
                                          .replace('capacitive_load_unit (1.0, pf)', 'capacitive_load_unit (1.0, ff)')
    lib2 = parser.parse_string(content_ps_ff)
    norm2 = lib2.unit_normalizer

    # 1ps -> 0.001ns
    assert pytest.approx(norm2.time_multiplier) == 0.001
    # 1ff -> 0.001pf
    assert pytest.approx(norm2.cap_multiplier) == 0.001

def test_fo4_calculation(sample_liberty_content):
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    # INV_X1 input cap is 0.002
    # FO4 load = 4 * 0.002 = 0.008

    slew, load = lib.fo4_operating_point

    assert pytest.approx(load) == 0.008
    # Slew should be calculated based on this load.
    # Since 0.008 is small (between 0.001 and 0.01 in index_2), it will interpolate.
    assert slew > 0

def test_linear_model_fit():
    # Model: D = 0.1 + 0.5 * Load
    lut = LookupTable(
        index_1=[0.1], # Fixed slew
        index_2=[0.0, 1.0, 2.0],
        values=[
            [0.1, 0.6, 1.1]
        ]
    )

    d0, k, r_squared = lut.fit_linear_model(0.1)
    assert pytest.approx(d0) == 0.1
    assert pytest.approx(k) == 0.5
    assert pytest.approx(r_squared) == 1.0  # Perfect linear fit

def test_missing_baseline_cell(sample_liberty_content):
    # Rename INV_X1 to OTHER_X1.
    # Since OTHER_X1 is logically an inverter (has function !A),
    # the classifier should identify it as an inverter and use it as baseline.
    content = sample_liberty_content.replace("cell(INV_X1)", "cell(OTHER_X1)")
    parser = LibertyParser()
    lib = parser.parse_string(content)

    # It should find it via classifier
    assert lib.baseline_cell is not None
    assert lib.baseline_cell.name == "OTHER_X1"

    # If we make it NOT an inverter (e.g. BUFFER), then it should return None
    content_buf = content.replace('function : "!A"', 'function : "A"')
    lib_buf = parser.parse_string(content_buf)

    # Now it's a buffer, so no baseline inverter should be found
    # (assuming no other cells)
    assert lib_buf.baseline_cell is None

    # Should still calculate a fallback operating point
    slew, load = lib_buf.fo4_operating_point
    assert slew > 0
    assert load > 0
