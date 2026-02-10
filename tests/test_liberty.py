from pathlib import Path
import math

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import example

from parsfet.models.liberty import Cell, LibertyLibrary, LookupTable
from parsfet.parsers.liberty import LibertyParser


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


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.1, 0.1),  # Exact match (min)
        (0.5, 0.5),  # Exact match (mid)
        (1.0, 1.0),  # Exact match (max)
        (0.3, 0.3),  # Interpolation
        (0.0, 0.1),  # Extrapolation (clamped low)
        (2.0, 1.0),  # Extrapolation (clamped high - assumption: clamped to nearest)
    ],
)
def test_lut_interpolation_1d_cases(x, expected):
    """Parametrized tests for 1D interpolation."""
    lut = LookupTable(index_1=[0.1, 0.5, 1.0], values=[0.1, 0.5, 1.0])
    assert pytest.approx(lut.interpolate(x)) == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (0.0, 0.0, 0.0),  # Corner 0,0
        (1.0, 1.0, 2.0),  # Corner 1,1
        (0.5, 0.5, 1.0),  # Center
        (0.5, 0.0, 0.5),  # Edge mid
        (0.0, 0.5, 0.5),  # Edge mid
    ],
)
def test_lut_interpolation_2d_cases(x, y, expected):
    """Parametrized tests for 2D interpolation."""
    # Simple bilinear surface: z = x + y
    lut = LookupTable(
        index_1=[0.0, 1.0],
        index_2=[0.0, 1.0],
        values=[
            [0.0, 1.0],  # x=0: z=0, z=1
            [1.0, 2.0],  # x=1: z=1, z=2
        ],
    )
    assert pytest.approx(lut.interpolate(x, y)) == expected


@given(st.floats(min_value=0.1, max_value=1.0))
def test_lut_interpolation_1d_hypothesis(x):
    """Property-based test: Multi-linear interpolation on f(x)=x should recover x."""
    lut = LookupTable(
        index_1=[0.1, 0.5, 1.0],
        values=[0.1, 0.5, 1.0]
    )
    result = lut.interpolate(x)
    assert pytest.approx(result) == x


@given(
    load=st.floats(min_value=0.001, max_value=1.0),
    slew=st.floats(min_value=0.01, max_value=1.0)
)
def test_lut_monotonicity_hypothesis(load, slew):
    """
    Property: For a physically standard LUT (delay increases with load and slew),
    interpolation should preserve monotonicity.
    """
    # Create a synthetic delay table where delay = slew + load
    # index_1 = slew, index_2 = load
    # values[i][j] = index_1[i] + index_2[j]
    
    slews = [0.01, 0.1, 0.5, 1.0]
    loads = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    values = []
    for s in slews:
        row = []
        for l in loads:
            row.append(s + l)
        values.append(row)
            
    lut = LookupTable(
        index_1=slews,
        index_2=loads,
        values=values
    )
    
    val = lut.interpolate(slew, load)
    assert val > 0
    
    # Check local monotonicity with a small delta
    delta = 0.0001
    
    # Increase slew -> delay should increase
    if slew + delta <= 1.0:
        val_next_slew = lut.interpolate(slew + delta, load)
        assert val_next_slew >= val
        
    # Increase load -> delay should increase
    if load + delta <= 1.0:
        val_next_load = lut.interpolate(slew, load + delta)
        assert val_next_load >= val


def test_parse_from_file(sample_liberty_file):
    parser = LibertyParser()
    lib = parser.parse(sample_liberty_file)
    assert (
        lib.name == "test_lib"
    ) 
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
    content_ps_ff = sample_liberty_content.replace(
        'time_unit : "1ns"', 'time_unit : "1ps"'
    ).replace("capacitive_load_unit (1.0, pf)", "capacitive_load_unit (1.0, ff)")
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

    slew, load = lib.fo4_operating_point()

    assert pytest.approx(load) == 0.008
    # Slew should be calculated based on this load.
    assert slew > 0


def test_linear_model_fit():
    # Model: D = 0.1 + 0.5 * Load
    lut = LookupTable(
        index_1=[0.1],  # Fixed slew
        index_2=[0.0, 1.0, 2.0],
        values=[[0.1, 0.6, 1.1]],
    )

    d0, k, r_squared = lut.fit_linear_model(0.1)
    assert pytest.approx(d0) == 0.1
    assert pytest.approx(k) == 0.5
    assert pytest.approx(r_squared) == 1.0  # Perfect linear fit


def test_missing_baseline_cell(sample_liberty_content):
    # Rename INV_X1 to OTHER_X1.
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
    assert lib_buf.baseline_cell is None

    # Should still calculate a fallback operating point
    slew, load = lib_buf.fo4_operating_point()
    assert slew > 0
    assert load > 0


def test_json_round_trip(sample_liberty_content):
    """Test round-trip serialization: Parse -> Dict -> Object."""
    parser = LibertyParser()
    original_lib = parser.parse_string(sample_liberty_content)
    
    # Dump to dict (JSON-like)
    data = original_lib.model_dump()
    
    # Reconstruct from dict
    restored_lib = LibertyLibrary(**data)
    
    # Verify key attributes equality
    assert restored_lib.name == original_lib.name
    assert restored_lib.technology == original_lib.technology
    assert restored_lib.time_unit == original_lib.time_unit
    
    # Verify deep structure (cells)
    assert len(restored_lib.cells) == len(original_lib.cells)
    assert "INV_X1" in restored_lib.cells
    
    inv_orig = original_lib.cells["INV_X1"]
    inv_rest = restored_lib.cells["INV_X1"]
    
    assert inv_rest.area == inv_orig.area
    assert len(inv_rest.pins) == len(inv_orig.pins)
    assert inv_rest.pins["Y"].function == inv_orig.pins["Y"].function
