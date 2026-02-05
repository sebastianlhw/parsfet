"""Tests for the Lark-based Liberty parser.

These tests mirror test_liberty.py but use LibertyParser (Lark-based).
"""

from pathlib import Path

import pytest

from parsfet.models.liberty import Cell, LibertyLibrary, LookupTable
from parsfet.parsers.liberty import LibertyParser


def test_lark_parse_liberty_library_attributes(sample_liberty_content):
    """Verifies that library-level attributes are parsed correctly by Lark.

    Checks:
        - Library name, technology, delay model.
        - Units (time, voltage, capacitance).
        - Nominal operating conditions.
    """
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


def test_lark_parse_cells_and_pins(sample_liberty_content):
    """Verifies that cells and pins are parsed correctly by Lark.

    Checks:
        - Cell attributes (area, leakage).
        - Pin attributes (direction, capacitance, function).
        - Sequential cell identification.
    """
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


def test_lark_timing_arcs_and_lut(sample_liberty_content):
    """Verifies that timing arcs and LUTs are parsed correctly by Lark.

    Checks:
        - Timing arc structure.
        - 2D Lookup Table dimensions and values.
    """
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    inv = lib.cells["INV_X1"]
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


def test_lark_parse_from_file(sample_liberty_file):
    """Verifies Lark parsing from a file path."""
    parser = LibertyParser()
    lib = parser.parse(sample_liberty_file)
    assert lib.name == "test_lib"
    assert "INV_X1" in lib.cells


def test_lark_unit_normalizer(sample_liberty_content):
    """Verifies unit normalizer creation from Lark-parsed library."""
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    normalizer = lib.unit_normalizer
    assert normalizer.time_multiplier == 1.0
    assert normalizer.cap_multiplier == 1.0


def test_lark_fo4_calculation(sample_liberty_content):
    """Verifies FO4 operating point calculation on Lark-parsed library."""
    parser = LibertyParser()
    lib = parser.parse_string(sample_liberty_content)

    slew, load = lib.fo4_operating_point()

    assert pytest.approx(load) == 0.008
    assert slew > 0


def test_lark_missing_baseline_cell(sample_liberty_content):
    """Verifies baseline fallback logic with Lark parser."""
    content = sample_liberty_content.replace("cell(INV_X1)", "cell(OTHER_X1)")
    parser = LibertyParser()
    lib = parser.parse_string(content)

    assert lib.baseline_cell is not None
    assert lib.baseline_cell.name == "OTHER_X1"

    content_buf = content.replace('function : "!A"', 'function : "A"')
    lib_buf = parser.parse_string(content_buf)

    assert lib_buf.baseline_cell is None

    slew, load = lib_buf.fo4_operating_point()
    assert slew > 0
    assert load > 0
