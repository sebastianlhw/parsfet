"""Tests for parsfet.data module."""

import numpy as np
import pandas as pd
import pytest

from parsfet.data import FEATURE_COLUMNS, Dataset, load_files, load_from_pattern


def test_load_files_single(sample_liberty_file):
    """Test loading a single Liberty file."""
    ds = load_files([sample_liberty_file])

    assert len(ds.entries) == 1
    assert ds.entries[0].library is not None
    assert len(ds.entries[0].metrics) > 0


def test_to_dataframe_columns(sample_liberty_file):
    """Test that to_dataframe returns correct columns."""
    df = load_files([sample_liberty_file]).to_dataframe()

    # Check required columns
    expected_cols = [
        "library",
        "cell",
        "cell_type",
        "area_ratio",
        "d0_ratio",
        "k_ratio",
        "raw_area_um2",
        "raw_d0_ns",
        "voltage",
        "temperature",
        "baseline_d0_ns",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_to_dataframe_cell_type_categorical(sample_liberty_file):
    """Test that cell_type is a categorical column."""
    df = load_files([sample_liberty_file]).to_dataframe()

    assert df["cell_type"].dtype.name == "category"


def test_to_dataframe_content(sample_liberty_file):
    """Test that DataFrame contains expected cells."""
    df = load_files([sample_liberty_file]).to_dataframe()

    # Sample lib has INV_X1 and DFF_X1
    cell_names = df["cell"].tolist()
    assert "INV_X1" in cell_names
    assert "DFF_X1" in cell_names


def test_to_numpy_shape(sample_liberty_file):
    """Test to_numpy returns correct shapes."""
    X, y, label_map = load_files([sample_liberty_file]).to_numpy()

    # Should have 2 cells (INV_X1, DFF_X1)
    assert X.shape[0] == 2
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert y.shape[0] == 2
    assert len(label_map) > 0


def test_to_numpy_label_map(sample_liberty_file):
    """Test that label_map contains cell types."""
    X, y, label_map = load_files([sample_liberty_file]).to_numpy()

    # All labels in y should be valid keys in label_map
    for label in y:
        assert label in label_map


def test_empty_dataset():
    """Test empty dataset returns empty structures."""
    ds = Dataset()
    df = ds.to_dataframe()
    assert df.empty

    X, y, label_map = ds.to_numpy()
    assert X.size == 0
    assert y.size == 0
    assert label_map == {}


def test_load_files_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_files(["nonexistent.lib"])


def test_feature_columns_count():
    """Verify FEATURE_COLUMNS matches to_feature_vector size."""
    # This ensures the constant stays in sync with NormalizedMetrics
    from parsfet.normalizers.invd1 import NormalizedMetrics

    m = NormalizedMetrics(cell_name="test")
    assert len(m.to_feature_vector()) == len(FEATURE_COLUMNS)


# --- Integration tests for LEF/TechLEF loading ---


def test_load_tech_lef(sample_liberty_file, sample_lef_file):
    """Test loading TechLEF adds tech_info to entries."""
    ds = Dataset()
    ds.load_files([sample_liberty_file])
    ds.load_tech_lef(sample_lef_file)  # Use LEF file as TechLEF (has layers)

    assert ds.entries[0].tech_info is not None
    assert ds.entries[0].tech_info.units_database == 1000
    assert "M1" in ds.entries[0].tech_info.layers


def test_to_dataframe_lef_columns(sample_liberty_file, sample_lef_file):
    """Test that LEF columns are present after loading LEF."""
    ds = Dataset()
    ds.load_files([sample_liberty_file])
    ds.load_lef([sample_lef_file])  # Sample LEF has INV_X1 macro
    ds.load_tech_lef(sample_lef_file)

    df = ds.to_dataframe()

    # LEF columns should be present
    assert "lef_width" in df.columns
    assert "lef_height" in df.columns
    assert "lef_area" in df.columns
    assert "pin_layers_json" in df.columns
    assert "metal_stack_height" in df.columns

    # INV_X1 should have LEF data matched
    inv_row = df[df["cell"] == "INV_X1"]
    if not inv_row.empty:
        assert inv_row["lef_width"].iloc[0] == 1.0
        assert inv_row["lef_height"].iloc[0] == 2.0


def test_load_lef_not_found():
    """Test that FileNotFoundError is raised for missing LEF file."""
    ds = Dataset()
    with pytest.raises(FileNotFoundError):
        ds.load_lef(["nonexistent.lef"])


def test_load_tech_lef_not_found():
    """Test that FileNotFoundError is raised for missing TechLEF file."""
    ds = Dataset()
    with pytest.raises(FileNotFoundError):
        ds.load_tech_lef("nonexistent.tlef")


def test_export_to_json(sample_liberty_file, sample_lef_file):
    """Test export_to_json produces proper structure."""
    ds = Dataset()
    ds.load_files([sample_liberty_file])
    ds.load_lef([sample_lef_file])
    ds.load_tech_lef(sample_lef_file)

    data = ds.export_to_json()

    # Check basic structure
    assert "library" in data
    assert "cells" in data
    assert "technology" in data

    # Check technology section
    assert "metal_stack_height" in data["technology"]
    assert "layers" in data["technology"]

    # Check cell physical data with pin use type
    if "INV_X1" in data["cells"]:
        cell = data["cells"]["INV_X1"]
        assert "physical" in cell
        assert "pins" in cell["physical"]
        for pin_name, pin in cell["physical"]["pins"].items():
            assert "direction" in pin
            assert "use" in pin  # New: pin use type
            assert "layers" in pin


def test_export_to_json_include_port_geometry(sample_liberty_file, sample_lef_file):
    """Test export_to_json produces proper structure with include_port_geometry=True."""
    ds = Dataset()
    ds.load_files([sample_liberty_file])
    ds.load_lef([sample_lef_file])
    ds.load_tech_lef(sample_lef_file)

    data = ds.export_to_json(include_port_geometry=True)

    assert "INV_X1" in data.get("cells", {})
    cell = data["cells"]["INV_X1"]
    assert "physical" in cell
    assert "pins" in cell["physical"]
    
    assert "A" in cell["physical"]["pins"]
    pin_a = cell["physical"]["pins"]["A"]
    assert "ports" in pin_a
    assert len(pin_a["ports"]) > 0
    
    port = pin_a["ports"][0]
    assert "layer" in port
    assert "x1" in port
    assert "y1" in port
    assert "x2" in port
    assert "y2" in port
    assert port["layer"] == "M1"


def test_export_to_json_empty():
    """Test export_to_json on empty dataset."""
    ds = Dataset()
    data = ds.export_to_json()
    assert "error" in data


# --- Tests for raw_power_fo4 column ---


def test_to_dataframe_power_fo4_column(sample_liberty_file):
    """raw_power_fo4 column exists in the DataFrame; may be None for cells with no power tables."""
    df = load_files([sample_liberty_file]).to_dataframe()
    assert "raw_power_fo4" in df.columns


# --- Tests for list_cell_arcs() ---


def test_list_cell_arcs_inv(sample_liberty_file):
    """list_cell_arcs returns one arc for INV_X1 (A → Y)."""
    ds = load_files([sample_liberty_file])
    arcs = ds.list_cell_arcs("INV_X1")

    assert len(arcs) >= 1
    # Should contain the A→Y arc
    from_pins = [a["from_pin"] for a in arcs]
    assert "A" in from_pins

    arc_a = next(a for a in arcs if a["from_pin"] == "A")
    assert "timing_type" in arc_a
    assert "timing_sense" in arc_a
    assert arc_a["timing_sense"] == "negative_unate"


def test_list_cell_arcs_dff(sample_liberty_file):
    """list_cell_arcs returns multiple arc types for DFF_X1 (clk-to-Q + constraints)."""
    ds = load_files([sample_liberty_file])
    arcs = ds.list_cell_arcs("DFF_X1")

    assert len(arcs) >= 1
    from_pins = [a["from_pin"] for a in arcs]
    assert "CLK" in from_pins

    timing_types = [a["timing_type"] for a in arcs]
    # Should have at least a setup/hold or rising_edge arc
    assert any(t in ("setup_rising", "hold_rising", "rising_edge") for t in timing_types)


def test_list_cell_arcs_missing_cell(sample_liberty_file):
    """list_cell_arcs raises KeyError for unknown cell name."""
    ds = load_files([sample_liberty_file])
    with pytest.raises(KeyError, match="NONEXISTENT"):
        ds.list_cell_arcs("NONEXISTENT")


def test_list_cell_arcs_out_of_range(sample_liberty_file):
    """list_cell_arcs raises IndexError for bad entry_index."""
    ds = load_files([sample_liberty_file])
    with pytest.raises(IndexError):
        ds.list_cell_arcs("INV_X1", entry_index=99)


# --- Tests for query_cell_at() ---


def test_query_cell_at_all_arcs(sample_liberty_file):
    """query_cell_at with no filters returns all timing arcs for the cell."""
    ds = load_files([sample_liberty_file])
    arcs = ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.008)

    assert len(arcs) >= 1
    for arc in arcs:
        assert "from_pin" in arc
        assert "timing_type" in arc
        assert "timing_sense" in arc
        assert "delay_ns" in arc
        assert "output_slew_ns" in arc
        assert "energy_fj" in arc
        # Delay and slew should be positive for a real arc
        assert arc["delay_ns"] >= 0.0
        assert arc["output_slew_ns"] >= 0.0


def test_query_cell_at_from_pin_filter(sample_liberty_file):
    """query_cell_at(from_pin='A') returns only arc(s) driven by pin A."""
    ds = load_files([sample_liberty_file])
    arcs = ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.008, from_pin="A")

    assert len(arcs) >= 1
    for arc in arcs:
        assert arc["from_pin"] == "A"
    # INV_X1 A→Y arc has delay tables — should be positive
    assert arcs[0]["delay_ns"] > 0.0


def test_query_cell_at_clk_to_q(sample_liberty_file):
    """query_cell_at with timing_type='rising_edge' returns the clk-to-Q arc."""
    ds = load_files([sample_liberty_file])
    arcs = ds.query_cell_at(
        "DFF_X1", slew_ns=0.1, load_pf=0.01, timing_type="rising_edge"
    )

    assert len(arcs) >= 1
    for arc in arcs:
        assert arc["timing_type"] == "rising_edge"
    assert arcs[0]["delay_ns"] > 0.0


def test_query_cell_at_bad_filter_raises(sample_liberty_file):
    """query_cell_at raises ValueError when no arcs match the filter."""
    ds = load_files([sample_liberty_file])
    with pytest.raises(ValueError, match="No timing arcs match"):
        ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.008, from_pin="NONEXISTENT_PIN")


def test_query_cell_at_missing_cell(sample_liberty_file):
    """query_cell_at raises KeyError for unknown cell name."""
    ds = load_files([sample_liberty_file])
    with pytest.raises(KeyError, match="NONEXIST"):
        ds.query_cell_at("NONEXIST", slew_ns=0.05, load_pf=0.01)


def test_query_cell_at_canonical_units(sample_liberty_file):
    """query_cell_at returns delay in ns regardless of source library units (ns library)."""
    ds = load_files([sample_liberty_file])
    # The sample lib uses 1ns time unit and 1pf cap unit.
    # Querying at a point inside the table should give sensible ns values (~0.05 - 0.5ns).
    arcs = ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.008)
    delay = arcs[0]["delay_ns"]
    # Should be in a reasonable ns range, not in ps (i.e. not ~50000)
    assert 0.001 < delay < 10.0


def test_query_cell_at_energy_no_power_tables(sample_liberty_file):
    """energy_fj is 0.0 when the library has no power tables."""
    ds = load_files([sample_liberty_file])
    arcs = ds.query_cell_at("INV_X1", slew_ns=0.05, load_pf=0.008)
    # Sample lib has no power arcs, so energy should be 0.0
    for arc in arcs:
        assert arc["energy_fj"] == 0.0

