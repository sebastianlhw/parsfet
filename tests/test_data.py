"""Tests for parsfet.data module."""

import pytest
import numpy as np
import pandas as pd

from parsfet.data import Dataset, load_files, load_from_pattern, FEATURE_COLUMNS


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


def test_export_to_json_empty():
    """Test export_to_json on empty dataset."""
    ds = Dataset()
    data = ds.export_to_json()
    assert "error" in data


