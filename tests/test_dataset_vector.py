"""Tests for Dataset.to_vector() and Dataset.to_summary_dict()."""

import pytest
from pathlib import Path
from parsfet.data import Dataset


def test_to_vector_returns_15_elements(sample_liberty_path):
    """Test that to_vector() returns exactly 15 features."""
    ds = Dataset().load_files([sample_liberty_path])
    vector = ds.to_vector()
    
    assert len(vector) == 15
    assert all(isinstance(v, (int, float)) for v in vector)


def test_to_vector_mean_ratios(sample_liberty_path):
    """Test that mean ratios are computed correctly."""
    ds = Dataset().load_files([sample_liberty_path])
    vector = ds.to_vector()
    
    # Elements 0-3 are mean ratios (area, d0, k, leakage)
    mean_area, mean_d0, mean_k, mean_leakage = vector[0:4]
    
    # All should be positive for a real library
    assert mean_area > 0
    assert mean_d0 > 0
    assert mean_k >= 0  # k can be 0 in some cases


def test_to_vector_std_ratios(sample_liberty_path):
    """Test that std ratios are included (elements 4-7)."""
    ds = Dataset().load_files([sample_liberty_path])
    vector = ds.to_vector()
    
    # Elements 4-7 are std ratios
    std_area, std_d0, std_k, std_leakage = vector[4:8]
    
    # Std should be >= 0
    assert std_area >= 0
    assert std_d0 >= 0
    assert std_k >= 0
    assert std_leakage >= 0


def test_to_vector_cell_ratios(sample_liberty_path):
    """Test that cell count ratios are between 0 and 1."""
    ds = Dataset().load_files([sample_liberty_path])
    vector = ds.to_vector()
    
    # Elements 9-13 are ratios (combinational, sequential, inv, nand, dff)
    comb_ratio = vector[9]
    seq_ratio = vector[10]
    inv_ratio = vector[11]
    nand_ratio = vector[12]
    dff_ratio = vector[13]
    
    # All ratios should be between 0 and 1
    assert 0 <= comb_ratio <= 1
    assert 0 <= seq_ratio <= 1
    assert 0 <= inv_ratio <= 1
    assert 0 <= nand_ratio <= 1
    assert 0 <= dff_ratio <= 1
    
    # Combinational + sequential should sum to ~1
    assert abs((comb_ratio + seq_ratio) - 1.0) < 0.01


def test_to_vector_empty_dataset():
    """Test to_vector() on empty dataset returns zeros."""
    ds = Dataset()
    vector = ds.to_vector()
    
    assert len(vector) == 15
    assert all(v == 0.0 for v in vector)


def test_to_summary_dict_structure(sample_liberty_path):
    """Test that to_summary_dict() has expected structure."""
    ds = Dataset().load_files([sample_liberty_path])
    summary = ds.to_summary_dict()
    
    assert "library" in summary
    assert "baseline" in summary
    assert "normalized_stats" in summary
    assert "cell_counts" in summary
    assert "function_types" in summary
    assert "metadata" in summary
    
    # Check baseline sub-structure
    baseline = summary["baseline"]
    assert "cell" in baseline
    assert "area_um2" in baseline
    assert "d0_ns" in baseline
    assert "k_ns_per_pf" in baseline
    
    # Check normalized_stats has std
    assert "std" in summary["normalized_stats"]["area"]
    assert "mean" in summary["normalized_stats"]["area"]


def test_to_summary_dict_cell_counts(sample_liberty_path):
    """Test that cell counts are consistent."""
    ds = Dataset().load_files([sample_liberty_path])
    summary = ds.to_summary_dict()
    
    counts = summary["cell_counts"]
    assert counts["total"] == counts["combinational"] + counts["sequential"]
    assert counts["total"] > 0


def test_to_summary_dict_empty_dataset():
    """Test to_summary_dict() on empty dataset."""
    ds = Dataset()
    summary = ds.to_summary_dict()
    
    assert "error" in summary
    assert summary["error"] == "No entries loaded"


def test_vector_comparison_same_library(sample_liberty_path):
    """Test that same library produces identical vectors."""
    ds1 = Dataset().load_files([sample_liberty_path])
    ds2 = Dataset().load_files([sample_liberty_path])
    
    vec1 = ds1.to_vector()
    vec2 = ds2.to_vector()
    
    # Should be exactly equal
    assert vec1 == vec2


@pytest.fixture
def sample_liberty_path():
    """Fixture providing path to sample liberty file."""
    # Use existing test data
    path = Path("testdata/example.lib")
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path
