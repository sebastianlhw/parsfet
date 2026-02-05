"""Tests for combining JSON exports with Liberty files.

These tests verify that:
1. JSON exports can be loaded via Dataset.load_files()
2. JSON + lib files can be combined with unified normalization
3. Re-normalization works correctly for JSON-sourced cells
"""

import json
import tempfile
from pathlib import Path

import pytest

from parsfet.data import Dataset


@pytest.fixture
def sample_json_export(tmp_path: Path) -> Path:
    """Create a minimal JSON export file for testing.

    Contains two cells: INVD1 (baseline) and NAND2D1.
    """
    export_data = {
        "library": "test_export",
        "baseline": {
            "cell": "INVD1",
            "area_um2": 1.0,
            "d0_ns": 0.01,
            "k_ns_per_pf": 0.5,
            "leakage": 1.0,
            "input_cap_pf": 0.001,
        },
        "cells": {
            "INVD1": {
                "cell_name": "INVD1",
                "cell_type": "inverter",
                "area_ratio": 1.0,
                "d0_ratio": 1.0,
                "k_ratio": 1.0,
                "leakage_ratio": 1.0,
                "input_cap_ratio": 1.0,
                "num_inputs": 1,
                "num_outputs": 1,
                "is_sequential": False,
                "raw": {
                    "area_um2": 1.0,
                    "d0_ns": 0.01,
                    "k_ns_per_pf": 0.5,
                    "leakage": 1.0,
                    "input_cap_pf": 0.001,
                },
            },
            "NAND2D1": {
                "cell_name": "NAND2D1",
                "cell_type": "nand",
                "area_ratio": 2.0,
                "d0_ratio": 1.5,
                "k_ratio": 1.2,
                "leakage_ratio": 2.0,
                "input_cap_ratio": 2.0,
                "num_inputs": 2,
                "num_outputs": 1,
                "is_sequential": False,
                "raw": {
                    "area_um2": 2.0,
                    "d0_ns": 0.015,
                    "k_ns_per_pf": 0.6,
                    "leakage": 2.0,
                    "input_cap_pf": 0.002,
                },
            },
        },
    }

    json_path = tmp_path / "test_export.json"
    json_path.write_text(json.dumps(export_data, indent=2))
    return json_path


class TestLoadJsonFile:
    """Tests for loading JSON exports."""

    def test_load_json_creates_entry(self, sample_json_export: Path):
        """Test that loading a JSON file creates a valid LibraryEntry."""
        ds = Dataset()
        ds.load_files([sample_json_export])

        assert len(ds.entries) == 1
        entry = ds.entries[0]
        assert entry.library.name == "test_export"
        assert len(entry.library.cells) == 2

    def test_load_json_marks_from_json(self, sample_json_export: Path):
        """Test that JSON entries are correctly marked with from_json=True."""
        ds = Dataset()
        ds.load_files([sample_json_export])

        entry = ds.entries[0]
        assert entry.from_json is True

    def test_load_json_stores_raw_metrics(self, sample_json_export: Path):
        """Test that raw metrics are cached for potential re-normalization."""
        ds = Dataset()
        ds.load_files([sample_json_export])

        entry = ds.entries[0]
        assert len(entry.raw_metrics_cache) == 2
        assert "INVD1" in entry.raw_metrics_cache
        assert entry.raw_metrics_cache["INVD1"]["area"] == 1.0
        assert entry.raw_metrics_cache["INVD1"]["d0_ns"] == 0.01


class TestCombineJsonWithLib:
    """Tests for combining JSON exports with Liberty files."""

    def test_combine_json_only(self, sample_json_export: Path):
        """Test combining a single JSON file (should work like combine with 1 lib)."""
        ds = Dataset()
        ds.load_files([sample_json_export], normalize=False)
        combined = ds.combine()

        assert len(combined.entries) == 1
        entry = combined.entries[0]
        assert entry.normalizer is not None
        assert len(entry.metrics) == 2

    def test_combine_reuses_raw_metrics(self, sample_json_export: Path):
        """Test that combine uses cached raw metrics from JSON for normalization.

        This verifies that we don't need the original .lib file to re-normalize
        cells if we have the raw metrics in the JSON export.
        """
        ds = Dataset()
        ds.load_files([sample_json_export], normalize=False)
        combined = ds.combine()

        entry = combined.entries[0]
        # NAND2D1 should have metrics calculated from raw values
        nand_metrics = entry.metrics["NAND2D1"]
        assert nand_metrics.raw_area == 2.0
        assert nand_metrics.raw_d0_ns == 0.015


class TestLibraryEntryFields:
    """Tests for LibraryEntry dataclass fields."""

    def test_from_json_default_false(self):
        """Test that from_json defaults to False for standard Liberty files."""
        from parsfet.data import LibraryEntry
        from parsfet.models.liberty import LibertyLibrary

        lib = LibertyLibrary(name="test", cells={})
        entry = LibraryEntry(library=lib, normalizer=None)

        assert entry.from_json is False
        assert entry.raw_metrics_cache == {}
