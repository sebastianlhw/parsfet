"""Tests for multi-file combining functionality in parsfet.data module."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from parsfet.data import Dataset, load_files
from parsfet.exceptions import DuplicateCellError


@pytest.fixture
def lib1_content():
    """Liberty file with INV_X1 and NAND2_X1."""
    return textwrap.dedent("""
    library(lib1) {
      technology (cmos);
      delay_model : table_lookup;
      time_unit : "1ns";
      voltage_unit : "1V";
      current_unit : "1mA";
      pulling_resistance_unit : "1kohm";
      leakage_power_unit : "1nW";
      capacitive_load_unit (1.0, pf);

      nom_process : 1.0;
      nom_temperature : 25.0;
      nom_voltage : 1.2;

      lu_table_template(delay_template_5x5) {
        variable_1 : input_net_transition;
        variable_2 : total_output_net_capacitance;
        index_1 ("0.01, 0.05, 0.1, 0.5, 1.0");
        index_2 ("0.001, 0.01, 0.1, 0.5, 1.0");
      }

      cell(INV_X1) {
        area : 1.5;
        cell_leakage_power : 0.05;
        pin(A) {
          direction : input;
          capacitance : 0.002;
        }
        pin(Y) {
          direction : output;
          function : "!A";
          timing() {
            related_pin : "A";
            timing_sense : negative_unate;
            cell_rise(delay_template_5x5) {
              values("0.05, 0.06, 0.08, 0.15, 0.25", \\
                     "0.06, 0.07, 0.09, 0.16, 0.26", \\
                     "0.08, 0.09, 0.11, 0.18, 0.28", \\
                     "0.15, 0.16, 0.18, 0.25, 0.35", \\
                     "0.25, 0.26, 0.28, 0.35, 0.45");
            }
            cell_fall(delay_template_5x5) {
               values("0.04, 0.05, 0.07, 0.14, 0.24", \\
                      "0.05, 0.06, 0.08, 0.15, 0.25", \\
                      "0.07, 0.08, 0.10, 0.17, 0.27", \\
                      "0.14, 0.15, 0.17, 0.24, 0.34", \\
                      "0.24, 0.25, 0.27, 0.34, 0.44");
            }
          }
        }
      }

      cell(NAND2_X1) {
        area : 2.0;
        cell_leakage_power : 0.08;
        pin(A) {
          direction : input;
          capacitance : 0.003;
        }
        pin(B) {
          direction : input;
          capacitance : 0.003;
        }
        pin(Y) {
          direction : output;
          function : "!(A & B)";
        }
      }
    }
    """)


@pytest.fixture
def lib2_content():
    """Liberty file with NOR2_X1 and BUF_X1 (no duplicates with lib1)."""
    return textwrap.dedent("""
    library(lib2) {
      technology (cmos);
      delay_model : table_lookup;
      time_unit : "1ns";
      voltage_unit : "1V";
      current_unit : "1mA";
      pulling_resistance_unit : "1kohm";
      leakage_power_unit : "1nW";
      capacitive_load_unit (1.0, pf);

      nom_process : 1.0;
      nom_temperature : 25.0;
      nom_voltage : 1.2;

      lu_table_template(delay_template_5x5) {
        variable_1 : input_net_transition;
        variable_2 : total_output_net_capacitance;
        index_1 ("0.01, 0.05, 0.1, 0.5, 1.0");
        index_2 ("0.001, 0.01, 0.1, 0.5, 1.0");
      }

      cell(NOR2_X1) {
        area : 2.0;
        cell_leakage_power : 0.08;
        pin(A) {
          direction : input;
          capacitance : 0.003;
        }
        pin(B) {
          direction : input;
          capacitance : 0.003;
        }
        pin(Y) {
          direction : output;
          function : "!(A | B)";
        }
      }

      cell(BUF_X1) {
        area : 2.5;
        cell_leakage_power : 0.06;
        pin(A) {
          direction : input;
          capacitance : 0.002;
        }
        pin(Y) {
          direction : output;
          function : "A";
        }
      }
    }
    """)


@pytest.fixture
def lib3_duplicate_content():
    """Liberty file with INV_X1 (duplicates with lib1)."""
    return textwrap.dedent("""
    library(lib3) {
      technology (cmos);
      delay_model : table_lookup;
      time_unit : "1ns";
      voltage_unit : "1V";
      current_unit : "1mA";
      pulling_resistance_unit : "1kohm";
      leakage_power_unit : "1nW";
      capacitive_load_unit (1.0, pf);

      nom_process : 1.0;
      nom_temperature : 25.0;
      nom_voltage : 1.2;

      lu_table_template(delay_template_5x5) {
        variable_1 : input_net_transition;
        variable_2 : total_output_net_capacitance;
        index_1 ("0.01, 0.05, 0.1, 0.5, 1.0");
        index_2 ("0.001, 0.01, 0.1, 0.5, 1.0");
      }

      cell(INV_X1) {
        area : 1.8;
        cell_leakage_power : 0.06;
        pin(A) {
          direction : input;
          capacitance : 0.0025;
        }
        pin(Y) {
          direction : output;
          function : "!A";
          timing() {
            related_pin : "A";
            timing_sense : negative_unate;
            cell_rise(delay_template_5x5) {
              values("0.06, 0.07, 0.09, 0.16, 0.26", \\
                     "0.07, 0.08, 0.10, 0.17, 0.27", \\
                     "0.09, 0.10, 0.12, 0.19, 0.29", \\
                     "0.16, 0.17, 0.19, 0.26, 0.36", \\
                     "0.26, 0.27, 0.29, 0.36, 0.46");
            }
            cell_fall(delay_template_5x5) {
               values("0.05, 0.06, 0.08, 0.15, 0.25", \\
                      "0.06, 0.07, 0.09, 0.16, 0.26", \\
                      "0.08, 0.09, 0.11, 0.18, 0.28", \\
                      "0.15, 0.16, 0.18, 0.25, 0.35", \\
                      "0.25, 0.26, 0.28, 0.35, 0.45");
            }
          }
        }
      }
    }
    """)


@pytest.fixture
def lib1_file(lib1_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f:
        f.write(lib1_content)
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def lib2_file(lib2_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f:
        f.write(lib2_content)
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def lib3_file(lib3_duplicate_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f:
        f.write(lib3_duplicate_content)
        path = Path(f.name)
    yield path
    path.unlink()


class TestFindDuplicates:
    """Tests for Dataset.find_duplicates() method."""

    def test_no_duplicates(self, lib1_file, lib2_file):
        """Two files with no overlapping cells should have no duplicates."""
        ds = Dataset()
        ds.load_files([lib1_file, lib2_file], normalize=False)

        dups = ds.find_duplicates()
        assert dups == {}

    def test_with_duplicates(self, lib1_file, lib3_file):
        """Two files with overlapping INV_X1 should detect duplicate."""
        ds = Dataset()
        ds.load_files([lib1_file, lib3_file], normalize=False)

        dups = ds.find_duplicates()
        assert "INV_X1" in dups
        assert len(dups["INV_X1"]) == 2

    def test_single_file_no_duplicates(self, lib1_file):
        """Single file should never have duplicates."""
        ds = Dataset()
        ds.load_files([lib1_file], normalize=False)

        dups = ds.find_duplicates()
        assert dups == {}


class TestCombine:
    """Tests for Dataset.combine() method."""

    def test_combine_no_duplicates(self, lib1_file, lib2_file):
        """Combining files with distinct cells should succeed."""
        ds = Dataset()
        ds.load_files([lib1_file, lib2_file], normalize=False)

        combined = ds.combine()

        assert len(combined.entries) == 1
        # Should have all 4 cells: INV_X1, NAND2_X1, NOR2_X1, BUF_X1
        assert len(combined.entries[0].library.cells) == 4
        assert combined.entries[0].normalizer is not None

    def test_combine_with_duplicates_raises(self, lib1_file, lib3_file):
        """Combining files with duplicates should raise DuplicateCellError."""
        ds = Dataset()
        ds.load_files([lib1_file, lib3_file], normalize=False)

        with pytest.raises(DuplicateCellError) as exc_info:
            ds.combine()

        assert "INV_X1" in str(exc_info.value)
        assert "INV_X1" in exc_info.value.cell_names()

    def test_combine_allow_duplicates(self, lib1_file, lib3_file):
        """Combining with allow_duplicates=True should use first occurrence."""
        ds = Dataset()
        ds.load_files([lib1_file, lib3_file], normalize=False)

        combined = ds.combine(allow_duplicates=True)

        assert len(combined.entries) == 1
        # INV_X1 should come from lib1 (area=1.5, not 1.8)
        inv_cell = combined.entries[0].library.cells["INV_X1"]
        assert inv_cell.area == 1.5

    def test_combine_empty_raises(self):
        """Combining empty dataset should raise ValueError."""
        ds = Dataset()

        with pytest.raises(ValueError, match="No entries loaded"):
            ds.combine()

    def test_combine_preserves_metrics(self, lib1_file, lib2_file):
        """Combined dataset should have normalized metrics for all cells."""
        ds = Dataset()
        ds.load_files([lib1_file, lib2_file], normalize=False)

        combined = ds.combine()

        # All cells should have metrics
        metrics = combined.entries[0].metrics
        assert "INV_X1" in metrics
        assert "NAND2_X1" in metrics
        assert "NOR2_X1" in metrics
        assert "BUF_X1" in metrics

    def test_combine_explicit_baseline(self, lib1_file):
        """Combining with explicit baseline should use that cell."""
        ds = Dataset()
        ds.load_files([lib1_file], normalize=False)

        # INV_X1 is in lib1
        combined = ds.combine(baseline="INV_X1")

        assert combined.entries[0].normalizer.baseline_cell.name == "INV_X1"

    def test_combine_explicit_baseline_missing(self, lib1_file):
        """Combining with missing baseline should raise ValueError."""
        ds = Dataset()
        ds.load_files([lib1_file], normalize=False)

        with pytest.raises(ValueError, match="No baseline cell found"):
            ds.combine(baseline="MISSING_CELL")


class TestDataFrameSourceFile:
    """Tests for source_file column in to_dataframe()."""

    def test_source_file_single_lib(self, lib1_file):
        """Single file should have source_file in DataFrame."""
        ds = load_files([lib1_file])
        df = ds.to_dataframe()

        assert "source_file" in df.columns
        # All cells should have same source file
        assert df["source_file"].nunique() == 1
        assert str(lib1_file) in df["source_file"].iloc[0]

    def test_source_file_combined(self, lib1_file, lib2_file):
        """Combined dataset should track source files per cell."""
        ds = Dataset()
        ds.load_files([lib1_file, lib2_file], normalize=False)
        combined = ds.combine()

        df = combined.to_dataframe()

        assert "source_file" in df.columns
        # Cells from different libs should have different source files
        inv_row = df[df["cell"] == "INV_X1"]
        nor_row = df[df["cell"] == "NOR2_X1"]

        assert str(lib1_file) in inv_row["source_file"].iloc[0]
        assert str(lib2_file) in nor_row["source_file"].iloc[0]
