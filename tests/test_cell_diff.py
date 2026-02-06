import pytest
from parsfet.comparators.cell_diff import _normalize_cell_name, compare_cell_coverage
from parsfet.models.liberty import LibertyLibrary, Cell

def test_normalize_cell_name_prefixes():
    # Test common prefixes removal
    assert _normalize_cell_name("sky130_fd_sc_hd__nand2_1") == "NAND2_1"
    assert _normalize_cell_name("asap7_inv_x1") == "INV_X1"
    assert _normalize_cell_name("gf180mcu_nor2") == "NOR2"

    # Test case insensitivity for prefixes
    assert _normalize_cell_name("SKY130_FD_SC_HD__nand2_1") == "NAND2_1"
    assert _normalize_cell_name("AsAp7_inv_x1") == "INV_X1"

def test_normalize_cell_name_suffixes():
    # Test common suffixes removal
    assert _normalize_cell_name("nand2_1_lvt") == "NAND2_1"
    assert _normalize_cell_name("inv_x1_hvt") == "INV_X1"
    assert _normalize_cell_name("buf_x2_svt") == "BUF_X2"
    assert _normalize_cell_name("dffbwp") == "DFF"
    assert _normalize_cell_name("dff_bwp") == "DFF_"

    # Test combined prefix and suffix
    assert _normalize_cell_name("sky130_fd_sc_hd__nand2_1_lvt") == "NAND2_1"

def test_normalize_cell_name_no_change():
    # Test names that shouldn't change (except uppercasing)
    assert _normalize_cell_name("NAND2_X1") == "NAND2_X1"
    assert _normalize_cell_name("unknown_prefix_cell") == "UNKNOWN_PREFIX_CELL"

def test_compare_cell_coverage():
    # Create two dummy libraries
    lib_a = LibertyLibrary(
        name="lib_a",
        cells={
            "sky130_fd_sc_hd__nand2_1": Cell(name="sky130_fd_sc_hd__nand2_1"),
            "sky130_fd_sc_hd__inv_1": Cell(name="sky130_fd_sc_hd__inv_1"),
            "sky130_fd_sc_hd__xor2_1": Cell(name="sky130_fd_sc_hd__xor2_1"),
        }
    )

    lib_b = LibertyLibrary(
        name="lib_b",
        cells={
            "asap7_nand2_1": Cell(name="asap7_nand2_1"), # Should match NAND2_1
            "asap7_inv_1": Cell(name="asap7_inv_1"),   # Should match INV_1
            "asap7_nor2_1": Cell(name="asap7_nor2_1"),  # Unique to B
        }
    )

    result = compare_cell_coverage(lib_a, lib_b, normalize_names=True)

    # Normalized names:
    # A: NAND2_1, INV_1, XOR2_1
    # B: NAND2_1, INV_1, NOR2_1

    # Common: NAND2_1, INV_1 (2)
    # Only A: XOR2_1 (1)
    # Only B: NOR2_1 (1)

    assert len(result.common) == 2
    # The result.common contains original names from lib_a for common keys
    assert "sky130_fd_sc_hd__nand2_1" in result.common
    assert "sky130_fd_sc_hd__inv_1" in result.common

    assert len(result.only_in_a) == 1
    assert "sky130_fd_sc_hd__xor2_1" in result.only_in_a

    assert len(result.only_in_b) == 1
    assert "asap7_nor2_1" in result.only_in_b

    # Check coverage metrics
    # Jaccard = 2 / 4 = 0.5
    assert result.jaccard_similarity == 0.5
    # A in B = 2 / 3
    assert result.coverage_a_in_b == pytest.approx(2/3)
    # B in A = 2 / 3
    assert result.coverage_b_in_a == pytest.approx(2/3)

def test_compare_cell_coverage_no_normalization():
    lib_a = LibertyLibrary(
        name="lib_a",
        cells={
            "nand2_1": Cell(name="nand2_1"),
        }
    )
    lib_b = LibertyLibrary(
        name="lib_b",
        cells={
            "NAND2_1": Cell(name="NAND2_1"),
        }
    )

    result = compare_cell_coverage(lib_a, lib_b, normalize_names=False)
    assert len(result.common) == 0
    assert len(result.only_in_a) == 1
    assert len(result.only_in_b) == 1
