import pytest
from parsfet.models.liberty import LibertyLibrary, Cell
from parsfet.comparators.cell_diff import compare_cell_coverage, find_equivalent_cells, CellDiffResult
from parsfet.comparators.fingerprint import create_fingerprint, compare_fingerprints, TechnologyFingerprint

@pytest.fixture
def lib_a():
    l = LibertyLibrary(name="lib_a")
    l.cells["INV_X1"] = Cell(name="INV_X1", area=1.0, pins={})
    l.cells["NAND2_X1"] = Cell(name="NAND2_X1", area=2.0, pins={})
    l.cells["DFF_X1"] = Cell(name="DFF_X1", area=4.0, pins={})
    return l

@pytest.fixture
def lib_b():
    l = LibertyLibrary(name="lib_b")
    l.cells["INV_X1"] = Cell(name="INV_X1", area=1.2, pins={}) # Same name
    l.cells["NAND2_X2"] = Cell(name="NAND2_X2", area=2.4, pins={}) # Different suffix (drive strength)
    l.cells["NOR2_X1"] = Cell(name="NOR2_X1", area=2.0, pins={}) # Unique to B
    # DFF_X1 is missing in B
    return l

def test_compare_cell_coverage(lib_a, lib_b):
    # With normalization (default)
    # INV_X1 matches INV_X1
    # NAND2_X1 should match NAND2_X2 if suffixes are stripped?
    # Let's check _normalize_cell_name logic in cell_diff.py.
    # It strips suffixes: _lvt, _hvt, etc. but X1/X2 are usually part of name or handled?
    # Actually, the suffixes list in cell_diff.py is ["_lvt", "_hvt", "_svt", "_ulvt", "bwp", "BWP"].
    # It does NOT strip _X1, _X2 drive strengths automatically in the provided code unless specifically added.
    # So NAND2_X1 and NAND2_X2 will likely be different.

    diff = compare_cell_coverage(lib_a, lib_b, normalize_names=True)

    assert diff.library_a == "lib_a"
    assert diff.library_b == "lib_b"

    # Common
    assert "INV_X1" in diff.common

    # Only in A
    assert "DFF_X1" in diff.only_in_a
    assert "NAND2_X1" in diff.only_in_a # Because X1 != X2 and X* is not stripped

    # Only in B
    assert "NOR2_X1" in diff.only_in_b
    assert "NAND2_X2" in diff.only_in_b

def test_jaccard_similarity(lib_a, lib_b):
    diff = compare_cell_coverage(lib_a, lib_b)

    # A: {INV_X1, NAND2_X1, DFF_X1}
    # B: {INV_X1, NAND2_X2, NOR2_X1}
    # Intersection: {INV_X1} (1)
    # Union: {INV_X1, NAND2_X1, DFF_X1, NAND2_X2, NOR2_X1} (5)

    assert diff.jaccard_similarity == 1/5
    assert diff.coverage_a_in_b == 1/3 # 1 common out of 3 in A
    assert diff.coverage_b_in_a == 1/3 # 1 common out of 3 in B

def test_find_equivalent_cells(lib_a, lib_b):
    # Exact match
    equivs = find_equivalent_cells("INV_X1", lib_a, lib_b)
    assert len(equivs) > 0
    assert equivs[0][0] == "INV_X1"
    assert equivs[0][1] == 1.0

    # Partial match (substring)
    # Let's add a cell to lib_b that is a substring match
    lib_b.cells["NAND2"] = Cell(name="NAND2", area=2.0, pins={})

    equivs = find_equivalent_cells("NAND2_X1", lib_a, lib_b)
    # NAND2 is in NAND2_X1 (normalized or raw)
    # If normalized: NAND2_X1 -> NAND2_X1. NAND2 -> NAND2.
    # "NAND2" in "NAND2_X1" -> True.
    assert any(e[0] == "NAND2" for e in equivs)

    # Function match
    # Since we didn't define functions in the cells, function matching won't trigger based on logic,
    # but the code uses _same_function_type which checks names!
    # "NAND" in name_a and "NAND" in name_b

    equivs_nand = find_equivalent_cells("NAND2_X1", lib_a, lib_b)
    # Should match NAND2_X2 because both have "NAND"
    names = [e[0] for e in equivs_nand]
    assert "NAND2_X2" in names

def test_fingerprint_creation(lib_a):
    # Need to set up baseline cell or it will try to detect it
    # lib_a has INV_X1, which is a standard name for baseline

    fp = create_fingerprint(lib_a)

    assert fp.name == "lib_a"
    assert fp.baseline_cell == "INV_X1"
    assert fp.total_cells == 3

    # Counts
    # Since we didn't set pins/functions, classifier returns 'unknown'
    # But for basic fingerprint (fallback), it counts by name "INV"
    # So inverter_count should be 1
    assert fp.inverter_count == 1

    # Update INV_X1 to be classified as inverter
    from parsfet.models.liberty import Pin
    lib_a.cells["INV_X1"].pins = {
        "A": Pin(name="A", direction="input"),
        "Y": Pin(name="Y", direction="output", function="!A")
    }

    fp2 = create_fingerprint(lib_a)
    assert fp2.inverter_count == 1
    assert fp2.baseline_area == 1.0

def test_fingerprint_comparison():
    fp_a = TechnologyFingerprint(
        name="A",
        baseline_cell="INV",
        baseline_area=1.0,
        baseline_d0=0.01,
        baseline_leakage=1.0,
        total_cells=100,
        combinational_cells=80,
        sequential_cells=20,
        inverter_count=10,
        buffer_count=10,
        nand_count=20,
        nor_count=20,
        dff_count=20,
        latch_count=0
    )

    fp_b = TechnologyFingerprint(
        name="B",
        baseline_cell="INV",
        baseline_area=0.5, # Smaller tech
        baseline_d0=0.005, # Faster
        baseline_leakage=2.0,
        total_cells=100,
        combinational_cells=80,
        sequential_cells=20,
        inverter_count=10,
        buffer_count=10,
        nand_count=20,
        nor_count=20,
        dff_count=20,
        latch_count=0
    )

    res = compare_fingerprints(fp_a, fp_b)

    # Cosine similarity of distribution vector
    # Since distributions are identical (counts are same), similarity should be 1.0
    assert pytest.approx(res["similarity"]["cosine"]) == 1.0

    # Metrics comparison
    comp = res["comparison"]
    # baseline_d0_ratio is fp_a / fp_b = 0.01 / 0.005 = 2.0
    assert comp["baseline_d0_ratio"] == 2.0
