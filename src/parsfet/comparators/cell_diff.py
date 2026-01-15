"""
Cell Coverage Comparison

Compare what cells exist in different libraries.
This is useful for:
- Understanding library completeness
- Finding gaps when porting designs
- Identifying unique cells
"""

from dataclasses import dataclass, field
from typing import Optional

from ..models.liberty import LibertyLibrary


@dataclass
class CellDiffResult:
    """Result of comparing cell coverage between two libraries"""

    library_a: str
    library_b: str

    only_in_a: set[str] = field(default_factory=set)
    only_in_b: set[str] = field(default_factory=set)
    common: set[str] = field(default_factory=set)

    @property
    def jaccard_similarity(self) -> float:
        """
        Jaccard index: |A ∩ B| / |A ∪ B|

        1.0 = identical cell sets
        0.0 = no overlap
        """
        intersection = len(self.common)
        union = len(self.only_in_a) + len(self.only_in_b) + len(self.common)
        return intersection / union if union > 0 else 0.0

    @property
    def coverage_a_in_b(self) -> float:
        """What fraction of A's cells are also in B?"""
        total_a = len(self.only_in_a) + len(self.common)
        return len(self.common) / total_a if total_a > 0 else 0.0

    @property
    def coverage_b_in_a(self) -> float:
        """What fraction of B's cells are also in A?"""
        total_b = len(self.only_in_b) + len(self.common)
        return len(self.common) / total_b if total_b > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "library_a": self.library_a,
            "library_b": self.library_b,
            "only_in_a": sorted(self.only_in_a),
            "only_in_b": sorted(self.only_in_b),
            "common": sorted(self.common),
            "counts": {
                "only_in_a": len(self.only_in_a),
                "only_in_b": len(self.only_in_b),
                "common": len(self.common),
            },
            "metrics": {
                "jaccard_similarity": self.jaccard_similarity,
                "coverage_a_in_b": self.coverage_a_in_b,
                "coverage_b_in_a": self.coverage_b_in_a,
            },
        }


def compare_cell_coverage(
    lib_a: LibertyLibrary,
    lib_b: LibertyLibrary,
    normalize_names: bool = True,
) -> CellDiffResult:
    """
    Compare cell coverage between two libraries.

    Args:
        lib_a: First library
        lib_b: Second library
        normalize_names: If True, normalize cell names for comparison
                        (e.g., ignore prefixes like "sky130_fd_sc_hd__")

    Returns:
        CellDiffResult with sets of unique and common cells
    """
    if normalize_names:
        cells_a = {_normalize_cell_name(name): name for name in lib_a.cells.keys()}
        cells_b = {_normalize_cell_name(name): name for name in lib_b.cells.keys()}
    else:
        cells_a = {name: name for name in lib_a.cells.keys()}
        cells_b = {name: name for name in lib_b.cells.keys()}

    keys_a = set(cells_a.keys())
    keys_b = set(cells_b.keys())

    # Use original names in the result
    only_in_a = {cells_a[k] for k in keys_a - keys_b}
    only_in_b = {cells_b[k] for k in keys_b - keys_a}
    common_keys = keys_a & keys_b
    common = {cells_a[k] for k in common_keys}

    return CellDiffResult(
        library_a=lib_a.name,
        library_b=lib_b.name,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        common=common,
    )


def _normalize_cell_name(name: str) -> str:
    """
    Normalize cell name for comparison across libraries.

    Removes common prefixes and suffixes that vary between foundries.

    Examples:
        sky130_fd_sc_hd__inv_1 -> inv_1
        INVD1BWP -> INVD1
        INV_X1_LVT -> INV_X1
    """
    # Remove common foundry prefixes
    prefixes = [
        "sky130_fd_sc_hd__",
        "sky130_fd_sc_hs__",
        "sky130_fd_sc_ms__",
        "sky130_fd_sc_ls__",
        "sky130_fd_sc_hdll__",
        "asap7_",
        "gf180mcu_",
    ]

    normalized = name
    for prefix in prefixes:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix) :]
            break

    # Remove common suffixes (Vt flavors, etc.)
    suffixes = ["_lvt", "_hvt", "_svt", "_ulvt", "bwp", "BWP"]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    return normalized.upper()


def find_equivalent_cells(
    cell_name: str,
    lib_a: LibertyLibrary,
    lib_b: LibertyLibrary,
) -> list[tuple[str, float]]:
    """
    Find cells in lib_b that are equivalent to cell_name in lib_a.

    Returns list of (cell_name, similarity_score) sorted by similarity.
    """
    if cell_name not in lib_a.cells:
        return []

    source_cell = lib_a.cells[cell_name]
    normalized_source = _normalize_cell_name(cell_name)

    candidates = []
    for target_name, target_cell in lib_b.cells.items():
        # Check name similarity
        normalized_target = _normalize_cell_name(target_name)

        if normalized_source == normalized_target:
            # Exact name match (after normalization)
            candidates.append((target_name, 1.0))
        elif normalized_source in normalized_target or normalized_target in normalized_source:
            # Partial name match
            candidates.append((target_name, 0.8))
        elif _same_function_type(normalized_source, normalized_target):
            # Same function type (e.g., both are NANDs)
            candidates.append((target_name, 0.5))

    return sorted(candidates, key=lambda x: -x[1])


def _same_function_type(name_a: str, name_b: str) -> bool:
    """Check if two cell names represent the same logical function type"""
    function_types = [
        "INV",
        "BUF",
        "NAND",
        "NOR",
        "AND",
        "OR",
        "XOR",
        "XNOR",
        "MUX",
        "AOI",
        "OAI",
        "DFF",
        "LATCH",
        "SDFF",
        "DLAT",
    ]

    for func in function_types:
        a_has = func in name_a.upper()
        b_has = func in name_b.upper()
        if a_has and b_has:
            return True

    return False
