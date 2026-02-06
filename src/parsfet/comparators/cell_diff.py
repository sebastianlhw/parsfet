"""Cell Coverage Comparison.

This module provides tools to compare the cell content of two libraries.
It identifies common cells, unique cells, and calculates similarity metrics
like the Jaccard index. This is essential for:
- Checking library completeness when porting designs.
- Identifying technology-specific or custom cells.
- Verifying if two libraries are compatible or equivalent.
"""

from dataclasses import dataclass, field

from ..models.liberty import LibertyLibrary


@dataclass
class CellDiffResult:
    """Result of comparing cell coverage between two libraries.

    Attributes:
        library_a: Name of the first library.
        library_b: Name of the second library.
        only_in_a: Set of cell names present only in library A.
        only_in_b: Set of cell names present only in library B.
        common: Set of cell names present in both libraries.
    """

    library_a: str
    library_b: str

    only_in_a: set[str] = field(default_factory=set)
    only_in_b: set[str] = field(default_factory=set)
    common: set[str] = field(default_factory=set)

    @property
    def jaccard_similarity(self) -> float:
        """Calculates the Jaccard similarity index.

        J(A, B) = |A ∩ B| / |A ∪ B|
        Values range from 0.0 (no overlap) to 1.0 (identical sets).
        """
        intersection = len(self.common)
        union = len(self.only_in_a) + len(self.only_in_b) + len(self.common)
        return intersection / union if union > 0 else 0.0

    @property
    def coverage_a_in_b(self) -> float:
        """Calculates the fraction of cells from library A that are also in B."""
        total_a = len(self.only_in_a) + len(self.common)
        return len(self.common) / total_a if total_a > 0 else 0.0

    @property
    def coverage_b_in_a(self) -> float:
        """Calculates the fraction of cells from library B that are also in A."""
        total_b = len(self.only_in_b) + len(self.common)
        return len(self.common) / total_b if total_b > 0 else 0.0

    def to_dict(self) -> dict:
        """Converts the result to a dictionary for JSON serialization."""
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
    """Compares the cell sets of two libraries.

    Args:
        lib_a: The first Liberty library.
        lib_b: The second Liberty library.
        normalize_names: If True (default), removes foundry-specific prefixes and
            suffix variations (e.g., Vt flavors) to compare core logical cell names.

    Returns:
        A CellDiffResult object containing the comparison details.
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


# Common foundry prefixes
PREFIXES = [
    "sky130_fd_sc_hd__",
    "sky130_fd_sc_hs__",
    "sky130_fd_sc_ms__",
    "sky130_fd_sc_ls__",
    "sky130_fd_sc_hdll__",
    "asap7_",
    "gf180mcu_",
]

# Pre-calculated lowercased prefixes to avoid repeated lower() calls
PREFIX_TUPLES = [(p, p.lower()) for p in PREFIXES]


def _normalize_cell_name(name: str) -> str:
    """Normalizes a cell name for robust comparison.

    Strips known foundry prefixes and Vt suffixes to isolate the core cell function name.
    e.g., 'sky130_fd_sc_hd__nand2_1' -> 'NAND2_1'.
    """
    normalized = name
    name_lower = name.lower()

    for prefix, prefix_lower in PREFIX_TUPLES:
        if name_lower.startswith(prefix_lower):
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
    """Finds cells in lib_b that are likely equivalent to a cell in lib_a.

    Uses name normalization and functional heuristics to suggest matches.

    Args:
        cell_name: Name of the cell in lib_a to match.
        lib_a: The source library.
        lib_b: The target library to search.

    Returns:
        A list of tuples (target_cell_name, confidence_score), sorted by score.
        Score ranges from 0.0 to 1.0.
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
    """Checks if two cell names likely represent the same logical function."""
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
