"""
Technology Fingerprinting

Create compact signatures that identify and characterize process technologies.
These fingerprints enable:
- Clustering similar technologies
- Quick identification of library characteristics
- ML-based process prediction
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional

from ..models.liberty import LibertyLibrary
from ..normalizers.invd1 import INVD1Normalizer


@dataclass
class TechnologyFingerprint:
    """
    Compact signature of a technology/library.

    This captures the essential characteristics of a library
    in a way that enables comparison and clustering.
    """

    name: str

    # Baseline cell info
    baseline_cell: str = ""
    baseline_area: float = 0.0
    baseline_d0: float = 0.0  # Intrinsic delay (D₀)
    baseline_k: float = 0.0   # Load slope (k)
    baseline_leakage: float = 0.0

    # Statistical summaries of normalized metrics
    mean_area_ratio: float = 0.0
    mean_d0_ratio: float = 0.0  # Intrinsic delay ratio
    mean_k_ratio: float = 0.0   # Load slope ratio
    mean_leakage_ratio: float = 0.0

    std_area_ratio: float = 0.0
    std_d0_ratio: float = 0.0
    std_k_ratio: float = 0.0
    std_leakage_ratio: float = 0.0

    # Cell count metrics
    total_cells: int = 0
    combinational_cells: int = 0
    sequential_cells: int = 0

    # Function type counts
    inverter_count: int = 0
    buffer_count: int = 0
    nand_count: int = 0
    nor_count: int = 0
    aoi_count: int = 0
    oai_count: int = 0
    mux_count: int = 0
    dff_count: int = 0
    latch_count: int = 0

    # Drive strength diversity
    min_drive_area: float = 0.0  # Smallest cell area (relative to baseline)
    max_drive_area: float = 0.0  # Largest cell area

    # Additional metadata
    process_node: Optional[str] = None
    foundry: Optional[str] = None
    vt_flavor: Optional[str] = None

    def to_vector(self) -> list[float]:
        """
        Convert to feature vector for ML.

        The vector contains normalized metrics that work well
        for clustering and classification.
        """
        return [
            self.mean_area_ratio,
            self.mean_d0_ratio,
            self.mean_k_ratio,
            self.mean_leakage_ratio,
            self.std_area_ratio,
            self.std_d0_ratio,
            self.std_k_ratio,
            self.std_leakage_ratio,
            float(self.total_cells) / 1000.0,  # Normalize cell count
            float(self.combinational_cells) / max(1, self.total_cells),
            float(self.sequential_cells) / max(1, self.total_cells),
            float(self.inverter_count) / max(1, self.total_cells),
            float(self.nand_count) / max(1, self.total_cells),
            float(self.dff_count) / max(1, self.total_cells),
            self.max_drive_area / max(1.0, self.min_drive_area),
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "baseline": {
                "cell": self.baseline_cell,
                "area": self.baseline_area,
                "d0_ns": self.baseline_d0,
                "k_ns_per_pf": self.baseline_k,
                "leakage": self.baseline_leakage,
            },
            "normalized_stats": {
                "area": {
                    "mean": self.mean_area_ratio,
                    "std": self.std_area_ratio,
                },
                "d0": {
                    "mean": self.mean_d0_ratio,
                    "std": self.std_d0_ratio,
                },
                "k": {
                    "mean": self.mean_k_ratio,
                    "std": self.std_k_ratio,
                },
                "leakage": {
                    "mean": self.mean_leakage_ratio,
                    "std": self.std_leakage_ratio,
                },
            },
            "cell_counts": {
                "total": self.total_cells,
                "combinational": self.combinational_cells,
                "sequential": self.sequential_cells,
            },
            "function_types": {
                "inverter": self.inverter_count,
                "buffer": self.buffer_count,
                "nand": self.nand_count,
                "nor": self.nor_count,
                "aoi": self.aoi_count,
                "oai": self.oai_count,
                "mux": self.mux_count,
                "dff": self.dff_count,
                "latch": self.latch_count,
            },
            "drive_diversity": {
                "min_area_ratio": self.min_drive_area,
                "max_area_ratio": self.max_drive_area,
            },
            "metadata": {
                "process_node": self.process_node,
                "foundry": self.foundry,
                "vt_flavor": self.vt_flavor,
            },
        }


def create_fingerprint(library: LibertyLibrary) -> TechnologyFingerprint:
    """
    Generate fingerprint for a library.

    Args:
        library: Parsed Liberty library

    Returns:
        TechnologyFingerprint capturing library characteristics
    """
    # Create normalizer to get baseline and normalized metrics
    try:
        normalizer = INVD1Normalizer(library)
        metrics = normalizer.normalize_all()
    except ValueError:
        # No baseline cell found - create basic fingerprint
        return _create_basic_fingerprint(library)

    fp = TechnologyFingerprint(name=library.name)

    # Baseline info
    fp.baseline_cell = normalizer.baseline.cell_name
    fp.baseline_area = normalizer.baseline.area
    fp.baseline_d0 = normalizer.baseline.d0
    fp.baseline_k = normalizer.baseline.k
    fp.baseline_leakage = normalizer.baseline.leakage

    # Compute statistics
    areas = [m.area_ratio for m in metrics.values() if m.area_ratio > 0]
    d0s = [m.d0_ratio for m in metrics.values() if m.d0_ratio > 0]
    ks = [m.k_ratio for m in metrics.values() if m.k_ratio > 0]
    leakages = [m.leakage_ratio for m in metrics.values() if m.leakage_ratio > 0]

    if areas:
        fp.mean_area_ratio = statistics.mean(areas)
        fp.std_area_ratio = statistics.stdev(areas) if len(areas) > 1 else 0.0
        fp.min_drive_area = min(areas)
        fp.max_drive_area = max(areas)

    if d0s:
        fp.mean_d0_ratio = statistics.mean(d0s)
        fp.std_d0_ratio = statistics.stdev(d0s) if len(d0s) > 1 else 0.0

    if ks:
        fp.mean_k_ratio = statistics.mean(ks)
        fp.std_k_ratio = statistics.stdev(ks) if len(ks) > 1 else 0.0

    if leakages:
        fp.mean_leakage_ratio = statistics.mean(leakages)
        fp.std_leakage_ratio = statistics.stdev(leakages) if len(leakages) > 1 else 0.0

    # Cell counts
    fp.total_cells = len(library.cells)
    fp.sequential_cells = sum(1 for c in library.cells.values() if c.is_sequential)
    fp.combinational_cells = fp.total_cells - fp.sequential_cells

    # Function type counts
    for name in library.cells.keys():
        name_upper = name.upper()
        if "INV" in name_upper:
            fp.inverter_count += 1
        elif "BUF" in name_upper:
            fp.buffer_count += 1
        if "NAND" in name_upper:
            fp.nand_count += 1
        if "NOR" in name_upper:
            fp.nor_count += 1
        if "AOI" in name_upper:
            fp.aoi_count += 1
        if "OAI" in name_upper:
            fp.oai_count += 1
        if "MUX" in name_upper:
            fp.mux_count += 1
        if "DFF" in name_upper or "SDFF" in name_upper:
            fp.dff_count += 1
        if "LAT" in name_upper:
            fp.latch_count += 1

    # Metadata from library
    fp.process_node = library.process_node
    fp.foundry = library.foundry
    if library.vt_flavor:
        fp.vt_flavor = library.vt_flavor.value

    return fp


def _create_basic_fingerprint(library: LibertyLibrary) -> TechnologyFingerprint:
    """Create fingerprint when no baseline cell is available"""
    fp = TechnologyFingerprint(name=library.name)

    # Cell counts only
    fp.total_cells = len(library.cells)
    fp.sequential_cells = sum(1 for c in library.cells.values() if c.is_sequential)
    fp.combinational_cells = fp.total_cells - fp.sequential_cells

    # Function type counts
    for name in library.cells.keys():
        name_upper = name.upper()
        if "INV" in name_upper:
            fp.inverter_count += 1
        if "BUF" in name_upper:
            fp.buffer_count += 1
        if "NAND" in name_upper:
            fp.nand_count += 1
        if "NOR" in name_upper:
            fp.nor_count += 1
        if "DFF" in name_upper:
            fp.dff_count += 1

    return fp


def compare_fingerprints(fp_a: TechnologyFingerprint, fp_b: TechnologyFingerprint) -> dict:
    """
    Compare two fingerprints and compute similarity metrics.

    Returns:
        Dictionary with comparison results
    """
    # Vector similarity (cosine)
    vec_a = fp_a.to_vector()
    vec_b = fp_b.to_vector()

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5

    cosine_sim = dot / (mag_a * mag_b) if (mag_a > 0 and mag_b > 0) else 0.0

    # Euclidean distance
    euclidean_dist = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5

    # Specific metric comparisons
    return {
        "fingerprint_a": fp_a.name,
        "fingerprint_b": fp_b.name,
        "similarity": {
            "cosine": cosine_sim,
            "euclidean_distance": euclidean_dist,
        },
        "comparison": {
            "area_ratio_diff": fp_a.mean_area_ratio - fp_b.mean_area_ratio,
            "d0_ratio_diff": fp_a.mean_d0_ratio - fp_b.mean_d0_ratio,
            "k_ratio_diff": fp_a.mean_k_ratio - fp_b.mean_k_ratio,
            "cell_count_diff": fp_a.total_cells - fp_b.total_cells,
            "baseline_d0_ratio": fp_a.baseline_d0 / fp_b.baseline_d0
            if fp_b.baseline_d0 > 0
            else 0,
        },
    }

