"""Plotting module for PPA comparison charts.

This module provides functions to generate comparison plots between
two or more normalized JSON outputs from Pars-FET, focusing on
PPA metrics: Delay, Size (Area), Capacitance, and Power.


Design Philosophy:
- High data-ink ratio
- No chartjunk (3D effects, unnecessary gridlines)
- Clear axis labels with units
- 45° parity lines on scatter plots

Example:
    >>> from parsfet.plotting import load_comparison_data, plot_summary_dashboard
    >>> data = load_comparison_data(["lib_a.json", "lib_b.json"])
    >>> plot_summary_dashboard(data, Path("output/charts"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Color palette: distinguishable, grayscale-friendly
CELL_TYPE_COLORS = {
    "inverter": "#2563eb",  # blue
    "buffer": "#16a34a",    # green
    "nand": "#dc2626",      # red
    "nor": "#9333ea",       # purple
    "and": "#ea580c",       # orange
    "or": "#0891b2",        # cyan
    "xor": "#be185d",       # pink
    "mux": "#854d0e",       # brown
    "other": "#6b7280",     # gray
}

# Marker styles per cell type (for grayscale distinction)
CELL_TYPE_MARKERS = {
    "inverter": "o",
    "buffer": "s",
    "nand": "^",
    "nor": "v",
    "and": "D",
    "or": "p",
    "xor": "h",
    "mux": "*",
    "other": "x",
}


def load_comparison_data(json_paths: list[Path | str]) -> dict[str, Any]:
    """Load multiple JSON files for comparison.

    Args:
        json_paths: List of paths to normalized JSON files.

    Returns:
        Dictionary with keys 'libraries' (list of loaded JSON data)
        and 'names' (list of library names).
    """
    libraries = []
    names = []
    
    for path in json_paths:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        libraries.append(data)
        names.append(data.get("library", path.stem))
    
    return {"libraries": libraries, "names": names}


def _extract_cell_metrics(lib_data: dict) -> dict[str, dict]:
    """Extract per-cell metrics from a library JSON.

    Returns:
        Dict mapping cell_name -> {d0_ns, k_ns_per_pf, fo4_delay_ns, 
                                    area_um2, input_cap_pf, leakage, cell_type}
    """
    cells = lib_data.get("cells", {})
    fo4_op = lib_data.get("fo4_operating_point", {})
    fo4_load = fo4_op.get("load_pf", 0)
    
    result = {}
    for cell_name, cell_data in cells.items():
        raw = cell_data.get("raw", {})
        d0 = raw.get("d0_ns", 0)
        k = raw.get("k_ns_per_pf", 0)
        # FO4 delay = d0 + k * fo4_load
        fo4_delay = d0 + k * fo4_load
        
        result[cell_name] = {
            "d0_ns": d0,
            "k_ns_per_pf": k,
            "fo4_delay_ns": fo4_delay,
            "area_um2": raw.get("area_um2", 0),
            "input_cap_pf": raw.get("input_cap_pf", 0),
            "leakage": raw.get("leakage", 0),
            "cell_type": cell_data.get("cell_type", "other"),
        }
    
    return result


def _match_cells(lib_a_cells: dict, lib_b_cells: dict) -> list[str]:
    """Find cells present in both libraries."""
    return sorted(set(lib_a_cells.keys()) & set(lib_b_cells.keys()))


def plot_delay_scatter(
    data: dict[str, Any],
    output_path: Path | str,
    metric: str = "fo4_delay_ns",
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """Generate scatter plot comparing delays between two libraries.

    Args:
        data: Output from load_comparison_data().
        output_path: Path to save the PNG file.
        metric: Which metric to plot ('fo4_delay_ns', 'd0_ns', 'k_ns_per_pf').
        figsize: Figure size in inches.
    """
    if len(data["libraries"]) < 2:
        raise ValueError("Need at least 2 libraries for comparison")
    
    lib_a = _extract_cell_metrics(data["libraries"][0])
    lib_b = _extract_cell_metrics(data["libraries"][1])
    common_cells = _match_cells(lib_a, lib_b)
    
    if not common_cells:
        raise ValueError("No common cells found between libraries")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points by cell type
    for cell_type in CELL_TYPE_COLORS:
        x_vals = []
        y_vals = []
        labels = []
        
        for cell in common_cells:
            if lib_a[cell]["cell_type"] == cell_type:
                x_vals.append(lib_a[cell][metric])
                y_vals.append(lib_b[cell][metric])
                labels.append(cell)
        
        if x_vals:
            ax.scatter(
                x_vals, y_vals,
                c=CELL_TYPE_COLORS[cell_type],
                marker=CELL_TYPE_MARKERS[cell_type],
                label=cell_type,
                alpha=0.7,
                s=50,
            )
    
    # Add parity line (45°)
    all_vals = [lib_a[c][metric] for c in common_cells] + [lib_b[c][metric] for c in common_cells]
    min_val, max_val = min(all_vals), max(all_vals)
    margin = (max_val - min_val) * 0.05
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        'k--', alpha=0.5, linewidth=1, label='Parity'
    )
    
    # Labels and formatting
    metric_labels = {
        "fo4_delay_ns": "FO4 Delay (ns)",
        "d0_ns": "D₀ Intrinsic Delay (ns)",
        "k_ns_per_pf": "k Load Sensitivity (ns/pF)",
    }
    label = metric_labels.get(metric, metric)
    
    ax.set_xlabel(f"{data['names'][0]} - {label}")
    ax.set_ylabel(f"{data['names'][1]} - {label}")
    ax.set_title(f"Delay Comparison: {data['names'][0]} vs {data['names'][1]}")
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_area_scatter(
    data: dict[str, Any],
    output_path: Path | str,
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """Generate scatter plot comparing cell areas between two libraries."""
    if len(data["libraries"]) < 2:
        raise ValueError("Need at least 2 libraries for comparison")
    
    lib_a = _extract_cell_metrics(data["libraries"][0])
    lib_b = _extract_cell_metrics(data["libraries"][1])
    common_cells = _match_cells(lib_a, lib_b)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for cell_type in CELL_TYPE_COLORS:
        x_vals = []
        y_vals = []
        
        for cell in common_cells:
            if lib_a[cell]["cell_type"] == cell_type:
                x_vals.append(lib_a[cell]["area_um2"])
                y_vals.append(lib_b[cell]["area_um2"])
        
        if x_vals:
            ax.scatter(
                x_vals, y_vals,
                c=CELL_TYPE_COLORS[cell_type],
                marker=CELL_TYPE_MARKERS[cell_type],
                label=cell_type,
                alpha=0.7,
                s=50,
            )
    
    # Parity line
    all_vals = [lib_a[c]["area_um2"] for c in common_cells] + [lib_b[c]["area_um2"] for c in common_cells]
    min_val, max_val = min(all_vals), max(all_vals)
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'k--', alpha=0.5, linewidth=1, label='Parity')
    
    ax.set_xlabel(f"{data['names'][0]} - Cell Area (µm²)")
    ax.set_ylabel(f"{data['names'][1]} - Cell Area (µm²)")
    ax.set_title(f"Area Comparison: {data['names'][0]} vs {data['names'][1]}")
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cap_scatter(
    data: dict[str, Any],
    output_path: Path | str,
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """Generate scatter plot comparing input capacitances."""
    if len(data["libraries"]) < 2:
        raise ValueError("Need at least 2 libraries for comparison")
    
    lib_a = _extract_cell_metrics(data["libraries"][0])
    lib_b = _extract_cell_metrics(data["libraries"][1])
    common_cells = _match_cells(lib_a, lib_b)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for cell_type in CELL_TYPE_COLORS:
        x_vals = []
        y_vals = []
        
        for cell in common_cells:
            if lib_a[cell]["cell_type"] == cell_type:
                # Convert pF to fF for readability
                x_vals.append(lib_a[cell]["input_cap_pf"] * 1000)
                y_vals.append(lib_b[cell]["input_cap_pf"] * 1000)
        
        if x_vals:
            ax.scatter(
                x_vals, y_vals,
                c=CELL_TYPE_COLORS[cell_type],
                marker=CELL_TYPE_MARKERS[cell_type],
                label=cell_type,
                alpha=0.7,
                s=50,
            )
    
    # Parity line
    all_vals = ([lib_a[c]["input_cap_pf"] * 1000 for c in common_cells] + 
                [lib_b[c]["input_cap_pf"] * 1000 for c in common_cells])
    min_val, max_val = min(all_vals), max(all_vals)
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'k--', alpha=0.5, linewidth=1, label='Parity')
    
    ax.set_xlabel(f"{data['names'][0]} - Input Capacitance (fF)")
    ax.set_ylabel(f"{data['names'][1]} - Input Capacitance (fF)")
    ax.set_title(f"Capacitance Comparison: {data['names'][0]} vs {data['names'][1]}")
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_leakage_bar(
    data: dict[str, Any],
    output_path: Path | str,
    top_n: int = 20,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Generate bar chart comparing leakage power.

    Args:
        data: Output from load_comparison_data().
        output_path: Path to save the PNG file.
        top_n: Number of top cells to show (sorted by max leakage).
        figsize: Figure size in inches.
    """
    if len(data["libraries"]) < 2:
        raise ValueError("Need at least 2 libraries for comparison")
    
    lib_a = _extract_cell_metrics(data["libraries"][0])
    lib_b = _extract_cell_metrics(data["libraries"][1])
    common_cells = _match_cells(lib_a, lib_b)
    
    # Sort by maximum leakage across both libraries
    cell_leakages = [
        (cell, max(lib_a[cell]["leakage"], lib_b[cell]["leakage"]))
        for cell in common_cells
    ]
    cell_leakages.sort(key=lambda x: x[1], reverse=True)
    top_cells = [c[0] for c in cell_leakages[:top_n]]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(top_cells))
    width = 0.35
    
    vals_a = [lib_a[c]["leakage"] for c in top_cells]
    vals_b = [lib_b[c]["leakage"] for c in top_cells]
    
    ax.bar(x - width/2, vals_a, width, label=data["names"][0], color="#2563eb", alpha=0.8)
    ax.bar(x + width/2, vals_b, width, label=data["names"][1], color="#dc2626", alpha=0.8)
    
    ax.set_xlabel("Cell")
    ax.set_ylabel("Leakage (normalized)")
    ax.set_title(f"Leakage Comparison: {data['names'][0]} vs {data['names'][1]}")
    ax.set_xticks(x)
    ax.set_xticklabels(top_cells, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary_dashboard(
    data: dict[str, Any],
    output_dir: Path | str,
) -> list[Path]:
    """Generate all comparison charts in a single dashboard.

    Args:
        data: Output from load_comparison_data().
        output_dir: Directory to save PNG files.

    Returns:
        List of paths to generated chart files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    # FO4 Delay scatter
    delay_path = output_dir / "delay_comparison.png"
    plot_delay_scatter(data, delay_path)
    output_files.append(delay_path)
    
    # Area scatter
    area_path = output_dir / "area_comparison.png"
    plot_area_scatter(data, area_path)
    output_files.append(area_path)
    
    # Capacitance scatter
    cap_path = output_dir / "cap_comparison.png"
    plot_cap_scatter(data, cap_path)
    output_files.append(cap_path)
    
    # Leakage bar chart
    leakage_path = output_dir / "leakage_comparison.png"
    plot_leakage_bar(data, leakage_path)
    output_files.append(leakage_path)
    
    return output_files
