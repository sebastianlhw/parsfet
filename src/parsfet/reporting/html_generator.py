
import json
import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import asdict

from parsfet.normalizers.invd1 import INVD1Normalizer
from parsfet.models.liberty import Cell, LibertyLibrary

def interpolate_1d_at_slew(table, target_slew):
    """Interpolates a 2D table to get 1D curve at target slew.
    Returns: (loads_list, values_list)
    """
    if not table or not table.values:
        return None, None
        
    slew_axis = np.array(table.index_1)
    values = np.array(table.values)
    
    idx = np.searchsorted(slew_axis, target_slew)
    
    if idx == 0:
        y_values = values[0, :]
    elif idx >= len(slew_axis):
        y_values = values[-1, :]
    else:
        s0, s1 = slew_axis[idx-1], slew_axis[idx]
        w1 = (target_slew - s0) / (s1 - s0)
        w0 = 1.0 - w1
        y_values = w0 * values[idx-1, :] + w1 * values[idx, :]
        
    # JSON serialization requires native lists/floats, not numpy types
    return list(table.index_2), [float(v) for v in y_values]

def generate_report(library: LibertyLibrary, normalizer: INVD1Normalizer, output_path: Path):
    """Generates the interactive HTML report."""
    
    # 1. Build Data Structure
    # Use canonical units (ns) from baseline metrics, NOT raw units
    target_slew = normalizer.baseline.fo4_slew
    
    cells_data = []
    
    for cell_name, cell in library.cells.items():
        # Get Normalized Metrics (re-calculate or access cache if possible)
        # Note: We assume normalizer is pre-configured/loaded
        try:
            metrics = normalizer.normalize(cell)
            metrics_dict = metrics.to_dict()
        except Exception:
            continue # Skip problematic cells
            
        # --- Prepare Delay Plot Data ---
        d0, k_d, r2_d = cell.linear_delay_model(target_slew)
        delay_scatters = []
        max_load = 0.0
        
        # We need timing arcs for raw data.
        # Note: If parsed with include_timing=False, this will be empty.
        if hasattr(cell, 'timing_arcs'):
            for i, arc in enumerate(cell.timing_arcs):
                label_base = f"{arc.related_pin or 'In'} ({arc.timing_type})"
                for table_name, table in [("Rise", arc.cell_rise), ("Fall", arc.cell_fall)]:
                    loads, values = interpolate_1d_at_slew(table, target_slew)
                    if loads:
                        max_load = max(max_load, max(loads))
                        delay_scatters.append({
                            "name": f"{label_base} {table_name}",
                            "x": loads,
                            "y": values
                        })
        
        # Model Line (Delay)
        if max_load == 0: max_load = 0.1 # Default if no arcs
        x_model = [0, max_load * 1.1]
        y_model = [d0 + k_d * x for x in x_model]
        
        delay_plot = {
            "scatters": delay_scatters,
            "model": {"x": x_model, "y": y_model}
        }
        
        # --- Prepare Power Plot Data ---
        e0, k_p, r2_p = cell.linear_power_model(target_slew)
        power_scatters = []
        max_load_p = 0.0
        
        if hasattr(cell, 'power_arcs'):
            for i, arc in enumerate(cell.power_arcs):
                label_base = f"{arc.related_pin or 'In'}"
                for table_name, table in [("Rise", arc.rise_power), ("Fall", arc.fall_power)]:
                    loads, values = interpolate_1d_at_slew(table, target_slew)
                    if loads:
                        max_load_p = max(max_load_p, max(loads))
                        power_scatters.append({
                            "name": f"{label_base} {table_name}",
                            "x": loads,
                            "y": values
                        })

        # Model Line (Power)
        if max_load_p == 0: max_load_p = max_load # usage fallback
        x_model_p = [0, max_load_p * 1.1]
        y_model_p = [e0 + k_p * x for x in x_model_p]
        
        power_plot = {
            "scatters": power_scatters,
            "model": {"x": x_model_p, "y": y_model_p}
        }

        # Combine
        cells_data.append({
            "name": cell_name,
            "type": metrics.cell_type,
            "delay_r2": r2_d,
            "power_r2": r2_p,
            "metrics": metrics_dict, # Full details for the JSON viewer
            "plots": {
                "delay": delay_plot,
                "power": power_plot
            }
        })

    # Sort by name
    cells_data.sort(key=lambda x: x["name"])

    # Extract Baseline LUT (Transition) for Methodology Chart
    baseline_lut = None
    if normalizer.baseline_cell and hasattr(normalizer.baseline_cell, 'timing_arcs'):
        # Find the first arc with a transition table
        for arc in normalizer.baseline_cell.timing_arcs:
            table = arc.rise_transition or arc.fall_transition
            if table and table.values:
                # Normalize units to canonical (ns, pF)
                norm = library.unit_normalizer
                input_slews = [norm.normalize_time(v) for v in table.index_1]
                loads = [norm.normalize_capacitance(v) for v in table.index_2]
                values = [[norm.normalize_time(v) for v in row] for row in table.values]

                baseline_lut = {
                    "name": f"{arc.related_pin} -> {normalizer.baseline.cell_name} ({'Rise' if arc.rise_transition else 'Fall'} Trans)",
                    "input_slews": input_slews,
                    "loads": loads,
                    "values": values
                }
                break

    # Global Payload
    payload = {
        "library_name": library.name,
        "baseline_name": normalizer.baseline.cell_name, # Kept for backward compat if needed
        "baseline": asdict(normalizer.baseline), # Full baseline object
        "baseline_lut": baseline_lut,
        "target_slew_ns": target_slew,
        "cells": cells_data
    }
    
    # 2. Render Template
    template_path = Path(__file__).parent.parent / "templates" / "report_master_detail.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
        
    template_content = template_path.read_text(encoding="utf-8")
    
    # Simple JSON injection
    json_str = json.dumps(payload, default=str)
    
    final_html = template_content.replace(
        "{{ lib_data_json }}", 
        json_str
    )
    
    output_path.write_text(final_html, encoding="utf-8")
