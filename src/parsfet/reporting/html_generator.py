
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
    values = np.array(table.values) # Ensure 2D
    if len(values.shape) == 1:
         return None, None

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

def interpolate_1d_at_load(table, target_load):
    """Interpolates a 2D table to get Output Slew vs Input Slew at fixed Load.
    Returns: (input_slews_list, output_slews_list)
    """
    if not table or not table.values:
        return None, None
        
    # Check if table is 2D
    if not table.index_2 or not table.is_2d:
        return None, None
        
    load_axis = np.array(table.index_2)
    slew_axis = np.array(table.index_1)
    values = np.array(table.values) # Shape: (num_slews, num_loads)
    if len(values.shape) == 1:
        return None, None
    
    idx = np.searchsorted(load_axis, target_load)
    
    output_slews = []
    
    # Calculate weights once since load is constant
    if idx == 0:
        w1, w0 = 0.0, 1.0
        c_idx = 0
    elif idx >= len(load_axis):
        w1, w0 = 0.0, 1.0 # Clamp to end
        c_idx = len(load_axis) - 1
    else:
        l0, l1 = load_axis[idx-1], load_axis[idx]
        w1 = (target_load - l0) / (l1 - l0)
        w0 = 1.0 - w1
        c_idx = idx
        
    for r in range(len(slew_axis)):
        row_vals = values[r, :]
        if idx == 0 or idx >= len(load_axis):
            val = row_vals[c_idx]
        else:
            val = w0 * row_vals[idx-1] + w1 * row_vals[idx]
        output_slews.append(val)
        
    return list(table.index_1), output_slews

def generate_report(entries: list[Any], output_path: Path):
    """Generates the interactive HTML report for one or more libraries.
    
    Args:
        entries: List of DatasetEntry objects (from parsfet.data).
        output_path: Path to write the HTML file.
    """
    
    # 1. Build Data Structure
    libraries_data = []

    for entry in entries:
        library = entry.library
        normalizer = entry.normalizer
        
        if not normalizer:
            continue
            
        # Use canonical units (ns) from baseline metrics, NOT raw units
        target_slew = normalizer.baseline.fo4_slew
        
        cells_data = []
        
        for cell_name, cell in library.cells.items():
            # Get Normalized Metrics (re-calculate or access cache if possible)
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
            # Find the first arc with ANY transition table
            target_arc = None
            for arc in normalizer.baseline_cell.timing_arcs:
                if (arc.rise_transition and arc.rise_transition.values) or \
                   (arc.fall_transition and arc.fall_transition.values):
                    target_arc = arc
                    break
            
            if target_arc:
                # Get the PRIMARY table for the main visualization (usually Rise)
                primary_table = target_arc.rise_transition or target_arc.fall_transition
                
                # Normalize units to canonical (ns, pF)
                norm = library.unit_normalizer
                input_slews = [norm.normalize_time(v) for v in primary_table.index_1]
                loads = [norm.normalize_capacitance(v) for v in primary_table.index_2]
                values = [[norm.normalize_time(v) for v in row] for row in primary_table.values]

                # --- Calculate Convergence Curve (S_out vs S_in) at FO4 Load ---
                # CRITICAL: The FO4 point is derived from the AVERAGE of Rise and Fall transitions.
                # To make the chart accurate, we must plot the AVERAGE transfer curve.
                
                raw_fo4_load = normalizer.fo4_load # Raw unit
                
                # Get Rise Curve (if exists)
                rise_in, rise_out = [], []
                if target_arc.rise_transition:
                    rise_in, rise_out = interpolate_1d_at_load(target_arc.rise_transition, raw_fo4_load)
                
                # Get Fall Curve (if exists)
                fall_in, fall_out = [], []
                if target_arc.fall_transition:
                    fall_in, fall_out = interpolate_1d_at_load(target_arc.fall_transition, raw_fo4_load)
                
                # Calculate Average Curve
                avg_out = []
                avg_in = rise_in if rise_in else fall_in # Assuming generic axis is same
                
                if rise_out and fall_out:
                    # element-wise average
                    avg_out = [(r + f) / 2.0 for r, f in zip(rise_out, fall_out)]
                elif rise_out:
                    avg_out = rise_out
                elif fall_out:
                    avg_out = fall_out
                
                # Normalize Convergence Data
                if avg_in:
                    conv_in = [norm.normalize_time(v) for v in avg_in]
                    conv_out = [norm.normalize_time(v) for v in avg_out]
                else:
                    conv_in, conv_out = [], []

                baseline_lut = {
                    "name": f"{target_arc.related_pin} -> {normalizer.baseline.cell_name} (Average Trans)",
                    "input_slews": input_slews,
                    "loads": loads,
                    "values": values, # Still showing the primary surface in the "Matrix", which is fine
                    "convergence": {
                        "input_slew": conv_in,
                        "output_slew": conv_out
                    }
                }
        
        libraries_data.append({
            "library_name": library.name,
            "baseline_name": normalizer.baseline.cell_name,
            "baseline": asdict(normalizer.baseline),
            "baseline_lut": baseline_lut,
            "target_slew_ns": target_slew,
            "cells": cells_data
        })

    # Global Payload
    # Wrap in "libraries" array. 
    # For backward compatibility with the template (if unchanged), we select the FIRST library as primary 'data',
    # but we ALSO provide 'libraries' for the multi-lib switcher.
    if not libraries_data:
        # Fallback empty payload
        payload = {"libraries": [], "cells": []}
    else:
        payload = {
            # Primary view (first library) - keeps existing template working initially
            **libraries_data[0], 
            "libraries": libraries_data
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
