
import csv
from typing import TextIO, Any
from parsfet.models.export import ExportedLibrary


def sanitize_csv_field(value: Any) -> Any:
    """Sanitizes a field to prevent CSV Injection (Formula Injection).

    If the value is a string and starts with one of the trigger characters
    (=, +, -, @), it prepends a single quote to force it to be treated as text.
    """
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def generate_csv(library: ExportedLibrary, output_file: TextIO):
    """Generates a CSV export of the library data.

    Args:
        library: Reduced ExportedLibrary object.
        output_file: File-like object to write CSV to.
    """
    writer = csv.writer(output_file)
    
    # Define Headers
    headers = [
        "cell_name",
        "cell_type",
        "drive_strength",
        "num_inputs",
        "num_outputs",
        # Raw physical
        "area_um2",
        "leakage",
        "input_cap_pf",
        # Delay (Linear Model)
        "d0_ns",
        "k_ns_per_pf",
        "delay_r2",
        "delay_fo4_resid_pct",
        # Power (Linear Model)
        "e0_unit",
        "k_unit_per_pf",
        "power_r2",
        # Normalized Ratios
        "area_ratio",
        "d0_ratio",
        "leakage_ratio",
    ]
    
    writer.writerow(headers)
    
    # Iterate and write rows
    for cell_name, cell in library.cells.items():
        # Helpers to safely get nested attributes
        delay_model = cell.delay_model
        power_model = cell.power_model
        delay_fit = cell.delay_fit_quality
        power_fit = cell.power_fit_quality
        raw = cell.raw
        
        row = [
            cell.cell_name,
            cell.cell_type,
            cell.drive_strength,
            cell.num_inputs,
            cell.num_outputs,
            # Raw
            raw.area_um2 if raw else "",
            raw.leakage if raw else "",
            raw.input_cap_pf if raw else "",
            # Delay
            delay_model.d0_ns if delay_model else "",
            delay_model.k_ns_per_pf if delay_model else "",
            delay_fit.r_squared if delay_fit else "",
            delay_fit.fo4_residual_pct if delay_fit else "",
            # Power
            power_model.e0_unit if power_model else "",
            power_model.k_unit_per_pf if power_model else "",
            power_fit.r_squared if power_fit else "",
            # Ratios
            cell.area_ratio,
            cell.d0_ratio,
            cell.leakage_ratio,
        ]

        # Sanitize all fields in the row
        row = [sanitize_csv_field(item) for item in row]

        writer.writerow(row)
