
import pytest
import csv
import io
from parsfet.reporting.csv_generator import generate_csv
from parsfet.models.export import ExportedLibrary, ExportedBaseline, ExportedCell

def test_csv_injection_prevention():
    """Verifies that cell names starting with dangerous characters are escaped in CSV output."""

    # Define malicious cell names
    injection_cases = [
        "=cmd|' /C calc'!A0",
        "+cmd|' /C calc'!A0",
        "-cmd|' /C calc'!A0",
        "@cmd|' /C calc'!A0",
        " =cmd|' /C calc'!A0",  # Leading space bypass
        "\t=cmd|' /C calc'!A0", # Tab bypass
    ]
    safe_name = "INV_X1"
    quoted_name = "'AlreadyQuoted"

    # Create mock library
    baseline = ExportedBaseline(cell="INV_X1")
    cells = {
        safe_name: ExportedCell(cell_name=safe_name),
        quoted_name: ExportedCell(cell_name=quoted_name)
    }
    for name in injection_cases:
        cells[name] = ExportedCell(cell_name=name)

    library = ExportedLibrary(
        library="TestLib",
        baseline=baseline,
        cells=cells
    )

    # Capture output
    output = io.StringIO()
    generate_csv(library, output)

    # Parse CSV output
    output.seek(0)
    reader = csv.reader(output)
    headers = next(reader)
    rows = list(reader)

    found_cases = 0
    found_quoted = False
    
    for row in rows:
        cell_name_csv = row[0]
        
        # Check injection cases
        # Logic: If it starts with ', check if the rest (or stripped rest) matches
        if cell_name_csv.startswith("'"):
            # Check for double quoting of already quoted value
            if cell_name_csv == quoted_name: 
                # Ideally we want to preserve 'AlreadyQuoted as is, or as ''AlreadyQuoted?
                # If we sanitize it, it becomes ''AlreadyQuoted.
                # If we don't, it stays 'AlreadyQuoted.
                # Let's assert based on current behavior first, or intended behavior.
                # The user suggested: "If a value already starts with ', you’ll end up with ''value."
                # "Optional guard: if value.startswith("'"): return value"
                found_quoted = True
            
            # Check identifying which injection case this is
            # We strip the leading ' added by sanitization, then check if it matches an input
            sanitized_val = cell_name_csv[1:]
            if sanitized_val in injection_cases:
                found_cases += 1
                continue
                
        # If it's one of the dangerous chars unescaped (even with whitespace), FAIL
        # Excel treats leading whitespace as formula if trimmed.
        if cell_name_csv.strip().startswith(("=", "+", "-", "@")) and cell_name_csv in injection_cases:
             pytest.fail(f"Found unescaped malicious cell name: '{cell_name_csv}'")
             
        # Check quoted name handling
        if cell_name_csv == quoted_name:
             found_quoted = True

    assert found_cases == len(injection_cases), f"Expected {len(injection_cases)} escaped injection cases, found {found_cases}"
    assert found_quoted, "Already quoted value was lost or mangled beyond recognition"
