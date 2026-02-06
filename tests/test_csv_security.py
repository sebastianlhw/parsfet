import pytest
import csv
import io
from parsfet.reporting.csv_generator import generate_csv
from parsfet.models.export import ExportedLibrary, ExportedBaseline, ExportedCell

def test_csv_injection_prevention():
    """Verifies that cell names starting with dangerous characters are escaped in CSV output."""

    # Define a malicious cell name
    malicious_name = "=cmd|' /C calc'!A0"
    safe_name = "INV_X1"

    # Create mock library
    baseline = ExportedBaseline(cell="INV_X1")
    cells = {
        malicious_name: ExportedCell(cell_name=malicious_name),
        safe_name: ExportedCell(cell_name=safe_name)
    }

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

    # We want to ensure that the value *read back* starts with ' if the original started with =

    malicious_row_escaped = [r for r in rows if r[0] == f"'{malicious_name}"]
    malicious_row_unescaped = [r for r in rows if r[0] == malicious_name]

    if malicious_row_unescaped:
        pytest.fail(f"Found unescaped malicious cell name: {malicious_name}")

    assert len(malicious_row_escaped) == 1, "Malicious cell name should be escaped with a single quote"
