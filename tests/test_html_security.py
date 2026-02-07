
import pytest
from pathlib import Path
from parsfet.reporting.html_generator import generate_report
from parsfet.data import LibraryEntry
from parsfet.models.liberty import LibertyLibrary, Cell
from parsfet.normalizers.invd1 import INVD1Normalizer

def test_csp_header_present(tmp_path):
    """Verifies that the generated HTML report contains a Content-Security-Policy meta tag."""

    # Create a minimal dummy library/entry
    lib = LibertyLibrary(name="TestLib", technology="cmos")
    cell = Cell(name="INV_X1", area=1.0)
    lib.cells["INV_X1"] = cell

    # We need a normalizer attached to the entry
    normalizer = INVD1Normalizer(lib, baseline_name="INV_X1")
    entry = LibraryEntry(library=lib, normalizer=normalizer)

    output_file = tmp_path / "report.html"

    # Generate report
    generate_report([entry], output_file)

    content = output_file.read_text(encoding="utf-8")

    # Assert CSP is present
    assert '<meta http-equiv="Content-Security-Policy"' in content

    # Assert specific directives
    assert "default-src 'none'" in content
    assert "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.plot.ly https://cdn.jsdelivr.net" in content
    assert "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com" in content
    assert "font-src https://fonts.gstatic.com" in content
    assert "img-src 'self' data:" in content
