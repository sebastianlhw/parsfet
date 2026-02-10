
import pytest
import re
from pathlib import Path
from parsfet.reporting.html_generator import generate_report
from parsfet.models.liberty import LibertyLibrary, Cell
from parsfet.normalizers.invd1 import INVD1Normalizer

def test_sri_hashes_present(tmp_path):
    """Verifies that the external scripts have the correct SRI integrity attributes."""

    # Create a minimal dummy library/entry to generate a report
    lib = LibertyLibrary(name="TestLib", technology="cmos")
    cell = Cell(name="INV_X1", area=1.0)
    lib.cells["INV_X1"] = cell

    # We need a normalizer attached to the entry
    normalizer = INVD1Normalizer(lib, baseline_name="INV_X1")
    # Mock linear models to avoid errors during generation
    cell.linear_delay_model = lambda slew: (0.1, 0.5, 0.99)
    cell.linear_power_model = lambda slew: (0.01, 0.2, 0.95)

    entry = type("Entry", (), {"library": lib, "normalizer": normalizer})

    output_file = tmp_path / "sri_report.html"

    # Generate report
    generate_report([entry], output_file)

    content = output_file.read_text(encoding="utf-8")

    # Define expected hashes
    expected_integrity = {
        "https://cdn.tailwindcss.com/3.4.1": "sha384-SOMLQz+nKv/ORIYXo3J3NrWJ33oBgGvkHlV9t8i70QVLq8ZtST9Np1gDsVUkk4xN",
        "https://cdn.plot.ly/plotly-2.27.0.min.js": "sha384-Hl48Kq2HifOWdXEjMsKo6qxqvRLTYqIGbvlENBmkHAxZKIGCXv43H6W1jA671RzC",
        "https://cdn.jsdelivr.net/npm/alpinejs@3.13.3/dist/cdn.min.js": "sha384-Rpe/8orFUm5Q1GplYBHxbuA8Az8O8C5sAoOsdbRWkqPjKFaxPgGZipj4zeHL7lxX"
    }

    for url, hash_val in expected_integrity.items():
        # Construct the expected pattern: src="..." integrity="..."
        # Attributes order might not be guaranteed by HTML parsers but here we check the raw string or use regex
        # The template has: src="..." integrity="..." crossorigin="..."

        # Regex to find the script tag with the src
        pattern = rf'<script[^>]*src="{re.escape(url)}"[^>]*>'
        match = re.search(pattern, content)
        assert match, f"Script tag for {url} not found"

        tag_content = match.group(0)

        # Check integrity attribute
        integrity_match = re.search(r'integrity="([^"]+)"', tag_content)
        assert integrity_match, f"Integrity attribute missing for {url}"
        assert integrity_match.group(1) == hash_val, f"Integrity hash mismatch for {url}"

        # Check crossorigin attribute
        assert 'crossorigin="anonymous"' in tag_content, f"crossorigin='anonymous' missing for {url}"
