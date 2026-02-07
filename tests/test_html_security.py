
import pytest
import re
from pathlib import Path
from parsfet.reporting.html_generator import generate_report
from parsfet.data import LibraryEntry
from parsfet.models.liberty import LibertyLibrary, Cell
from parsfet.normalizers.invd1 import INVD1Normalizer

def test_csp_nonce_present(tmp_path):
    """Verifies that the generated HTML report uses a nonce for CSP and avoids unsafe-inline for scripts."""

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

    # 1. Check for CSP meta tag presence
    csp_match = re.search(r'<meta http-equiv="Content-Security-Policy" content="(.*?)">', content)
    assert csp_match, "CSP meta tag not found"
    csp_content = csp_match.group(1)

    # 2. Extract script-src directive
    script_src_match = re.search(r"script-src ([^;]+)", csp_content)
    assert script_src_match, "script-src directive missing"
    script_src = script_src_match.group(1)

    # 3. Assert 'unsafe-inline' is NOT in script-src
    assert "'unsafe-inline'" not in script_src, "script-src should not contain 'unsafe-inline'"

    # Assert 'unsafe-eval' IS in script-src (required for Alpine.js standard build)
    assert "'unsafe-eval'" in script_src, "script-src should contain 'unsafe-eval' for Alpine.js"

    # 4. Extract nonce from script-src
    nonce_match = re.search(r"'nonce-([a-fA-F0-9]+)'", script_src)
    assert nonce_match, "script-src should contain a nonce (e.g., 'nonce-<hex>')"
    nonce_value = nonce_match.group(1)

    # 5. Verify script tags use the correct nonce
    # Find all script tags that are NOT external (src=...)
    # Actually, even external scripts might need nonce if strict-dynamic is used, but here we just check our inline scripts
    # Our template has inline scripts like <script>window.LIB_DATA = ...</script>

    script_tags = re.findall(r'<script(.*?)>', content)
    for tag_attr in script_tags:
        # Check if it has a src attribute (external)
        if 'src="' in tag_attr:
            continue

        # Check for nonce attribute
        nonce_attr_match = re.search(r'nonce="([^"]+)"', tag_attr)
        assert nonce_attr_match, f"Inline script tag missing nonce attribute: <script{tag_attr}>"
        assert nonce_attr_match.group(1) == nonce_value, f"Script nonce mismatch: expected {nonce_value}, got {nonce_attr_match.group(1)}"
