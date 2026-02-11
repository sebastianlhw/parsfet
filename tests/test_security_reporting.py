
import json
import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

from parsfet.reporting.html_generator import generate_report
from unittest.mock import patch

# Mocks for the test to avoid full dependency on library parsing
@dataclass
class MockCell:
    name: str
    timing_arcs: list = field(default_factory=list)
    power_arcs: list = field(default_factory=list)

    def linear_delay_model(self, slew):
        return 0.1, 0.01, 0.99

    def linear_power_model(self, slew):
        return 0.1, 0.01, 0.99

@dataclass
class MockLibrary:
    name: str
    cells: Dict[str, MockCell] = field(default_factory=dict)
    unit_normalizer: Any = None

@dataclass
class MockBaseline:
    cell_name: str = "INV_X1"
    fo4_slew: float = 0.02
    input_cap: float = 0.001

@dataclass
class MockMetrics:
    cell_type: str = "INV"
    def to_dict(self):
        return {"area_ratio": 1.0}

@dataclass
class MockNormalizer:
    baseline: MockBaseline = field(default_factory=MockBaseline)
    baseline_cell: MockCell = field(default_factory=lambda: MockCell(name="INV_X1"))
    fo4_load: float = 0.004

    def normalize(self, cell):
        return MockMetrics()

@dataclass
class MockEntry:
    library: MockLibrary
    normalizer: MockNormalizer

@patch("parsfet.reporting.html_generator.validate_assets")
def test_html_report_xss_prevention(mock_validate, tmp_path):
    """Verifies that malicious payloads in cell names are escaped in the HTML report."""

    # Payload that would break out of script tag if unescaped
    # Payload with multiple dangerous characters
    malicious_payload = "<script>alert('XSS & more')</script>"

    cell = MockCell(name=malicious_payload)
    library = MockLibrary(name="MaliciousLib", cells={malicious_payload: cell})

    # Mock unit normalizer
    class MockUnitNormalizer:
        def normalize_time(self, t): return t
        def normalize_capacitance(self, c): return c
    library.unit_normalizer = MockUnitNormalizer()

    entry = MockEntry(library=library, normalizer=MockNormalizer())

    output_path = tmp_path / "xss_report.html"

    generate_report([entry], output_path)

    content = output_path.read_text(encoding="utf-8")

    # The malicious sequence should NOT be present literally as a contiguous string
    # (We cannot just check for "<script>" because the template itself has valid script tags)
    assert malicious_payload not in content
    
    # The escaped version should be present
    # < -> \u003c, > -> \u003e, & -> \u0026
    expected_escaped = r"\u003cscript\u003ealert('XSS \u0026 more')\u003c/script\u003e"
    assert expected_escaped in content

    # Verify that the JSON is valid and decodes correctly
    start_marker = "window.LIB_DATA = "
    start_idx = content.find(start_marker)
    assert start_idx != -1

    # Find the end of the JSON object (followed by ;</script>)
    end_marker = ";</script>"
    end_idx = content.find(end_marker, start_idx)
    assert end_idx != -1

    json_str = content[start_idx + len(start_marker):end_idx]

    data = json.loads(json_str)

    # Check that the name comes back correctly
    # Note: malicious_payload has raw < characters.
    # The JSON string has \u003c.
    # json.loads should decode \u003c back to <.
    found_name = data["libraries"][0]["cells"][0]["name"]
    assert found_name == malicious_payload

@patch("parsfet.reporting.html_generator.validate_assets")
def test_json_unicode_separators_escaped(mock_validate, tmp_path):
    """Verifies that U+2028 and U+2029 are escaped to prevent JS syntax errors."""

    # Payload with Line Separator and Paragraph Separator
    payload_ls = "Line\u2028Break"
    payload_ps = "Para\u2029Graph"

    cell_ls = MockCell(name=payload_ls)
    cell_ps = MockCell(name=payload_ps)

    library = MockLibrary(name="UnicodeLib", cells={
        payload_ls: cell_ls,
        payload_ps: cell_ps
    })

    # Mock unit normalizer
    class MockUnitNormalizer:
        def normalize_time(self, t): return t
        def normalize_capacitance(self, c): return c
    library.unit_normalizer = MockUnitNormalizer()

    entry = MockEntry(library=library, normalizer=MockNormalizer())

    output_path = tmp_path / "unicode_report.html"
    generate_report([entry], output_path)

    content = output_path.read_text(encoding="utf-8")

    # The literal characters should NOT be present in the HTML source
    # (Because they are invalid in JS string literals)
    assert "\u2028" not in content
    assert "\u2029" not in content

    # The escaped versions should be present
    assert r"\u2028" in content
    assert r"\u2029" in content

    # Verify JSON validity
    start_marker = "window.LIB_DATA = "
    start_idx = content.find(start_marker)
    end_marker = ";</script>"
    end_idx = content.find(end_marker, start_idx)

    json_str = content[start_idx + len(start_marker):end_idx]
    data = json.loads(json_str)

    names = [c["name"] for c in data["libraries"][0]["cells"]]
    assert payload_ls in names
    assert payload_ps in names
