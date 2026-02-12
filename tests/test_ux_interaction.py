
import pytest
import sys
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from parsfet.reporting.html_generator import generate_report
from parsfet.models.liberty import LibertyLibrary, Cell
from playwright.sync_api import Page, expect

# --- Minimal Mocks ---

@dataclass
class MockMetrics:
    cell_type: str = "inverter"
    def to_dict(self):
        return {"area_ratio": 1.0, "d0_ratio": 1.0, "p0_ratio": 1.0, "k_ratio": 1.0, "e0_ratio": 1.0, "k_power_ratio": 1.0, "raw": {"d0_ns": 0.1, "e0_unit": 0.01, "area_um2": 1.0}}

@dataclass
class MockBaseline:
    cell_name: str = "INV_X1"
    fo4_slew: float = 0.01
    input_cap: float = 0.001

@dataclass
class MockNormalizer:
    baseline: MockBaseline = field(default_factory=MockBaseline)
    fo4_load: float = 0.005
    baseline_cell: MagicMock = field(default_factory=MagicMock)

    def normalize(self, cell):
        return MockMetrics()

    def normalize_time(self, val): return val
    def normalize_capacitance(self, val): return val

@dataclass
class MockDatasetEntry:
    library: LibertyLibrary
    normalizer: MockNormalizer

@pytest.fixture(scope="module")
def report_file(tmp_path_factory):
    output_file = tmp_path_factory.mktemp("report") / "test_report.html"

    lib = LibertyLibrary(
        name="test_lib",
        technology="cmos",
        delay_model="table_lookup",
        time_unit="1ns",
        voltage_unit="1V",
        current_unit="1mA",
        leakage_power_unit="1nW",
        pulling_resistance_unit="1kohm",
        capacitive_load_unit=(1.0, "pf"),
        cells={}
    )

    cell = MagicMock(spec=Cell)
    cell.name = "INV_X1"
    cell.area = 1.0
    cell.linear_delay_model.return_value = (0.1, 0.5, 0.99)
    cell.linear_power_model.return_value = (0.01, 0.2, 0.95)
    cell.timing_arcs = []
    cell.power_arcs = []
    cell.plots = {
        "delay": {"scatters": [], "model": {"x": [0, 1], "y": [0, 1]}},
        "power": {"scatters": [], "model": {"x": [0, 1], "y": [0, 1]}}
    }

    lib.cells["INV_X1"] = cell

    entry = MockDatasetEntry(
        library=lib,
        normalizer=MockNormalizer()
    )

    generate_report([entry], output_file)
    return output_file

def test_keyboard_shortcuts(page: Page, report_file):
    """Verifies keyboard shortcuts for search and navigation."""
    page.goto(f"file://{report_file.absolute()}")

    # 1. Verify Initial State (Dashboard active)
    expect(page.locator("#panel-dashboard")).to_be_visible()
    expect(page.locator("#panel-browser")).not_to_be_visible()

    # 2. Test '/' Shortcut -> Focus Search
    # Note: Search input is in the browser tab, but focusing it might switch tab or just focus?
    # Actually, the search input is physically inside the "browser" panel HTML structure.
    # If the browser panel is hidden (x-show=false), can we focus it?
    # Browsers usually prevent focusing hidden elements.
    # So pressing '/' should probably switch to browser tab AND focus search, OR just focus if visible.
    # My plan was just "focus search".
    # If I am on Dashboard, search is hidden.
    # So I should probably make '/' switch to browser tab as well?
    # Or maybe the search input is global?
    # Let's check the HTML.
    # The search input is inside `id="panel-browser"`.
    # So if I am on dashboard, `#panel-browser` has `display: none` (via x-show).
    # So I cannot focus it.
    # I should update my implementation plan to: If '/' is pressed, switch to 'browser' tab AND focus search.

    # For now, let's just try to switch to browser tab using '2' first, then test '/'.

    # 3. Test '2' Shortcut -> Switch to Cell Browser
    page.keyboard.press("2")
    expect(page.locator("#panel-browser")).to_be_visible()
    expect(page.locator("#panel-dashboard")).not_to_be_visible()

    # 4. Test '/' Shortcut -> Focus Search (now that it is visible)
    # First, blur everything
    page.mouse.click(0, 0)
    expect(page.locator("#cell-search")).not_to_be_focused()

    page.keyboard.press("/")
    expect(page.locator("#cell-search")).to_be_focused()

    # 5. Test Esc Shortcut -> Clear Search / Blur
    page.keyboard.type("inv")
    expect(page.locator("#cell-search")).to_have_value("inv")

    page.keyboard.press("Escape")
    expect(page.locator("#cell-search")).to_have_value("")
    # It should still be focused after clearing?
    # If I implement: if value -> clear; else -> blur.
    expect(page.locator("#cell-search")).to_be_focused()

    page.keyboard.press("Escape")
    expect(page.locator("#cell-search")).not_to_be_focused()

    # 6. Test '1' Shortcut -> Switch to Dashboard
    page.keyboard.press("1")
    expect(page.locator("#panel-dashboard")).to_be_visible()
