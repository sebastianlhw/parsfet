
import pytest
import json
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from parsfet.reporting.html_generator import generate_report
from parsfet.models.liberty import LibertyLibrary, Cell

# --- Minimal Mocks for Dependencies ---

@dataclass
class MockMetrics:
    cell_type: str = "inverter"
    def to_dict(self):
        return {"area_ratio": 1.0, "d0_ratio": 1.0, "p0_ratio": 1.0}

@dataclass
class MockBaseline:
    cell_name: str = "INV_X1"
    fo4_slew: float = 0.01

@dataclass
class MockNormalizer:
    baseline: MockBaseline = field(default_factory=MockBaseline)
    fo4_load: float = 0.005
    baseline_cell: MagicMock = field(default_factory=MagicMock)

    def normalize(self, cell):
        return MockMetrics()
    
    # Mocking unit conversion methods
    def normalize_time(self, val): return val
    def normalize_capacitance(self, val): return val

@dataclass
class MockDatasetEntry:
    library: LibertyLibrary
    normalizer: MockNormalizer

@pytest.fixture
def mock_library():
    # Setup minimal library structure
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
    return lib

@pytest.fixture
def mock_cell():
    cell = MagicMock(spec=Cell)
    cell.name = "INV_X1"
    cell.area = 1.0
    # Mock linear model returns: (d0, k, r2)
    cell.linear_delay_model.return_value = (0.1, 0.5, 0.99)
    cell.linear_power_model.return_value = (0.01, 0.2, 0.95)
    
    # Mock arcs to avoid AttributeError during iteration
    cell.timing_arcs = []
    cell.power_arcs = []
    
    return cell

@patch("parsfet.reporting.html_generator.validate_assets")
def test_generate_report_structure(mock_validate, tmp_path, mock_library, mock_cell):
    """Test standard report generation with mocked data."""
    output_file = tmp_path / "report.html"
    
    # Add mock cell to library
    mock_library.cells["INV_X1"] = mock_cell
    
    entry = MockDatasetEntry(
        library=mock_library,
        normalizer=MockNormalizer()
    )
    
    generate_report([entry], output_file)
    
    assert output_file.exists()
    content = output_file.read_text("utf-8")
    
    # Verify HTML Structure
    assert "<!DOCTYPE html>" in content
    assert "<html" in content
    
    # Check for embedded JSON payload
    assert "test_lib" in content
    assert "INV_X1" in content
    
    # Extract JSON from the variable assignment
    # window.LIB_DATA = {...};
    import re
    match = re.search(r'window\.LIB_DATA = ({.*?});', content, re.DOTALL)
    assert match, "Could not find JSON payload in HTML"
    
    data = json.loads(match.group(1))
    assert data["library_name"] == "test_lib"
    assert len(data["cells"]) == 1
    cell_data = data["cells"][0]
    assert cell_data["name"] == "INV_X1"
    assert cell_data["delay_r2"] == 0.99
    assert cell_data["power_r2"] == 0.95

@patch("parsfet.reporting.html_generator.validate_assets")
def test_generate_report_empty_entries(mock_validate, tmp_path):
    """Verify behavior with no entries."""
    output_file = tmp_path / "empty_report.html"
    generate_report([], output_file)
    
    assert output_file.exists()
    content = output_file.read_text("utf-8")
    
    # Should still have valid HTML shell
    assert "libraries" in content
