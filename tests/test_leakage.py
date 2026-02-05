"""Tests for leakage power extraction.

Verifies that the parser correctly identifies the worst-case leakage power
from multiple conditional `leakage_power` groups.
"""

def test_worst_case_leakage_liberty():
    """Verifies that the Liberty parser picks the maximum leakage value.

    Scenario: A cell has a default `cell_leakage_power` of 1.0, but a conditional
    leakage group specifies 5.0. The parsed `cell_leakage_power` should be 5.0.
    """
    from parsfet.parsers.liberty import LibertyParser

    lib_content = """
    library(test) {
        cell(test_cell) {
            cell_leakage_power : 1.0;
            leakage_power() {
                when : "!A !B";
                value : 5.0;
            }
            leakage_power() {
                when : "A !B";
                value : 0.5;
            }
        }
    }
    """
    parser = LibertyParser()
    lib = parser.parse_string(lib_content)
    cell = lib.cells["test_cell"]

    # Should be 5.0 (the max value), not 1.0 (default) or 0.5 (conditional)
    assert cell.cell_leakage_power == 5.0


def test_worst_case_leakage_json():
    """Verifies that the JSON parser picks the maximum leakage value.

    Scenario: Similar to the Liberty test, but input is in JSON format.
    Checks that conditional leakage values override the base value if higher.
    """
    import json

    from parsfet.parsers.liberty_json import LibertyJSONParser

    # JSON content mimicking Skywater format for a single cell
    # Note: Structure flattened as per _build_cell expectations in test context
    # but normally parse_string expects full library JSON or cell JSON.
    # checking parse_string implementation: expects dict with keys.

    cell_data = {
        "area": 0.0,
        "cell_leakage_power": 1.0,
        "leakage_power": [{"when": "!A !B", "value": 5.0}, {"when": "A !B", "value": 0.5}],
    }

    json_content = json.dumps(cell_data)
    parser = LibertyJSONParser()
    # parse_string treats content as a single cell library if it's the cell file format
    # effectively calling _build_cell
    lib = parser.parse_string(json_content, name="test_cell")
    cell = lib.cells["test_cell"]

    assert cell.cell_leakage_power == 5.0
