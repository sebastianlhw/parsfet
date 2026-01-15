"""
Liberty JSON Parser

Parses the JSON representation of Liberty files used by SkyWater PDK.
These files store cell-level timing data in JSON format rather than
raw Liberty syntax.
"""

import json
from pathlib import Path
from typing import Any, Optional

from ..models.common import OperatingCondition, ProcessCorner
from ..models.liberty import (Cell, LibertyLibrary, LookupTable, Pin, PowerArc,
                              TimingArc)
from .base import BaseParser


class LibertyJSONParser(BaseParser[LibertyLibrary]):
    """
    Parser for Liberty JSON format (.lib.json) used by SkyWater PDK.

    This format stores cell data as JSON objects with keys like:
    - "area": 3.7536
    - "pin,A": { "direction": "input", "capacitance": 0.002302 }
    - "pin,Y": { "direction": "output", "timing": { ... } }
    """

    def parse(self, path: Path) -> LibertyLibrary:
        """Parse a single cell JSON file"""
        content = self._read_file(path, encoding="utf-8")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> LibertyLibrary:
        """Parse JSON content as a single-cell library"""
        data = json.loads(content)

        library = LibertyLibrary(name=name)
        cell = self._build_cell(data, name)
        library.cells[cell.name] = cell

        return library

    def parse_library_dir(self, path: Path, corner: str = "tt_025C_1v80") -> LibertyLibrary:
        """
        Parse a SkyWater PDK library directory.

        Structure expected:
            path/
                timing/
                    sky130_fd_sc_hd__tt_025C_1v80.lib.json  # library header
                cells/
                    inv/
                        sky130_fd_sc_hd__inv_1__tt_025C_1v80.lib.json
                        sky130_fd_sc_hd__inv_2__tt_025C_1v80.lib.json
                    nand2/
                        sky130_fd_sc_hd__nand2_1__tt_025C_1v80.lib.json
        """
        # Determine library name from path
        lib_name = path.name

        # Try to load library header
        library = LibertyLibrary(name=lib_name)
        header_base = path / "timing" / f"{lib_name}__{corner}.lib.json"
        header_path = header_base if header_base.exists() else Path(str(header_base) + ".gz")

        if header_path.exists():
            header_content = self._read_file(header_path)
            header_data = json.loads(header_content)
            library.nom_voltage = header_data.get("nom_voltage")
            library.nom_temperature = header_data.get("nom_temperature")

            # Parse operating conditions
            op_key = f"operating_conditions,{corner}"
            if op_key in header_data:
                op_data = header_data[op_key]
                library.operating_conditions = OperatingCondition(
                    name=corner,
                    voltage=op_data.get("voltage", library.nom_voltage or 1.0),
                    temperature=op_data.get("temperature", library.nom_temperature or 25.0),
                )

        # Find all cell JSON files for this corner
        cells_dir = path / "cells"
        if not cells_dir.exists():
            return library

        cell_patterns = [f"*__{corner}.lib.json", f"*__{corner}.lib.json.gz"]

        for cell_dir in cells_dir.iterdir():
            if not cell_dir.is_dir():
                continue

            for pattern in cell_patterns:
                for json_file in cell_dir.glob(pattern):
                    try:
                        cell_content = self._read_file(json_file)
                        cell_data = json.loads(cell_content)
                        cell_name = json_file.name.split(".")[0]
                        cell = self._build_cell(cell_data, cell_name)
                        library.cells[cell.name] = cell
                    except Exception as e:
                        print(f"Warning: Failed to parse {json_file}: {e}")

        return library

    def _build_cell(self, data: dict[str, Any], name: str) -> Cell:
        """Build Cell from JSON data"""
        cell = Cell(
            name=name,
            area=data.get("area", 0.0),
            cell_leakage_power=data.get("cell_leakage_power"),
        )

        # Parse leakage power conditions
        if "leakage_power" in data:
            for lp in data["leakage_power"]:
                cell.leakage_power_values.append(
                    {
                        "when": lp.get("when"),
                        "value": lp.get("value", 0.0),
                    }
                )

        # Parse pins (keys like "pin,A", "pin,Y")
        for key, value in data.items():
            if key.startswith("pin,"):
                pin_name = key.split(",", 1)[1]
                if not isinstance(value, dict):
                    continue  # Skip non-dict pin data
                pin = self._build_pin(value, pin_name)
                cell.pins[pin_name] = pin

                # Extract timing arcs from output pins
                if pin.direction == "output" and "timing" in value:
                    timing_data = value["timing"]
                    # Handle both single arc (dict) and multiple arcs (list)
                    if isinstance(timing_data, list):
                        for arc_data in timing_data:
                            if isinstance(arc_data, dict):
                                arc = self._build_timing_arc(arc_data)
                                cell.timing_arcs.append(arc)
                    elif isinstance(timing_data, dict):
                        arc = self._build_timing_arc(timing_data)
                        cell.timing_arcs.append(arc)

                # Extract power arcs
                if "internal_power" in value:
                    power_data = value["internal_power"]
                    if isinstance(power_data, list):
                        for pd in power_data:
                            if isinstance(pd, dict):
                                power_arc = self._build_power_arc(pd)
                                cell.power_arcs.append(power_arc)
                    elif isinstance(power_data, dict):
                        power_arc = self._build_power_arc(power_data)
                        cell.power_arcs.append(power_arc)

        # Check for sequential markers
        if "ff," in str(data.keys()) or "latch," in str(data.keys()):
            cell.is_sequential = True

        return cell

    def _build_pin(self, data: dict[str, Any], name: str) -> Pin:
        """Build Pin from JSON data"""
        return Pin(
            name=name,
            direction=data.get("direction", "input"),
            capacitance=data.get("capacitance"),
            max_capacitance=data.get("max_capacitance"),
            function=data.get("function"),
            clock=data.get("clock", "false").lower() == "true"
            if isinstance(data.get("clock"), str)
            else bool(data.get("clock", False)),
            rise_capacitance=data.get("rise_capacitance"),
            fall_capacitance=data.get("fall_capacitance"),
        )

    def _build_timing_arc(self, data: dict[str, Any]) -> TimingArc:
        """Build TimingArc from JSON timing data"""
        arc = TimingArc(
            related_pin=data.get("related_pin", ""),
            timing_sense=data.get("timing_sense", "positive_unate"),
            timing_type=data.get("timing_type"),
        )

        # Parse lookup tables
        for key, value in data.items():
            if isinstance(value, dict) and "values" in value:
                lut = self._build_lut(value)

                if key.startswith("cell_rise"):
                    arc.cell_rise = lut
                elif key.startswith("cell_fall"):
                    arc.cell_fall = lut
                elif key.startswith("rise_transition"):
                    arc.rise_transition = lut
                elif key.startswith("fall_transition"):
                    arc.fall_transition = lut
                elif key.startswith("rise_constraint"):
                    arc.rise_constraint = lut
                elif key.startswith("fall_constraint"):
                    arc.fall_constraint = lut

        return arc

    def _build_power_arc(self, data: dict[str, Any]) -> PowerArc:
        """Build PowerArc from JSON internal_power data"""
        arc = PowerArc(
            related_pin=data.get("related_pin"),
        )

        for key, value in data.items():
            if isinstance(value, dict) and "values" in value:
                lut = self._build_lut(value)

                if "rise_power" in key:
                    arc.rise_power = lut
                elif "fall_power" in key:
                    arc.fall_power = lut

        return arc

    def _build_lut(self, data: dict[str, Any]) -> LookupTable:
        """Build LookupTable from JSON data"""
        return LookupTable(
            index_1=data.get("index_1", []),
            index_2=data.get("index_2", []),
            values=data.get("values", []),
        )

    def validate(self, data: LibertyLibrary) -> list[str]:
        """Validate parsed library"""
        warnings = []

        if not data.cells:
            warnings.append("Library contains no cells")

        if not data.baseline_cell:
            warnings.append("No baseline inverter found")

        return warnings
