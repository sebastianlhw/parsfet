"""Liberty JSON Parser.

Parses the JSON representation of Liberty files, specifically tailored for the
SkyWater PDK format. This format separates timing data into individual JSON files
per cell, offering a more granular approach than monolithic .lib files.
"""

import json
from pathlib import Path
from typing import Any, Optional

from ..models.common import OperatingCondition, ProcessCorner
from ..models.liberty import (Cell, LibertyLibrary, LookupTable, Pin, PowerArc,
                              TimingArc)
from .base import BaseParser


class LibertyJSONParser(BaseParser[LibertyLibrary]):
    """Parser for Liberty JSON format (.lib.json) used by SkyWater PDK.

    Handles the JSON structure where cell attributes and timing arcs are stored
    as key-value pairs (e.g., "pin,A": {...}).
    """

    def parse(self, path: Path) -> LibertyLibrary:
        """Parses a single cell JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A LibertyLibrary object containing the single parsed cell.
        """
        content = self._read_file(path, encoding="utf-8")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> LibertyLibrary:
        """Parses JSON content as a single-cell library.

        Args:
            content: JSON string content.
            name: Name for the library/cell.

        Returns:
            A LibertyLibrary object containing the parsed cell.
        """
        data = json.loads(content)

        library = LibertyLibrary(name=name)
        cell = self._build_cell(data, name)
        library.cells[cell.name] = cell

        return library

    def parse_library_dir(self, path: Path, corner: str = "tt_025C_1v80") -> LibertyLibrary:
        """Parses a complete library directory (e.g., SkyWater PDK style).

        Expects a directory structure with a 'timing' folder for headers and a 'cells'
        folder containing subdirectories for each cell type.

        Args:
            path: Path to the library root directory.
            corner: The specific PVT corner to parse (e.g., "tt_025C_1v80").

        Returns:
            A populated LibertyLibrary object containing all found cells for the corner.
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
        """Builds a Cell object from JSON data."""
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
        """Builds a Pin object from JSON data."""
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
        """Builds a TimingArc object from JSON timing data."""
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
        """Builds a PowerArc object from JSON internal_power data."""
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
        """Builds a LookupTable object from JSON data."""
        return LookupTable(
            index_1=data.get("index_1", []),
            index_2=data.get("index_2", []),
            values=data.get("values", []),
        )

    def validate(self, data: LibertyLibrary) -> list[str]:
        """Validates the parsed library.

        Checks for basic validity like presence of cells and baseline inverter.
        """
        warnings = []

        if not data.cells:
            warnings.append("Library contains no cells")

        if not data.baseline_cell:
            warnings.append("No baseline inverter found")

        return warnings
