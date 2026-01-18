"""Liberty (.lib) file parser.

Liberty format is a hierarchical attribute/group structure used for timing and power
characterization of standard cells. This parser handles the full complexity of
Liberty syntax, including:
- Nested groups (library, cell, pin, timing, etc.)
- Lookup tables (2D arrays for timing/power) with complex value structures
- Attributes and expressions

Reference: Liberty User Guide (Synopsys)
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

from ..models.liberty import Cell, LibertyLibrary, LookupTable, Pin, PowerArc, TimingArc
from .base import BaseParser

logger = logging.getLogger(__name__)


class LegacyLibertyParser(BaseParser[LibertyLibrary]):
    """Legacy Liberty (.lib) parser using regex-based tokenization.

    This is the original parser implementation, kept for reference and comparison.
    For new code, use LibertyParser (Lark-based) instead.
    """

    # Pre-compiled token pattern for Liberty format
    _TOKEN_PATTERN = re.compile(
        r"""
        "(?:[^"\\]|\\.)*"             # Quoted string (with escapes)
        |'(?:[^'\\]|\\.)*'            # Single-quoted string
        |[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?  # Number
        |[a-zA-Z_][a-zA-Z0-9_\.\[\]]*  # Identifier (including array notation)
        |[{}();:,\\]                   # Punctuation
    """,
        re.VERBOSE,
    )

    def parse(self, path: Path) -> LibertyLibrary:
        """Parses a Liberty file from a given path.

        Args:
            path: Path to the Liberty file.

        Returns:
            A populated LibertyLibrary object.
        """
        logger.info(f"Parsing Liberty file: {path}")
        content = self._read_file(path, encoding="utf-8", errors="replace")
        # name is stem, but if it's .lib.gz we want the name without .lib
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> LibertyLibrary:
        """Parses Liberty content from a string.

        Args:
            content: The Liberty file content.
            name: Name for the library.

        Returns:
            A populated LibertyLibrary object.
        """
        logger.debug(f"Parsing content string, length: {len(content)}")
        # Preprocess: remove comments
        content = self._remove_comments(content)

        # Tokenize and initialize token stream
        self._init_tokens(self._tokenize(content))

        # Parse top-level library group
        if self._length == 0:
            return LibertyLibrary(name=name)

        ast = self._parse_group()

        # Build model from AST
        return self._build_library(ast, name)

    def _remove_comments(self, content: str) -> str:
        """Removes C-style (/* ... */) and line (// ...) comments.

        Also removes backslash line continuations to handle multi-line values.
        """
        # Remove /* ... */ comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        # Remove // ... comments
        content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
        # Remove backslash line continuations (fixes multi-line values parsing)
        content = re.sub(r"\\\s*\n\s*", " ", content)
        return content

    def _tokenize(self, content: str) -> list[str]:
        """Converts the content string into a stream of tokens using pre-compiled pattern."""
        tokens = self._TOKEN_PATTERN.findall(content)
        logger.debug(f"Generated {len(tokens)} tokens")
        return tokens

    def _skip_to(self, target: str) -> None:
        """Skips tokens until the target token is found (consumes the target)."""
        while self._pos < self._length:
            if self._consume() == target:
                return
        raise ValueError(f"Could not find '{target}'")

    def _parse_group(self) -> dict[str, Any]:
        """Parses a group structure: name (qualifier) { contents }.

        Returns:
            A dictionary containing:
            - _name: The group name (e.g., "cell", "pin").
            - _qualifier: The optional qualifier (e.g., cell name).
            - Parsed attributes and nested groups.
        """
        result: dict[str, Any] = {}

        # Get group name
        group_name = self._consume()
        result["_name"] = group_name

        # Check for qualifier in parentheses
        if self._peek() == "(":
            self._consume()  # (
            qualifier_parts = []
            while self._peek() != ")":
                token = self._consume()
                if token is None:
                    break
                if token not in (",",):
                    qualifier_parts.append(token.strip("\"'"))
            self._expect(")")
            result["_qualifier"] = " ".join(qualifier_parts) if qualifier_parts else None

        # Check for group body
        if self._peek() == "{":
            self._consume()  # {
            result.update(self._parse_group_body())
            # } is consumed by _parse_group_body

        return result

    def _parse_group_body(self) -> dict[str, Any]:
        """Parses the contents inside the curly braces { } of a group."""
        result: dict[str, Any] = {}
        result["_groups"] = []  # List to hold nested groups of same type

        while self._pos < self._length:
            token = self._peek()

            if token == "}":
                self._consume()
                break

            if token is None:
                break

            # Look ahead to determine if this is a group or attribute
            next_token = self._peek(1)

            if next_token == "(":
                # This is a nested group
                nested = self._parse_group()
                nested_name = nested.get("_name", "unknown")

                # Store groups in a list by type
                if nested_name not in result:
                    result[nested_name] = []
                if not isinstance(result[nested_name], list):
                    result[nested_name] = [result[nested_name]]
                result[nested_name].append(nested)
                result["_groups"].append(nested)

            elif next_token == ":":
                # This is an attribute: name : value ;
                attr_name = self._consume()
                self._expect(":")
                attr_value = self._parse_attribute_value()
                result[attr_name] = attr_value

                # Consume optional semicolon
                if self._peek() == ";":
                    self._consume()

            elif next_token == "{":
                # Group without qualifier: name { }
                nested = self._parse_group()
                nested_name = nested.get("_name", "unknown")
                if nested_name not in result:
                    result[nested_name] = []
                if not isinstance(result[nested_name], list):
                    result[nested_name] = [result[nested_name]]
                result[nested_name].append(nested)
                result["_groups"].append(nested)

            else:
                # Unknown token, skip
                self._consume()

        return result

    def _parse_attribute_value(self) -> Any:
        """Parses the value of an attribute."""
        token = self._peek()

        if token is None:
            return None

        # Check for quoted string (possibly containing comma-separated values)
        if token.startswith('"') or token.startswith("'"):
            value = self._consume()
            # Strip quotes
            value = value[1:-1] if len(value) >= 2 else value

            # Check if it's a comma-separated list of numbers
            if "," in value:
                try:
                    parts = [p.strip() for p in value.split(",")]
                    return [float(p) for p in parts if p]
                except ValueError:
                    return value

            # Try to convert to number
            try:
                if "." in value or "e" in value.lower():
                    return float(value)
                return int(value)
            except ValueError:
                return value

        # Check for number
        try:
            if "." in token or "e" in token.lower():
                # Try converting the token first without consuming
                val = float(token)
                self._consume()
                return val
            # Check if it looks like a number
            if token.lstrip("-+").isdigit():
                val = int(token)
                self._consume()
                return val
        except (ValueError, AttributeError):
            pass

        # Otherwise, return as identifier
        return self._consume()

    def _build_library(self, ast: dict[str, Any], default_name: str) -> LibertyLibrary:
        """Converts the Abstract Syntax Tree (AST) to a LibertyLibrary model."""
        # Get library name from qualifier or default
        lib_name = ast.get("_qualifier", default_name) or default_name
        if isinstance(lib_name, str):
            lib_name = lib_name.strip("\"'")

        library = LibertyLibrary(
            name=lib_name,
            technology=self._get_str(ast, "technology"),
            delay_model=self._get_str(ast, "delay_model", "table_lookup"),
            time_unit=self._get_str(ast, "time_unit", "1ns"),
            voltage_unit=self._get_str(ast, "voltage_unit", "1V"),
            nom_voltage=self._get_float(ast, "nom_voltage"),
            nom_temperature=self._get_float(ast, "nom_temperature"),
            nom_process=self._get_float(ast, "nom_process"),
        )

        # Parse capacitive load unit (can be complex)
        cap_unit = ast.get("capacitive_load_unit")
        if cap_unit:
            # Handle group-like format: e.g. capacitive_load_unit(1.0, pf)
            if isinstance(cap_unit, list) and len(cap_unit) > 0 and isinstance(cap_unit[0], dict):
                qualifier = cap_unit[0].get("_qualifier", "")
                if qualifier:
                    library.capacitive_load_unit = self._parse_cap_unit(qualifier)
                else:
                    library.capacitive_load_unit = self._parse_cap_unit(cap_unit)
            else:
                library.capacitive_load_unit = self._parse_cap_unit(cap_unit)

        # Parse lookup table templates
        for template in ast.get("lu_table_template", []):
            if isinstance(template, dict):
                template_name = template.get("_qualifier", "")
                library.lu_table_templates[template_name] = template

        # Parse cells
        for cell_ast in ast.get("cell", []):
            if isinstance(cell_ast, dict):
                cell = self._build_cell(cell_ast, library.lu_table_templates)
                library.cells[cell.name] = cell

        # Store raw attributes for completeness
        library.attributes = {
            k: v
            for k, v in ast.items()
            if not k.startswith("_") and k not in ("cell", "lu_table_template")
        }

        return library

    def _build_cell(self, ast: dict[str, Any], templates: dict[str, Any] = None) -> Cell:
        """Build Cell from AST"""
        cell_name = ast.get("_qualifier", "unknown")
        if isinstance(cell_name, str):
            cell_name = cell_name.strip("\"'")

        cell = Cell(
            name=cell_name,
            area=self._get_float(ast, "area", 0.0),
            cell_leakage_power=self._get_float(ast, "cell_leakage_power"),
            dont_use=self._get_bool(ast, "dont_use"),
            dont_touch=self._get_bool(ast, "dont_touch"),
            clock_gating_integrated_cell=self._get_str(ast, "clock_gating_integrated_cell"),
        )

        # Check if sequential (has ff or latch group)
        if ast.get("ff") or ast.get("latch"):
            cell.is_sequential = True

        # Parse pins
        for pin_ast in ast.get("pin", []):
            if isinstance(pin_ast, dict):
                pin = self._build_pin(pin_ast)
                cell.pins[pin.name] = pin

                # Extract timing arcs from pin
                for timing_ast in pin_ast.get("timing", []):
                    if isinstance(timing_ast, dict):
                        arc = self._build_timing_arc(timing_ast, templates)
                        cell.timing_arcs.append(arc)

                # Extract power arcs
                for power_ast in pin_ast.get("internal_power", []):
                    if isinstance(power_ast, dict):
                        power_arc = self._build_power_arc(power_ast, templates)
                        cell.power_arcs.append(power_arc)

        # Parse leakage power values
        for leak_ast in ast.get("leakage_power", []):
            if isinstance(leak_ast, dict):
                cell.leakage_power_values.append(
                    {
                        "when": leak_ast.get("when"),
                        "value": self._get_float(leak_ast, "value", 0.0),
                    }
                )

        # Parse power/ground pins (pg_pin)
        for pg_pin_ast in ast.get("pg_pin", []):
            if isinstance(pg_pin_ast, dict):
                pg_name = pg_pin_ast.get("_qualifier", "unknown")
                if isinstance(pg_name, str):
                    pg_name = pg_name.strip("\"'")
                cell.pg_pins[pg_name] = {
                    "pg_type": self._get_str(pg_pin_ast, "pg_type"),
                    "voltage_name": self._get_str(pg_pin_ast, "voltage_name"),
                }

        # Calculate worst-case leakage power
        max_leakage = cell.cell_leakage_power or 0.0
        for entry in cell.leakage_power_values:
            val = entry.get("value", 0.0)
            if val > max_leakage:
                max_leakage = val
        cell.cell_leakage_power = max_leakage

        # Collect undefined attributes (not in known set)
        known_cell_attrs = {
            "_name",
            "_qualifier",
            "_groups",
            "area",
            "cell_leakage_power",
            "dont_use",
            "dont_touch",
            "clock_gating_integrated_cell",
            "pin",
            "ff",
            "latch",
            "leakage_power",
            "pg_pin",
            "statetable",
            "bundle",
            "bus",
            "test_cell",
            "mode_definition",
        }
        for key, value in ast.items():
            if key not in known_cell_attrs and not key.startswith("_"):
                cell.undefined_attributes[key] = value

        return cell

    def _build_pin(self, ast: dict[str, Any]) -> Pin:
        """Builds a Pin object from the AST."""
        pin_name = ast.get("_qualifier", "unknown")
        if isinstance(pin_name, str):
            pin_name = pin_name.strip("\"'")

        pin = Pin(
            name=pin_name,
            direction=self._get_str(ast, "direction", "input"),
            capacitance=self._get_float(ast, "capacitance"),
            max_capacitance=self._get_float(ast, "max_capacitance"),
            min_capacitance=self._get_float(ast, "min_capacitance"),
            function=self._get_str(ast, "function"),
            clock=self._get_bool(ast, "clock"),
            clock_gate_clock_pin=self._get_bool(ast, "clock_gate_clock_pin"),
            clock_gate_enable_pin=self._get_bool(ast, "clock_gate_enable_pin"),
            rise_capacitance=self._get_float(ast, "rise_capacitance"),
            fall_capacitance=self._get_float(ast, "fall_capacitance"),
        )

        # Collect undefined attributes (not in known set)
        known_pin_attrs = {
            "_name",
            "_qualifier",
            "_groups",
            "direction",
            "capacitance",
            "max_capacitance",
            "min_capacitance",
            "function",
            "clock",
            "clock_gate_clock_pin",
            "clock_gate_enable_pin",
            "rise_capacitance",
            "fall_capacitance",
            "timing",
            "internal_power",
            "driver_waveform_rise",
            "driver_waveform_fall",
            "input_voltage",
            "output_voltage",
            "related_power_pin",
            "related_ground_pin",
        }
        for key, value in ast.items():
            if key not in known_pin_attrs and not key.startswith("_"):
                pin.undefined_attributes[key] = value

        return pin

    def _build_timing_arc(self, ast: dict[str, Any], templates: dict[str, Any] = None) -> TimingArc:
        """Build TimingArc from AST"""
        arc = TimingArc(
            related_pin=self._get_str(ast, "related_pin", ""),
            timing_sense=self._get_str(ast, "timing_sense", "positive_unate"),
            timing_type=self._get_str(ast, "timing_type"),
        )

        # Parse lookup tables
        arc.cell_rise = self._build_lut(ast.get("cell_rise"), templates)
        arc.cell_fall = self._build_lut(ast.get("cell_fall"), templates)
        arc.rise_transition = self._build_lut(ast.get("rise_transition"), templates)
        arc.fall_transition = self._build_lut(ast.get("fall_transition"), templates)
        arc.rise_constraint = self._build_lut(ast.get("rise_constraint"), templates)
        arc.fall_constraint = self._build_lut(ast.get("fall_constraint"), templates)

        return arc

    def _build_power_arc(self, ast: dict[str, Any], templates: dict[str, Any] = None) -> PowerArc:
        """Build PowerArc from AST"""
        arc = PowerArc(
            related_pin=self._get_str(ast, "related_pin"),
            when=self._get_str(ast, "when"),
        )
        arc.rise_power = self._build_lut(ast.get("rise_power"), templates)
        arc.fall_power = self._build_lut(ast.get("fall_power"), templates)
        return arc

    def _build_lut(self, lut_list: Any, templates: dict[str, Any] = None) -> Optional[LookupTable]:
        """Build LookupTable from AST"""
        if not lut_list:
            return None

        # lut_list might be a list with one element
        if isinstance(lut_list, list):
            if len(lut_list) == 0:
                return None
            lut_ast = lut_list[0]
        else:
            lut_ast = lut_list

        if not isinstance(lut_ast, dict):
            return None

        lut = LookupTable()

        # Check for template
        template_name = lut_ast.get("_qualifier", "")
        if isinstance(template_name, str):
            template_name = template_name.strip("\"'")

        template = templates.get(template_name) if templates else None

        # Parse index_1, index_2 - they may be stored as groups with qualifier
        idx1 = self._extract_lut_index(lut_ast.get("index_1"))
        idx2 = self._extract_lut_index(lut_ast.get("index_2"))

        # If indices not in LUT, check template
        if not idx1 and template:
            idx1 = self._extract_lut_index(template.get("index_1"))
        if not idx2 and template:
            idx2 = self._extract_lut_index(template.get("index_2"))

        if idx1:
            lut.index_1 = idx1
        if idx2:
            lut.index_2 = idx2

        # Parse values - may be stored as groups with qualifier
        values = self._extract_lut_values(lut_ast.get("values"))
        if values:
            # Check if values need reshaping: if we have both indices and
            # values is a single flattened row, reshape to 2D
            if idx1 and idx2 and len(values) == 1:
                flat_values = values[0]
                expected_size = len(idx1) * len(idx2)
                if len(flat_values) == expected_size:
                    # Reshape: values are stored row-major (index_1 is outer, index_2 is inner)
                    # Each row corresponds to one slew value (index_1)
                    # Each column corresponds to one load value (index_2)
                    reshaped = []
                    for i in range(len(idx1)):
                        row_start = i * len(idx2)
                        row_end = row_start + len(idx2)
                        reshaped.append(flat_values[row_start:row_end])
                    lut.values = reshaped
                else:
                    lut.values = values
            else:
                lut.values = values

        return lut

    def _extract_lut_index(self, data: Any) -> list[float]:
        """Extracts index values from LUT AST.

        Handles both direct lists and group format.
        """
        if data is None:
            return []

        # Direct list of numbers
        if isinstance(data, list):
            # Check if it's a list of dicts (group format)
            if data and isinstance(data[0], dict):
                # Get qualifier from first group which contains the values
                group = data[0]
                qualifier = group.get("_qualifier", "")
                if qualifier:
                    return self._parse_number_list(qualifier)
            # Direct list of numbers
            result = []
            for x in data:
                if isinstance(x, (int, float)):
                    result.append(float(x))
                elif isinstance(x, str):
                    result.extend(self._parse_number_list(x))
            return result

        # String of comma-separated values
        if isinstance(data, str):
            return self._parse_number_list(data)

        return []

    def _extract_lut_values(self, data: Any) -> list[Any]:
        """Extracts values from LUT AST.

        Handles both direct lists and group format, including multi-line values.
        """
        if data is None:
            return []

        result = []

        # Direct list
        if isinstance(data, list):
            # Check if it's a list of dicts (group format)
            if data and isinstance(data[0], dict):
                # Get qualifier from first group
                group = data[0]
                qualifier = group.get("_qualifier", "")
                if qualifier:
                    # Split multi-line values (joined with backslash in Liberty)
                    lines = qualifier.replace("\\", "").split(",")
                    # Try to parse as 2D - each quoted string is a row
                    # But if it's all in one qualifier, treat as single row
                    row = self._parse_number_list(qualifier)
                    if row:
                        result.append(row)
            else:
                # Direct list of rows
                for row in data:
                    if isinstance(row, list):
                        result.append(
                            [float(x) if isinstance(x, (int, float)) else 0.0 for x in row]
                        )
                    elif isinstance(row, str):
                        result.append(self._parse_number_list(row))
                    elif isinstance(row, (int, float)):
                        result.append([float(row)])

        elif isinstance(data, str):
            # Single row or comma-separated
            result.append(self._parse_number_list(data))

        return result

    def _parse_2d_values(self, values: Any) -> list[list[float]]:
        """Parses 2D lookup table values."""
        result = []

        if isinstance(values, list):
            for row in values:
                if isinstance(row, list):
                    result.append([float(x) for x in row])
                elif isinstance(row, str):
                    result.append(self._parse_number_list(row))
                elif isinstance(row, (int, float)):
                    result.append([float(row)])
        elif isinstance(values, str):
            # Single row or comma-separated
            result.append(self._parse_number_list(values))

        return result

    def _parse_number_list(self, s: str) -> list[float]:
        """Parses comma-separated numbers from a string."""
        s = s.strip("\"'")
        parts = re.split(r"[,\s]+", s)
        numbers = []
        for p in parts:
            p = p.strip()
            if p:
                try:
                    numbers.append(float(p))
                except ValueError:
                    pass
        return numbers

    def _parse_cap_unit(self, value: Any) -> tuple[float, str]:
        """Parses capacitive_load_unit value."""
        if isinstance(value, list) and len(value) >= 2:
            return (float(value[0]), str(value[1]).strip("\"'"))
        if isinstance(value, str):
            # Try to parse "1,pf" or similar
            parts = re.split(r"[,\s]+", value.strip("\"'"))
            if len(parts) >= 2:
                try:
                    return (float(parts[0]), parts[1])
                except ValueError:
                    pass
        return (1.0, "pf")

    def _get_str(self, d: dict, key: str, default: str = None) -> Optional[str]:
        """Gets a string value from a dictionary, handling quotes."""
        val = d.get(key)
        if val is None:
            return default

        # Handle group format: e.g. technology(cmos) -> [{'_name': 'technology', '_qualifier': 'cmos'}]
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return val[0].get("_qualifier", default)

        if isinstance(val, str):
            # Clean up trailing semicolons if they got attached (due to tokenizer)
            val = val.strip().rstrip(";")
            val = val.strip("\"'")
            if not val and default:
                # Return explicit empty string if present, otherwise fallback
                pass
            return val
        return str(val)

    def _get_float(self, d: dict, key: str, default: float = None) -> Optional[float]:
        """Gets a float value from a dictionary."""
        val = d.get(key)
        if val is None:
            return default
        try:
            if isinstance(val, str):
                val = val.strip("\"'")
            return float(val)
        except (ValueError, TypeError):
            return default

    def _get_bool(self, d: dict, key: str, default: bool = False) -> bool:
        """Gets a boolean value from a dictionary."""
        val = d.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            # Clean up trailing semicolons
            val = val.strip().rstrip(";").strip("\"'")
            return val.lower() in ("true", "yes", "1")
        return bool(val)

    def validate(self, data: LibertyLibrary) -> list[str]:
        """Validates the parsed Liberty library.

        Checks for:
        - No cells in library.
        - Missing baseline inverter (crucial for logical effort).
        - Cells with zero or missing area.
        """
        logger.debug(f"Validating library: {data.name}")
        warnings = []

        if not data.cells:
            warnings.append("Library contains no cells")

        if not data.baseline_cell:
            warnings.append("No baseline inverter (INVD1) found for normalization")

        # Check for cells without area
        no_area = [name for name, cell in data.cells.items() if cell.area <= 0]
        if no_area:
            warnings.append(f"{len(no_area)} cells {no_area} have zero or missing area")

        if warnings:
            logger.warning(f"Validation warnings for {data.name}: {warnings}")

        return warnings
