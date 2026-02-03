"""Liberty (.lib) file parser using Lark.

This module provides an alternative Liberty parser implementation using Lark,
a modern Python parsing toolkit. It can be used side-by-side with the existing
regex-based parser for comparison and evaluation.

Key benefits over the legacy parser:
- Formal EBNF grammar (liberty.lark) that matches the Liberty specification
- Better error messages with line/column information
- Cleaner separation of syntax (grammar) and semantics (transformer)
- Direct model construction via Transformer (no intermediate AST)
"""

import functools
import gzip
import logging
import re
from pathlib import Path
from typing import Any, Optional

from lark import Lark, Token, Transformer, Tree

from ..models.liberty import Cell, LibertyLibrary, LookupTable, Pin, PowerArc, TimingArc
from .base import BaseParser

logger = logging.getLogger(__name__)

# Load grammar from file (relative to this module)
GRAMMAR_PATH = Path(__file__).parent / "liberty.lark"

# Pre-compiled regex for backslash line continuation (performance optimization)
_BACKSLASH_CONTINUATION = re.compile(r"\\\s*\n\s*")

# Derive known attributes from model fields (single source of truth)
_KNOWN_CELL_ATTRS = frozenset(Cell.model_fields.keys())
_KNOWN_PIN_ATTRS = frozenset(Pin.model_fields.keys())


@functools.cache
def _get_lark_parser() -> Lark:
    """Returns a cached Lark parser instance.

    Uses functools.cache to ensure the parser is only created once,
    improving performance for batch processing.
    """
    return Lark(
        GRAMMAR_PATH.read_text(),
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


class LibertyTransformer(Transformer):
    """Transforms Lark parse tree into LibertyLibrary model objects.

    This transformer builds our data model directly during tree traversal,
    avoiding the creation of an intermediate AST structure.
    """

    def __init__(self, name: str = "unknown"):
        super().__init__()
        self._name = name
        self._templates: dict[str, dict] = {}  # LUT templates

    # === Value transformations ===

    def string_value(self, items) -> str:
        """Handle quoted strings, stripping quotes."""
        val = str(items[0])
        return val.strip("\"'")

    def number_value(self, items) -> float:
        """Convert numeric tokens to float."""
        return float(items[0])

    def name_value(self, items) -> str:
        """Handle unquoted identifiers."""
        return str(items[0])

    def arg(self, items) -> Any:
        """Single argument."""
        return items[0] if items else None

    def arg_list(self, items) -> list:
        """List of arguments."""
        return list(items)

    def value(self, items) -> Any:
        """Generic value - pass through."""
        return items[0] if items else None

    # === Attribute transformations ===

    def simple_attr(self, items) -> tuple[str, Any]:
        """Simple attribute: name : value ;"""
        name = str(items[0])
        value = items[1] if len(items) > 1 else None
        return (name, value)

    def complex_attr(self, items) -> tuple[str, list]:
        """Complex attribute: name ( args ) ;"""
        name = str(items[0])
        args = items[1] if len(items) > 1 else []
        return (name, args)

    # === Group transformations ===

    def group_body(self, items) -> dict:
        """Collect group body items into a dictionary."""
        result = {
            "_attributes": {},
            "_groups": [],
        }
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                # Attribute (simple or complex)
                name, value = item
                result["_attributes"][name] = value
            elif isinstance(item, dict) and "_type" in item:
                # Nested group
                result["_groups"].append(item)
        return result

    def group(self, items) -> dict:
        """Parse a group structure."""
        # Items: NAME, [arg_list], group_body
        name = str(items[0])

        if len(items) == 2:
            # Group without qualifier: NAME { body }
            qualifier = None
            body = items[1]
        else:
            # Group with args: NAME ( args ) { body }
            args = items[1] if isinstance(items[1], list) else None
            qualifier = " ".join(str(a) for a in args) if args else None
            body = items[2] if len(items) > 2 else items[1]

        return {
            "_type": name,
            "_qualifier": qualifier,
            "_attributes": body.get("_attributes", {}) if isinstance(body, dict) else {},
            "_groups": body.get("_groups", []) if isinstance(body, dict) else [],
        }

    def library_body(self, items) -> dict:
        """Same as group_body for library level."""
        return self.group_body(items)

    def statement(self, items) -> Any:
        """Pass through statement."""
        return items[0] if items else None

    def library(self, items) -> dict:
        """Parse top-level library group."""
        name = str(items[0])
        body = items[1] if len(items) > 1 else {}
        return {
            "_type": "library",
            "_qualifier": name,
            "_attributes": body.get("_attributes", {}) if isinstance(body, dict) else {},
            "_groups": body.get("_groups", []) if isinstance(body, dict) else [],
        }

    def start(self, items) -> dict:
        """Entry point - return library AST."""
        return items[0] if items else {}


class LibertyParser(BaseParser[LibertyLibrary]):
    """Liberty parser using Lark grammar.

    Parses Liberty (.lib) files using a formal EBNF grammar for correctness
    and maintainability. This is the default parser implementation.
    """

    def __init__(self):
        """Initialize the Lark parser with the Liberty grammar."""
        super().__init__()
        # Use cached parser instance for performance
        self._parser = _get_lark_parser()

    def parse(self, path: Path) -> LibertyLibrary:
        """Parses a Liberty file from a given path.

        Args:
            path: Path to the Liberty file.

        Returns:
            A populated LibertyLibrary object.
        """
        logger.info(f"[Lark] Parsing Liberty file: {path}")

        # Handle gzip files
        if path.suffix == ".gz" or str(path).endswith(".lib.gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            content = self._read_file(path, encoding="utf-8", errors="replace")

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
        logger.debug(f"[Lark] Parsing content string, length: {len(content)}")

        # Preprocess: remove backslash line continuations (use compiled regex)
        content = _BACKSLASH_CONTINUATION.sub(" ", content)

        # Parse using Lark
        tree = self._parser.parse(content)

        # Transform to intermediate representation
        transformer = LibertyTransformer(name)
        ast = transformer.transform(tree)

        # Build model from AST
        return self._build_library(ast, name)

    def _build_library(self, ast: dict, default_name: str) -> LibertyLibrary:
        """Converts the transformed AST to a LibertyLibrary model."""
        attrs = ast.get("_attributes", {})
        groups = ast.get("_groups", [])

        lib_name = ast.get("_qualifier", default_name) or default_name
        if isinstance(lib_name, str):
            lib_name = lib_name.strip("\"'")

        library = LibertyLibrary(
            name=lib_name,
            technology=self._get_str(attrs, "technology"),
            delay_model=self._get_str(attrs, "delay_model", "table_lookup"),
            time_unit=self._get_str(attrs, "time_unit", "1ns"),
            voltage_unit=self._get_str(attrs, "voltage_unit", "1V"),
            nom_voltage=self._get_float(attrs, "nom_voltage"),
            nom_temperature=self._get_float(attrs, "nom_temperature"),
            nom_process=self._get_float(attrs, "nom_process"),
        )

        # Parse capacitive load unit
        cap_unit = attrs.get("capacitive_load_unit")
        if cap_unit:
            library.capacitive_load_unit = self._parse_cap_unit(cap_unit)

        # Build lookup table templates
        templates = {}
        for grp in groups:
            if grp.get("_type") == "lu_table_template":
                template_name = grp.get("_qualifier", "")
                templates[template_name] = grp.get("_attributes", {})
        library.lu_table_templates = templates

        # Parse cells
        for grp in groups:
            if grp.get("_type") == "cell":
                cell = self._build_cell(grp, templates)
                library.cells[cell.name] = cell

        # Store raw attributes
        library.attributes = {k: v for k, v in attrs.items()}

        return library

    def _build_cell(self, grp: dict, templates: dict) -> Cell:
        """Build Cell from group AST."""
        attrs = grp.get("_attributes", {})
        nested = grp.get("_groups", [])

        cell_name = grp.get("_qualifier", "unknown")
        if isinstance(cell_name, str):
            cell_name = cell_name.strip("\"'")

        cell = Cell(
            name=cell_name,
            area=self._get_float(attrs, "area", 0.0),
            cell_leakage_power=self._get_float(attrs, "cell_leakage_power"),
            dont_use=self._get_bool(attrs, "dont_use"),
            dont_touch=self._get_bool(attrs, "dont_touch"),
            clock_gating_integrated_cell=self._get_str(attrs, "clock_gating_integrated_cell"),
        )

        # Check if sequential
        for n in nested:
            if n.get("_type") in ("ff", "latch"):
                cell.is_sequential = True
                break

        # Parse pins
        for n in nested:
            if n.get("_type") == "pin":
                pin = self._build_pin(n)
                cell.pins[pin.name] = pin

                # Extract timing arcs from pin
                for timing in n.get("_groups", []):
                    if timing.get("_type") == "timing":
                        arc = self._build_timing_arc(timing, templates)
                        cell.timing_arcs.append(arc)

                # Extract power arcs
                for power in n.get("_groups", []):
                    if power.get("_type") == "internal_power":
                        power_arc = self._build_power_arc(power, templates)
                        cell.power_arcs.append(power_arc)

        # Parse leakage power values
        for n in nested:
            if n.get("_type") == "leakage_power":
                n_attrs = n.get("_attributes", {})
                cell.leakage_power_values.append(
                    {
                        "when": n_attrs.get("when"),
                        "value": self._get_float(n_attrs, "value", 0.0),
                    }
                )

        # Parse power/ground pins (pg_pin)
        for n in nested:
            if n.get("_type") == "pg_pin":
                n_attrs = n.get("_attributes", {})
                pg_name = n.get("_qualifier", "unknown")
                if isinstance(pg_name, str):
                    pg_name = pg_name.strip("\"'")
                cell.pg_pins[pg_name] = {
                    "pg_type": self._get_str(n_attrs, "pg_type"),
                    "voltage_name": self._get_str(n_attrs, "voltage_name"),
                }

        # Calculate worst-case leakage power
        max_leakage = cell.cell_leakage_power or 0.0
        for entry in cell.leakage_power_values:
            val = entry.get("value", 0.0)
            if val > max_leakage:
                max_leakage = val
        cell.cell_leakage_power = max_leakage

        # Collect undefined attributes (derived from model fields)
        for key, value in attrs.items():
            if key not in _KNOWN_CELL_ATTRS:
                cell.undefined_attributes[key] = value

        return cell

    def _build_pin(self, grp: dict) -> Pin:
        """Build Pin from group AST."""
        attrs = grp.get("_attributes", {})

        pin_name = grp.get("_qualifier", "unknown")
        if isinstance(pin_name, str):
            pin_name = pin_name.strip("\"'")

        pin = Pin(
            name=pin_name,
            direction=self._get_str(attrs, "direction", "input"),
            capacitance=self._get_float(attrs, "capacitance"),
            max_capacitance=self._get_float(attrs, "max_capacitance"),
            min_capacitance=self._get_float(attrs, "min_capacitance"),
            function=self._get_str(attrs, "function"),
            clock=self._get_bool(attrs, "clock"),
            clock_gate_clock_pin=self._get_bool(attrs, "clock_gate_clock_pin"),
            clock_gate_enable_pin=self._get_bool(attrs, "clock_gate_enable_pin"),
            rise_capacitance=self._get_float(attrs, "rise_capacitance"),
            fall_capacitance=self._get_float(attrs, "fall_capacitance"),
        )

        # Collect undefined attributes (derived from model fields)
        for key, value in attrs.items():
            if key not in _KNOWN_PIN_ATTRS:
                pin.undefined_attributes[key] = value

        return pin

    def _build_timing_arc(self, grp: dict, templates: dict) -> TimingArc:
        """Build TimingArc from group AST."""
        attrs = grp.get("_attributes", {})
        nested = grp.get("_groups", [])

        arc = TimingArc(
            related_pin=self._get_str(attrs, "related_pin", ""),
            timing_sense=self._get_str(attrs, "timing_sense", "positive_unate"),
            timing_type=self._get_str(attrs, "timing_type"),
        )

        # Find LUT groups
        for n in nested:
            lut_type = n.get("_type")
            if lut_type == "cell_rise":
                arc.cell_rise = self._build_lut(n, templates)
            elif lut_type == "cell_fall":
                arc.cell_fall = self._build_lut(n, templates)
            elif lut_type == "rise_transition":
                arc.rise_transition = self._build_lut(n, templates)
            elif lut_type == "fall_transition":
                arc.fall_transition = self._build_lut(n, templates)
            elif lut_type == "rise_constraint":
                arc.rise_constraint = self._build_lut(n, templates)
            elif lut_type == "fall_constraint":
                arc.fall_constraint = self._build_lut(n, templates)

        return arc

    def _build_power_arc(self, grp: dict, templates: dict) -> PowerArc:
        """Build PowerArc from group AST."""
        attrs = grp.get("_attributes", {})
        nested = grp.get("_groups", [])

        arc = PowerArc(
            related_pin=self._get_str(attrs, "related_pin"),
            when=self._get_str(attrs, "when"),
        )

        for n in nested:
            if n.get("_type") == "rise_power":
                arc.rise_power = self._build_lut(n, templates)
            elif n.get("_type") == "fall_power":
                arc.fall_power = self._build_lut(n, templates)

        return arc

    def _build_lut(self, grp: dict, templates: dict) -> Optional[LookupTable]:
        """Build LookupTable from group AST."""
        attrs = grp.get("_attributes", {})

        lut = LookupTable()

        # Check for template
        template_name = grp.get("_qualifier", "")
        if isinstance(template_name, str):
            template_name = template_name.strip("\"'")
        template = templates.get(template_name, {})

        # Parse indices
        # Parse indices
        if "index_1" in attrs:
            idx1 = self._extract_index(attrs.get("index_1"))
        else:
            idx1 = self._extract_index(template.get("index_1"))

        if "index_2" in attrs:
            idx2 = self._extract_index(attrs.get("index_2"))
        else:
            idx2 = self._extract_index(template.get("index_2"))

        if idx1:
            lut.index_1 = idx1
        if idx2:
            lut.index_2 = idx2

        # Parse values
        values = self._extract_values(attrs.get("values"))
        if values:
            # Reshape if needed
            if idx1 and idx2 and len(values) == 1:
                flat_values = values[0]
                expected_size = len(idx1) * len(idx2)
                if len(flat_values) == expected_size:
                    reshaped = []
                    for i in range(len(idx1)):
                        row_start = i * len(idx2)
                        row_end = row_start + len(idx2)
                        reshaped.append(flat_values[row_start:row_end])
                    lut.values = reshaped
                else:
                    lut.values = values
            elif idx1 and not idx2 and len(values) == 1:
                # 1D table - flatten the single row
                lut.values = values[0]
            else:
                lut.values = values

        return lut

    def _unwrap_value(self, val: Any) -> Any:
        """Unwrap Lark Tree/Token objects to plain Python values."""
        if val is None:
            return None
        if isinstance(val, Tree):
            # Get the first child's value
            if val.children:
                return self._unwrap_value(val.children[0])
            return None
        if isinstance(val, Token):
            return str(val).strip("\"'")
        return val

    def _extract_index(self, data: Any) -> list[float]:
        """Extract index values from attribute."""
        if data is None:
            return []

        # Unwrap Tree/Token if needed
        data = self._unwrap_value(data)
        if data is None:
            return []

        if isinstance(data, list):
            result = []
            for x in data:
                x = self._unwrap_value(x)
                if isinstance(x, (int, float)):
                    result.append(float(x))
                elif isinstance(x, str):
                    result.extend(self._parse_number_list(x))
            return result

        if isinstance(data, str):
            return self._parse_number_list(data)

        return []

    def _extract_values(self, data: Any) -> list[list[float]]:
        """Extract LUT values from attribute."""
        if data is None:
            return []

        # Unwrap Tree/Token if needed
        data = self._unwrap_value(data)
        if data is None:
            return []

        if isinstance(data, str):
            # Multi-line values are joined by the transformer
            return [self._parse_number_list(data)]

        if isinstance(data, list):
            result = []
            for row in data:
                row = self._unwrap_value(row)
                if isinstance(row, str):
                    result.append(self._parse_number_list(row))
                elif isinstance(row, list):
                    result.append([float(x) for x in row])
            return result

        return []

    def _parse_number_list(self, s: str) -> list[float]:
        """Parse comma-separated numbers from a string."""
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
        """Parse capacitive_load_unit value."""
        if isinstance(value, list) and len(value) >= 2:
            return (float(value[0]), str(value[1]).strip("\"'"))
        if isinstance(value, str):
            parts = re.split(r"[,\s]+", value.strip("\"'"))
            if len(parts) >= 2:
                try:
                    return (float(parts[0]), parts[1])
                except ValueError:
                    pass
        return (1.0, "pf")

    def _get_str(self, d: dict, key: str, default: str = None) -> Optional[str]:
        """Get string value from dictionary."""
        val = d.get(key)
        if val is None:
            return default
        # Unwrap single-element lists (from complex_attr like 'technology(cmos)')
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        # Unwrap Tree/Token if needed
        val = self._unwrap_value(val)
        if val is None:
            return default
        if isinstance(val, str):
            return val.strip("\"'")
        return str(val)

    def _get_float(self, d: dict, key: str, default: float = None) -> Optional[float]:
        """Get float value from dictionary."""
        val = d.get(key)
        if val is None:
            return default
        # Unwrap Tree/Token if needed
        val = self._unwrap_value(val)
        if val is None:
            return default
        try:
            if isinstance(val, str):
                val = val.strip("\"'")
            return float(val)
        except (ValueError, TypeError):
            return default

    def _get_bool(self, d: dict, key: str, default: bool = False) -> bool:
        """Get boolean value from dictionary."""
        val = d.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip("\"'").lower() in ("true", "yes", "1")
        return bool(val)

    def validate(self, data: LibertyLibrary) -> list[str]:
        """Validates the parsed Liberty library.

        Same validation logic as the legacy parser.
        """
        logger.debug(f"[Lark] Validating library: {data.name}")
        warnings = []

        if not data.cells:
            warnings.append("Library contains no cells")

        if not data.baseline_cell:
            warnings.append("No baseline inverter (INVD1) found for normalization")

        no_area = [name for name, cell in data.cells.items() if cell.area <= 0]
        if no_area:
            warnings.append(f"{len(no_area)} cells {no_area} have zero or missing area")

        if warnings:
            logger.warning(f"[Lark] Validation warnings for {data.name}: {warnings}")

        return warnings
