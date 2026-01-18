"""LEF (.lef) and TechLEF (.techlef) file parsers using Lark.

This module provides LEF parsers using a formal EBNF grammar for correctness
and maintainability. This is the default parser implementation.

Reference: LEF/DEF Language Reference (Cadence)
"""

import functools
import logging
import re
from pathlib import Path
from typing import Any, Optional

from lark import Lark, Transformer, Token, Tree

from ..models.lef import (
    LayerDirection,
    LayerType,
    LEFLibrary,
    Macro,
    MacroPin,
    MetalLayer,
    Rect,
    Site,
    TechLEF,
    Via,
)
from .base import BaseParser

logger = logging.getLogger(__name__)

# Load grammar from file (relative to this module)
GRAMMAR_PATH = Path(__file__).parent / "lef.lark"

# Pre-compiled regex for comment removal
_BLOCK_COMMENT = re.compile(r'/\*[\s\S]*?\*/')
_LINE_COMMENT = re.compile(r'#[^\n]*')


@functools.cache
def _get_lark_parser() -> Lark:
    """Returns a cached Lark parser instance."""
    return Lark(
        GRAMMAR_PATH.read_text(),
        parser='lalr',
        propagate_positions=True,
        maybe_placeholders=False,
    )


class LEFTransformer(Transformer):
    """Transforms Lark parse tree into LEFLibrary model objects.
    
    Uses a simple approach: collect all terminal values from each rule,
    then process based on rule name.
    """
    
    def __init__(self, name: str = "unknown"):
        super().__init__()
        self._name = name
        self._library = LEFLibrary(name=name)
    
    # === Terminal transformations ===
    
    def NUMBER(self, token) -> float:
        return float(token)
    
    def NAME(self, token) -> str:
        return str(token)
    
    def QUOTED_STRING(self, token) -> str:
        return str(token).strip('"')
    
    # === Passthrough rules ===
    
    def value(self, items):
        return items[0] if items else None
    
    def statement(self, items):
        return items[0] if items else None
    
    def symmetry_flag(self, items):
        # Token is the first item - it's SYM_X, SYM_Y, or SYM_R90
        if items:
            return str(items[0]).upper()
        return ""
    
    def SYM_X(self, token):
        return "X"
    
    def SYM_Y(self, token):
        return "Y"
    
    def SYM_R90(self, token):
        return "R90"
    
    def spacing_option(self, items):
        return items
    
    def resistance_option(self, items):
        return "RPERSQ"
    
    def capacitance_option(self, items):
        return "CPERSQDIST"
    
    def macro_subclass(self, items):
        return items[0] if items else None
    
    def unknown_layer_stmt(self, items):
        return None  # Ignore unknown statements
    
    def unknown_stmt(self, items):
        return None  # Ignore unknown top-level statements (TechLEF compat)
    
    def viarule_block(self, items):
        return None  # Ignore VIARULE blocks for now
    
    def viarule_body(self, items):
        return None
    
    def viarule_stmt(self, items):
        return None
    
    def property_stmt(self, items):
        return None  # Ignore property statements
    
    # === Header statements ===
    
    def version_stmt(self, items):
        self._library.version = str(items[0])
        return None
    
    def manufacturing_grid(self, items):
        self._library.manufacturing_grid = float(items[0])
        return None
    
    def busbitchars_stmt(self, items):
        self._library.bus_bit_chars = str(items[0]).strip('"')
        return None
    
    def dividerchar_stmt(self, items):
        self._library.divider_char = str(items[0]).strip('"')
        return None
    
    # === UNITS block ===
    
    def unit_stmt(self, items):
        # DATABASE MICRONS <value>
        if items:
            self._library.units_database = int(items[0])
        return None
    
    def units_block(self, items):
        return None
    
    # === LAYER block ===
    
    def layer_stmt(self, items):
        """Capture layer statement with all values."""
        return list(items)
    
    def layer_body(self, items):
        return [i for i in items if i is not None]
    
    def layer_block(self, items):
        """Build MetalLayer from parsed statements."""
        # items[0] = name, items[-1] = end name, items[1:-1] = body statements
        name = str(items[0])
        layer = MetalLayer(name=name)
        
        # Get body items (skip name at start and end)
        body = items[1] if len(items) > 2 else []
        if isinstance(body, list):
            for stmt in body:
                if not isinstance(stmt, list) or not stmt:
                    continue
                # First item is the value, identify type by checking values
                self._apply_layer_stmt(layer, stmt)
        
        self._library.layers[name] = layer
        return layer
    
    def _apply_layer_stmt(self, layer: MetalLayer, values: list):
        """Apply statement values to layer based on content analysis."""
        if not values:
            return
        
        first = str(values[0]).upper() if values else ""
        
        # TYPE statement: value is layer type name
        if first in ("ROUTING", "CUT", "MASTERSLICE", "OVERLAP", "IMPLANT"):
            type_map = {
                "ROUTING": LayerType.ROUTING,
                "CUT": LayerType.CUT,
                "MASTERSLICE": LayerType.MASTERSLICE,
                "OVERLAP": LayerType.OVERLAP,
                "IMPLANT": LayerType.IMPLANT,
            }
            layer.layer_type = type_map.get(first, LayerType.ROUTING)
        # DIRECTION statement
        elif first in ("HORIZONTAL", "VERTICAL"):
            if first == "HORIZONTAL":
                layer.direction = LayerDirection.HORIZONTAL
            else:
                layer.direction = LayerDirection.VERTICAL
        # RPERSQ means RESISTANCE statement
        elif first == "RPERSQ" and len(values) > 1:
            layer.resistance = float(values[1])
        # CPERSQDIST means CAPACITANCE statement
        elif first == "CPERSQDIST" and len(values) > 1:
            layer.capacitance = float(values[1])
        # Numeric value - could be PITCH, WIDTH, etc
        elif isinstance(values[0], (int, float)):
            # Single numeric - check context from previously set values
            # This is a limitation - we'll set based on what's not yet set
            val = float(values[0])
            if layer.pitch is None:
                layer.pitch = val
            elif layer.width is None:
                layer.width = val
            elif layer.spacing is None:
                layer.spacing = val
    
    # === VIA block ===
    
    def via_stmt(self, items):
        return list(items)
    
    def via_body(self, items):
        return [i for i in items if i is not None]
    
    def via_block(self, items):
        name = str(items[0])
        via = Via(name=name, layers=[])
        
        # Body is items[1] (or items[2] if DEFAULT keyword present)
        for item in items[1:]:
            if isinstance(item, list):
                for stmt in item:
                    if isinstance(stmt, list) and stmt:
                        # Check if it's a layer name (string, not number)
                        if isinstance(stmt[0], str):
                            via.layers.append(stmt[0])
                        elif isinstance(stmt[0], (int, float)) and len(stmt) == 1:
                            via.resistance = float(stmt[0])
        
        self._library.vias[name] = via
        return via
    
    # === SITE block ===
    
    def site_stmt(self, items):
        return list(items)
    
    def site_body(self, items):
        return [i for i in items if i is not None]
    
    def site_block(self, items):
        name = str(items[0])
        class_type = "core"
        width, height = 0.0, 0.0
        symmetry = []
        
        body = items[1] if len(items) > 2 else []
        if isinstance(body, list):
            for stmt in body:
                if not isinstance(stmt, list) or not stmt:
                    continue
                first = str(stmt[0]).upper() if stmt else ""
                if first == "CORE" or first == "PAD" or first == "IO":
                    class_type = first.lower()
                elif first in ("X", "Y", "R90"):
                    symmetry = [str(s).upper() for s in stmt]
                elif isinstance(stmt[0], (int, float)):
                    if len(stmt) >= 2:
                        width, height = float(stmt[0]), float(stmt[1])
        
        site = Site(name=name, class_type=class_type, width=width, height=height, symmetry=symmetry)
        self._library.sites[name] = site
        return site
    
    # === MACRO block ===
    
    def macro_stmt(self, items):
        return list(items)
    
    def macro_body(self, items):
        return [i for i in items if i is not None]
    
    def macro_block(self, items):
        name = str(items[0])
        class_type = "core"
        class_seen = False  # Track if CLASS statement was encountered
        origin = (0.0, 0.0)
        size = (0.0, 0.0)
        symmetry = []
        site = None
        foreign = None
        pins = {}
        obstructions = []
        
        body = items[1] if len(items) > 2 else []
        if isinstance(body, list):
            for item in body:
                # MacroPin directly
                if isinstance(item, MacroPin):
                    pins[item.name] = item
                # pin_block result wrapped in macro_stmt list
                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], MacroPin):
                    pins[item[0].name] = item[0]
                # Obstructions (list of Rect directly from obs_block)
                elif isinstance(item, list) and item and isinstance(item[0], Rect):
                    obstructions.extend(item)
                # Obstructions wrapped in macro_stmt list [[Rect, ...]]
                elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], list) and item[0] and isinstance(item[0][0], Rect):
                    obstructions.extend(item[0])
                # Other list-based statements
                elif isinstance(item, list) and item:
                    first_val = item[0]
                    first_str = str(first_val).upper() if isinstance(first_val, str) else ""
                    
                    # CLASS statement (first match sets class, flag prevents duplicate)
                    if first_str in ("CORE", "BLOCK", "PAD", "ENDCAP", "COVER") and not class_seen:
                        class_type = first_str.lower()
                        class_seen = True
                    # SYMMETRY statement (contains X, Y, R90)
                    elif first_str in ("X", "Y", "R90"):
                        symmetry = [str(s).upper() for s in item if str(s).upper() in ("X", "Y", "R90")]
                    # SIZE or ORIGIN statement (two numbers)
                    elif isinstance(first_val, (int, float)) and len(item) == 2:
                        if size == (0.0, 0.0):
                            size = (float(item[0]), float(item[1]))
                        else:
                            origin = (float(item[0]), float(item[1]))
                    # SITE statement (single name - CLASS was already processed if present)
                    elif isinstance(first_val, str) and len(item) == 1:
                        site = str(first_val)
        
        macro = Macro(
            name=name, class_type=class_type, origin=origin, size=size,
            symmetry=symmetry, site=site, foreign=foreign, pins=pins,
            obstructions=obstructions
        )
        self._library.macros[name] = macro
        return macro
    
    # === PIN block ===
    
    def pin_stmt(self, items):
        return list(items)
    
    def pin_body(self, items):
        return [i for i in items if i is not None]
    
    def pin_block(self, items):
        name = str(items[0])
        direction = "input"
        use = None
        shape = None
        ports = []
        
        body = items[1] if len(items) > 2 else []
        if isinstance(body, list):
            for item in body:
                # Port result is list of Rect wrapped in list from pin_stmt
                if isinstance(item, list) and item:
                    first = item[0]
                    # Check if it's a list of Rect (from port_block)
                    if isinstance(first, Rect):
                        ports.extend(item)
                    # Or wrapped one more level
                    elif isinstance(first, list) and first and isinstance(first[0], Rect):
                        ports.extend(first)
                    # Direction/use/shape statements
                    elif isinstance(first, str):
                        first_upper = str(first).upper()
                        if first_upper in ("INPUT", "OUTPUT", "INOUT", "FEEDTHRU"):
                            direction = first_upper.lower()
                        elif first_upper in ("SIGNAL", "POWER", "GROUND", "CLOCK"):
                            use = first_upper.lower()
                        elif first_upper in ("ABUTMENT", "RING"):
                            shape = first_upper.lower()
        
        return MacroPin(name=name, direction=direction, use=use, shape=shape, ports=ports)
    
    # === PORT block ===
    
    def port_stmt(self, items):
        return list(items)
    
    def port_body(self, items):
        return [i for i in items if i is not None]
    
    def port_block(self, items):
        rects = []
        current_layer = None
        
        # items is [port_body_result] where port_body_result is list of port_stmt results
        body = items[0] if items else []
        # Unwrap if double-nested
        if isinstance(body, list) and len(body) == 1 and isinstance(body[0], list):
            body = body[0]
        
        if isinstance(body, list):
            for stmt in body:
                if not isinstance(stmt, list) or not stmt:
                    continue
                first = stmt[0]
                # LAYER statement - first is a string NAME
                if isinstance(first, str):
                    current_layer = first
                # RECT statement - first is a number
                elif isinstance(first, (int, float)) and len(stmt) >= 4 and current_layer:
                    rects.append(Rect(
                        layer=current_layer,
                        x1=float(stmt[0]),
                        y1=float(stmt[1]),
                        x2=float(stmt[2]),
                        y2=float(stmt[3])
                    ))
        return rects
    
    # === OBS block ===
    
    def obs_stmt(self, items):
        return list(items)
    
    def obs_body(self, items):
        return [i for i in items if i is not None]
    
    def obs_block(self, items):
        rects = []
        current_layer = None
        
        body = items[0] if items else []
        if isinstance(body, list):
            for stmt in body:
                if not isinstance(stmt, list) or not stmt:
                    continue
                first = stmt[0]
                if isinstance(first, str):
                    current_layer = first
                elif isinstance(first, (int, float)) and len(stmt) >= 4 and current_layer:
                    rects.append(Rect(
                        layer=current_layer,
                        x1=float(stmt[0]),
                        y1=float(stmt[1]),
                        x2=float(stmt[2]),
                        y2=float(stmt[3])
                    ))
        return rects
    
    # === Entry point ===
    
    def start(self, items):
        return self._library


class LEFParser(BaseParser[LEFLibrary]):
    """LEF parser using Lark grammar.
    
    Parses LEF (.lef) files using a formal EBNF grammar for correctness
    and maintainability. This is the default parser implementation.
    """
    
    def __init__(self):
        """Initialize the Lark parser with the LEF grammar."""
        super().__init__()
        self._parser = _get_lark_parser()
    
    def parse(self, path: Path) -> LEFLibrary:
        """Parses a LEF file from a given path."""
        logger.info(f"Parsing LEF file: {path}")
        content = self._read_file(path, encoding='utf-8', errors='replace')
        name = path.name.split('.')[0]
        return self.parse_string(content, name)
    
    def parse_string(self, content: str, name: str = "unknown") -> LEFLibrary:
        """Parses LEF content from a string."""
        logger.debug(f"Parsing LEF content string, length: {len(content)}")
        
        # Remove comments
        content = _BLOCK_COMMENT.sub('', content)
        content = _LINE_COMMENT.sub('', content)
        
        # Parse using Lark
        tree = self._parser.parse(content)
        
        # Transform to model
        transformer = LEFTransformer(name)
        return transformer.transform(tree)
    
    def validate(self, data: LEFLibrary) -> list[str]:
        """Validates the parsed LEF library."""
        warnings = []
        
        if not data.layers:
            warnings.append("No layers defined")
        
        if not data.sites:
            warnings.append("No sites defined")
        
        for macro_name, macro in data.macros.items():
            if macro.size[0] <= 0 or macro.size[1] <= 0:
                warnings.append(f"Macro {macro_name} has invalid size: {macro.size}")
        
        if warnings:
            logger.warning(f"Validation warnings for {data.name}: {warnings}")
        
        return warnings


class TechLEFParser(BaseParser[TechLEF]):
    """TechLEF parser using Lark grammar.
    
    Parses Technology LEF files which contain layer, via, and site definitions
    but typically no macro definitions.
    """
    
    def __init__(self):
        """Initialize the TechLEFParser."""
        self._lef_parser = LEFParser()
    
    def parse(self, path: Path) -> TechLEF:
        """Parses a TechLEF file from a given path."""
        content = self._read_file(path, encoding='utf-8', errors='replace')
        name = path.name.split('.')[0]
        return self.parse_string(content, name)
    
    def parse_string(self, content: str, name: str = "unknown") -> TechLEF:
        """Parses TechLEF content from a string."""
        lef = self._lef_parser.parse_string(content, name)
        
        return TechLEF(
            name=name,
            version=lef.version,
            units_database=lef.units_database,
            manufacturing_grid=lef.manufacturing_grid,
            layers=lef.layers,
            vias=lef.vias,
            sites=lef.sites,
        )
    
    def validate(self, data: TechLEF) -> list[str]:
        """Validates the parsed TechLEF."""
        warnings = []
        
        if not data.layers:
            warnings.append("No layers defined in technology LEF")
        
        routing_layers = [l for l in data.layers.values() if l.layer_type == LayerType.ROUTING]
        if not routing_layers:
            warnings.append("No routing layers found")
        
        return warnings
