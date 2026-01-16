"""LEF (.lef) and TechLEF (.techlef) file parsers.

LEF files describe the physical layout rules and macro definitions of a technology.
This module provides parsers for both standard LEF (containing macros) and
Technology LEF (containing only layer/via definitions).

Reference: LEF/DEF Language Reference (Cadence)
"""

import re
from pathlib import Path
from typing import Optional

from ..models.lef import (LayerDirection, LayerType, LEFLibrary, Macro,
                          MacroPin, MetalLayer, Rect, Site, TechLEF, Via)
from .base import BaseParser


class LEFParser(BaseParser[LEFLibrary]):
    """Parser for LEF (Library Exchange Format) files.

    Implements a recursive descent parser using tokenization, matching
    the architecture of LibertyParser for consistency and maintainability.
    """

    # Pre-compiled token pattern for lexing
    _TOKEN_PATTERN = re.compile(r"""
        "(?:[^"\\]|\\.)*"             # Double-quoted string
        |'(?:[^'\\]|\\.)*'            # Single-quoted string
        |[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?  # Number
        |[a-zA-Z_][a-zA-Z0-9_\.\[\]]*  # Identifier
        |[;()]                         # Punctuation
    """, re.VERBOSE)


    def parse(self, path: Path) -> LEFLibrary:
        """Parses a LEF file from a given path.

        Args:
            path: Path to the LEF file.

        Returns:
            A populated LEFLibrary object.
        """
        content = self._read_file(path, encoding="utf-8", errors="replace")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> LEFLibrary:
        """Parses LEF content from a string.

        Args:
            content: The LEF file content.
            name: Name for the library.

        Returns:
            A populated LEFLibrary object.
        """
        # Preprocess: remove comments
        content = self._remove_comments(content)

        # Tokenize and initialize token stream
        self._init_tokens(self._tokenize(content))

        library = LEFLibrary()

        # Parse top-level statements
        while self._pos < self._length:
            token = self._peek()

            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "VERSION":
                self._consume()
                library.version = self._consume()
                self._skip_semicolon()

            elif token_upper == "UNITS":
                self._consume()
                self._parse_units(library)

            elif token_upper == "MANUFACTURINGGRID":
                self._consume()
                library.manufacturing_grid = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "LAYER":
                layer = self._parse_layer()
                library.layers[layer.name] = layer

            elif token_upper == "VIA":
                via = self._parse_via()
                library.vias[via.name] = via

            elif token_upper == "SITE":
                site = self._parse_site()
                library.sites[site.name] = site

            elif token_upper == "MACRO":
                macro = self._parse_macro()
                library.macros[macro.name] = macro

            elif token_upper == "END":
                # End of a top-level block (e.g., END LIBRARY)
                self._consume()
                if self._peek():
                    self._consume()  # Consume the name after END
                break

            else:
                # Skip unknown tokens (e.g., BUSBITCHARS, DIVIDERCHAR)
                self._consume()
                # If followed by quoted string or value, consume until semicolon
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

        return library

    def _remove_comments(self, content: str) -> str:
        """Removes both # line comments and /* */ block comments."""
        # Remove /* ... */ comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        # Remove # ... comments
        content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
        return content

    def _tokenize(self, content: str) -> list[str]:
        """Converts the content string into a stream of tokens."""
        return self._TOKEN_PATTERN.findall(content)

    def _expect(self, expected: str, case_sensitive: bool = False) -> str:
        """LEF uses case-insensitive keywords by default."""
        return super()._expect(expected, case_sensitive=case_sensitive)

    def _skip_to_end(self, end_name: str) -> None:
        """Skips tokens until END <end_name> is found."""
        while self._pos < self._length:
            if self._peek() and self._peek().upper() == "END":
                self._consume()
                if self._peek() and self._peek().upper() == end_name.upper():
                    self._consume()
                    return
            else:
                self._consume()

    def _parse_units(self, library: LEFLibrary) -> None:
        """Parses UNITS ... END UNITS block."""
        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                if self._peek() and self._peek().upper() == "UNITS":
                    self._consume()
                break

            elif token_upper == "DATABASE":
                self._consume()
                if self._peek() and self._peek().upper() == "MICRONS":
                    self._consume()
                    library.units_database = int(self._consume())
                self._skip_semicolon()

            else:
                # Skip unknown unit types
                self._consume()
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

    def _parse_layer(self) -> MetalLayer:
        """Parses LAYER ... END <name> block."""
        self._expect("LAYER")
        name = self._consume()
        layer = MetalLayer(name=name)

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                self._consume()  # layer name
                break

            elif token_upper == "TYPE":
                self._consume()
                layer_type = self._consume().lower()
                if layer_type == "routing":
                    layer.layer_type = LayerType.ROUTING
                elif layer_type == "cut":
                    layer.layer_type = LayerType.CUT
                elif layer_type == "masterslice":
                    layer.layer_type = LayerType.MASTERSLICE
                elif layer_type == "overlap":
                    layer.layer_type = LayerType.OVERLAP
                elif layer_type == "implant":
                    layer.layer_type = LayerType.IMPLANT
                self._skip_semicolon()

            elif token_upper == "DIRECTION":
                self._consume()
                direction = self._consume().lower()
                if direction == "horizontal":
                    layer.direction = LayerDirection.HORIZONTAL
                elif direction == "vertical":
                    layer.direction = LayerDirection.VERTICAL
                self._skip_semicolon()

            elif token_upper == "PITCH":
                self._consume()
                layer.pitch = float(self._consume())
                # Handle PITCH x y format
                if self._peek() and self._peek() != ";":
                    self._consume()  # Skip second value
                self._skip_semicolon()

            elif token_upper == "WIDTH":
                self._consume()
                layer.width = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "MINWIDTH":
                self._consume()
                layer.min_width = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "SPACING":
                self._consume()
                layer.spacing = float(self._consume())
                # Skip any additional spacing constraints
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

            elif token_upper == "RESISTANCE":
                self._consume()
                if self._peek() and self._peek().upper() == "RPERSQ":
                    self._consume()
                    layer.resistance = float(self._consume())
                else:
                    self._consume()  # Skip value
                self._skip_semicolon()

            elif token_upper == "CAPACITANCE":
                self._consume()
                if self._peek() and self._peek().upper() == "CPERSQDIST":
                    self._consume()
                    layer.capacitance = float(self._consume())
                else:
                    self._consume()  # Skip value
                self._skip_semicolon()

            elif token_upper == "EDGECAPACITANCE":
                self._consume()
                layer.edge_capacitance = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "THICKNESS":
                self._consume()
                layer.thickness = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "HEIGHT":
                self._consume()
                layer.height = float(self._consume())
                self._skip_semicolon()

            else:
                # Skip unknown layer properties
                self._consume()
                while self._peek() and self._peek() != ";" and self._peek().upper() != "END":
                    self._consume()
                self._skip_semicolon()

        return layer

    def _parse_via(self) -> Via:
        """Parses VIA ... END <name> block."""
        self._expect("VIA")
        name = self._consume()

        # Handle optional DEFAULT keyword
        if self._peek() and self._peek().upper() == "DEFAULT":
            self._consume()

        via = Via(name=name)
        via.layers = []

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                self._consume()  # via name
                break

            elif token_upper == "LAYER":
                self._consume()
                layer_name = self._consume()
                via.layers.append(layer_name)
                self._skip_semicolon()

            elif token_upper == "RECT":
                self._consume()
                # Consume 4 coordinates
                for _ in range(4):
                    if self._peek() and self._peek() != ";":
                        self._consume()
                self._skip_semicolon()

            elif token_upper == "RESISTANCE":
                self._consume()
                via.resistance = float(self._consume())
                self._skip_semicolon()

            else:
                # Skip unknown via properties
                self._consume()
                while self._peek() and self._peek() != ";" and self._peek().upper() != "END":
                    self._consume()
                self._skip_semicolon()

        return via

    def _parse_site(self) -> Site:
        """Parses SITE ... END <name> block."""
        self._expect("SITE")
        name = self._consume()

        class_type = "core"
        width, height = 0.0, 0.0
        symmetry = []

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                self._consume()  # site name
                break

            elif token_upper == "CLASS":
                self._consume()
                class_type = self._consume().lower()
                self._skip_semicolon()

            elif token_upper == "SIZE":
                self._consume()
                width = float(self._consume())
                self._expect("BY")
                height = float(self._consume())
                self._skip_semicolon()

            elif token_upper == "SYMMETRY":
                self._consume()
                while self._peek() and self._peek() != ";":
                    symmetry.append(self._consume())
                self._skip_semicolon()

            else:
                # Skip unknown site properties
                self._consume()
                while self._peek() and self._peek() != ";" and self._peek().upper() != "END":
                    self._consume()
                self._skip_semicolon()

        return Site(
            name=name,
            class_type=class_type,
            width=width,
            height=height,
            symmetry=symmetry
        )

    def _parse_macro(self) -> Macro:
        """Parses MACRO ... END <name> block."""
        self._expect("MACRO")
        name = self._consume()

        class_type = "core"
        origin = (0.0, 0.0)
        size = (0.0, 0.0)
        symmetry = []
        site = None
        foreign = None
        pins = {}
        obstructions = []

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                end_name = self._consume()  # macro name
                if end_name and end_name.upper() == name.upper():
                    break
                # If END doesn't match macro name, it's nested END, continue

            elif token_upper == "CLASS":
                self._consume()
                class_type = self._consume().lower()
                self._skip_semicolon()

            elif token_upper == "ORIGIN":
                self._consume()
                x = float(self._consume())
                y = float(self._consume())
                origin = (x, y)
                self._skip_semicolon()

            elif token_upper == "SIZE":
                self._consume()
                w = float(self._consume())
                self._expect("BY")
                h = float(self._consume())
                size = (w, h)
                self._skip_semicolon()

            elif token_upper == "SYMMETRY":
                self._consume()
                while self._peek() and self._peek() != ";":
                    symmetry.append(self._consume())
                self._skip_semicolon()

            elif token_upper == "SITE":
                self._consume()
                site = self._consume()
                self._skip_semicolon()

            elif token_upper == "FOREIGN":
                self._consume()
                foreign = self._consume()
                # Skip optional coordinates
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

            elif token_upper == "PIN":
                pin = self._parse_macro_pin()
                pins[pin.name] = pin

            elif token_upper == "OBS":
                obs_rects = self._parse_obstruction()
                obstructions.extend(obs_rects)

            else:
                # Skip unknown macro properties
                self._consume()
                while self._peek() and self._peek() != ";" and self._peek().upper() != "END" and self._peek().upper() != "PIN" and self._peek().upper() != "OBS":
                    self._consume()
                self._skip_semicolon()

        return Macro(
            name=name,
            class_type=class_type,
            origin=origin,
            size=size,
            symmetry=symmetry,
            site=site,
            foreign=foreign,
            pins=pins,
            obstructions=obstructions
        )

    def _parse_macro_pin(self) -> MacroPin:
        """Parses PIN ... END <name> block within a macro."""
        self._expect("PIN")
        name = self._consume()

        direction = "input"
        use = None
        shape = None
        ports = []

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                end_name = self._consume()  # pin name
                if end_name and end_name.upper() == name.upper():
                    break

            elif token_upper == "DIRECTION":
                self._consume()
                direction = self._consume().lower()
                self._skip_semicolon()

            elif token_upper == "USE":
                self._consume()
                use = self._consume().lower()
                self._skip_semicolon()

            elif token_upper == "SHAPE":
                self._consume()
                shape = self._consume().lower()
                self._skip_semicolon()

            elif token_upper == "PORT":
                self._consume()
                port_rects = self._parse_port()
                ports.extend(port_rects)

            else:
                # Skip unknown pin properties
                self._consume()
                while self._peek() and self._peek() != ";" and self._peek().upper() != "END" and self._peek().upper() != "PORT":
                    self._consume()
                self._skip_semicolon()

        return MacroPin(
            name=name,
            direction=direction,
            use=use,
            shape=shape,
            ports=ports
        )

    def _parse_port(self) -> list[Rect]:
        """Parses PORT ... END block within a pin."""
        rects = []
        current_layer = None

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                # PORT blocks just have "END" without name
                break

            elif token_upper == "LAYER":
                self._consume()
                current_layer = self._consume()
                self._skip_semicolon()

            elif token_upper == "RECT":
                self._consume()
                x1 = float(self._consume())
                y1 = float(self._consume())
                x2 = float(self._consume())
                y2 = float(self._consume())
                if current_layer:
                    rects.append(Rect(layer=current_layer, x1=x1, y1=y1, x2=x2, y2=y2))
                self._skip_semicolon()

            elif token_upper == "POLYGON":
                self._consume()
                # Skip polygon coordinates until semicolon
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

            elif token_upper == "VIA":
                self._consume()
                # Skip via reference until semicolon
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

            else:
                # Skip unknown port elements
                self._consume()

        return rects

    def _parse_obstruction(self) -> list[Rect]:
        """Parses OBS ... END block within a macro."""
        self._expect("OBS")
        rects = []
        current_layer = None

        while self._pos < self._length:
            token = self._peek()
            if token is None:
                break

            token_upper = token.upper()

            if token_upper == "END":
                self._consume()
                # OBS blocks just have "END" without name
                break

            elif token_upper == "LAYER":
                self._consume()
                current_layer = self._consume()
                self._skip_semicolon()

            elif token_upper == "RECT":
                self._consume()
                x1 = float(self._consume())
                y1 = float(self._consume())
                x2 = float(self._consume())
                y2 = float(self._consume())
                if current_layer:
                    rects.append(Rect(layer=current_layer, x1=x1, y1=y1, x2=x2, y2=y2))
                self._skip_semicolon()

            elif token_upper == "POLYGON":
                self._consume()
                # Skip polygon coordinates until semicolon
                while self._peek() and self._peek() != ";":
                    self._consume()
                self._skip_semicolon()

            else:
                # Skip unknown obstruction elements
                self._consume()

        return rects

    def validate(self, data: LEFLibrary) -> list[str]:
        """Validates the parsed LEF library.

        Checks for basic consistency:
        - At least one layer defined.
        - At least one site defined.
        - All macros have positive dimensions.

        Args:
            data: The parsed LEFLibrary object.

        Returns:
            A list of warning messages.
        """
        warnings = []

        if not data.layers:
            warnings.append("No layers defined")

        if not data.sites:
            warnings.append("No sites defined")

        # Check for macros without valid size
        for name, macro in data.macros.items():
            if macro.size[0] <= 0 or macro.size[1] <= 0:
                warnings.append(f"Macro {name} has invalid size: {macro.size}")

        return warnings


class TechLEFParser(BaseParser[TechLEF]):
    """Parser for Technology LEF files.

    Technology LEF files contain layer, via, and site definitions but typically
    no macro definitions.
    """

    def __init__(self):
        """Initializes the TechLEFParser."""
        self._lef_parser = LEFParser()

    def parse(self, path: Path) -> TechLEF:
        """Parses a TechLEF file from a given path."""
        content = self._read_file(path, encoding="utf-8", errors="replace")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> TechLEF:
        """Parses TechLEF content from a string.

        Delegates to LEFParser for the underlying parsing logic but wraps the result
        in a TechLEF object.
        """
        # Use LEF parser for the heavy lifting
        lef = self._lef_parser.parse_string(content, name)

        return TechLEF(
            version=lef.version,
            units_database=lef.units_database,
            manufacturing_grid=lef.manufacturing_grid,
            layers=lef.layers,
            vias=lef.vias,
            sites=lef.sites,
        )

    def validate(self, data: TechLEF) -> list[str]:
        """Validates the parsed TechLEF.

        Checks that layers and routing layers are defined.
        """
        warnings = []

        if not data.layers:
            warnings.append("No layers defined in technology LEF")

        routing_layers = [l for l in data.layers.values() if l.layer_type == LayerType.ROUTING]
        if not routing_layers:
            warnings.append("No routing layers found")

        return warnings
