"""
LEF (.lef) and TechLEF (.techlef) file parsers.

LEF uses a keyword-based section format:
    LAYER M1
        TYPE ROUTING ;
        DIRECTION HORIZONTAL ;
        PITCH 0.14 ;
    END M1

Reference: LEF/DEF Language Reference (Cadence)
"""

import re
from pathlib import Path
from typing import Any, Optional

from ..models.lef import (LayerDirection, LayerType, LEFLibrary, Macro,
                          MacroPin, MetalLayer, Rect, Site, TechLEF, Via)
from .base import BaseParser


class LEFParser(BaseParser[LEFLibrary]):
    """
    Parser for LEF (Library Exchange Format) files.

    Handles both cell LEF (with MACROs) and technology LEF (layers/vias only).
    """

    def parse(self, path: Path) -> LEFLibrary:
        """Parse LEF file from path"""
        content = self._read_file(path, encoding="utf-8", errors="replace")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> LEFLibrary:
        """Parse LEF from string content"""
        # Preprocess: remove comments
        content = self._remove_comments(content)

        library = LEFLibrary()

        # Parse version
        version_match = re.search(r"VERSION\s+([\d.]+)\s*;", content)
        if version_match:
            library.version = version_match.group(1)

        # Parse units
        units_match = re.search(r"UNITS\s+DATABASE\s+MICRONS\s+(\d+)\s*;", content)
        if units_match:
            library.units_database = int(units_match.group(1))

        # Parse manufacturing grid
        grid_match = re.search(r"MANUFACTURINGGRID\s+([\d.]+)\s*;", content)
        if grid_match:
            library.manufacturing_grid = float(grid_match.group(1))

        # Parse layers
        library.layers = self._parse_layers(content)

        # Parse vias
        library.vias = self._parse_vias(content)

        # Parse sites
        library.sites = self._parse_sites(content)

        # Parse macros
        library.macros = self._parse_macros(content)

        return library

    def _remove_comments(self, content: str) -> str:
        """Remove # line comments"""
        return re.sub(r"#.*$", "", content, flags=re.MULTILINE)

    def _parse_layers(self, content: str) -> dict[str, MetalLayer]:
        """Extract LAYER sections"""
        layers = {}

        # Pattern for LAYER ... END layername
        pattern = r"LAYER\s+(\S+)\s*\n(.*?)END\s+\1"

        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            name = match.group(1)
            body = match.group(2)

            layer = MetalLayer(name=name)

            # Parse TYPE
            type_match = re.search(r"TYPE\s+(\w+)\s*;", body, re.IGNORECASE)
            if type_match:
                layer_type = type_match.group(1).lower()
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

            # Parse DIRECTION
            dir_match = re.search(r"DIRECTION\s+(\w+)\s*;", body, re.IGNORECASE)
            if dir_match:
                direction = dir_match.group(1).lower()
                if direction == "horizontal":
                    layer.direction = LayerDirection.HORIZONTAL
                elif direction == "vertical":
                    layer.direction = LayerDirection.VERTICAL

            # Parse PITCH
            pitch_match = re.search(r"PITCH\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if pitch_match:
                layer.pitch = float(pitch_match.group(1))

            # Parse WIDTH
            width_match = re.search(r"WIDTH\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if width_match:
                layer.width = float(width_match.group(1))

            # Parse MINWIDTH
            minwidth_match = re.search(r"MINWIDTH\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if minwidth_match:
                layer.min_width = float(minwidth_match.group(1))

            # Parse SPACING
            spacing_match = re.search(r"SPACING\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if spacing_match:
                layer.spacing = float(spacing_match.group(1))

            # Parse RESISTANCE (RPERSQ)
            res_match = re.search(r"RESISTANCE\s+RPERSQ\s+([\d.eE+-]+)\s*;", body, re.IGNORECASE)
            if res_match:
                layer.resistance = float(res_match.group(1))

            # Parse CAPACITANCE (CPERSQDIST)
            cap_match = re.search(
                r"CAPACITANCE\s+CPERSQDIST\s+([\d.eE+-]+)\s*;", body, re.IGNORECASE
            )
            if cap_match:
                layer.capacitance = float(cap_match.group(1))

            # Parse EDGECAPACITANCE
            edge_cap_match = re.search(r"EDGECAPACITANCE\s+([\d.eE+-]+)\s*;", body, re.IGNORECASE)
            if edge_cap_match:
                layer.edge_capacitance = float(edge_cap_match.group(1))

            # Parse THICKNESS
            thick_match = re.search(r"THICKNESS\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if thick_match:
                layer.thickness = float(thick_match.group(1))

            # Parse HEIGHT
            height_match = re.search(r"HEIGHT\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if height_match:
                layer.height = float(height_match.group(1))

            layers[name] = layer

        return layers

    def _parse_vias(self, content: str) -> dict[str, Via]:
        """Extract VIA sections"""
        vias = {}

        # Pattern for VIA ... END vianame
        pattern = r"VIA\s+(\S+)\s*(DEFAULT)?\s*\n(.*?)END\s+\1"

        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            name = match.group(1)
            body = match.group(3)

            via = Via(name=name)

            # Extract layer references from RECT statements
            layer_pattern = r"LAYER\s+(\S+)\s*;"
            via.layers = re.findall(layer_pattern, body, re.IGNORECASE)

            # Parse RESISTANCE
            res_match = re.search(r"RESISTANCE\s+([\d.eE+-]+)\s*;", body, re.IGNORECASE)
            if res_match:
                via.resistance = float(res_match.group(1))

            vias[name] = via

        return vias

    def _parse_sites(self, content: str) -> dict[str, Site]:
        """Extract SITE sections"""
        sites = {}

        # Pattern for SITE ... END sitename
        pattern = r"SITE\s+(\S+)\s*\n(.*?)END\s+\1"

        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            name = match.group(1)
            body = match.group(2)

            # Parse CLASS
            class_match = re.search(r"CLASS\s+(\w+)\s*;", body, re.IGNORECASE)
            class_type = class_match.group(1).lower() if class_match else "core"

            # Parse SIZE
            size_match = re.search(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if size_match:
                width = float(size_match.group(1))
                height = float(size_match.group(2))
            else:
                width, height = 0.0, 0.0

            # Parse SYMMETRY
            sym_match = re.search(r"SYMMETRY\s+([^;]+);", body, re.IGNORECASE)
            symmetry = sym_match.group(1).split() if sym_match else []

            sites[name] = Site(
                name=name, class_type=class_type, width=width, height=height, symmetry=symmetry
            )

        return sites

    def _parse_macros(self, content: str) -> dict[str, Macro]:
        """Extract MACRO sections"""
        macros = {}

        # Pattern for MACRO ... END macroname
        pattern = r"MACRO\s+(\S+)\s*\n(.*?)END\s+\1"

        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            name = match.group(1)
            body = match.group(2)

            # Parse CLASS
            class_match = re.search(r"CLASS\s+(\w+)\s*;", body, re.IGNORECASE)
            class_type = class_match.group(1).lower() if class_match else "core"

            # Parse ORIGIN
            origin_match = re.search(r"ORIGIN\s+([\d.+-]+)\s+([\d.+-]+)\s*;", body, re.IGNORECASE)
            origin = (
                (float(origin_match.group(1)), float(origin_match.group(2)))
                if origin_match
                else (0.0, 0.0)
            )

            # Parse SIZE
            size_match = re.search(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;", body, re.IGNORECASE)
            if size_match:
                size = (float(size_match.group(1)), float(size_match.group(2)))
            else:
                size = (0.0, 0.0)

            # Parse SYMMETRY
            sym_match = re.search(r"SYMMETRY\s+([^;]+);", body, re.IGNORECASE)
            symmetry = sym_match.group(1).split() if sym_match else []

            # Parse SITE
            site_match = re.search(r"SITE\s+(\S+)\s*;", body, re.IGNORECASE)
            site = site_match.group(1) if site_match else None

            # Parse FOREIGN
            foreign_match = re.search(r"FOREIGN\s+(\S+)", body, re.IGNORECASE)
            foreign = foreign_match.group(1) if foreign_match else None

            # Parse PINs
            pins = self._parse_macro_pins(body)

            # Parse OBSTRUCTIONs
            obstructions = self._parse_obstructions(body)

            macros[name] = Macro(
                name=name,
                class_type=class_type,
                origin=origin,
                size=size,
                symmetry=symmetry,
                site=site,
                foreign=foreign,
                pins=pins,
                obstructions=obstructions,
            )

        return macros

    def _parse_macro_pins(self, macro_body: str) -> dict[str, MacroPin]:
        """Extract PIN sections from macro body"""
        pins = {}

        # Pattern for PIN ... END pinname
        pattern = r"PIN\s+(\S+)\s*\n(.*?)END\s+\1"

        for match in re.finditer(pattern, macro_body, re.DOTALL | re.IGNORECASE):
            name = match.group(1)
            body = match.group(2)

            # Parse DIRECTION
            dir_match = re.search(r"DIRECTION\s+(\w+)\s*;", body, re.IGNORECASE)
            direction = dir_match.group(1).lower() if dir_match else "input"

            # Parse USE
            use_match = re.search(r"USE\s+(\w+)\s*;", body, re.IGNORECASE)
            use = use_match.group(1).lower() if use_match else None

            # Parse SHAPE
            shape_match = re.search(r"SHAPE\s+(\w+)\s*;", body, re.IGNORECASE)
            shape = shape_match.group(1).lower() if shape_match else None

            # Parse PORT rectangles
            ports = []
            port_pattern = r"PORT\s*(.*?)END"
            for port_match in re.finditer(port_pattern, body, re.DOTALL | re.IGNORECASE):
                port_body = port_match.group(1)

                # Find all RECT statements
                rect_pattern = r"LAYER\s+(\S+)\s*;\s*RECT\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*;"
                for rect_match in re.finditer(rect_pattern, port_body, re.IGNORECASE):
                    layer = rect_match.group(1)
                    x1, y1, x2, y2 = (float(rect_match.group(i)) for i in range(2, 6))
                    ports.append(Rect(layer=layer, x1=x1, y1=y1, x2=x2, y2=y2))

            pins[name] = MacroPin(name=name, direction=direction, use=use, shape=shape, ports=ports)

        return pins

    def _parse_obstructions(self, macro_body: str) -> list[Rect]:
        """Extract OBS (obstruction) rectangles"""
        obstructions = []

        obs_pattern = r"OBS\s*(.*?)END"
        for obs_match in re.finditer(obs_pattern, macro_body, re.DOTALL | re.IGNORECASE):
            obs_body = obs_match.group(1)

            # Find all RECT statements
            rect_pattern = (
                r"LAYER\s+(\S+)\s*;\s*RECT\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*;"
            )
            for rect_match in re.finditer(rect_pattern, obs_body, re.IGNORECASE):
                layer = rect_match.group(1)
                x1, y1, x2, y2 = (float(rect_match.group(i)) for i in range(2, 6))
                obstructions.append(Rect(layer=layer, x1=x1, y1=y1, x2=x2, y2=y2))

        return obstructions

    def validate(self, data: LEFLibrary) -> list[str]:
        """Validate parsed LEF library"""
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
    """
    Parser for Technology LEF files.

    TechLEF contains only layer and via definitions (no macros).
    """

    def __init__(self):
        self._lef_parser = LEFParser()

    def parse(self, path: Path) -> TechLEF:
        """Parse TechLEF file from path"""
        content = self._read_file(path, encoding="utf-8", errors="replace")
        name = path.name.split(".")[0]
        return self.parse_string(content, name)

    def parse_string(self, content: str, name: str = "unknown") -> TechLEF:
        """Parse TechLEF from string content"""
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
        """Validate parsed TechLEF"""
        warnings = []

        if not data.layers:
            warnings.append("No layers defined in technology LEF")

        routing_layers = [l for l in data.layers.values() if l.layer_type == LayerType.ROUTING]
        if not routing_layers:
            warnings.append("No routing layers found")

        return warnings
