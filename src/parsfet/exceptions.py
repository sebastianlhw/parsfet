"""Pars-FET Exceptions.

This module defines custom exceptions for the Pars-FET framework.
"""

from pathlib import Path
from typing import Any


class DuplicateCellError(Exception):
    """Raised when duplicate cell names are found across files during combine.

    Attributes:
        duplicates: Dictionary mapping cell name to list of (entry_index, source_path).
    """

    def __init__(self, duplicates: dict[str, list[tuple[int, Path]]]):
        self.duplicates = duplicates
        msg = self._format_message()
        super().__init__(msg)

    def _format_message(self) -> str:
        lines = [f"Found {len(self.duplicates)} duplicate cell(s) across files:"]
        for cell, sources in self.duplicates.items():
            files = ", ".join(str(p) for _, p in sources)
            lines.append(f"  - {cell}: {files}")
        lines.append("")
        lines.append(
            "Use allow_duplicates=True to keep first occurrence, or rename cells before combining."
        )
        return "\n".join(lines)

    def cell_names(self) -> list[str]:
        """Returns list of duplicate cell names."""
        return list(self.duplicates.keys())
