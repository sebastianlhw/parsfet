import gzip
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class BaseParser(ABC, Generic[T]):
    """
    Abstract base for all technology file parsers.

    Design principles (per Linus):
    - Modular: One parser per file type
    - Lazy: Stream where possible, don't load entire file into memory
    - Reproducible: Same input → same output
    """

    def _read_file(self, path: Path, encoding: str = "utf-8", errors: str = "strict") -> str:
        """
        Read file content, handling .gz compression automatically.
        """
        if path.suffix == ".gz":
            with gzip.open(path, mode="rt", encoding=encoding, errors=errors) as f:
                return f.read()
        return path.read_text(encoding=encoding, errors=errors)

    @abstractmethod
    def parse(self, path: Path) -> T:
        """
        Parse file from path and return structured data.

        Args:
            path: Path to the technology file

        Returns:
            Parsed and validated data structure
        """
        ...

    @abstractmethod
    def parse_string(self, content: str, name: str = "unknown") -> T:
        """
        Parse from string content.

        Args:
            content: File content as string
            name: Optional name for the library/file

        Returns:
            Parsed and validated data structure
        """
        ...

    def validate(self, data: T) -> list[str]:
        """
        Validate parsed data and return list of warnings.

        Override in subclasses to add format-specific validation.

        Returns:
            List of warning messages (empty if valid)
        """
        return []
