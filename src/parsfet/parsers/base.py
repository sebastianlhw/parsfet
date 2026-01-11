import gzip
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseParser(ABC, Generic[T]):
    """Abstract base class for all technology file parsers.

    Adheres to the following design principles:
    - **Modular:** One parser per file type.
    - **Lazy:** Streams content where possible (though full parsing typically requires memory).
    - **Reproducible:** Same input yields same output model.

    Attributes:
        Generic[T]: The type of the model returned by the parser (e.g., LibertyLibrary).
    """

    def _read_file(self, path: Path, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Reads file content, automatically handling .gz compression.

        Args:
            path: Path to the file.
            encoding: Text encoding (default: utf-8).
            errors: Error handling scheme for encoding errors (default: strict).

        Returns:
            The content of the file as a string.
        """
        if path.suffix == ".gz":
            with gzip.open(path, mode="rt", encoding=encoding, errors=errors) as f:
                return f.read()
        return path.read_text(encoding=encoding, errors=errors)

    @abstractmethod
    def parse(self, path: Path) -> T:
        """Parses a file from a given path.

        Args:
            path: Path to the technology file.

        Returns:
            The parsed and validated data model.
        """
        ...

    @abstractmethod
    def parse_string(self, content: str, name: str = "unknown") -> T:
        """Parses a file from string content.

        Args:
            content: The raw content string.
            name: Optional name for the parsed object (e.g., library name).

        Returns:
            The parsed and validated data model.
        """
        ...

    def validate(self, data: T) -> list[str]:
        """Validates the parsed data model.

        Subclasses should override this to provide format-specific validation logic.

        Args:
            data: The parsed data model.

        Returns:
            A list of warning messages (empty list if valid).
        """
        return []
