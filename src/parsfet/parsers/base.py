"""Base parser module with shared tokenizer infrastructure.

Provides the abstract base class for all technology file parsers,
including common tokenization utilities used by recursive descent parsers.
"""

import gzip
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class BaseParser(ABC, Generic[T]):
    """Abstract base class for all technology file parsers.

    Provides shared tokenizer infrastructure for recursive descent parsing,
    including token stream management and common helper methods.

    Adheres to the following design principles:
    - **Modular:** One parser per file type.
    - **Consistent:** Shared tokenizer infrastructure across all parsers.
    - **Reproducible:** Same input yields same output model.

    Attributes:
        Generic[T]: The type of the model returned by the parser (e.g., LibertyLibrary).
    """

    def __init__(self):
        """Initializes the parser with empty token stream."""
        self._pos: int = 0
        self._tokens: list[str] = []
        self._length: int = 0

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

    def _init_tokens(self, tokens: list[str]) -> None:
        """Initializes the token stream for parsing.

        Args:
            tokens: List of tokens from the tokenizer.
        """
        self._tokens = tokens
        self._pos = 0
        self._length = len(tokens)

    def _peek(self, offset: int = 0) -> Optional[str]:
        """Looks ahead at the token at the given offset without consuming it.

        Args:
            offset: Number of tokens to look ahead (default: 0 = current token).

        Returns:
            The token at the offset position, or None if past end.
        """
        idx = self._pos + offset
        return self._tokens[idx] if idx < self._length else None

    def _consume(self) -> Optional[str]:
        """Consumes and returns the current token.

        Returns:
            The current token, or None if at end of stream.
        """
        if self._pos >= self._length:
            return None
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, expected: str, case_sensitive: bool = True) -> str:
        """Consumes the current token and verifies it matches the expected value.

        Args:
            expected: The expected token value.
            case_sensitive: If False, comparison is case-insensitive.

        Returns:
            The consumed token.

        Raises:
            ValueError: If the token doesn't match the expected value.
        """
        token = self._consume()
        if case_sensitive:
            matches = token == expected
        else:
            matches = token is not None and token.upper() == expected.upper()

        if not matches:
            raise ValueError(f"Expected '{expected}', got '{token}' at position {self._pos}")
        return token

    def _skip_semicolon(self) -> None:
        """Consumes a semicolon if present at the current position."""
        if self._peek() == ";":
            self._consume()

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
