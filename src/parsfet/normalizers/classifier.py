"""Cell Type Classifier.

This module categorizes standard cells based on their logical function.
It parses boolean function strings into an AST, evaluates truth tables,
and classifies cells by comparing signatures to known gate patterns.

This semantic approach correctly handles:
- De Morgan equivalents: !(A|B) == !A & !B → 'nor'
- Multiple syntax variants: A', ~A, !A → all recognized as NOT
- Arbitrary input counts up to MAX_INPUTS
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.liberty import Cell


# --- Configuration ---

MAX_INPUTS = 6  # Beyond this, fall back to 'unknown' (2^6 = 64 evaluations)


# --- Cell Type Enum ---


class CellType(Enum):
    """Standard cell type classification.

    Enum values represent the semantic category of a standard cell
    based on its boolean function or sequential behavior.
    """

    INVERTER = auto()
    BUFFER = auto()
    AND = auto()
    OR = auto()
    NAND = auto()
    NOR = auto()
    XOR = auto()
    XNOR = auto()
    FLIP_FLOP = auto()
    LATCH = auto()
    CONSTANT = auto()
    UNKNOWN = auto()


# --- AST Node Classes ---


class ASTNode(ABC):
    """Abstract base class for boolean expression AST nodes."""

    @abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        """Evaluate node given variable assignments (0 or 1)."""

    @abstractmethod
    def get_variables(self) -> set[str]:
        """Return set of all variable names in this subtree."""


class VarNode(ASTNode):
    """Variable reference node."""

    def __init__(self, name: str):
        self.name = name.upper()  # Normalize to uppercase

    def evaluate(self, env: dict[str, int]) -> int:
        return env.get(self.name, 0)

    def get_variables(self) -> set[str]:
        return {self.name}


class NotNode(ASTNode):
    """Logical NOT node."""

    def __init__(self, child: ASTNode):
        self.child = child

    def evaluate(self, env: dict[str, int]) -> int:
        return 1 - self.child.evaluate(env)

    def get_variables(self) -> set[str]:
        return self.child.get_variables()


class BinaryOpNode(ASTNode):
    """Base class for binary operations."""

    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def get_variables(self) -> set[str]:
        return self.left.get_variables() | self.right.get_variables()


class AndNode(BinaryOpNode):
    """Logical AND node."""

    def evaluate(self, env: dict[str, int]) -> int:
        return self.left.evaluate(env) & self.right.evaluate(env)


class OrNode(BinaryOpNode):
    """Logical OR node."""

    def evaluate(self, env: dict[str, int]) -> int:
        return self.left.evaluate(env) | self.right.evaluate(env)


class XorNode(BinaryOpNode):
    """Logical XOR node."""

    def evaluate(self, env: dict[str, int]) -> int:
        return self.left.evaluate(env) ^ self.right.evaluate(env)


# --- Tokenizer ---


def _tokenize(func: str) -> list[str]:
    """Convert a function string to a list of tokens.

    Handles operators: ! ~ (NOT), & * (AND), | + (OR), ^ (XOR), ( )
    Also handles postfix negation: A' → becomes ['A', "!'"]
    """
    # Normalize whitespace
    func = func.replace(" ", "")

    # Replace ~ with ! for consistency
    func = func.replace("~", "!")

    tokens: list[str] = []
    i = 0

    while i < len(func):
        c = func[i]

        if c in "!&*|+^()":
            tokens.append(c)
            i += 1
        elif c == "'":
            # Postfix negation: apply NOT to previous token/expression
            # We'll handle this as a special marker
            tokens.append("!'")
            i += 1
        elif c.isalnum() or c == "_":
            # Variable name: consume all alphanumeric chars
            j = i
            while j < len(func) and (func[j].isalnum() or func[j] == "_"):
                j += 1
            tokens.append(func[i:j])
            i = j
        else:
            # Skip unknown characters
            i += 1

    return tokens


# --- Recursive Descent Parser ---
#
# Grammar (with correct operator precedence):
#   expr      = xor_term (('|' | '+') xor_term)*
#   xor_term  = term ('^' term)*
#   term      = factor (('&' | '*') factor)*
#   factor    = '!' factor | primary ("!'")*
#   primary   = '(' expr ')' | VAR


class ParseError(Exception):
    """Raised when parsing fails."""


def _parse(tokens: list[str]) -> ASTNode:
    """Parse tokens into an AST."""
    pos = [0]  # Mutable position counter

    def peek() -> str | None:
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume() -> str:
        token = tokens[pos[0]]
        pos[0] += 1
        return token

    def parse_expr() -> ASTNode:
        """expr = xor_term (('|' | '+') xor_term)*"""
        left = parse_xor_term()
        while peek() in ("|", "+"):
            consume()
            right = parse_xor_term()
            left = OrNode(left, right)
        return left

    def parse_xor_term() -> ASTNode:
        """xor_term = term ('^' term)*"""
        left = parse_term()
        while peek() == "^":
            consume()
            right = parse_term()
            left = XorNode(left, right)
        return left

    def parse_term() -> ASTNode:
        """term = factor (('&' | '*') factor)*"""
        left = parse_factor()
        while peek() in ("&", "*"):
            consume()
            right = parse_factor()
            left = AndNode(left, right)
        return left

    def parse_factor() -> ASTNode:
        """factor = '!' factor | primary ("!'")*"""
        if peek() == "!":
            consume()
            return NotNode(parse_factor())
        return parse_primary()

    def parse_primary() -> ASTNode:
        """primary = '(' expr ')' | VAR, with optional postfix negation"""
        token = peek()

        if token == "(":
            consume()  # '('
            node = parse_expr()
            if peek() == ")":
                consume()  # ')'
            # Handle postfix negation on parenthesized expression
            while peek() == "!'":
                consume()
                node = NotNode(node)
            return node
        elif token is not None and token not in "!&*|+^()":
            consume()
            node: ASTNode = VarNode(token)
            # Handle postfix negation: A' → NOT(A)
            while peek() == "!'":
                consume()
                node = NotNode(node)
            return node
        else:
            raise ParseError(f"Unexpected token: {token}")

    if not tokens:
        raise ParseError("Empty expression")

    result = parse_expr()

    # Handle any remaining postfix negations
    while peek() == "!'":
        consume()
        result = NotNode(result)

    return result


# --- Truth Table Evaluation ---


def _compute_signature(ast: ASTNode) -> tuple[int, ...]:
    """Compute truth table signature for the AST.

    Variables are sorted alphabetically for canonical ordering.
    Returns tuple of output values for all 2^N input combinations.
    """
    variables = sorted(ast.get_variables())  # Canonical ordering

    if len(variables) > MAX_INPUTS:
        raise ValueError(f"Too many inputs: {len(variables)} > {MAX_INPUTS}")

    if len(variables) == 0:
        # Constant expression
        return (ast.evaluate({}),)

    results: list[int] = []
    for i in range(2 ** len(variables)):
        env = {var: (i >> j) & 1 for j, var in enumerate(variables)}
        results.append(ast.evaluate(env))

    return tuple(results)


# --- Signature Lookup Tables ---

# 1-input signatures (2 entries)
SIGNATURES_1: dict[tuple[int, ...], CellType] = {
    (1, 0): CellType.INVERTER,  # !A
    (0, 1): CellType.BUFFER,  # A
}

# 2-input signatures (4 entries: A=bit0, B=bit1)
SIGNATURES_2: dict[tuple[int, ...], CellType] = {
    (0, 0, 0, 1): CellType.AND,  # A & B
    (0, 1, 1, 1): CellType.OR,  # A | B
    (1, 1, 1, 0): CellType.NAND,  # !(A & B)
    (1, 0, 0, 0): CellType.NOR,  # !(A | B)
    (0, 1, 1, 0): CellType.XOR,  # A ^ B
    (1, 0, 0, 1): CellType.XNOR,  # !(A ^ B)
}

# 3-input signatures (8 entries)
SIGNATURES_3: dict[tuple[int, ...], CellType] = {
    (0, 0, 0, 0, 0, 0, 0, 1): CellType.AND,  # A & B & C
    (0, 1, 1, 1, 1, 1, 1, 1): CellType.OR,  # A | B | C
    (1, 1, 1, 1, 1, 1, 1, 0): CellType.NAND,  # !(A & B & C)
    (1, 0, 0, 0, 0, 0, 0, 0): CellType.NOR,  # !(A | B | C)
}

# 4-input signatures (16 entries)
SIGNATURES_4: dict[tuple[int, ...], CellType] = {
    tuple([0] * 15 + [1]): CellType.AND,  # A & B & C & D
    tuple([0] + [1] * 15): CellType.OR,  # A | B | C | D
    tuple([1] * 15 + [0]): CellType.NAND,  # !(A & B & C & D)
    tuple([1] + [0] * 15): CellType.NOR,  # !(A | B | C | D)
}


def _lookup_signature(sig: tuple[int, ...]) -> CellType:
    """Look up gate type from signature."""
    n = len(sig)

    # Constant outputs
    if all(v == 0 for v in sig):
        return CellType.CONSTANT
    if all(v == 1 for v in sig):
        return CellType.CONSTANT

    if n == 2:
        return SIGNATURES_1.get(sig, CellType.UNKNOWN)
    elif n == 4:
        return SIGNATURES_2.get(sig, CellType.UNKNOWN)
    elif n == 8:
        return SIGNATURES_3.get(sig, CellType.UNKNOWN)
    elif n == 16:
        return SIGNATURES_4.get(sig, CellType.UNKNOWN)
    else:
        return CellType.UNKNOWN


# --- Main Classification Functions ---


@lru_cache(maxsize=2048)
def classify_function(func: str) -> CellType:
    """Classify a boolean function string by semantic analysis.

    Parses the function into an AST, evaluates its truth table,
    and looks up the signature to determine the gate type.

    Args:
        func: Boolean function string (e.g., "!(A & B)", "A | B", "!A")

    Returns:
        CellType enum: INVERTER, BUFFER, AND, OR, NAND, NOR,
                       XOR, XNOR, CONSTANT, or UNKNOWN
    """
    if not func or not func.strip():
        return CellType.UNKNOWN

    try:
        tokens = _tokenize(func)
        if not tokens:
            return CellType.UNKNOWN

        ast = _parse(tokens)
        signature = _compute_signature(ast)
        return _lookup_signature(signature)

    except (ParseError, ValueError, IndexError):
        return CellType.UNKNOWN


def classify_cell(cell: "Cell") -> CellType:
    """Classifies a cell type based on its pin directions and logical functions.

    Identifies standard gates (INV, NAND, NOR, etc.) and sequential elements
    (DFF, Latch). Cells with complex or unrecognizable logic are labeled UNKNOWN.

    Args:
        cell: The Liberty Cell object to classify.

    Returns:
        CellType enum: INVERTER, BUFFER, NAND, NOR, AND, OR, XOR, XNOR,
                       FLIP_FLOP, LATCH, CONSTANT, or UNKNOWN
    """
    # Check for sequential elements first (ff/latch groups)
    if cell.is_sequential:
        # Check timing types for latch vs flip-flop
        if any("latch" in str(arc.timing_type or "").lower() for arc in cell.timing_arcs):
            return CellType.LATCH
        return CellType.FLIP_FLOP

    # Get output pins to analyze combinational functions
    output_pins = [p for p in cell.pins.values() if p.direction == "output"]

    # Classify based on output pin functions
    for pin in output_pins:
        if not pin.function:
            continue

        result = classify_function(pin.function.strip())
        if result != CellType.UNKNOWN:
            return result

    return CellType.UNKNOWN
