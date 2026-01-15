"""Cell Type Classifier.

This module categorizes standard cells based on their logical function.
It analyzes pin directions and boolean function strings to determine if a cell
is a basic gate (INV, NAND, NOR, etc.), a sequential element (DFF, Latch),
or a complex logic cell.

This classification is crucial for:
- Finding equivalent cells across libraries.
- Generating technology fingerprints.
- Normalizing metrics by cell type.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.liberty import Cell


# Pattern matchers for logic functions
# Handle multiple syntax variants: Synopsys, Cadence, Verilog

def _normalize_function(func: str) -> str:
    """Normalizes boolean function strings for consistent pattern matching.

    Removes whitespace, standardizes negation symbols (e.g., A', ~A -> !A),
    and strips redundant outer parentheses.
    """
    # Remove whitespace
    func = func.replace(" ", "")
    # Normalize negation syntax: A' → !A, ~A → !A
    func = re.sub(r"(\w)'", r"!\1", func)
    func = func.replace("~", "!")
    # Strip outer parentheses: (expr) → expr
    while func.startswith("(") and func.endswith(")"):
        # Check if the parens are balanced (not like "(A)|(B)")
        depth = 0
        balanced = True
        for i, c in enumerate(func[1:-1]):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
        if balanced and depth == 0:
            func = func[1:-1]
        else:
            break
    return func


def _get_input_vars(func: str) -> set[str]:
    """Extracts variable names from a function string."""
    # Find all single uppercase letters that aren't operators
    return set(re.findall(r"\b([A-Z])\b", func.upper()))


def is_negation_only(func: str) -> bool:
    """Checks if the function represents a pure inverter (negation)."""
    func = _normalize_function(func)
    # Patterns: !A, (!A), !(A), !VAR, (!VAR), !(VAR)
    # Support multi-char variable names like A1, IN
    patterns = [
        r"^!([A-Za-z]\w*)$",        # !A, !A1, !IN
        r"^\(!([A-Za-z]\w*)\)$",    # (!A), (!A1)
        r"^!\(([A-Za-z]\w*)\)$",    # !(A), !(A1)
    ]
    return any(re.match(p, func, re.I) for p in patterns)


def is_identity_only(func: str) -> bool:
    """Checks if the function represents a buffer (identity)."""
    func = _normalize_function(func)
    # Just a single variable: A, A1, IN
    return bool(re.match(r"^[A-Za-z]\w*$", func))


def is_and_gate(func: str) -> bool:
    """Checks if the function represents an AND gate."""
    func = _normalize_function(func)
    # Patterns: A&B, A*B, (A&B), A1&A2, etc.
    # Support multi-char variable names
    var = r"[A-Za-z]\w*"  # Variable pattern
    patterns = [
        rf"^\(?{var}[&*]{var}([&*]{var})?\)?$",
    ]
    if any(re.match(p, func, re.I) for p in patterns):
        # Verify no negations
        if "!" not in func:
            return True
    return False


def is_or_gate(func: str) -> bool:
    """Checks if the function represents an OR gate."""
    func = _normalize_function(func)
    var = r"[A-Za-z]\w*"
    patterns = [
        rf"^\(?{var}[|+]{var}([|+]{var})?\)?$",
    ]
    if any(re.match(p, func, re.I) for p in patterns):
        if "!" not in func:
            return True
    return False


def is_nand_gate(func: str) -> bool:
    """Checks if the function represents a NAND gate."""
    func = _normalize_function(func)
    var = r"[A-Za-z]\w*"
    # Patterns: !(A&B), !(A*B), !(A1&A2)
    patterns = [
        rf"^!\({var}[&*]{var}([&*]{var})?\)$",
    ]
    return any(re.match(p, func, re.I) for p in patterns)


def is_nor_gate(func: str) -> bool:
    """Checks if the function represents a NOR gate."""
    func = _normalize_function(func)
    var = r"[A-Za-z]\w*"
    # Patterns: !(A|B), !(A+B), !(A1|A2)
    patterns = [
        rf"^!\({var}[|+]{var}([|+]{var})?\)$",
    ]
    return any(re.match(p, func, re.I) for p in patterns)


def is_xor_gate(func: str) -> bool:
    """Checks if the function represents an XOR gate."""
    func = _normalize_function(func)
    # XOR is often written as: A^B or complex expressions
    if "^" in func:
        return True
    # Also check for XOR expansion: (A&!B)|(!A&B)
    xor_expanded = r"^\(?[A-Za-z][&*]![A-Za-z]\)?[|+]\(?![A-Za-z][&*][A-Za-z]\)?$"
    return bool(re.match(xor_expanded, func, re.I))


def classify_cell(cell: "Cell") -> str:
    """Classifies a cell type based on its pin directions and logical functions.

    Identifies standard gates (INV, NAND, NOR, etc.) and sequential elements (DFF, Latch).
    Cells with complex or unrecognizable logic are labeled 'unknown'.

    Args:
        cell: The Liberty Cell object to classify.

    Returns:
        A string representing the cell type: 'inverter', 'buffer', 'nand', 'nor',
        'and', 'or', 'xor', 'flip_flop', 'latch', or 'unknown'.
    """
    # Get input/output pins by direction
    input_pins = [p for p in cell.pins.values() if p.direction == "input"]
    output_pins = [p for p in cell.pins.values() if p.direction == "output"]

    # Check for sequential elements first (ff/latch groups)
    if cell.is_sequential:
        # Check timing types for latch vs flip-flop
        if any('latch' in str(arc.timing_type or '').lower()
               for arc in cell.timing_arcs):
            return "latch"
        return "flip_flop"

    # Parse combinational functions from output pins
    for pin in output_pins:
        if not pin.function:
            continue

        func = pin.function.strip()
        num_inputs = len(input_pins)

        # Inverter: single input + pure negation
        if num_inputs == 1 and is_negation_only(func):
            return "inverter"

        # Buffer: single input + identity
        if num_inputs == 1 and is_identity_only(func):
            return "buffer"

        # Simple gates: 2-3 inputs, basic operations
        if num_inputs in (2, 3):
            if is_nand_gate(func):
                return "nand"
            if is_nor_gate(func):
                return "nor"
            if is_and_gate(func):
                return "and"
            if is_or_gate(func):
                return "or"
            if is_xor_gate(func):
                return "xor"

    return "unknown"  # Complex or unrecognized
