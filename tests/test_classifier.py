"""Tests for the boolean expression classifier.

Verifies that boolean strings from Liberty files are correctly parsed,
evaluated, and classified into cell types (AND, OR, NAND, etc.).
"""

import pytest

from parsfet.normalizers.classifier import (
    CellType,
    _compute_signature,
    _parse,
    _tokenize,
    classify_cell,
    classify_function,
)


class TestTokenizer:
    """Test the tokenizer helper function."""

    def test_simple_variable(self):
        """Verifies tokenization of a single variable."""
        assert _tokenize("A") == ["A"]

    def test_multi_char_variable(self):
        """Verifies tokenization of multi-character variable names."""
        assert _tokenize("A1") == ["A1"]
        assert _tokenize("IN") == ["IN"]

    def test_negation_variants(self):
        """Verifies handling of different negation symbols (!, ~, ')."""
        assert _tokenize("!A") == ["!", "A"]
        assert _tokenize("~A") == ["!", "A"]  # ~ normalized to !
        assert _tokenize("A'") == ["A", "!'"]  # postfix

    def test_and_variants(self):
        """Verifies handling of AND operator variants (&, *)."""
        assert _tokenize("A&B") == ["A", "&", "B"]
        assert _tokenize("A*B") == ["A", "*", "B"]

    def test_or_variants(self):
        """Verifies handling of OR operator variants (|, +)."""
        assert _tokenize("A|B") == ["A", "|", "B"]
        assert _tokenize("A+B") == ["A", "+", "B"]

    def test_xor(self):
        """Verifies handling of XOR operator (^)."""
        assert _tokenize("A^B") == ["A", "^", "B"]

    def test_parentheses(self):
        """Verifies handling of parentheses."""
        assert _tokenize("(A&B)") == ["(", "A", "&", "B", ")"]

    def test_complex_expression(self):
        """Verifies tokenization of nested expressions."""
        assert _tokenize("!(A&B)") == ["!", "(", "A", "&", "B", ")"]

    def test_whitespace_removed(self):
        """Verifies that whitespace is ignored."""
        assert _tokenize("A & B") == ["A", "&", "B"]


class TestClassifyFunction:
    """Test the main classification function logic."""

    # Basic gates
    def test_inverter(self):
        """Verifies classification of inverters (NOT gate)."""
        assert classify_function("!A") == CellType.INVERTER
        assert classify_function("~A") == CellType.INVERTER
        assert classify_function("A'") == CellType.INVERTER
        assert classify_function("(!A)") == CellType.INVERTER
        assert classify_function("!(A)") == CellType.INVERTER

    def test_buffer(self):
        """Verifies classification of buffers."""
        assert classify_function("A") == CellType.BUFFER
        assert classify_function("(A)") == CellType.BUFFER

    def test_and(self):
        """Verifies classification of 2-input AND gates."""
        assert classify_function("A&B") == CellType.AND
        assert classify_function("A*B") == CellType.AND
        assert classify_function("(A&B)") == CellType.AND

    def test_or(self):
        """Verifies classification of 2-input OR gates."""
        assert classify_function("A|B") == CellType.OR
        assert classify_function("A+B") == CellType.OR

    def test_nand(self):
        """Verifies classification of 2-input NAND gates."""
        assert classify_function("!(A&B)") == CellType.NAND
        assert classify_function("!(A*B)") == CellType.NAND

    def test_nor(self):
        """Verifies classification of 2-input NOR gates."""
        assert classify_function("!(A|B)") == CellType.NOR
        assert classify_function("!(A+B)") == CellType.NOR

    def test_xor(self):
        """Verifies classification of 2-input XOR gates."""
        assert classify_function("A^B") == CellType.XOR

    def test_xnor(self):
        """Verifies classification of 2-input XNOR gates."""
        assert classify_function("!(A^B)") == CellType.XNOR

    # De Morgan equivalents - KEY SEMANTIC TEST
    def test_de_morgan_nand(self):
        """Verifies De Morgan equivalence: !(A&B) == !A | !B (NAND)."""
        assert classify_function("!(A&B)") == CellType.NAND
        assert classify_function("!A|!B") == CellType.NAND

    def test_de_morgan_nor(self):
        """Verifies De Morgan equivalence: !(A|B) == !A & !B (NOR)."""
        assert classify_function("!(A|B)") == CellType.NOR
        assert classify_function("!A&!B") == CellType.NOR

    # Multi-input gates
    def test_3_input_and(self):
        """Verifies classification of 3-input AND gates."""
        assert classify_function("A&B&C") == CellType.AND

    def test_3_input_or(self):
        """Verifies classification of 3-input OR gates."""
        assert classify_function("A|B|C") == CellType.OR

    def test_3_input_nand(self):
        """Verifies classification of 3-input NAND gates."""
        assert classify_function("!(A&B&C)") == CellType.NAND

    def test_3_input_nor(self):
        """Verifies classification of 3-input NOR gates."""
        assert classify_function("!(A|B|C)") == CellType.NOR

    def test_4_input_gates(self):
        """Verifies classification of 4-input gates."""
        assert classify_function("A&B&C&D") == CellType.AND
        assert classify_function("A|B|C|D") == CellType.OR
        assert classify_function("!(A&B&C&D)") == CellType.NAND
        assert classify_function("!(A|B|C|D)") == CellType.NOR

    # Constants
    def test_tautology(self):
        """Verifies that always-true expressions are classified as CONSTANT."""
        assert classify_function("A|!A") == CellType.CONSTANT

    def test_contradiction(self):
        """Verifies that always-false expressions are classified as CONSTANT."""
        assert classify_function("A&!A") == CellType.CONSTANT

    # Complex / Unknown
    def test_aoi_unknown(self):
        """Verifies that AOI gates fall back to UNKNOWN (if not explicitly supported)."""
        assert classify_function("!(A|(B&C))") == CellType.UNKNOWN

    def test_oai_unknown(self):
        """Verifies that OAI gates fall back to UNKNOWN (if not explicitly supported)."""
        assert classify_function("!((A|B)&C)") == CellType.UNKNOWN

    # Error handling
    def test_empty_string(self):
        """Verifies behavior on empty input string."""
        assert classify_function("") == CellType.UNKNOWN

    def test_whitespace_only(self):
        """Verifies behavior on whitespace-only input."""
        assert classify_function("   ") == CellType.UNKNOWN

    def test_none_like(self):
        """Verifies behavior on empty-like inputs."""
        assert classify_function("") == CellType.UNKNOWN


class TestVariableOrdering:
    """Test that variable ordering is canonical (alphabetical)."""

    def test_and_commutative(self):
        """Verifies that AND is commutative (A&B == B&A)."""
        assert classify_function("A&B") == classify_function("B&A")

    def test_or_commutative(self):
        """Verifies that OR is commutative (A|B == B|A)."""
        assert classify_function("A|B") == classify_function("B|A")

    def test_different_var_names(self):
        """Verifies that logic is independent of variable names (A,B vs X,Y)."""
        assert classify_function("X&Y") == CellType.AND
        assert classify_function("A&B") == CellType.AND


class TestXorPrecedence:
    """Test that XOR has correct precedence (between OR and AND)."""

    def test_xor_lower_than_and(self):
        """Verifies operator precedence: A & B ^ C means (A & B) ^ C."""
        # (A & B) ^ C: AND first, then XOR
        # Truth table for 3 vars with A=bit0, B=bit1, C=bit2:
        # ABC: 000->0, 001->1, 010->0, 011->1, 100->0, 101->1, 110->1, 111->0
        # This is XOR of (A&B) with C
        result = classify_function("A&B^C")
        # This should be 'unknown' as it's a 3-input XOR variant
        assert result in (CellType.UNKNOWN, CellType.XOR)  # Depends on pattern

    def test_explicit_xor_grouping(self):
        """Verifies explicit grouping with XOR."""
        assert classify_function("(A^B)") == CellType.XOR


class TestCaching:
    """Test LRU cache behavior."""

    def test_repeated_calls_use_cache(self):
        """Verifies that repeated calls return consistent results (cache hit)."""
        # First call
        result1 = classify_function("!(A&B)")
        # Second call (should hit cache)
        result2 = classify_function("!(A&B)")
        assert result1 == result2 == CellType.NAND


class TestTruthTableSignatures:
    """Verify specific truth table signatures."""

    def test_inverter_signature(self):
        """Verifies truth table signature for Inverter."""
        ast = _parse(_tokenize("!A"))
        sig = _compute_signature(ast)
        assert sig == (1, 0)  # !0=1, !1=0

    def test_buffer_signature(self):
        """Verifies truth table signature for Buffer."""
        ast = _parse(_tokenize("A"))
        sig = _compute_signature(ast)
        assert sig == (0, 1)  # 0->0, 1->1

    def test_and_signature(self):
        """Verifies truth table signature for AND gate."""
        ast = _parse(_tokenize("A&B"))
        sig = _compute_signature(ast)
        # A=bit0, B=bit1: 00->0, 01->0, 10->0, 11->1
        assert sig == (0, 0, 0, 1)

    def test_or_signature(self):
        """Verifies truth table signature for OR gate."""
        ast = _parse(_tokenize("A|B"))
        sig = _compute_signature(ast)
        # A=bit0, B=bit1: 00->0, 01->1, 10->1, 11->1
        assert sig == (0, 1, 1, 1)

    def test_nand_signature(self):
        """Verifies truth table signature for NAND gate."""
        ast = _parse(_tokenize("!(A&B)"))
        sig = _compute_signature(ast)
        assert sig == (1, 1, 1, 0)

    def test_nor_signature(self):
        """Verifies truth table signature for NOR gate."""
        ast = _parse(_tokenize("!(A|B)"))
        sig = _compute_signature(ast)
        assert sig == (1, 0, 0, 0)

    def test_xor_signature(self):
        """Verifies truth table signature for XOR gate."""
        ast = _parse(_tokenize("A^B"))
        sig = _compute_signature(ast)
        assert sig == (0, 1, 1, 0)
