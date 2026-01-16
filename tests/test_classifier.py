import pytest
from parsfet.models.liberty import Cell, Pin, TimingArc
from parsfet.normalizers.classifier import classify_cell

def create_cell(name, pins_dict, is_sequential=False):
    """Helper to create a cell with pins and function."""
    pins = {}
    for pin_name, attrs in pins_dict.items():
        direction = attrs.get("direction", "input")
        function = attrs.get("function", None)
        clock = attrs.get("clock", False)

        pins[pin_name] = Pin(
            name=pin_name,
            direction=direction,
            function=function,
            clock=clock
        )

    timing_arcs = []
    if is_sequential:
        # Add a dummy timing arc for latch detection
        # Logic in classifier: if any('latch' in str(arc.timing_type or '').lower() ...)
        pass

    return Cell(
        name=name,
        pins=pins,
        is_sequential=is_sequential,
        timing_arcs=timing_arcs
    )

def test_classify_inverter():
    # Standard inverter
    cell = create_cell("INV", {
        "A": {"direction": "input"},
        "Y": {"direction": "output", "function": "!A"}
    })
    assert classify_cell(cell) == "inverter"

    # With whitespace
    cell = create_cell("INV_S", {
        "A": {"direction": "input"},
        "Y": {"direction": "output", "function": " ! A "}
    })
    assert classify_cell(cell) == "inverter"

    # Different syntax
    cell = create_cell("INV_P", {
        "A": {"direction": "input"},
        "Y": {"direction": "output", "function": "!(A)"}
    })
    assert classify_cell(cell) == "inverter"

    # Apostrophe syntax
    cell = create_cell("INV_AP", {
        "A": {"direction": "input"},
        "Y": {"direction": "output", "function": "A'"}
    })
    assert classify_cell(cell) == "inverter"

def test_classify_buffer():
    cell = create_cell("BUF", {
        "A": {"direction": "input"},
        "Y": {"direction": "output", "function": "A"}
    })
    assert classify_cell(cell) == "buffer"

def test_classify_nand():
    # 2-input NAND
    cell = create_cell("NAND2", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "!(A&B)"}
    })
    assert classify_cell(cell) == "nand"

    # 3-input NAND
    cell = create_cell("NAND3", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "C": {"direction": "input"},
        "Y": {"direction": "output", "function": "!(A*B*C)"}
    })
    assert classify_cell(cell) == "nand"

def test_classify_nor():
    # 2-input NOR
    cell = create_cell("NOR2", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "!(A+B)"}
    })
    assert classify_cell(cell) == "nor"

    # With pipe
    cell = create_cell("NOR2_P", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "!(A|B)"}
    })
    assert classify_cell(cell) == "nor"

def test_classify_and():
    cell = create_cell("AND2", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "A&B"}
    })
    assert classify_cell(cell) == "and"

def test_classify_or():
    cell = create_cell("OR2", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "A+B"}
    })
    assert classify_cell(cell) == "or"

def test_classify_xor():
    # Using caret
    cell = create_cell("XOR2", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "A^B"}
    })
    assert classify_cell(cell) == "xor"

    # Expanded form (A&!B)|(!A&B)
    cell = create_cell("XOR2_E", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "(A&!B)+(!A&B)"}
    })
    assert classify_cell(cell) == "xor"

def test_classify_sequential():
    # D Flip-Flop
    cell = create_cell("DFF", {
        "D": {"direction": "input"},
        "CLK": {"direction": "input", "clock": True},
        "Q": {"direction": "output", "function": "IQ"}
    }, is_sequential=True)

    assert classify_cell(cell) == "flip_flop"

    # Latch (detected by timing_type)
    cell_latch = create_cell("DLAT", {
        "D": {"direction": "input"},
        "G": {"direction": "input"},
        "Q": {"direction": "output"}
    }, is_sequential=True)

    # Add timing arc with 'latch' type
    cell_latch.timing_arcs.append(
        TimingArc(related_pin="G", timing_type="clear_latch")
    )

    assert classify_cell(cell_latch) == "latch"

def test_classify_unknown():
    # Complex function not in standard list
    cell = create_cell("COMPLEX", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "C": {"direction": "input"},
        "Y": {"direction": "output", "function": "(A+B)&C"} # AOI/OAI type logic
    })
    # Current classifier might not handle AOI/OAI specifically, or treat as unknown
    # The code checks for basic gates. If not matched, returns unknown.
    # Note: src/parsfet/normalizers/classifier.py implementation doesn't seem to have AOI/OAI matchers in classify_cell
    # although _same_function_type in cell_diff.py lists them.
    assert classify_cell(cell) == "unknown"

    # Mismatch inputs count (e.g. Inverter with 2 inputs?)
    cell = create_cell("BAD_INV", {
        "A": {"direction": "input"},
        "B": {"direction": "input"},
        "Y": {"direction": "output", "function": "!A"}
    })
    # 2 inputs, but function only uses A. classify_cell checks num_inputs matches expected for gate.
    # For inverter, expects num_inputs == 1.
    assert classify_cell(cell) == "unknown"
