
import numpy as np
import pytest
from hypothesis import given, strategies as st
from parsfet.models.liberty import LookupTable
from parsfet.reporting.html_generator import interpolate_1d_at_slew, interpolate_1d_at_load

# Strategies for generating valid 2D LookupTables
@st.composite
def lookup_table_2d_strategy(draw):
    # Monotonic indices
    size_1 = draw(st.integers(min_value=2, max_value=5))
    size_2 = draw(st.integers(min_value=2, max_value=5))
    
    index_1 = sorted(draw(st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=size_1, max_size=size_1, unique=True)))
    index_2 = sorted(draw(st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=size_2, max_size=size_2, unique=True)))
    
    # Values array (flattened or nested? Model expects nested list usually, but code converts to np.array)
    # The code does `values = np.array(table.values)`
    # Let's generate a flat list of lists
    values = []
    for _ in range(size_1):
        row = draw(st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=size_2, max_size=size_2))
        values.append(row)
        
    return LookupTable(
        index_1=index_1,
        index_2=index_2,
        values=values
    )

def test_interpolate_at_slew_exact_match():
    """Verify exact match when target slew exists in index_1."""
    lut = LookupTable(
        index_1=[1.0, 2.0],
        index_2=[10.0, 20.0],
        values=[[5.0, 6.0], [7.0, 8.0]] 
    )
    
    # At slew=1.0, should return first row
    loads, values = interpolate_1d_at_slew(lut, 1.0)
    assert loads == [10.0, 20.0]
    assert values == [5.0, 6.0]
    
    # At slew=2.0, should return second row
    loads, values = interpolate_1d_at_slew(lut, 2.0)
    assert values == [7.0, 8.0]

def test_interpolate_at_slew_midpoint():
    """Verify linear interpolation at midpoint."""
    # z = x + y
    # x (slew) = [0, 10], y (load) = [0, 10]
    # values:
    # x=0: [0, 10]
    # x=10: [10, 20]
    lut = LookupTable(
        index_1=[0.0, 10.0],
        index_2=[0.0, 10.0],
        values=[[0.0, 10.0], [10.0, 20.0]]
    )
    
    # At x=5.0, should be average of rows: [5.0, 15.0]
    loads, values = interpolate_1d_at_slew(lut, 5.0)
    assert loads == [0.0, 10.0]
    assert values == [5.0, 15.0]

@given(lut=lookup_table_2d_strategy(), target_slew=st.floats(min_value=0.0, max_value=12.0))
def test_interpolate_at_slew_structure(lut, target_slew):
    """Hypothesis test: Ensure structural validity of output."""
    loads, values = interpolate_1d_at_slew(lut, target_slew)
    
    assert loads == lut.index_2
    assert len(values) == len(loads)
    assert all(isinstance(v, float) for v in values)

def test_interpolate_at_load_exact_match():
    """Verify exact match when target load exists in index_2."""
    lut = LookupTable(
        index_1=[1.0, 2.0], # Slew
        index_2=[10.0, 20.0], # Load
        values=[[5.0, 6.0],  # Slew 1
                [7.0, 8.0]]  # Slew 2
    )
    
    # At load=10.0, should get column 0: [5.0, 7.0]
    slews, values = interpolate_1d_at_load(lut, 10.0)
    assert slews == [1.0, 2.0]
    assert values == [5.0, 7.0]

def test_interpolate_at_load_midpoint():
    """Verify interpolation at load midpoint."""
    # Same z = x + y setup
    lut = LookupTable(
        index_1=[0.0, 10.0],
        index_2=[0.0, 10.0],
        values=[[0.0, 10.0], [10.0, 20.0]]
    )
    
    # At load=5.0
    # Row 0 (x=0): interp(0, 10) at 0.5 -> 5.0
    # Row 1 (x=10): interp(10, 20) at 0.5 -> 15.0
    slews, values = interpolate_1d_at_load(lut, 5.0)
    assert slews == [0.0, 10.0]
    assert values == [5.0, 15.0]

def test_empty_or_invalid_table():
    """Verify graceful handling of empty/invalid tables."""
    assert interpolate_1d_at_slew(None, 1.0) == (None, None)
    
    empty_lut = LookupTable(index_1=[], index_2=[], values=[])
    assert interpolate_1d_at_slew(empty_lut, 1.0) == (None, None)
