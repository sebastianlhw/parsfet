
import pytest
from parsfet.models.liberty import Cell, PowerArc, LookupTable
from parsfet.models.export import ExportedCell, ExportedPowerModel

def test_power_arc_linear_model():
    """Test standard linear power model extraction from a PowerArc."""
    # Data: E = 0.5 + 0.1 * Load
    loads = [0.0, 10.0, 20.0]
    energies = [0.5, 1.5, 2.5]  # Matches exactly

    arc = PowerArc(
        related_pin="A",
        rise_power=LookupTable(
            index_1=[0.1],  # Slew
            index_2=loads,  # Load
            values=[energies] # 2D table: 1 slew x 3 loads
        ),
        fall_power=LookupTable(
            index_1=[0.1],
            index_2=loads,
            values=[energies]
        )
    )

    # Test at slew=0.1
    e0, k, r2 = arc.linear_power_model(slew=0.1)

    assert e0 == pytest.approx(0.5)
    assert k == pytest.approx(0.1)
    assert r2 == pytest.approx(1.0)

def test_cell_power_aggregation():
    """Test that Cell.linear_power_model picks the worst-case (max E0) arc."""
    # Arc 1: E = 0.1 + 0.05 * Load (Low energy)
    arc1 = PowerArc(
        related_pin="A",
        rise_power=LookupTable(
            index_1=[0.1],
            index_2=[0, 10],
            values=[[0.1, 0.6]]
        )
    )

    # Arc 2: E = 1.0 + 0.2 * Load (High energy)
    arc2 = PowerArc(
        related_pin="B",
        rise_power=LookupTable(
            index_1=[0.1],
            index_2=[0, 10],
            values=[[1.0, 3.0]]
        )
    )

    cell = Cell(
        name="TEST_CELL",
        power_arcs=[arc1, arc2]
    )

    e0, k, r2 = cell.linear_power_model(slew=0.1)

    # Should pick Arc 2 because E0=1.0 > E0=0.1
    assert e0 == pytest.approx(1.0)
    assert k == pytest.approx(0.2)
    assert r2 == pytest.approx(1.0)

def test_export_model_structure():
    """Test that ExportedCell can hold the new power model."""
    pm = ExportedPowerModel(
        e0_unit=0.5,
        k_unit_per_pf=0.01
    )
    
    ec = ExportedCell(
        cell_name="TEST",
        power_model=pm
    )

    assert ec.power_model is not None
    assert ec.power_model.e0_unit == 0.5
    assert ec.power_model.k_unit_per_pf == 0.01
