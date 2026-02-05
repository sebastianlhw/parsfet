"""Pytest configuration and fixtures.

Provides shared fixtures for Liberty and LEF content/files used across multiple tests.
"""

import tempfile
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def sample_liberty_content():
    """Provides a sample Liberty file content as a string.

    Contains:
    - Library header (units, operating conditions).
    - Lookup table template (5x5).
    - Cells:
        - INV_X1: Inverter with timing arcs (rise/fall).
        - DFF_X1: Sequential cell (Flip-Flop) with clock pin.
    """
    return textwrap.dedent("""
    library(test_lib) {
      technology (cmos);
      delay_model : table_lookup;
      time_unit : "1ns";
      voltage_unit : "1V";
      current_unit : "1mA";
      pulling_resistance_unit : "1kohm";
      leakage_power_unit : "1nW";
      capacitive_load_unit (1.0, pf);

      nom_process : 1.0;
      nom_temperature : 25.0;
      nom_voltage : 1.2;

      lu_table_template(delay_template_5x5) {
        variable_1 : input_net_transition;
        variable_2 : total_output_net_capacitance;
        index_1 ("0.01, 0.05, 0.1, 0.5, 1.0");
        index_2 ("0.001, 0.01, 0.1, 0.5, 1.0");
      }

      cell(INV_X1) {
        area : 1.5;
        cell_leakage_power : 0.05;
        pin(A) {
          direction : input;
          capacitance : 0.002;
        }
        pin(Y) {
          direction : output;
          function : "!A";
          timing() {
            related_pin : "A";
            timing_sense : negative_unate;
            cell_rise(delay_template_5x5) {
              values("0.05, 0.06, 0.08, 0.15, 0.25", \
                     "0.06, 0.07, 0.09, 0.16, 0.26", \
                     "0.08, 0.09, 0.11, 0.18, 0.28", \
                     "0.15, 0.16, 0.18, 0.25, 0.35", \
                     "0.25, 0.26, 0.28, 0.35, 0.45");
            }
            cell_fall(delay_template_5x5) {
               values("0.04, 0.05, 0.07, 0.14, 0.24", \
                      "0.05, 0.06, 0.08, 0.15, 0.25", \
                      "0.07, 0.08, 0.10, 0.17, 0.27", \
                      "0.14, 0.15, 0.17, 0.24, 0.34", \
                      "0.24, 0.25, 0.27, 0.34, 0.44");
            }
          }
        }
      }

      cell(DFF_X1) {
        area : 4.0;
        ff("IQ", "IQN") {
            next_state : "D";
            clocked_on : "CLK";
        }
        pin(D) {
            direction : input;
            capacitance : 0.003;
        }
        pin(CLK) {
            direction : input;
            capacitance : 0.003;
            clock : true;
        }
        pin(Q) {
            direction : output;
            function : "IQ";
        }
      }
    }
    """)


@pytest.fixture
def sample_lef_content():
    """Provides a sample LEF file content as a string.

    Contains:
    - Header (version, units, grid).
    - Layers (M1, M2).
    - Vias (M1_M2).
    - Sites (core).
    - Macro (INV_X1) with pins and obstructions.
    """
    return textwrap.dedent("""
    VERSION 5.8 ;
    BUSBITCHARS "[]" ;
    DIVIDERCHAR "/" ;

    UNITS
      DATABASE MICRONS 1000 ;
    END UNITS

    MANUFACTURINGGRID 0.005 ;

    LAYER M1
      TYPE ROUTING ;
      DIRECTION HORIZONTAL ;
      PITCH 0.2 ;
      WIDTH 0.1 ;
      SPACING 0.1 ;
      RESISTANCE RPERSQ 0.1 ;
      CAPACITANCE CPERSQDIST 0.2 ;
    END M1

    LAYER M2
      TYPE ROUTING ;
      DIRECTION VERTICAL ;
      PITCH 0.2 ;
      WIDTH 0.1 ;
      SPACING 0.1 ;
    END M2

    VIA M1_M2
      LAYER M1 ;
        RECT -0.05 -0.05 0.05 0.05 ;
      LAYER M2 ;
        RECT -0.05 -0.05 0.05 0.05 ;
    END M1_M2

    SITE core
      CLASS CORE ;
      SIZE 0.2 BY 2.0 ;
      SYMMETRY Y ;
    END core

    MACRO INV_X1
      CLASS CORE ;
      ORIGIN 0 0 ;
      SIZE 1.0 BY 2.0 ;
      SYMMETRY X Y ;
      SITE core ;
      PIN A
        DIRECTION INPUT ;
        PORT
          LAYER M1 ;
          RECT 0.1 0.5 0.2 1.5 ;
        END
      END A
      PIN Y
        DIRECTION OUTPUT ;
        PORT
          LAYER M1 ;
          RECT 0.8 0.5 0.9 1.5 ;
        END
      END Y
      OBS
        LAYER M1 ;
        RECT 0.0 0.0 1.0 2.0 ;
      END
    END INV_X1
    """)


@pytest.fixture
def sample_liberty_file(sample_liberty_content):
    """Creates a temporary .lib file populated with sample content.

    Yields:
        Path to the temporary file. Auto-deletes on cleanup.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lib", delete=False) as f:
        f.write(sample_liberty_content)
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.fixture
def sample_lef_file(sample_lef_content):
    """Creates a temporary .lef file populated with sample content.

    Yields:
        Path to the temporary file. Auto-deletes on cleanup.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lef", delete=False) as f:
        f.write(sample_lef_content)
        path = Path(f.name)
    yield path
    path.unlink()
