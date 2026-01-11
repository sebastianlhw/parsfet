"""Common type definitions and enumerations shared across Pars-FET models.

This module defines fundamental types like operating conditions, process corners,
threshold voltage flavors, and physical unit handling that are used throughout
the framework.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VtFlavor(str, Enum):
    """Enumeration of threshold voltage (Vt) flavors.

    Used to categorize cells or libraries based on their leakage/speed trade-off.
    """

    SVT = "svt"  # Standard Vt
    LVT = "lvt"  # Low Vt
    HVT = "hvt"  # High Vt
    ULVT = "ulvt"  # Ultra-low Vt
    SLVT = "slvt"  # Super-low Vt


class ProcessCorner(str, Enum):
    """Enumeration of process corners for PVT (Process, Voltage, Temperature) variation.

    Represents the manufacturing process variations:
    - TT: Typical-Typical (Nominal)
    - FF: Fast-Fast (Fast NMOS, Fast PMOS)
    - SS: Slow-Slow (Slow NMOS, Slow PMOS)
    - SF: Slow-Fast
    - FS: Fast-Slow
    """

    TT = "tt"
    FF = "ff"
    SS = "ss"
    SF = "sf"
    FS = "fs"


class OperatingCondition(BaseModel):
    """Specification of a PVT (Process, Voltage, Temperature) operating condition.

    Attributes:
        name: The name of the operating condition (e.g., "ss_0p72v_125c").
        process: The process corner (e.g., ProcessCorner.SS).
        voltage: The supply voltage in Volts.
        temperature: The junction temperature in Celsius.
    """

    name: str = ""
    process: ProcessCorner = ProcessCorner.TT
    voltage: float = Field(default=1.0, description="Nominal voltage in Volts")
    temperature: float = Field(default=25.0, description="Temperature in Celsius")

    model_config = {"frozen": False}


class PhysicalUnit(BaseModel):
    """Represents a physical quantity with a value and a unit.

    Used for dimensional analysis and unit conversion.

    Attributes:
        value: The numerical value.
        unit: The string representation of the unit (e.g., "ns", "pF", "um").
    """

    value: float
    unit: str

    def to(self, target_unit: str) -> "PhysicalUnit":
        """Converts the value to a target unit.

        Currently supports basic conversions for time, capacitance, and length.

        Args:
            target_unit: The unit string to convert to.

        Returns:
            A new PhysicalUnit instance with the converted value and target unit.

        Raises:
            ValueError: If the conversion between the current unit and target unit is not supported.
        """
        conversions = {
            ("ns", "ps"): 1000.0,
            ("ps", "ns"): 0.001,
            ("pf", "ff"): 1000.0,
            ("ff", "pf"): 0.001,
            ("um", "nm"): 1000.0,
            ("nm", "um"): 0.001,
        }
        key = (self.unit.lower(), target_unit.lower())
        if key in conversions:
            return PhysicalUnit(value=self.value * conversions[key], unit=target_unit)
        raise ValueError(f"Cannot convert {self.unit} to {target_unit}")

    def __repr__(self) -> str:
        return f"{self.value}{self.unit}"


# Canonical units for internal representation
CANONICAL_TIME_UNIT = "ns"
CANONICAL_CAP_UNIT = "pf"
CANONICAL_LENGTH_UNIT = "um"


def parse_time_unit(unit_str: str) -> float:
    """Parses a Liberty `time_unit` string to get a multiplier for nanoseconds.

    Args:
        unit_str: The time unit string from the Liberty file (e.g., "1ns", "100ps").

    Returns:
        A float multiplier to convert values in the given unit to nanoseconds (ns).
        For example, "100ps" returns 0.1.
    """
    unit_str = unit_str.strip().strip("\"'").lower()

    # Parse formats like "1ns", "1ps", "100ps"
    import re

    match = re.match(r"([\d.]+)\s*(ns|ps|fs)", unit_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "ns":
            return value
        elif unit == "ps":
            return value * 0.001
        elif unit == "fs":
            return value * 0.000001

    # Fallback
    if "ps" in unit_str:
        return 0.001
    return 1.0  # Default ns


def parse_cap_unit(cap_spec) -> float:
    """Parses a Liberty `capacitive_load_unit` spec to get a multiplier for picofarads.

    Args:
        cap_spec: The capacitance unit specification, which can be a tuple (value, unit)
            or a string "value, unit".

    Returns:
        A float multiplier to convert values to picofarads (pF).
    """
    if isinstance(cap_spec, tuple) and len(cap_spec) >= 2:
        value = float(cap_spec[0])
        unit = cap_spec[1].strip().strip("\"'").lower()
    elif isinstance(cap_spec, str):
        import re

        match = re.match(r"([\d.]+)\s*,?\s*(pf|ff|af)", cap_spec.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
        else:
            return 1.0
    else:
        return 1.0

    if unit == "pf":
        return value
    elif unit == "ff":
        return value * 0.001
    elif unit == "af":
        return value * 0.000001
    return 1.0


class UnitNormalizer:
    """Normalizes Liberty library values to canonical units.

    Canonical units are:
    - Time: nanoseconds (ns)
    - Capacitance: picofarads (pF)
    - Length/Area: micrometers (um)

    This ensures consistent comparison between libraries defined with different units.
    """

    def __init__(self, time_unit: str = "1ns", cap_unit=(1.0, "pf")):
        """Initializes the UnitNormalizer with the library's declared units.

        Args:
            time_unit: Liberty `time_unit` string (e.g., "1ns", "1ps").
            cap_unit: Liberty `capacitive_load_unit` tuple (e.g., (1, "pf")).
        """
        self.time_multiplier = parse_time_unit(time_unit)
        self.cap_multiplier = parse_cap_unit(cap_unit)

    def normalize_time(self, value: float) -> float:
        """Converts a time value from library units to nanoseconds."""
        return value * self.time_multiplier

    def normalize_capacitance(self, value: float) -> float:
        """Converts a capacitance value from library units to picofarads."""
        return value * self.cap_multiplier

    @classmethod
    def from_library(cls, library) -> "UnitNormalizer":
        """Creates a UnitNormalizer instance from a parsed LibertyLibrary object.

        Args:
            library: A `LibertyLibrary` instance.

        Returns:
            A configured `UnitNormalizer`.
        """
        return cls(time_unit=library.time_unit, cap_unit=library.capacitive_load_unit)
