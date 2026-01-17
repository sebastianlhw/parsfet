# Pars-FET 🌿🍨

<img src="docs/source/_static/logo.png" width="640">

The *parfait* way to parse technology input files into structured data.
A Python-based framework for parsing semiconductor technology files (`.lib`, `.lef`, `.techlef`) and enabling cross-process comparison and ML analysis.

## Key Features

- **Logic Function Classification**: Automatically identifies NAND, NOR, XOR, etc., from boolean formulas.
- **Linear Delay Modeling (D₀ + k)**: Separates intrinsic delay from drive capability for architecture-agnostic comparison.
- **Technology Fingerprinting**: Creates vector representations of libraries for ML applications.
- **Combined LEF/TechLEF Export**: Merges Liberty timing data with physical layout info (cell dimensions, pin layers, technology rules) into unified JSON.

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Parse a Liberty file
parsfet parse sample.lib

# Normalize with timing data only
parsfet normalize sample.lib --output normalized.json

# Normalize with combined physical + timing data
parsfet normalize sample.lib \
  --lef cells.lef \
  --tech-lef technology.lef \
  --output combined.json

# Compare two libraries
parsfet compare lib_a.lib lib_b.lib

# Generate technology fingerprint
parsfet fingerprint sample.lib
```

### Python API

```python
from parsfet.data import Dataset

# Load Liberty with LEF/TechLEF for combined analysis
ds = Dataset()
ds.load_files(["library.lib"])
ds.load_lef(["cells.lef"])
ds.load_tech_lef("tech.lef")

# Export combined JSON (includes cell dimensions, pin layers, tech rules)
ds.save_json("combined_output.json")

# Or work with DataFrames
df = ds.to_dataframe()
print(df[["cell", "area_ratio", "lef_width", "lef_height"]].head())
```

## Supported Formats

- **Liberty (.lib)**: Timing, power, and capacitance data
- **LEF (.lef)**: Physical cell definitions (dimensions, pins, layers)
- **TechLEF (.techlef)**: Metal layer and via definitions

## JSON Output Structure

When using combined export (`--lef` and `--tech-lef` options), the JSON includes:

```json
{
  "library": "my_lib",
  "cells": {
    "INV_X1": {
      "cell_type": "inverter",
      "area_ratio": 1.0,
      "d0_ratio": 1.0,
      "physical": {
        "width_um": 1.0,
        "height_um": 2.0,
        "pins": {
          "A": {"direction": "input", "use": "signal", "layers": ["M1"]},
          "Y": {"direction": "output", "use": "signal", "layers": ["M1"]},
          "VDD": {"direction": "inout", "use": "power", "layers": ["M1"]}
        }
      }
    }
  },
  "technology": {
    "metal_stack_height": 10,
    "layers": {
      "M1": {"type": "routing", "min_size_um": 0.072, "direction": "vertical"}
    }
  }
}
```

## Test Data

- **Primary**: SkyWater 130nm PDK
- **Verification**: NanGate45, ASAP7

# License

This project is licensed under the MIT License. You are free to use, modify, and distribute it—even in professional or commercial environments—provided that the original copyright and attribution remain intact.

## Third-Party Data

This project uses partial files from the following open-source projects for testing purposes:

- **SkyWater 130nm PDK**: Licensed under [Apache License 2.0](https://github.com/google/skywater-pdk/blob/main/LICENSE).
  - Source: [google/skywater-pdk](https://github.com/google/skywater-pdk)
- **ASAP7 PDK**: Licensed under [BSD 3-Clause License](https://github.com/The-OpenROAD-Project/asap7/blob/main/LICENSE).
  - Source: [The-OpenROAD-Project/asap7](https://github.com/The-OpenROAD-Project/asap7)

See corresponding `LICENSE` files for full terms.
