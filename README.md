# Pars-FET 🌿🍨

Mint-Pars-FET is a fresh, Pythonic technology file parser designed to turn messy, layered PDK specifications into clean, digestible data structures.

A Python-based framework for parsing semiconductor technology files (`.lib`, `.lef`, `.techlef`, `.ict`) and enabling cross-process comparison and ML analysis.

## Key Features

- **Logic Function Classification**: Automatically identifies NAND, NOR, XOR, etc., from boolean formulas.
- **Linear Delay Modeling (D₀ + k)**: Separates intrinsic delay from drive capability for architecture-agnostic comparison.
- **Fit Quality Metrics**: Validates linear delay assumptions with R² and residual analysis.
- **Technology Fingerprinting**: Creates vector representations of libraries for ML applications.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Parse a Liberty file
parsfet parse sample.lib

# Compare two libraries
parsfet compare lib_a.lib lib_b.lib

# Generate technology fingerprint
parsfet fingerprint sample.lib
```

## Supported Formats

- **Liberty (.lib)**: Timing, power, and capacitance data
- **LEF (.lef)**: Physical cell definitions
- **TechLEF (.techlef)**: Metal layer and via definitions
- **ICT (.ict)**: Interconnect technology parameters

## Test Data

- **Primary**: SkyWater 130nm PDK
- **Verification**: NanGate45, ASAP7

# License

This project is licensed under the MIT License. You are free to use, modify, and distribute it—even in professional or commercial environments—provided that the original copyright and attribution remain intact.

## Third-Party Data

This repository contains partial files from the following open-source projects for testing purposes:

- **SkyWater 130nm PDK**: Licensed under [Apache License 2.0](testdata/skywater/skywater-pdk/LICENSE).
  - Source: [google/skywater-pdk](https://github.com/google/skywater-pdk)
- **ASAP7 PDK**: Licensed under [BSD 3-Clause License](testdata/asap7/LICENSE).
  - Source: [The-OpenROAD-Project/asap7](https://github.com/The-OpenROAD-Project/asap7)

See the respective `LICENSE` files in `testdata/` for full terms.