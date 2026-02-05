t # Pars-FET 🌿🍨

![Security Scan](https://github.com/sebastianlhw/parsfet/actions/workflows/security.yml/badge.svg)

<img src="docs/source/_static/logo.png" width="240">

The *parfait* way to parse technology input files into structured data.
A Python-based framework for parsing semiconductor technology files (`.lib`, `.lef`) enabling cross-process comparison, visualization, and ML benchmarking.

## Key Features

- **Interactive Dashboard**: "No-Build" HTML report visualizing the speed vs. power pareto, sizing efficiency, and model quality.
- **Methodology Transparency**: "Show Your Work" philosophy. The dashboard mathematically proves the FO4 operating point derivation with live LUT curve visualization.
- **Linear Delay/Power Modeling**: Extracts architecture-agnostic parameters ($D = D_0 + k \cdot Load$, $E = E_0 + k \cdot Load$).
- **Logic Classification**: Automatically identifies NAND, NOR, XOR, etc., from boolean formulas.
- **Unified Export**: Merges Liberty timing with LEF physical data into JSON/CSV for analysis.

## Installation

```bash
pip install -e .
```

## Quick Start: Visualization

### 1. Interactive Report
Generate a self-contained HTML dashboard to explore your library.

```bash
parsfet report my_lib.lib --output dashboard.html
```

**What you get:**
- **Global Overview**: Pareto frontier (Speed vs Power) & Sizing efficiency plots.
- **Methodology Card**: 
    - Shows the **Baseline Inverter** used for normalization.
    - Visualizes the **Actual LUT Curves** used to derive the FO4 operating point.
    - Includes a **Raw Matrix Table** of the underlying library data for full transparency.
- **Cell Details**: Click any cell to see its linear fit quality ($R^2$), delay/power curves, and normalized metrics.

### 2. CSV Export
Export flattened metrics for Excel/Pandas analysis.

```bash
parsfet export-csv normalized.json output.csv
```

## Usage References

### CLI

```bash
# 1. Parse & Normalize (Standard JSON)
parsfet normalize sample.lib --output normalized.json

# 2. Parse & Normalize (Combined with LEF)
parsfet normalize sample.lib \
  --lef cells.lef \
  --tech-lef technology.lef \
  --output combined.json

# 3. Generate Report
parsfet report sample.lib

# 4. Export to CSV
parsfet export-csv combined.json flat_metrics.csv
```

### Python API

```python
from parsfet.data import Dataset

# Load Liberty + LEF
ds = Dataset()
ds.load_files(["library.lib"])
ds.load_lef(["cells.lef"])

# Export combined JSON
ds.save_json("combined.json")

# Convert to Pandas DataFrame
df = ds.to_dataframe()
print(df[["cell", "d0_ns", "e0_unit", "r_squared"]].head())
```

## Supported Formats

- **Liberty (.lib)**: Timing, power, capacitance.
- **LEF (.lef)**: Physical dimensions, pins, layers.
- **TechLEF (.techlef)**: Metal rules and via definitions.

## Test Data
- **Primary**: SkyWater 130nm PDK
- **Verification**: NanGate45, ASAP7

## Security

We take security seriously. Our pipeline ensures code safety and prevents data leaks using:

1.  **[Bandit](https://github.com/PyCQA/bandit)**: Scans Python code for common security issues (e.g., weak cryptography, potential injection vectors).
2.  **[Pip-Audit](https://github.com/pypa/pip-audit)**: Checks dependencies against the PyPI vulnerability database.
3.  **[Gitleaks](https://github.com/gitleaks/gitleaks)**: Scans for committed secrets (API keys, tokens) to prevent information leakage.

### Running Security Checks Locally

Contributors should run these checks before pushing:

```bash
# 1. Install checks
pip install bandit pip-audit

# 2. Run Code Scan (Exclude tests)
bandit -r src/

# 3. Check Dependencies
pip-audit
```

# License
MIT License. See `LICENSE` for details.
