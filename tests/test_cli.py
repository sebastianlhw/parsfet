from typer.testing import CliRunner
from parsfet.cli import app, detect_format
from pathlib import Path
import json

runner = CliRunner()

def test_detect_format():
    assert detect_format(Path("foo.lib")) == "lib"
    assert detect_format(Path("foo.lef")) == "lef"
    assert detect_format(Path("foo.techlef")) == "techlef"
    assert detect_format(Path("foo.ict")) == "ict"
    assert detect_format(Path("foo.txt"), forced_format="lib") == "lib"

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Pars-FET" in result.stdout

def test_cli_parse(sample_liberty_file):
    result = runner.invoke(app, ["parse", str(sample_liberty_file)])
    assert result.exit_code == 0
    assert "Liberty Summary" in result.stdout
    assert "test_lib" in result.stdout
    assert "INV_X1" in result.stdout

def test_cli_parse_verbose(sample_liberty_file):
    result = runner.invoke(app, ["parse", str(sample_liberty_file), "--verbose"])
    assert result.exit_code == 0
    assert "Cell Types" in result.stdout
    assert "Inverters" in result.stdout

def test_cli_parse_output(sample_liberty_file, tmp_path):
    output_file = tmp_path / "output.json"
    result = runner.invoke(app, ["parse", str(sample_liberty_file), "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert data["name"] == "test_lib"
        assert "INV_X1" in data["cells"]

def test_cli_parse_lef(sample_lef_file):
    result = runner.invoke(app, ["parse", str(sample_lef_file)])
    assert result.exit_code == 0
    assert "LEF Summary" in result.stdout
    # assert "M1" in result.stdout # Depending on summary output - only in verbose

def test_cli_normalize(sample_liberty_file, tmp_path):
    output_file = tmp_path / "normalized.json"
    result = runner.invoke(app, ["normalize", str(sample_liberty_file), "--output", str(output_file)])
    assert result.exit_code == 0
    assert "Normalization Summary" in result.stdout
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        # Structure is {baseline: {cell: ...}}
        assert "baseline" in data
        assert "cell" in data["baseline"]
        assert data["baseline"]["cell"] == "INV_X1"

def test_cli_normalize_with_lef(sample_liberty_file, sample_lef_file, tmp_path):
    output_file = tmp_path / "combined.json"
    # Using sample_lef_file as both LEF and TechLEF for simplicity (it has layers and macros)
    result = runner.invoke(app, [
        "normalize",
        str(sample_liberty_file),
        "--lef", str(sample_lef_file),
        "--tech-lef", str(sample_lef_file),
        "--output", str(output_file)
    ])
    assert result.exit_code == 0
    assert "Loaded LEF" in result.stdout
    assert output_file.exists()

def test_cli_compare(sample_liberty_file, tmp_path):
    # Compare file with itself
    result = runner.invoke(app, ["compare", str(sample_liberty_file), str(sample_liberty_file)])
    assert result.exit_code == 0
    assert "Jaccard Similarity" in result.stdout
    assert "100.00%" in result.stdout

def test_cli_fingerprint(sample_liberty_file, tmp_path):
    output_file = tmp_path / "fp.json"
    result = runner.invoke(app, ["fingerprint", str(sample_liberty_file), "--output", str(output_file)])
    assert result.exit_code == 0
    assert "Technology Fingerprint" in result.stdout
    assert output_file.exists()

def test_cli_export(sample_liberty_file, tmp_path):
    output_file = tmp_path / "export.json"
    result = runner.invoke(app, ["export", str(sample_liberty_file), str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)
        assert "cells" in data

def test_cli_not_found():
    result = runner.invoke(app, ["parse", "nonexistent.lib"])
    assert result.exit_code == 1
    assert "File not found" in result.stdout
