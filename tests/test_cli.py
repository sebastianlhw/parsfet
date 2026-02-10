import json
from pathlib import Path

import pytest
from parsfet.cli import app, detect_format


def test_detect_format():
    assert detect_format(Path("foo.lib")) == "lib"
    assert detect_format(Path("foo.lef")) == "lef"
    assert detect_format(Path("foo.techlef")) == "techlef"
    assert detect_format(Path("foo.ict")) == "ict"
    assert detect_format(Path("foo.txt"), forced_format="lib") == "lib"


def test_cli_help(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Pars-FET" in result.stdout


def test_cli_parse_liberty(runner, sample_liberty_file):
    result = runner.invoke(app, ["parse", str(sample_liberty_file)])
    assert result.exit_code == 0
    assert "Liberty Summary" in result.stdout
    assert "test_lib" in result.stdout
    assert "Cells" in result.stdout


def test_cli_parse_lef(runner, sample_lef_file):
    result = runner.invoke(app, ["parse", str(sample_lef_file)])
    assert result.exit_code == 0
    assert "LEF Summary" in result.stdout
    assert "Layers" in result.stdout


def test_cli_parse_not_found(runner):
    result = runner.invoke(app, ["parse", "nonexistent.lib"])
    assert result.exit_code == 1
    assert "File not found" in result.stdout


def test_cli_normalize(runner, sample_liberty_file):
    with runner.isolated_filesystem():
        # Copy sample file to current dir because invoked inside isolated fs
        # Actually easier to just pass absolute path to sample_liberty_file
        # But sample_liberty_file is a fixture that yields a path in a temp dir.
        
        output_file = Path("normalized.json")
        result = runner.invoke(
            app, 
            ["normalize", str(sample_liberty_file), "--output", str(output_file)]
        )
        assert result.exit_code == 0
        assert "Normalization Summary" in result.stdout
        assert output_file.exists()
        
        data = json.loads(output_file.read_text())
        assert "library" in data
        assert "cells" in data
        assert "INV_X1" in data["cells"]
        
        # Check normalization metrics
        cell = data["cells"]["INV_X1"]
        assert "area_ratio" in cell
        assert "d0_ratio" in cell


def test_cli_compare(runner, sample_liberty_content):
    # Create two versions of the library
    lib_a = Path("lib_a.lib")
    lib_b = Path("lib_b.lib")
    
    # Write files in a temp dir provided by runner.isolated_filesystem?
    # Or just use tempfile manually.
    # Conftest fixtures yield files in temp dir, let's use them.
    # But we need two different ones.
    
    with runner.isolated_filesystem():
        lib_a.write_text(sample_liberty_content)
        # Create lib_b with one extra cell
        content_b = sample_liberty_content.replace(
            "cell(INV_X1) {", "cell(INV_X1) {\n    cell_leakage_power : 0.1;"
        ).replace("library(test_lib)", "library(test_lib_b)")
        lib_b.write_text(content_b)
        
        result = runner.invoke(app, ["compare", str(lib_a), str(lib_b)])
        assert result.exit_code == 0
        assert "Cell Coverage Comparison" in result.stdout
        assert "Jaccard Similarity" in result.stdout
        
        # Test output
        output_file = Path("diff.json")
        result = runner.invoke(
            app, ["compare", str(lib_a), str(lib_b), "--output", str(output_file)]
        )
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "cell_diff" in data


def test_cli_fingerprint(runner, sample_liberty_file):
    output_file = Path("fingerprint.json")
    with runner.isolated_filesystem():
        # We need access to sample_liberty_file. 
        # Since it's an absolute path (from fixture), it should work fine.
        
        result = runner.invoke(
            app, 
            ["fingerprint", str(sample_liberty_file), "--output", str(output_file)]
        )
        assert result.exit_code == 0
        assert "Technology Fingerprint" in result.stdout
        assert "Baseline Metrics" in result.stdout
        
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "baseline" in data
        assert data["baseline"]["cell"] == "INV_X1"


def test_cli_combine(runner, sample_liberty_file):
    # Combine the same file with itself (duplicate check)
    result = runner.invoke(app, ["combine", str(sample_liberty_file), str(sample_liberty_file)])
    assert result.exit_code == 1
    assert "Duplicate cells found" in result.stdout
    
    # Allow duplicates
    output_file = Path("combined.json")
    with runner.isolated_filesystem():
        result = runner.invoke(
            app, 
            [
                "combine", 
                str(sample_liberty_file), 
                str(sample_liberty_file), 
                "--allow-duplicates", 
                "--output", 
                str(output_file)
            ]
        )
        assert result.exit_code == 0
        assert "Combined Dataset" in result.stdout
        assert output_file.exists()
        
        data = json.loads(output_file.read_text())
        assert len(data["cells"]) > 0


def test_cli_export(runner, sample_liberty_file):
    output_file = Path("export.json")
    with runner.isolated_filesystem():
        result = runner.invoke(
            app, ["export", str(sample_liberty_file), str(output_file)]
        )
        assert result.exit_code == 0
        assert output_file.exists()
        
        data = json.loads(output_file.read_text())
        assert data["name"] == "test_lib"
