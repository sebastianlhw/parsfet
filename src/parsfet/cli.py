"""Pars-FET CLI - Command Line Interface.

This module provides the command-line interface for the Pars-FET framework,
allowing users to parse, normalize, compare, and fingerprint semiconductor
technology files (.lib, .lef, .techlef).

The CLI is built using Typer and uses Rich for formatted output.

Typical usage example:

  $ parsfet parse my_tech.lib
  $ parsfet normalize my_tech.lib --output normalized.json
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="parsfet",
    help="Pars-FET: VLSI Technology Abstraction Framework",
    no_args_is_help=True,
)
console = Console()


def detect_format(path: Path, forced_format: Optional[str] = None) -> str:
    """Detects the file format from the extension or content.

    Args:
        path: The path to the file.
        forced_format: An optional string to force a specific format.
            Acceptable values are 'lib', 'lef', 'techlef', 'ict'.

    Returns:
        A string representing the detected format ('lib', 'lef', 'techlef', 'ict', or 'unknown').
    """
    if forced_format:
        return forced_format.lower()

    suffix = path.suffix.lower()

    if suffix == ".lib":
        return "lib"
    elif suffix == ".lef":
        return "lef"
    elif suffix == ".techlef":
        return "techlef"
    elif suffix == ".ict":
        return "ict"
    else:
        # Try to detect from content
        content = path.read_text(encoding="utf-8", errors="replace")[:1000]
        if "library" in content.lower() and "cell" in content.lower():
            return "lib"
        elif "LAYER" in content or "MACRO" in content:
            return "lef"
        return "unknown"


@app.command()
def parse(
    file: Path = typer.Argument(..., help="Path to technology file"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Force format: lib, lef, techlef"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Parses a technology file and displays a summary.

    This command supports Liberty (.lib), LEF (.lef), and TechLEF (.techlef) formats.
    It parses the file, displays a summary of the contents (e.g., number of cells,
    layers, operating conditions), and optionally saves the parsed data to a JSON file.

    Args:
        file: The path to the technology file to parse.
        format: Optional. Force the parser to use a specific format ('lib', 'lef', 'techlef').
            If not provided, the format is detected automatically.
        output: Optional. Path to save the parsed data as a JSON file.
        verbose: Optional. If True, displays more detailed information such as cell
            breakdowns or layer details.

    Raises:
        typer.Exit: If the file is not found or the format is unknown.
    """
    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    detected_format = detect_format(file, format)

    if detected_format == "lib":
        from .parsers.liberty import LibertyParser

        parser = LibertyParser()
        lib = parser.parse(file)

        # Display summary
        console.print(
            Panel.fit(
                f"[bold green]Library:[/] {lib.name}\n"
                f"[bold]Cells:[/] {len(lib.cells)}\n"
                f"[bold]Technology:[/] {lib.technology or 'N/A'}\n"
                f"[bold]Nom Voltage:[/] {lib.nom_voltage or 'N/A'} V\n"
                f"[bold]Nom Temp:[/] {lib.nom_temperature or 'N/A'} °C\n"
                f"[bold]Baseline Cell:[/] {lib.baseline_cell.name if lib.baseline_cell else 'Not found'}",
                title="Liberty Summary",
            )
        )

        if verbose:
            # Show cell breakdown
            table = Table(title="Cell Types")
            table.add_column("Type")
            table.add_column("Count", justify="right")

            inv_count = sum(1 for n in lib.cells if "INV" in n.upper())
            buf_count = sum(1 for n in lib.cells if "BUF" in n.upper())
            nand_count = sum(1 for n in lib.cells if "NAND" in n.upper())
            nor_count = sum(1 for n in lib.cells if "NOR" in n.upper())
            dff_count = sum(1 for n in lib.cells if "DFF" in n.upper())

            table.add_row("Inverters", str(inv_count))
            table.add_row("Buffers", str(buf_count))
            table.add_row("NANDs", str(nand_count))
            table.add_row("NORs", str(nor_count))
            table.add_row("DFFs", str(dff_count))

            console.print(table)

        # Validation warnings
        warnings = parser.validate(lib)
        if warnings:
            for w in warnings:
                console.print(f"[yellow]Warning:[/yellow] {w}")

        # Output to JSON
        if output:
            data = lib.model_dump()
            output.write_text(json.dumps(data, indent=2, default=str))
            console.print(f"[green]Saved to:[/green] {output}")

    elif detected_format in ("lef", "techlef"):
        from .parsers.lef import LEFParser, TechLEFParser

        if detected_format == "techlef":
            parser = TechLEFParser()
        else:
            parser = LEFParser()

        lef = parser.parse(file)

        console.print(
            Panel.fit(
                f"[bold green]LEF File:[/] {file.name}\n"
                f"[bold]Version:[/] {lef.version or 'N/A'}\n"
                f"[bold]Layers:[/] {len(lef.layers)}\n"
                f"[bold]Vias:[/] {len(lef.vias)}\n"
                f"[bold]Sites:[/] {len(lef.sites)}\n"
                f"[bold]Macros:[/] {len(getattr(lef, 'macros', {}))}",
                title="LEF Summary",
            )
        )

        if verbose and lef.layers:
            table = Table(title="Layers")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Direction")
            table.add_column("Pitch")

            for name, layer in lef.layers.items():
                table.add_row(
                    name,
                    layer.layer_type.value if layer.layer_type else "N/A",
                    layer.direction.value if layer.direction else "N/A",
                    f"{layer.pitch:.3f}" if layer.pitch else "N/A",
                )

            console.print(table)

        if output:
            data = lef.model_dump()
            output.write_text(json.dumps(data, indent=2, default=str))
            console.print(f"[green]Saved to:[/green] {output}")

    else:
        console.print(f"[red]Error:[/red] Unknown format: {detected_format}")
        raise typer.Exit(1)


@app.command()
def normalize(
    lib_file: Path = typer.Argument(..., help="Path to Liberty file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    baseline: Optional[str] = typer.Option(None, "--baseline", "-b", help="Baseline cell name"),
    lef: Optional[list[Path]] = typer.Option(None, "--lef", "-l", help="LEF file(s) for physical data"),
    tech_lef: Optional[Path] = typer.Option(None, "--tech-lef", "-t", help="TechLEF file for technology rules"),
):
    """Normalizes library metrics to the INVD1 baseline.

    This command calculates relative metrics (area, delay) for all cells in the library,
    normalized to the baseline inverter (typically INVD1). This allows for technology-agnostic
    comparisons between different libraries or process nodes.

    Optionally, LEF and TechLEF files can be provided to include physical data (cell dimensions,
    pin layers, technology rules) in the JSON output.

    Args:
        lib_file: Path to the Liberty (.lib) file.
        output: Optional. Path to save the normalized data as a JSON file.
        baseline: Optional. Name of the baseline cell to use. If not provided,
            the tool attempts to automatically detect the standard inverter.
        lef: Optional. One or more LEF files containing cell physical data.
        tech_lef: Optional. TechLEF file containing technology layer rules.

    Raises:
        typer.Exit: If the file is not found or normalization fails.

    Examples:
        # Basic normalization (Liberty only)
        $ parsfet normalize my_lib.lib --output normalized.json

        # With LEF and TechLEF for combined physical + timing data
        $ parsfet normalize my_lib.lib --lef cells.lef --tech-lef tech.lef -o combined.json
    """
    if not lib_file.exists():
        console.print(f"[red]Error:[/red] File not found: {lib_file}")
        raise typer.Exit(1)

    # If LEF/TechLEF provided, use Dataset API for combined export
    if lef or tech_lef:
        from .data import Dataset

        ds = Dataset()
        ds.load_files([lib_file])

        if lef:
            for lef_path in lef:
                if not lef_path.exists():
                    console.print(f"[red]Error:[/red] LEF file not found: {lef_path}")
                    raise typer.Exit(1)
            ds.load_lef(lef)
            console.print(f"[blue]Loaded LEF:[/blue] {len(lef)} file(s)")

        if tech_lef:
            if not tech_lef.exists():
                console.print(f"[red]Error:[/red] TechLEF file not found: {tech_lef}")
                raise typer.Exit(1)
            ds.load_tech_lef(tech_lef)
            console.print(f"[blue]Loaded TechLEF:[/blue] {tech_lef.name}")

        entry = ds.entries[0]
        if not entry.normalizer:
            console.print("[red]Error:[/red] No baseline inverter found in library")
            raise typer.Exit(1)

        summary = entry.normalizer.get_summary()

        # Display summary
        phys_info = ""
        if entry.lef_cells:
            phys_info += f"\n[bold]LEF Cells Matched:[/] {len(entry.lef_cells)}"
        if entry.tech_info:
            phys_info += f"\n[bold]Metal Stack:[/] {entry.tech_info.metal_stack_height} layers"

        console.print(
            Panel.fit(
                f"[bold green]Library:[/] {entry.library.name}\n"
                f"[bold]Baseline Cell:[/] {summary['baseline_cell']}\n"
                f"[bold]Total Cells:[/] {summary['total_cells']}\n\n"
                f"[bold]Area Ratios:[/]\n"
                f"  Mean: {summary['area_ratio_stats'].get('mean', 0):.2f}\n"
                f"  Min: {summary['area_ratio_stats'].get('min', 0):.2f}\n"
                f"  Max: {summary['area_ratio_stats'].get('max', 0):.2f}\n\n"
                f"[bold]D0 Ratios:[/]\n"
                f"  Mean: {summary['d0_ratio_stats'].get('mean', 0):.2f}\n"
                f"  Min: {summary['d0_ratio_stats'].get('min', 0):.2f}\n"
                f"  Max: {summary['d0_ratio_stats'].get('max', 0):.2f}"
                + phys_info,
                title="Normalization Summary",
            )
        )

        if output:
            ds.save_json(output)
            console.print(f"[green]Saved combined data to:[/green] {output}")

    else:
        # Original Liberty-only flow
        from .normalizers.invd1 import INVD1Normalizer
        from .parsers.liberty import LibertyParser

        parser = LibertyParser()
        lib = parser.parse(lib_file)

        try:
            normalizer = INVD1Normalizer(lib, baseline_name=baseline)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

        summary = normalizer.get_summary()

        console.print(
            Panel.fit(
                f"[bold green]Library:[/] {lib.name}\n"
                f"[bold]Baseline Cell:[/] {summary['baseline_cell']}\n"
                f"[bold]Total Cells:[/] {summary['total_cells']}\n\n"
                f"[bold]Area Ratios:[/]\n"
                f"  Mean: {summary['area_ratio_stats'].get('mean', 0):.2f}\n"
                f"  Min: {summary['area_ratio_stats'].get('min', 0):.2f}\n"
                f"  Max: {summary['area_ratio_stats'].get('max', 0):.2f}\n\n"
                f"[bold]D0 Ratios:[/]\n"
                f"  Mean: {summary['d0_ratio_stats'].get('mean', 0):.2f}\n"
                f"  Min: {summary['d0_ratio_stats'].get('min', 0):.2f}\n"
                f"  Max: {summary['d0_ratio_stats'].get('max', 0):.2f}",
                title="Normalization Summary",
            )
        )

        if output:
            data = normalizer.export_to_json()
            output.write_text(json.dumps(data, indent=2, default=str))
            console.print(f"[green]Saved to:[/green] {output}")


@app.command()
def compare(
    lib_a: Path = typer.Argument(..., help="First Liberty file"),
    lib_b: Path = typer.Argument(..., help="Second Liberty file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Compares two Liberty libraries.

    Performs a comparison between two libraries, checking for cell coverage overlap
    (Jaccard similarity) and comparing technological fingerprints (metrics at the
    baseline inverter level).

    Args:
        lib_a: Path to the first Liberty file.
        lib_b: Path to the second Liberty file.
        output: Optional. Path to save the comparison results as a JSON file.

    Raises:
        typer.Exit: If either file is not found.
    """
    for f in [lib_a, lib_b]:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    from .comparators.cell_diff import compare_cell_coverage
    from .comparators.fingerprint import (compare_fingerprints,
                                          create_fingerprint)
    from .parsers.liberty import LibertyParser

    parser = LibertyParser()
    a = parser.parse(lib_a)
    b = parser.parse(lib_b)

    # Cell coverage comparison
    diff = compare_cell_coverage(a, b)

    table = Table(title="Cell Coverage Comparison")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Library A", a.name)
    table.add_row("Library B", b.name)
    table.add_row("Only in A", str(len(diff.only_in_a)))
    table.add_row("Only in B", str(len(diff.only_in_b)))
    table.add_row("Common", str(len(diff.common)))
    table.add_row("Jaccard Similarity", f"{diff.jaccard_similarity:.2%}")
    table.add_row("A Coverage in B", f"{diff.coverage_a_in_b:.2%}")
    table.add_row("B Coverage in A", f"{diff.coverage_b_in_a:.2%}")

    console.print(table)

    # Fingerprint comparison
    try:
        fp_a = create_fingerprint(a)
        fp_b = create_fingerprint(b)
        fp_comparison = compare_fingerprints(fp_a, fp_b)

        console.print(
            f"\n[bold]Fingerprint Similarity:[/] {fp_comparison['similarity']['cosine']:.3f}"
        )

        if fp_a.baseline_d0 > 0 and fp_b.baseline_d0 > 0:
            speed_ratio = fp_a.baseline_d0 / fp_b.baseline_d0
            if speed_ratio > 1:
                console.print(f"[bold]Speed:[/] {b.name} is {speed_ratio:.2f}x faster")
            else:
                console.print(f"[bold]Speed:[/] {a.name} is {1 / speed_ratio:.2f}x faster")
    except Exception:
        pass  # Fingerprinting optional

    if output:
        data = {
            "cell_diff": diff.to_dict(),
            "fingerprints": {
                "a": fp_a.to_dict() if "fp_a" in dir() else None,
                "b": fp_b.to_dict() if "fp_b" in dir() else None,
            },
            "comparison": fp_comparison if "fp_comparison" in dir() else None,
        }
        output.write_text(json.dumps(data, indent=2, default=str))
        console.print(f"[green]Saved to:[/green] {output}")


@app.command()
def fingerprint(
    lib_file: Path = typer.Argument(..., help="Path to Liberty file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Generates a technology fingerprint for a library.

    A fingerprint is a compact representation of the technology's key characteristics,
    derived from the baseline inverter's performance (area, delay, leakage) and the
    distribution of cell types (combinational vs. sequential, function types).

    Args:
        lib_file: Path to the Liberty file.
        output: Optional. Path to save the fingerprint data as a JSON file.

    Raises:
        typer.Exit: If the file is not found.
    """
    if not lib_file.exists():
        console.print(f"[red]Error:[/red] File not found: {lib_file}")
        raise typer.Exit(1)

    from .comparators.fingerprint import create_fingerprint
    from .parsers.liberty import LibertyParser

    parser = LibertyParser()
    lib = parser.parse(lib_file)

    fp = create_fingerprint(lib)

    console.print(
        Panel.fit(
            f"[bold green]Library:[/] {fp.name}\n"
            f"[bold]Baseline Cell:[/] {fp.baseline_cell or 'N/A'}\n\n"
            f"[bold]Baseline Metrics:[/]\n"
            f"  Area: {fp.baseline_area:.4f} um²\n"
            f"  Delay: {fp.baseline_d0:.4f} ns\n"
            f"  Leakage: {fp.baseline_leakage:.4f} nW\n\n"
            f"[bold]Cell Counts:[/]\n"
            f"  Total: {fp.total_cells}\n"
            f"  Combinational: {fp.combinational_cells}\n"
            f"  Sequential: {fp.sequential_cells}\n\n"
            f"[bold]Function Types:[/]\n"
            f"  INV: {fp.inverter_count} | BUF: {fp.buffer_count}\n"
            f"  NAND: {fp.nand_count} | NOR: {fp.nor_count}\n"
            f"  DFF: {fp.dff_count} | LATCH: {fp.latch_count}",
            title="Technology Fingerprint",
        )
    )

    if output:
        data = fp.to_dict()
        output.write_text(json.dumps(data, indent=2, default=str))
        console.print(f"[green]Saved to:[/green] {output}")


@app.command()
def export(
    lib_file: Path = typer.Argument(..., help="Path to Liberty file"),
    output: Path = typer.Argument(..., help="Output JSON file"),
    include_timing: bool = typer.Option(True, help="Include timing arc data"),
):
    """Exports a Liberty library to a JSON format.

    This command converts the parsed Liberty structure into a JSON file, which is easier
    to consume by other tools or languages.

    Args:
        lib_file: Path to the Liberty file.
        output: Path to the output JSON file.
        include_timing: Optional. If True (default), includes timing and power arc data.
            Setting to False significantly reduces the output file size.

    Raises:
        typer.Exit: If the input file is not found.
    """
    if not lib_file.exists():
        console.print(f"[red]Error:[/red] File not found: {lib_file}")
        raise typer.Exit(1)

    from .parsers.liberty import LibertyParser

    parser = LibertyParser()
    lib = parser.parse(lib_file)

    # Export with options
    data = lib.model_dump(exclude_none=True)

    if not include_timing:
        # Remove timing arcs to reduce size
        for cell in data.get("cells", {}).values():
            cell.pop("timing_arcs", None)
            cell.pop("power_arcs", None)

    output.write_text(json.dumps(data, indent=2, default=str))
    console.print(f"[green]Exported {len(lib.cells)} cells to:[/green] {output}")


if __name__ == "__main__":
    app()
