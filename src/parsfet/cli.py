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
import logging
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

from .log_utils import setup_logging


@app.callback(invoke_without_command=False)
def main(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress debug logs (show warnings/errors only)"
    ),
):
    """Pars-FET: VLSI Technology Abstraction Framework."""
    setup_logging(quiet=quiet)


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
    elif suffix == ".json":
        return "json"
    else:
        # Try to detect from content
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(1000)
        except OSError:
            return "unknown"

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

    logger = logging.getLogger("parsfet.cli")
    logger.info(f"Starting parse for {file}")

    detected_format = detect_format(file, format)
    logger.debug(f"Detected format: {detected_format}")

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

        if logging.getLogger().isEnabledFor(logging.DEBUG):
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

        if logging.getLogger().isEnabledFor(logging.DEBUG) and lef.layers:
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
    lef: Optional[list[Path]] = typer.Option(
        None, "--lef", "-l", help="LEF file(s) for physical data"
    ),
    tech_lef: Optional[Path] = typer.Option(
        None, "--tech-lef", "-t", help="TechLEF file for technology rules"
    ),
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
                f"  Max: {summary['d0_ratio_stats'].get('max', 0):.2f}" + phys_info,
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
    from .data import Dataset
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

    # Fingerprint comparison using Dataset API
    try:
        ds_a = Dataset().load_files([lib_a])
        ds_b = Dataset().load_files([lib_b])

        vec_a = ds_a.to_vector()
        vec_b = ds_b.to_vector()

        # Compute cosine similarity
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        cosine_sim = dot / (mag_a * mag_b) if (mag_a > 0 and mag_b > 0) else 0.0

        console.print(f"\n[bold]Fingerprint Similarity:[/] {cosine_sim:.3f}")

        # Compare baseline speeds
        summary_a = ds_a.to_summary_dict()
        summary_b = ds_b.to_summary_dict()

        d0_a = summary_a.get("baseline", {}).get("d0_ns", 0)
        d0_b = summary_b.get("baseline", {}).get("d0_ns", 0)

        if d0_a > 0 and d0_b > 0:
            speed_ratio = d0_a / d0_b
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
                "a": summary_a if "summary_a" in dir() else None,
                "b": summary_b if "summary_b" in dir() else None,
            },
            "comparison": {
                "cosine_similarity": cosine_sim if "cosine_sim" in dir() else None,
            }
            if "cosine_sim" in dir()
            else None,
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

    from .data import Dataset

    # Load via Dataset (new recommended approach)
    ds = Dataset()
    ds.load_files([lib_file])

    if not ds.entries or not ds.entries[0].normalizer:
        console.print("[red]Error:[/red] No baseline inverter found in library")
        raise typer.Exit(1)

    # Generate summary using new Dataset API
    summary_dict = ds.to_summary_dict()

    baseline = summary_dict["baseline"]
    cell_counts = summary_dict["cell_counts"]
    func_types = summary_dict["function_types"]

    console.print(
        Panel.fit(
            f"[bold green]Library:[/] {summary_dict['library']}\n"
            f"[bold]Baseline Cell:[/] {baseline['cell']}\n\n"
            f"[bold]Baseline Metrics:[/]\n"
            f"  Area: {baseline['area_um2']:.4f} um²\n"
            f"  D0: {baseline['d0_ns']:.4f} ns\n"
            f"  k: {baseline['k_ns_per_pf']:.4f} ns/pF\n"
            f"  Leakage: {baseline['leakage']:.4e}\n\n"
            f"[bold]Cell Counts:[/]\n"
            f"  Total: {cell_counts['total']}\n"
            f"  Combinational: {cell_counts['combinational']}\n"
            f"  Sequential: {cell_counts['sequential']}\n\n"
            f"[bold]Function Types:[/]\n"
            f"  INV: {func_types['inverter']} | BUF: {func_types['buffer']}\n"
            f"  NAND: {func_types['nand']} | NOR: {func_types['nor']}\n"
            f"  DFF: {func_types['dff']} | LATCH: {func_types['latch']}",
            title="Technology Fingerprint",
        )
    )

    if output:
        data = summary_dict
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


@app.command()
def combine(
    lib_files: list[Path] = typer.Argument(
        ..., help="Liberty (.lib) or JSON (.json) files to combine"
    ),
    lef: Optional[list[Path]] = typer.Option(
        None, "--lef", "-l", help="LEF file(s) for physical data"
    ),
    tech_lef: Optional[Path] = typer.Option(None, "--tech-lef", "-t", help="TechLEF file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    check_duplicates: bool = typer.Option(
        False, "--check-duplicates", help="Only report duplicates, don't output"
    ),
    allow_duplicates: bool = typer.Option(
        False, "--allow-duplicates", help="Allow duplicates (first wins)"
    ),
    baseline: Optional[str] = typer.Option(None, "--baseline", "-b", help="Baseline cell name"),
):
    """Combines multiple Liberty or JSON files into one unified dataset.

    This command merges cells from multiple .lib files AND/OR previously-exported
    .json files into a single dataset with unified normalization. A baseline
    inverter is found from the combined cell pool, and ALL cells are re-normalized
    against this single baseline.

    When JSON exports are included, the raw cell metrics (area, d0, k, leakage,
    input_cap) are used for re-normalization against the new unified baseline.

    By default, an error is raised if duplicate cell names are found across files.
    Use --allow-duplicates to proceed (first occurrence wins) or --check-duplicates
    to just report duplicates without generating output.

    Args:
        lib_files: Liberty (.lib) or JSON (.json) files to combine.
        lef: Optional. LEF files for physical cell data.
        tech_lef: Optional. TechLEF file for technology rules.
        output: Optional. Path to save the combined JSON output.
        check_duplicates: If True, only check for duplicates and report.
        allow_duplicates: If True, allow duplicate cells (first wins).
        baseline: Optional. Name of the baseline cell to use.

    Examples:
        # Combine two Liberty libraries
        $ parsfet combine lib1.lib lib2.lib --output combined.json

        # Combine a JSON export with a Liberty file
        $ parsfet combine export.json lib2.lib -o merged.json --allow-duplicates

        # Check for duplicates first
        $ parsfet combine *.lib --check-duplicates

        # Force combine with duplicates
        $ parsfet combine lib1.lib lib2.lib --allow-duplicates -o merged.json
    """

    from .data import Dataset
    from .exceptions import DuplicateCellError

    # Validate files exist
    for f in lib_files:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    # Load files without immediate normalization
    ds = Dataset()
    ds.load_files(lib_files, normalize=False)

    console.print(f"[blue]Loaded {len(ds.entries)} libraries:[/blue]")
    total_cells = 0
    for entry in ds.entries:
        cell_count = len(entry.library.cells)
        total_cells += cell_count
        console.print(f"  • {entry.library.name}: {cell_count} cells")

    # Check for duplicates
    duplicates = ds.find_duplicates()
    if duplicates:
        console.print(f"\n[yellow]Found {len(duplicates)} duplicate cell(s):[/yellow]")
        for cell, sources in list(duplicates.items())[:10]:  # Show first 10
            files = ", ".join(s.name for _, s in sources)
            console.print(f"  • {cell}: {files}")
        if len(duplicates) > 10:
            console.print(f"  ... and {len(duplicates) - 10} more")

        if check_duplicates:
            console.print(
                "\n[blue]Use --allow-duplicates to proceed with first-occurrence-wins.[/blue]"
            )
            raise typer.Exit(0)

        if not allow_duplicates:
            console.print(
                "\n[red]Error:[/red] Duplicate cells found. Use --allow-duplicates to proceed."
            )
            raise typer.Exit(1)

    # Combine with unified normalization
    try:
        combined = ds.combine(allow_duplicates=allow_duplicates, baseline=baseline)
    except DuplicateCellError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Load LEF/TechLEF if provided
    if lef:
        for lef_path in lef:
            if not lef_path.exists():
                console.print(f"[red]Error:[/red] LEF file not found: {lef_path}")
                raise typer.Exit(1)
        combined.load_lef(lef)
        console.print(f"[blue]Loaded LEF:[/blue] {len(lef)} file(s)")

    if tech_lef:
        if not tech_lef.exists():
            console.print(f"[red]Error:[/red] TechLEF file not found: {tech_lef}")
            raise typer.Exit(1)
        combined.load_tech_lef(tech_lef)
        console.print(f"[blue]Loaded TechLEF:[/blue] {tech_lef.name}")

    # Display summary
    entry = combined.entries[0]
    baseline_name = entry.normalizer.baseline_cell.name if entry.normalizer else "N/A"

    console.print(
        Panel.fit(
            f"[bold green]Combined Dataset[/]\n"
            f"[bold]Total Cells:[/] {len(entry.library.cells)}\n"
            f"[bold]Baseline:[/] {baseline_name}\n"
            f"[bold]Source Files:[/] {len(lib_files)}",
            title="Combine Summary",
        )
    )

    # Save output
    if output:
        combined.save_json(output)
        console.print(f"[green]Saved to:[/green] {output}")
    elif not check_duplicates:
        console.print("[yellow]Tip:[/yellow] Use --output to save the combined data.")


@app.command()
def export_csv(
    input_file: Path = typer.Argument(..., help="Path to exported JSON file"),
    output: Path = typer.Argument(..., help="Output CSV file"),
):
    """Exports a Pars-FET JSON library to CSV format.

    This command converts the hierarchical JSON export into a flat CSV table,
    suitable for analysis in Excel, Matlab, or Pandas.
    """
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    from .models.export import ExportedLibrary
    from .reporting.csv_generator import generate_csv

    try:
        # Load the library model
        console.print(f"Loading {input_file}...")
        library = ExportedLibrary.from_json_file(str(input_file))
        
        # Generate CSV
        with open(output, "w", newline="") as f:
            generate_csv(library, f)
            
        console.print(f"[green]Exported CSV to:[/green] {output}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to export CSV: {e}")
        # raise  # Uncomment for debug trace
        raise typer.Exit(1)


@app.command()
def report(
    lib_files: list[Path] = typer.Argument(..., help="Path to Liberty (.lib) file(s)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file"),
    baseline: Optional[str] = typer.Option(None, "--baseline", "-b", help="Baseline cell name"),
    combine: bool = typer.Option(False, "--combine", "-c", help="Combine all libraries into one view"),
    allow_duplicates: bool = typer.Option(False, "--allow-duplicates", help="Allow duplicate cells when combining"),
):
    """Generates an interactive HTML verification report.

    This command parses one or more Liberty libraries, extracts linear model fits for all cells,
    and generates a single-file interactive HTML report.

    The report allows verifying the fit quality (R²) and inspecting outliers (red items).
    """
    for f in lib_files:
        if not f.exists():
            console.print(f"[red]Error:[/red] File not found: {f}")
            raise typer.Exit(1)

    if not output:
        # Default name based on first library
        output = lib_files[0].with_suffix(".html")

    from .data import Dataset
    from .reporting.html_generator import generate_report, validate_assets
    from .exceptions import DuplicateCellError

    try:
        # Check for assets BEFORE parsing to save time and give immediate feedback
        validate_assets()

        console.print(f"Loading {len(lib_files)} libraries...")
        
        # Use Dataset API to load and normalize all libraries consistently
        ds = Dataset()
        ds.load_files(lib_files)
        
        entries_to_report = ds.entries

        if combine:
            console.print("Combining libraries...")
            try:
                # combine() returns a new Dataset with a single entry
                combined_ds = ds.combine(allow_duplicates=allow_duplicates)
                entries_to_report = combined_ds.entries
            except DuplicateCellError as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("[yellow]Hint:[/yellow] Use --allow-duplicates to force merge (first wins).")
                raise typer.Exit(1)
        
        # Ensure we have a baseline (Dataset normalization handles comparison baselines if implemented, 
        # but here we rely on the primary entry's behavior or individual normalization)
        # Note: Dataset.load_files does NOT auto-normalize unless we tell it to, 
        # but the current implementation of Dataset.load_files might. 
        # Let's check Dataset implementation or just proceed assuming we normalize manually if needed.
        # Actually, Dataset.load_files defaults to normalize=True (inferred).
        
        # If the user specified a baseline name, we might need to re-normalize or hint it.
        # The current Dataset API might not propagate `baseline` arg to internal normalizers easily.
        # However, for this task, let's assume auto-detection or update normalized entries.
        
        if baseline:
             # Re-normalize with explicit baseline if requested
             from .normalizers.invd1 import INVD1Normalizer
             for entry in entries_to_report:
                 entry.normalizer = INVD1Normalizer(entry.library, baseline_name=baseline)
        
        console.print(f"Generating report to {output}...")
        generate_report(entries_to_report, output)
        
        console.print(f"[green]Report generated:[/green] {output}")

    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to generate report: {e}")
        # raise  # Uncomment for debug
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
