User Guide
==========

Pars-FET is a tool for looking inside semiconductor technology files. It helps you see what's actually in those ``.lib``, ``.lef``, and ``.techlef`` files without getting lost in the details.

The problem with technology files is they tell you absolute numbers—picoseconds, microns, nanowatts. Those numbers change every time you switch process nodes. Pars-FET normalizes everything to the simplest component: the inverter. This lets you compare the *structure* of a technology, not just its speed limit.

Installation
------------

.. code-block:: bash

   pip install parsfet

Basic Usage
-----------

The CLI gives you quick answers to common questions about your libraries.

See What You Have
^^^^^^^^^^^^^^^^^

Before you do anything else, you just want to know what you're dealing with. Is this a 7nm library or 180nm? How many cells? What are the layers?

.. code-block:: bash

   parsfet parse path/to/library.lib

This reads the file and prints a summary. It doesn't guess; it tells you exactly what counts of cells, layers, and operating conditions are defined in the file.

Normalize (The "Inverter Standard")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Absolute numbers are hard to compare. A "fast" NAND gate in 180nm is slow in 7nm. To see if a cell is *architecturally* good, you need to remove the process scaling.

.. code-block:: bash

   parsfet normalize path/to/library.lib --output normalized.json

This extracts a linear delay model (:math:`D = D_0 + k \cdot C_{load}`) for every cell and normalizes it against the baseline inverter.

*   **d0_ratio**: Intrinsic delay (unloaded speed) relative to INVD1.
*   **k_ratio**: Logical effort (drive capability) relative to INVD1.
*   **fit_quality**: Metrics (R²) to ensure the linear model is valid.

It also automatically classifies logic gates (NAND, NOR, etc.) based on their boolean function. If a NAND gate has a ``d0_ratio`` of 1.5, it means it's 50% slower internally than an inverter, regardless of the process node.

Combined Physical + Timing Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often you need both timing data (from Liberty) and physical layout data (from LEF) together. Pars-FET can combine these into a single JSON output:

.. code-block:: bash

   parsfet normalize path/to/library.lib \
     --lef path/to/cells.lef \
     --tech-lef path/to/technology.lef \
     --output combined.json

This adds to each cell:

*   **physical.width_um / height_um**: Cell dimensions from LEF
*   **physical.pins**: For each pin: direction, use type (signal/power/ground/clock), and metal layers used
*   **technology.layers**: Metal layer rules (min size, direction, pitch, spacing)

Compare Two Libraries
^^^^^^^^^^^^^^^^^^^^^

You have two libraries. Are they the same? Do they have the same cells? Is one missing the specialized flip-flops the other has?

.. code-block:: bash

   parsfet compare lib_a.lib lib_b.lib

This checks the overlap. It tells you which cells are unique to A, unique to B, and which are shared. It uses the Jaccard similarity index to give you a single number representing how much they overlap.

Fingerprinting
^^^^^^^^^^^^^^

If you want to feed library data into a machine learning model, you can't just dump the ``.lib`` text. You need a vector—a list of numbers that describes the "shape" of the technology.

.. code-block:: bash

   parsfet fingerprint path/to/library.lib

This creates a signature based on the statistical distribution of the normalized metrics. It turns the entire library into a compact vector that captures its essence (drive strength diversity, logic depth, etc.).

Combine Multiple Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes your cells are spread across multiple files. Maybe you have LVT, RVT, and HVT variants in separate libraries. Maybe your standard cells are in one file and your IO cells in another.

The ``combine`` command merges them into one unified dataset:

.. code-block:: bash

   parsfet combine lvt.lib rvt.lib hvt.lib --output combined.json

What happens under the hood:

1. All files are parsed and their cells pooled together
2. A baseline inverter is found from the *combined* cell pool
3. *Every* cell is normalized against that single baseline

This gives you a consistent reference point. An RVT inverter might show ``d0_ratio = 1.26``—meaning it's 26% slower than the LVT baseline. This is the kind of comparison that matters when you're choosing cells for a mixed-Vt design.

**Duplicate Detection**: If two files define the same cell name, the command stops and tells you. You can override with ``--allow-duplicates`` (first occurrence wins) or use ``--check-duplicates`` to just see the conflicts:

.. code-block:: bash

   # Check first
   parsfet combine *.lib --check-duplicates

   # Force it (first wins)
   parsfet combine lvt.lib rvt.lib --allow-duplicates -o merged.json

**Source Tracking**: The output JSON and DataFrame include a ``source_file`` field so you always know where each cell came from.

Combining JSON Exports with Liberty Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also combine previously-exported JSON files with Liberty files. This is useful when you want to incrementally build a dataset or merge exports from different analysis runs.

.. code-block:: bash

   # First export a normalized library
   parsfet normalize lvt.lib --output lvt_export.json

   # Later, combine it with another library
   parsfet combine lvt_export.json hvt.lib --output merged.json --allow-duplicates

All cells—whether from JSON or Liberty—are re-normalized against the unified baseline found from the combined pool. The raw metrics (area, d0, k, leakage, input_cap) stored in the JSON export are used for re-normalization.


Python API
----------

If you need to build your own analysis tools, import the core modules directly:

.. code-block:: python

   from parsfet.parsers.liberty import LibertyParser
   from parsfet.normalizers.invd1 import INVD1Normalizer

   # 1. Parse the raw file
   parser = LibertyParser()
   lib = parser.parse("my_tech.lib")

   # 2. Create the normalized view
   normalizer = INVD1Normalizer(lib)
   summary = normalizer.get_summary()

   # 3. Ask questions
   print(f"Baseline Cell: {summary['baseline_cell']}")
   print(f"Average Area Ratio: {summary['area_ratio_stats']['mean']}")

Dataset API (Combined LEF/TechLEF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For combined physical + timing analysis, use the ``Dataset`` class:

.. code-block:: python

   from parsfet.data import Dataset

   # Load Liberty with LEF/TechLEF
   ds = Dataset()
   ds.load_files(["library.lib"])
   ds.load_lef(["cells.lef"])
   ds.load_tech_lef("tech.lef")

   # Export combined JSON
   ds.save_json("combined_output.json")

   # Or work with DataFrames
   df = ds.to_dataframe()
   print(df[["cell", "area_ratio", "lef_width", "lef_height"]].head())

Multi-File Combining (Grand Dataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you have cells spread across multiple files, load them all and call ``combine()``:

.. code-block:: python

   from parsfet.data import Dataset

   ds = Dataset()

   # Load multiple libraries (delay normalization)
   ds.load_files(["lvt.lib", "rvt.lib", "hvt.lib"], normalize=False)

   # Check for duplicate cell names
   duplicates = ds.find_duplicates()
   if duplicates:
       print(f"Found {len(duplicates)} duplicates")

   # Combine into one grand dataset
   combined = ds.combine(allow_duplicates=True)

   # Now you have unified normalization
   df = combined.to_dataframe()

   # Compare LVT vs RVT inverters
   inv_df = df[df["cell"].str.contains("INVx1")]
   print(inv_df[["cell", "d0_ratio", "source_file"]])

The key insight: after ``combine()``, every cell is normalized to the same baseline. This makes cross-library comparisons meaningful. The ``source_file`` column tells you where each cell originated.

**Combining JSON Exports**: You can also load previously-exported JSON files:

.. code-block:: python

   ds = Dataset()
   ds.load_files(["export.json", "new_lib.lib"], normalize=False)
   combined = ds.combine(allow_duplicates=True)

The ``load_files()`` method auto-detects the file format (`.json` vs `.lib`). Raw metrics from JSON exports are used for re-normalization against the new unified baseline.


**Architecture**:

1. ``load_files(..., normalize=False)`` — Parse only, don't normalize yet
2. ``find_duplicates()`` — Detect naming conflicts
3. ``combine()`` — Merge cells, find unified baseline, normalize all
4. ``to_dataframe()`` / ``save_json()`` — Work with the combined data

If you try to combine files with duplicate cell names (same name, different files), it raises ``DuplicateCellError`` unless you pass ``allow_duplicates=True``.
