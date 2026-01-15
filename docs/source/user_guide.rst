Pars-FET User Guide
===================

Pars-FET is a VLSI technology abstraction framework designed to simplify the analysis, comparison, and characterization of semiconductor technology files. It supports Liberty (.lib), LEF (.lef), and TechLEF formats.

Installation
------------

.. code-block:: bash

   pip install parsfet

Basic Usage
-----------

Pars-FET provides a command-line interface (CLI) for common tasks.

Parsing a File
^^^^^^^^^^^^^^

To parse and summarize a technology file:

.. code-block:: bash

   parsfet parse path/to/library.lib

This command detects the format (lib, lef, techlef) and prints a summary of the contents, such as cell counts, operating conditions, and layers.

Normalization
^^^^^^^^^^^^^

To normalize a library to the standard inverter (INVD1) baseline:

.. code-block:: bash

   parsfet normalize path/to/library.lib --output normalized.json

This calculates relative metrics (area, delay, leakage) where the baseline inverter has a value of 1.0. This allows for process-node independent comparisons.

Comparing Libraries
^^^^^^^^^^^^^^^^^^^

To compare two libraries (e.g., to check for missing cells or compare performance):

.. code-block:: bash

   parsfet compare lib_a.lib lib_b.lib

This outputs the Jaccard similarity of the cell sets and compares the technological fingerprints.

Fingerprinting
^^^^^^^^^^^^^^

To generate a technological fingerprint:

.. code-block:: bash

   parsfet fingerprint path/to/library.lib

A fingerprint is a compact vector representation of the library's characteristics, useful for machine learning and clustering.

Python API
----------

You can also use Pars-FET as a Python library:

.. code-block:: python

   from parsfet.parsers.liberty import LibertyParser
   from parsfet.normalizers.invd1 import INVD1Normalizer

   # Parse
   parser = LibertyParser()
   lib = parser.parse("my_tech.lib")

   # Normalize
   normalizer = INVD1Normalizer(lib)
   summary = normalizer.get_summary()

   print(f"Baseline Cell: {summary['baseline_cell']}")
