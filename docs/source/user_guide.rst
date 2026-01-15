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

This divides every metric (area, delay, leakage) by the corresponding metric of the standard inverter (INVD1).
If a cell has a delay of 2.0, it means it's exactly twice as slow as an inverter. That fact remains true regardless of the process node.

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
