Pars-FET
========

**Pars-FET** is a fresh, Pythonic framework for parsing, normalizing, and comparing VLSI technology files.

It turns messy, layered PDK specifications (``.lib``, ``.lef``, ``.techlef``) into clean, digestible data structures. By normalizing everything to a baseline inverter, it allows you to compare the *structure* of different technologies—like comparing a 7nm node to a 180nm node—without getting lost in absolute units.

Quick Start
-----------

Get up and running in seconds:

.. code-block:: bash

   pip install parsfet

   # Parse a library to see what's inside
   parsfet parse my_library.lib

   # Normalize to the "Inverter Standard"
   parsfet normalize my_library.lib --output normalized.json

Key Features
------------

*   **Logic Classification**: Automatically identifies gates (NAND, NOR, etc.) from boolean formulas.
*   **Linear Delay Models**: Separates intrinsic delay (:math:`D_0`) from drive strength (:math:`k`).
*   **Fingerprinting**: Creates vector signatures of libraries for Machine Learning.
*   **Physical + Timing**: Merges LEF layout data with Liberty timing models.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   user_guide
   math_explanation
   modules

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
