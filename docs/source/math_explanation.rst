Mathematical Explanation
========================

Pars-FET uses specific mathematical models to normalize and analyze library data. This section explains the key concepts.

Logical Effort Normalization
----------------------------

To compare cells across different process nodes (e.g., 180nm vs. 7nm), we normalize all metrics to a "baseline inverter" (typically the 1X drive strength inverter, `INVD1`).

The fundamental idea is that the inverter is the atomic unit of digital logic. By expressing metrics relative to this atom, we derive process-independent characteristics.

Delay Model
^^^^^^^^^^^

We use a linearized delay model based on the concept of **Logical Effort**:

.. math::

   D = D_0 + k \cdot C_{load}

Where:
* :math:`D` is the total delay.
* :math:`D_0` is the **intrinsic delay** (parasitic delay) of the cell driving zero load.
* :math:`k` is the **load slope** (logical effort factor), representing how much delay increases per unit of load capacitance.
* :math:`C_{load}` is the output load capacitance.

Extraction
^^^^^^^^^^

For each cell, we extract :math:`D_0` and :math:`k` by performing a linear regression on the cell's timing lookup tables. We typically use the **FO4 (Fanout-of-4) slew** as the fixed input transition time for this extraction.

Normalization
^^^^^^^^^^^^^

Once :math:`D_0` and :math:`k` are extracted for a cell (let's say a NAND2) and the baseline inverter (INV), we calculate the normalized ratios:

.. math::

   \text{Ratio}_{D0} = \frac{D_{0, \text{cell}}}{D_{0, \text{INV}}}

.. math::

   \text{Ratio}_{k} = \frac{k_{\text{cell}}}{k_{\text{INV}}}

* A :math:`\text{Ratio}_{D0}` of 2.0 means the cell's internal delay is twice that of an inverter.
* A :math:`\text{Ratio}_{k}` of 1.5 means the cell struggles to drive load 1.5x more than an inverter (i.e., it has higher logical effort).

Interpolation
-------------

Liberty files store timing data in Non-Linear Delay Model (NLDM) lookup tables. Pars-FET uses **Bilinear Interpolation** to calculate values between table indices.

Given a table values :math:`V_{ij}` at indices :math:`x_i` (slew) and :math:`y_j` (load), the value at an arbitrary point :math:`(x, y)` is calculated as:

.. math::

   V(x, y) \approx \frac{1}{(x_1-x_0)(y_1-y_0)} \begin{bmatrix} x_1-x & x-x_0 \end{bmatrix} \begin{bmatrix} V_{00} & V_{01} \\ V_{10} & V_{11} \end{bmatrix} \begin{bmatrix} y_1-y \\ y-y_0 \end{bmatrix}

Where :math:`x_0, x_1` and :math:`y_0, y_1` are the table indices surrounding the query point.

Technology Fingerprinting
-------------------------

A technology fingerprint is a vector :math:`\mathbf{v}` that summarizes the library:

.. math::

   \mathbf{v} = [ \mu_{\text{area}}, \mu_{D0}, \mu_{k}, \mu_{\text{leak}}, \sigma_{\text{area}}, \dots, N_{\text{cells}}, P_{\text{comb}}, \dots ]

Where:
* :math:`\mu` and :math:`\sigma` are the mean and standard deviation of the normalized ratios across all cells in the library.
* :math:`N_{\text{cells}}` is the total cell count.
* :math:`P_{\text{comb}}` is the proportion of combinational cells.

This vector allows us to compute similarity between technologies using **Cosine Similarity**:

.. math::

   \text{Similarity}(A, B) = \frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\| \|\mathbf{v}_B\|}
