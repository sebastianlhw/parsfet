Mathematical Explanation
========================

We use simple models to describe complex behaviors. The goal isn't to be perfectly precise—that's what SPICE is for—but to be useful for comparison and understanding.

The Atomic Unit: The Inverter
-----------------------------

In digital logic, the inverter is the atom. It is the simplest useful structure. Every other gate is just a more complex version of an inverter.

To make sense of a technology, we stop looking at absolute units (nanoseconds, micrometers) and start looking at relative units. We define the standard inverter (typically `INVD1`) as our measuring stick.

*   **Unit Area:** 1.0 = Area of `INVD1`
*   **Unit Delay:** 1.0 = Intrinsic delay of `INVD1`
*   **Unit Drive:** 1.0 = Drive strength of `INVD1`

The Delay Model
---------------

We represent delay (:math:`D`) using a linear approximation. It's not perfect, but it captures the two things that matter: how fast the cell is by itself, and how much it slows down when you give it work to do.

.. math::

   D = D_0 + k \cdot C_{load}

Here is what these terms actually mean physically:

*   :math:`D_0` (**Intrinsic Delay**): This is the delay when the cell is driving *nothing*. It represents the internal parasitic capacitance and resistance of the cell itself. If you could disconnect the output wire entirely, it would still take this long to switch.
*   :math:`k` (**Drive Resistance** or **Logical Effort**): This tells you how much the cell "struggles" with a load. A low :math:`k` means the cell is strong; it can drive a heavy load without slowing down much. A high :math:`k` means the cell is weak.
*   :math:`C_{load}`: The capacitance of whatever is connected to the output.

Extraction
^^^^^^^^^^

Real timing data comes in lookup tables (NLDM), which are just grids of numbers measured by a simulator. To get our simple :math:`D_0` and :math:`k`, we take a slice of that table at a fixed input speed (slew) and fit a straight line through the delay vs. load points.

We usually pick the **FO4 (Fanout-of-4) slew**. This is the speed at which a signal transitions when an inverter drives 4 copies of itself. It's a "typical" situation in a real chip.

Normalization
^^^^^^^^^^^^^

Once we have :math:`D_0` and :math:`k` for a cell (like a NAND gate) and for our baseline inverter, we compare them:

.. math::

   \text{Delay Ratio} = \frac{D_{0, \text{cell}}}{D_{0, \text{INV}}}

If a NAND gate has a Delay Ratio of 1.5, it means "this gate is 50% slower than an inverter just because of its internal structure."

.. math::

   \text{Effort Ratio} = \frac{k_{\text{cell}}}{k_{\text{INV}}}

If the Effort Ratio is 1.3, it means "this gate struggles 30% more than an inverter to drive the same load." This is effectively the **Logical Effort** ($g$) from Sutherland's theory.

Interpolation
-------------

Sometimes we need a value that isn't exactly in the table. The library gives us points on a grid, but we need the value in the middle.

We use **Bilinear Interpolation**. Imagine holding a sheet of rubber at four corners (the table values). If you want to know the height of the sheet in the middle, it depends on how close you are to each corner.

Mathematically, if we are between grid points :math:`x_0, x_1` and :math:`y_0, y_1`, the value is a weighted average:

.. math::

   V(x, y) \approx w_{00}V_{00} + w_{01}V_{01} + w_{10}V_{10} + w_{11}V_{11}

The weights :math:`w` depend on the distance to the opposite corner. The closer you are to a point, the more influence it has.

Fingerprinting (The "Shape" of a Library)
-----------------------------------------

How do you describe a library to a computer? You can't just send the list of cells.

We create a "fingerprint" vector :math:`\mathbf{v}`. We look at the distribution of the normalized metrics across the whole library.

.. math::

   \mathbf{v} = [ \mu_{\text{area}}, \mu_{D0}, \sigma_{\text{area}}, \dots ]

*   **Mean Area Ratio:** Is the library mostly small cells or huge macros?
*   **Mean Intrinsic Delay:** Are the complex gates generally slow or fast compared to the inverter?
*   **Diversity (Sigma):** specific technologies might have a wide range of drive strengths (high sigma), while others offer only a few options (low sigma).

We compare these vectors using the angle between them (**Cosine Similarity**). If the angle is zero, the libraries have the same relative "shape," even if one is 7nm and the other is 180nm.
