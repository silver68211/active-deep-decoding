# -*- coding: utf-8 -*-
"""
alist_to_binary.py

Utility for converting an ALIST file representation of a sparse parity-check
matrix into its dense binary (NumPy) representation.

Author: hnksm
Created: Mar 24, 2023
Refactored: 2026

Description
-----------
ALIST is a standard format used to describe sparse parity-check matrices
(e.g., for LDPC/BCH codes). This function reads an ALIST file and constructs
the corresponding binary parity-check matrix H.

Input
-----
    filepath : str
        Path to the ALIST file.

Output
------
    H : np.ndarray of shape (m, n)
        Binary parity-check matrix with entries in {0,1}.
"""

from __future__ import annotations
import numpy as np


def Alist2bin(filepath: str) -> np.ndarray:
    """
    Convert an ALIST file to a binary parity-check matrix.

    Parameters
    ----------
    filepath : str
        Path to the ALIST file.

    Returns
    -------
    H : np.ndarray
        Parity-check matrix of shape (m, n) with dtype np.int32.
    """

    # ------------------------------------------------------------------
    # Read and parse file
    # ------------------------------------------------------------------
    with open(filepath, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    # Convert all entries to integers
    alist = [[int(x) for x in row] for row in lines]

    # ------------------------------------------------------------------
    # Extract matrix dimensions
    # ------------------------------------------------------------------
    # First line: n (columns), m (rows)
    n, m = alist[0]

    if n <= 0 or m <= 0:
        raise ValueError("Invalid ALIST file: non-positive matrix dimensions.")

    # Initialize parity-check matrix
    H = np.zeros((m, n), dtype=np.int32)

    # ------------------------------------------------------------------
    # Extract column weights
    # ------------------------------------------------------------------
    # Third line contains column degrees (number of ones in each column)
    column_weights = alist[2]

    if len(column_weights) != n:
        raise ValueError("Mismatch between declared column count and column weights.")

    # ------------------------------------------------------------------
    # Build parity-check matrix
    # ------------------------------------------------------------------
    # Starting from line index 4:
    # Each of the next n lines lists row indices of nonzero entries per column.
    #
    # Note: ALIST uses 1-based indexing → convert to 0-based.
    for col in range(n):
        # Row indices for this column (truncate to actual degree)
        row_indices = alist[4 + col][: column_weights[col]]

        for row in row_indices:
            if row <= 0 or row > m:
                raise ValueError("Invalid row index in ALIST file.")
            H[row - 1, col] = 1  # Convert to zero-based indexing

    return H

import numpy as np
import matplotlib.pyplot as plt
from typing import Union

try:
    from scipy.sparse import issparse
except ImportError:
    issparse = lambda x: False


from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse


def plot_pcm(
    H: Union[np.ndarray, "scipy.sparse.spmatrix"],
    title: str = "Parity-Check Matrix (PCM)",
    figsize: tuple = (8, 6),
    show_values: bool = False,
    show_grid: bool = True,
) -> None:
    """
    Visualize a parity-check matrix as a binary image.

    Parameters
    ----------
    H : np.ndarray or scipy.sparse matrix
        Parity-check matrix of shape (m, n).
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height).
    show_values : bool
        If True, overlay matrix values (only suitable for small matrices).
    show_grid : bool
        If True, draw grid lines at cell boundaries.
    """

    # Convert sparse to dense if needed
    if issparse(H):
        H = H.toarray()

    H = np.asarray(H)

    if H.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    m, n = H.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Display matrix
    ax.imshow(H,  aspect="equal")

    

    # Remove major ticks completely
    ax.set_xticks([])
    ax.set_yticks([])

    if show_grid:
        # Minor ticks at cell boundaries
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, m, 1), minor=True)

        # Draw grid along minor ticks
        ax.grid(which="minor", color="gray", linestyle="--", linewidth=1)

    ax.tick_params(which="both", direction="in", length=1, width=1, color="gray")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Optional test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    H = Alist2bin("codes/BCH_7_4_1_strip.alist.txt")
    print("Parity-check matrix shape:", H.shape)
    print(H)
    plot_pcm(H)

    pass
