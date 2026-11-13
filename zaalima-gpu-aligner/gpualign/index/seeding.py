import numpy as np


def best_cluster(diags: np.ndarray, band: int):
    """Return (diagonal, votes): the diagonal with most other diagonals within
    `band` to its right (an indel shifts the diagonal by its length)."""
    if diags.size == 0:
        return 0, 0
    d = np.sort(diags)
    right = np.searchsorted(d, d + band, side="right")
    counts = right - np.arange(d.size)
    i = int(counts.argmax())
    return int(d[i]), int(counts[i])
