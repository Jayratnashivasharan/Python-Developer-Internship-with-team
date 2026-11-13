"""Suffix-array construction by prefix doubling (numpy, O(n log^2 n)).

Practical to a few hundred Mbp per machine; see docs/HARDWARE_OPTIMIZATION.md
for the scaling limits and the production alternative (SA-IS / libdivsufsort).
`text` must end with a unique, smallest sentinel (code 0).
"""
import numpy as np


def build_suffix_array(text: np.ndarray) -> np.ndarray:
    n = text.shape[0]
    if n >= 2**31 - 1:
        raise ValueError("text too long for int32 suffix array")
    rank = text.astype(np.int64)
    sa = np.argsort(rank, kind="stable")
    srt = rank[sa]
    newr = np.concatenate(([0], np.cumsum(srt[1:] != srt[:-1])))
    rank[sa] = newr
    k = 1
    while newr[-1] < n - 1:
        second = np.zeros(n, np.int64)
        if k < n:
            second[: n - k] = rank[k:] + 1
        key = rank * (n + 1) + second
        sa = np.argsort(key, kind="stable")
        ks = key[sa]
        newr = np.concatenate(([0], np.cumsum(ks[1:] != ks[:-1])))
        rank[sa] = newr
        k <<= 1
    return sa.astype(np.int32)
