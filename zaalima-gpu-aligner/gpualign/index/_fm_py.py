"""Pure-Python/numpy implementation of the FM-index seed search.
Functionally identical to _fmcore.pyx (used when the extension is not built)."""
import numpy as np

OCC_SHIFT = 6   # checkpoint every 64 symbols


def _occ(bwt, occ, c, i):
    b = i >> OCC_SHIFT
    return int(occ[b, c]) + int(np.count_nonzero(bwt[b << OCC_SHIFT:i] == c))


def seed_diagonals(bwt, occ, C, sa, read, k, step, max_occ):
    """For every k-mer of `read` (stride `step`, ACGT only) do an exact
    backward search and return int64 array of (ref_pos - read_offset)."""
    m = read.shape[0]
    out = []
    n = bwt.shape[0]
    q0 = 0
    while q0 + k <= m:
        pat = read[q0:q0 + k]
        if pat.max() < 4:
            lo, hi = 0, n
            for t in range(k - 1, -1, -1):
                c = int(pat[t]) + 1
                lo = int(C[c]) + _occ(bwt, occ, c, lo)
                hi = int(C[c]) + _occ(bwt, occ, c, hi)
                if lo >= hi:
                    break
            if lo < hi and hi - lo <= max_occ:
                for r in range(lo, hi):
                    out.append(int(sa[r]) - q0)
        q0 += step
    return np.asarray(out, dtype=np.int64)
