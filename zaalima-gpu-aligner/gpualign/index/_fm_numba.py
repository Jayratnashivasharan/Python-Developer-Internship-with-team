"""Numba-JIT implementation of the FM-index seed search.

Used automatically when the Cython extension has not been compiled, so the
pipeline stays fast on machines without a C compiler. Same results as
_fmcore.pyx and _fm_py.py (tested)."""
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _occ(bwt, occ, c, i):
    b = i >> 6
    cnt = occ[b, c]
    p = b << 6
    while p < i:
        if bwt[p] == c:
            cnt += 1
        p += 1
    return cnt


@njit(nogil=True, cache=True)
def _seed(bwt, occ, C, sa, read, k, step, max_occ, out):
    m = read.shape[0]
    n = bwt.shape[0]
    cnt = 0
    q0 = 0
    while q0 + k <= m:
        ok = True
        for t in range(k):
            if read[q0 + t] >= 4:
                ok = False
                break
        if ok:
            lo = 0
            hi = n
            for t in range(k - 1, -1, -1):
                c = np.int64(read[q0 + t]) + 1
                lo = C[c] + _occ(bwt, occ, c, lo)
                hi = C[c] + _occ(bwt, occ, c, hi)
                if lo >= hi:
                    break
            if lo < hi and hi - lo <= max_occ:
                for r in range(lo, hi):
                    out[cnt] = np.int64(sa[r]) - q0
                    cnt += 1
        q0 += step
    return cnt


def seed_diagonals(bwt, occ, C, sa, read, k, step, max_occ):
    m = read.shape[0]
    nseeds = (m - k) // step + 1 if m >= k else 0
    out = np.empty(nseeds * max_occ + 1, dtype=np.int64)
    cnt = _seed(bwt, occ, C, sa, read, k, step, max_occ, out)
    return out[:cnt]
