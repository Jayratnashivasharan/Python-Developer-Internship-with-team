# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython FM-index backward search (GIL released in the hot loop).
Mirrors _fm_py.seed_diagonals exactly."""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t, int32_t, uint8_t

cdef inline int64_t _occ(const uint8_t[::1] bwt, const int64_t[:, ::1] occ,
                         int c, int64_t i) noexcept nogil:
    cdef int64_t b = i >> 6
    cdef int64_t cnt = occ[b, c]
    cdef int64_t p = b << 6
    while p < i:
        if bwt[p] == c:
            cnt += 1
        p += 1
    return cnt


def seed_diagonals(const uint8_t[::1] bwt, const int64_t[:, ::1] occ,
                   const int64_t[::1] C, const int32_t[::1] sa,
                   const uint8_t[::1] read, int k, int step, int max_occ):
    cdef Py_ssize_t m = read.shape[0]
    cdef Py_ssize_t nseeds = 0
    if m >= k:
        nseeds = (m - k) // step + 1
    out_arr = np.empty(nseeds * max_occ + 1, dtype=np.int64)
    cdef int64_t[::1] out = out_arr
    cdef Py_ssize_t cnt = 0
    cdef Py_ssize_t q0, t
    cdef int64_t lo, hi, r
    cdef int64_t n = bwt.shape[0]
    cdef int c
    cdef bint ok
    with nogil:
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
                    c = read[q0 + t] + 1
                    lo = C[c] + _occ(bwt, occ, c, lo)
                    hi = C[c] + _occ(bwt, occ, c, hi)
                    if lo >= hi:
                        break
                if lo < hi and hi - lo <= max_occ:
                    for r in range(lo, hi):
                        out[cnt] = <int64_t>sa[r] - q0
                        cnt += 1
            q0 += step
    return out_arr[:cnt]
