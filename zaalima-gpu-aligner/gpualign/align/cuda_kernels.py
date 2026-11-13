"""CUDA kernel (Numba @cuda.jit): one thread aligns one (read, ref-window) pair
with affine-gap Smith-Waterman, score-only, O(m) per-thread local memory.

  * Output per pair: (best score, query end, reference end). Traceback is only
    needed for reads that pass the score filter and is done on the CPU, so the
    GPU never wastes memory bandwidth on O(m*n) direction matrices.
  * Arithmetic is identical to baseline.py / cpu_kernels.py (validated by
    tests/test_kernel_equivalence.py).
  * Importing this module needs no GPU; kernels compile lazily at first launch.
    Set NUMBA_ENABLE_CUDASIM=1 to execute the same kernel on the CPU simulator.
"""
from __future__ import annotations
from numba import cuda, int32

from ..config import MAX_QUERY

NEG = -100000
THREADS_PER_BLOCK = 128


@cuda.jit
def sw_kernel(reads, rlen, refs, wlen, n, match, mismatch, gopen, gext, out):
    t = cuda.grid(1)
    if t >= n:
        return
    m = rlen[t]
    w = wlen[t]
    H = cuda.local.array(MAX_QUERY + 1, int32)
    E = cuda.local.array(MAX_QUERY + 1, int32)
    for j in range(m + 1):
        H[j] = 0
        E[j] = NEG
    best = 0
    bi = 0
    bj = 0
    for i in range(1, w + 1):
        rc = refs[t, i - 1]
        diag = 0
        left = 0
        f = NEG
        for j in range(1, m + 1):
            hold = H[j]
            e = max(E[j] - gext, hold - gopen)
            f = max(f - gext, left - gopen)
            if rc == reads[t, j - 1] and rc < 4:
                s = match
            else:
                s = -mismatch
            h = max(0, diag + s, e, f)
            H[j] = h
            E[j] = e
            diag = hold
            left = h
            if h > best:
                best = h
                bi = i
                bj = j
    out[t, 0] = best
    out[t, 1] = bj
    out[t, 2] = bi
