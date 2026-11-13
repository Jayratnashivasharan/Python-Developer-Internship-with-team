"""Numba-JIT CPU kernels: (1) score-only batch kernel (CPU fallback for the
GPU kernel), (2) full traceback, (3) variant-event extraction.
All use the exact conventions documented in baseline.py."""
from __future__ import annotations
import numpy as np
from numba import njit, prange

NEG = -100000


@njit(parallel=True, nogil=True, cache=True)
def sw_score_batch(reads, rlen, refs, wlen, n, match, mismatch, gopen, gext, out):
    """out[t] = (score, qend, rend). One pair per prange iteration."""
    for t in prange(n):
        m = rlen[t]
        w = wlen[t]
        H = np.zeros(m + 1, np.int32)
        E = np.full(m + 1, NEG, np.int32)
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


@njit(nogil=True, cache=True)
def sw_traceback(q, r, match, mismatch, gopen, gext):
    """Full local alignment with traceback.
    Returns (score, qstart, qend, rstart, rend, ops[uint8: 0=M 1=I 2=D])."""
    m = q.shape[0]
    n = r.shape[0]
    H = np.zeros((n + 1, m + 1), np.int32)
    E = np.full((n + 1, m + 1), NEG, np.int32)
    F = np.full((n + 1, m + 1), NEG, np.int32)
    best = 0
    bi = 0
    bj = 0
    for i in range(1, n + 1):
        rc = r[i - 1]
        for j in range(1, m + 1):
            e = max(E[i - 1, j] - gext, H[i - 1, j] - gopen)
            f = max(F[i, j - 1] - gext, H[i, j - 1] - gopen)
            if rc == q[j - 1] and rc < 4:
                s = match
            else:
                s = -mismatch
            h = max(0, H[i - 1, j - 1] + s, e, f)
            E[i, j] = e
            F[i, j] = f
            H[i, j] = h
            if h > best:
                best = h
                bi = i
                bj = j
    ops = np.empty(n + m + 1, np.uint8)
    k = 0
    i = bi
    j = bj
    state = 0
    while i > 0 and j > 0 and best > 0:
        if state == 0:
            h = H[i, j]
            if h == 0:
                break
            if r[i - 1] == q[j - 1] and r[i - 1] < 4:
                s = match
            else:
                s = -mismatch
            if h == H[i - 1, j - 1] + s:
                ops[k] = 0
                k += 1
                i -= 1
                j -= 1
            elif h == E[i, j]:
                state = 1
            else:
                state = 2
        elif state == 1:
            ops[k] = 2
            k += 1
            if E[i, j] == H[i - 1, j] - gopen:
                state = 0
            i -= 1
        else:
            ops[k] = 1
            k += 1
            if F[i, j] == H[i, j - 1] - gopen:
                state = 0
            j -= 1
    return best, j, bj, i, bi, ops[:k][::-1].copy()


@njit(nogil=True, cache=True)
def extract_events(ops, q, qstart, win, rstart):
    """Walk an alignment and return events as 4 int32 arrays:
    kind (0=SNP 1=INS 2=DEL), ref index (SNP pos / indel ANCHOR base in `win`),
    query index, length."""
    nops = ops.shape[0]
    kind = np.empty(nops, np.int32)
    rpos = np.empty(nops, np.int32)
    qpos = np.empty(nops, np.int32)
    ln = np.empty(nops, np.int32)
    ne = 0
    i = rstart
    j = qstart
    k = 0
    while k < nops:
        op = ops[k]
        if op == 0:
            if q[j] != win[i] and q[j] < 4 and win[i] < 4:
                kind[ne] = 0; rpos[ne] = i; qpos[ne] = j; ln[ne] = 1
                ne += 1
            i += 1
            j += 1
            k += 1
        else:
            run = 0
            while k + run < nops and ops[k + run] == op:
                run += 1
            kind[ne] = op
            rpos[ne] = i - 1
            qpos[ne] = j
            ln[ne] = run
            ne += 1
            if op == 1:
                j += run
            else:
                i += run
            k += run
    return kind[:ne], rpos[:ne], qpos[:ne], ln[:ne]
