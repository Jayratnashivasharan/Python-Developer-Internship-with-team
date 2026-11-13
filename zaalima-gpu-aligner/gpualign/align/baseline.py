"""Pure-Python affine-gap Smith-Waterman (Gotoh) -- the ACCURACY ORACLE.

Deliberately simple, list-based and independent of numpy/numba so that the
accelerated kernels can be validated against it cell-for-cell.

Conventions (shared by every kernel in this project)
  i = reference index (rows), j = query index (columns), both 1-based in the DP
  E: gap that consumes reference only ('D'),   F: gap that consumes query only ('I')
  H = max(0, H[i-1][j-1]+s, E, F)
  E[i][j] = max(E[i-1][j]-ext, H[i-1][j]-open)
  F[i][j] = max(F[i][j-1]-ext, H[i][j-1]-open)
  Best cell = FIRST strict maximum in row-major order (i outer, j inner).
  'N' never matches (not even N vs N).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

from ..config import ScoringParams

NEG = -100000
OP_M, OP_I, OP_D = 0, 1, 2
_OPCH = "MID"


@dataclass
class Alignment:
    score: int
    qstart: int          # 0-based, inclusive
    qend: int            # 0-based, exclusive
    rstart: int
    rend: int
    ops: List[int]       # per-base ops, start->end
    cigar: str


def ops_to_cigar(ops: Sequence[int], qstart: int, qend: int, qlen: int) -> str:
    out = []
    if qstart > 0:
        out.append(f"{qstart}S")
    run, prev = 0, None
    for op in ops:
        if op == prev:
            run += 1
        else:
            if prev is not None:
                out.append(f"{run}{_OPCH[prev]}")
            prev, run = op, 1
    if prev is not None:
        out.append(f"{run}{_OPCH[prev]}")
    if qlen - qend > 0:
        out.append(f"{qlen - qend}S")
    return "".join(out)


def smith_waterman(query: str, ref: str, p: ScoringParams = ScoringParams()) -> Alignment:
    m, n = len(query), len(ref)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    E = [[NEG] * (m + 1) for _ in range(n + 1)]
    F = [[NEG] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        rc = ref[i - 1]
        for j in range(1, m + 1):
            e = max(E[i - 1][j] - p.gap_extend, H[i - 1][j] - p.gap_open)
            f = max(F[i][j - 1] - p.gap_extend, H[i][j - 1] - p.gap_open)
            qc = query[j - 1]
            s = p.match if (rc == qc and rc != "N") else -p.mismatch
            h = max(0, H[i - 1][j - 1] + s, e, f)
            E[i][j], F[i][j], H[i][j] = e, f, h
            if h > best:
                best, bi, bj = h, i, j
    # ---- traceback ------------------------------------------------------
    ops: List[int] = []
    i, j, state = bi, bj, 0
    while i > 0 and j > 0 and best > 0:
        if state == 0:
            h = H[i][j]
            if h == 0:
                break
            s = p.match if (ref[i - 1] == query[j - 1] and ref[i - 1] != "N") else -p.mismatch
            if h == H[i - 1][j - 1] + s:
                ops.append(OP_M); i -= 1; j -= 1
            elif h == E[i][j]:
                state = 1
            else:
                state = 2
        elif state == 1:
            ops.append(OP_D)
            if E[i][j] == H[i - 1][j] - p.gap_open:
                state = 0
            i -= 1
        else:
            ops.append(OP_I)
            if F[i][j] == H[i][j - 1] - p.gap_open:
                state = 0
            j -= 1
    ops.reverse()
    return Alignment(best, j, bj, i, bi, ops, ops_to_cigar(ops, j, bj, m))


def smith_waterman_score(query: str, ref: str, p: ScoringParams = ScoringParams()):
    """Score-only variant: returns (score, qend, rend) like the GPU kernel."""
    a = smith_waterman(query, ref, p)
    return a.score, a.qend, a.rend
