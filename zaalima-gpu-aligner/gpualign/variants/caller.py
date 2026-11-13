"""Pileup-style variant accumulation from alignment deviations.

Events are keyed by (global_pos, REF, ALT) in VCF representation:
  SNP  pos, X, Y        INS  anchor, A, A+ins        DEL  anchor, A+del, A
Depth is kept as a difference array on disk (np.memmap) so reference size is
not limited by RAM. Indels are not left-normalised (documented limitation).
"""
from __future__ import annotations
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

_CH = "ACGTN"


class VariantAccumulator:
    def __init__(self, total_length: int, work_dir: str):
        path = os.path.join(work_dir, "depth_diff.npy")
        self.diff = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32,
                                              shape=(total_length + 2,))
        self.counts: Counter = Counter()
        self.path = path

    def add_alignment(self, rstart: int, rend: int, events: List[Tuple[int, str, str]]):
        self.diff[rstart] += 1
        self.diff[rend] -= 1
        for e in events:
            self.counts[e] += 1

    def depth(self) -> np.ndarray:
        return np.cumsum(self.diff, dtype=np.int64)

    def call(self, min_alt=3, min_af=0.25, min_depth=4) -> List[dict]:
        depth = self.depth()
        out = []
        for (pos, ref, alt), c in self.counts.items():
            dp = int(depth[pos])
            if c < min_alt or dp < min_depth or c / max(dp, 1) < min_af:
                continue
            af = min(1.0, c / dp)
            out.append({"gpos": pos, "ref": ref, "alt": alt, "ac": c, "dp": dp,
                        "af": af, "qual": _qual(c, dp), "gt": "1/1" if af >= 0.8 else "0/1"})
        out.sort(key=lambda v: (v["gpos"], v["alt"]))
        return out


def _qual(k: int, n: int, err=0.01, cap=999.0) -> float:
    """Phred-scaled P(>=k erroneous reads out of n | error rate `err`). Simple
    binomial model - a screening score, not a calibrated posterior."""
    n = max(n, k)
    p = 0.0
    for i in range(k, min(n, k + 40) + 1):
        p += math.comb(n, i) * err ** i * (1 - err) ** (n - i)
    return round(min(cap, -10 * math.log10(max(p, 1e-300))), 1)


def to_events(kind, rpos, qpos, ln, win: np.ndarray, q: np.ndarray, wstart: int):
    """numba event arrays -> list of (gpos, REF, ALT)."""
    ev = []
    for k, rp, qp, L in zip(kind.tolist(), rpos.tolist(), qpos.tolist(), ln.tolist()):
        if k == 0:
            ev.append((wstart + rp, _CH[win[rp]], _CH[q[qp]]))
        elif k == 1:
            ins = "".join(_CH[x] for x in q[qp:qp + L])
            a = _CH[win[rp]]
            if win[rp] < 4:
                ev.append((wstart + rp, a, a + ins))
        else:
            dele = "".join(_CH[x] for x in win[rp + 1: rp + 1 + L])
            a = _CH[win[rp]]
            if win[rp] < 4 and "N" not in dele:
                ev.append((wstart + rp, a + dele, a))
    return ev
