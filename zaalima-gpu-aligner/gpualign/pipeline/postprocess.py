"""Stage 3 (CPU threads, GIL released inside numba): for reads whose GPU score
passes the filter, run traceback on the small sub-window the GPU identified,
re-verify the score (GPU-vs-CPU self-check) and extract variant events."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..align.baseline import ops_to_cigar
from ..align.cpu_kernels import extract_events, sw_traceback
from ..config import PipelineConfig
from ..variants.caller import to_events


@dataclass
class PostResult:
    lines: List[str] = field(default_factory=list)
    alignments: List[Tuple[int, int, list]] = field(default_factory=list)  # rstart,rend,events
    aligned: int = 0
    lowscore: int = 0
    mismatch: int = 0
    n_events: int = 0


def postprocess(slot, index, cfg: PipelineConfig) -> PostResult:
    meta, _ = slot.meta
    p = cfg.scoring
    res = PostResult()
    for i in range(meta.n):
        score, qend, rend = (int(x) for x in slot.out[i])
        m = meta.qlens[i]
        if score < cfg.min_score_frac * p.match * m or score <= 0:
            res.lowscore += 1
            continue
        q = slot.reads[i, :m]
        win = slot.refs[i, :slot.wlen[i]]
        sc, qs, qe, rs, re_, ops = sw_traceback(q[:qend], win[:rend],
                                                p.match, p.mismatch, p.gap_open, p.gap_extend)
        if sc != score:
            res.mismatch += 1          # GPU/CPU disagreement -> trust the CPU traceback
        if sc <= 0:
            res.lowscore += 1
            continue
        kind, rpos, qpos, ln = extract_events(ops, q, qs, win, rs)
        ws = meta.wstarts[i]
        events = to_events(kind, rpos, qpos, ln, win, q, ws)
        cname, local = index.to_local(ws + rs)
        cigar = ops_to_cigar(ops.tolist(), qs, qe, m)
        res.lines.append(f"{meta.names[i]}\t{cname}\t{local + 1}\t{meta.strands[i]}\t{sc}\t{cigar}\n")
        res.alignments.append((ws + rs, ws + re_, events))
        res.aligned += 1
        res.n_events += len(events)
    return res
