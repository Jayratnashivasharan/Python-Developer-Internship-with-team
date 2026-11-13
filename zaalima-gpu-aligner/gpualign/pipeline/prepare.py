"""Stage 1 (CPU threads): FASTQ batch -> seeds (FM-index) -> locus -> fill a
pinned slot with (read, reference window) pairs. Reads without a consistent
seed locus are FILTERED here and never reach the GPU."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

from ..config import MAX_QUERY, PipelineConfig
from ..index.fm import FMIndex
from ..index.seeding import best_cluster
from ..seqio.encoding import encode, revcomp
from ..seqio.fastq import ReadBatch
from .stats import Stats


@dataclass
class BatchMeta:
    n: int
    names: List[str]
    strands: List[str]
    wstarts: List[int]
    qlens: List[int]


def prepare_batch(batch: ReadBatch, index: FMIndex, pool, cfg: PipelineConfig, stats: Stats):
    slot = pool.acquire()
    names, strands, wstarts, qlens = [], [], [], []
    row = skipped = filtered = 0
    for name, seq in zip(batch.names, batch.seqs):
        m = len(seq)
        if m < cfg.seed_k or m > MAX_QUERY:
            skipped += 1
            continue
        fwd = encode(seq)
        rev = revcomp(fwd)
        df = index.seed_diagonals(fwd, cfg.seed_k, cfg.seed_step, cfg.max_seed_occ)
        dr = index.seed_diagonals(rev, cfg.seed_k, cfg.seed_step, cfg.max_seed_occ)
        d1, v1 = best_cluster(df, cfg.band)
        d2, v2 = best_cluster(dr, cfg.band)
        nseeds = (m - cfg.seed_k) // cfg.seed_step + 1
        need = min(cfg.min_votes, nseeds)
        if max(v1, v2) < need:
            filtered += 1
            continue
        if v1 >= v2:
            q, d, strand = fwd, d1, "+"
        else:
            q, d, strand = rev, d2, "-"
        cs, ce = index.contig_bounds(index.contig_of(min(max(d + m // 2, 0), index.total_length - 1)))
        ws = max(cs, d - cfg.pad)
        we = min(ce, d + m + cfg.pad)
        if we - ws < m // 2:
            filtered += 1
            continue
        slot.reads[row, :m] = q
        slot.rlen[row] = m
        wl = we - ws
        slot.refs[row, :wl] = index.ref[ws:we]
        slot.wlen[row] = wl
        names.append(name); strands.append(strand); wstarts.append(ws); qlens.append(m)
        row += 1
    slot.meta = (BatchMeta(row, names, strands, wstarts, qlens), batch.seqs)
    stats.add(reads_total=len(batch), reads_seeded=row, reads_filtered=filtered,
              reads_skipped=skipped, bases=sum(map(len, batch.seqs)))
    stats.set(bytes_done=max(stats.bytes_done, batch.bytes_done), bytes_total=batch.bytes_total)
    return slot
