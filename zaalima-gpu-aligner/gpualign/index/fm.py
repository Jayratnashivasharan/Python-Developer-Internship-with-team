"""FM-index (Burrows-Wheeler Transform) used as the seeding PRE-FILTER:
only reads that share exact k-mers with the reference at a consistent locus
are sent to the (expensive) Smith-Waterman stage, and only against a small
reference window around that locus."""
from __future__ import annotations
import json
import os
from typing import List, Tuple

import numpy as np

from ..seqio.fasta import fasta_to_encoded
from .suffix_array import build_suffix_array

# Seed-search engine, fastest first: Cython (compiled) -> Numba JIT -> pure Python.
try:
    from ._fmcore import seed_diagonals as _seed
    SEED_ENGINE = "cython"
except ImportError:
    try:
        from ._fm_numba import seed_diagonals as _seed
        SEED_ENGINE = "numba"
    except Exception:                    # numba unusable for any reason
        from ._fm_py import seed_diagonals as _seed
        SEED_ENGINE = "python"
USING_CYTHON = SEED_ENGINE == "cython"

OCC_STEP = 64
SIGMA = 6        # $ A C G T N


def _bwt_tables(text: np.ndarray, sa: np.ndarray):
    n = text.shape[0]
    bwt = text[(sa.astype(np.int64) - 1) % n].astype(np.uint8)
    cnt = np.bincount(text, minlength=SIGMA).astype(np.int64)
    C = np.concatenate(([0], np.cumsum(cnt)[:-1])).astype(np.int64)
    occ = np.zeros((n // OCC_STEP + 1, SIGMA), dtype=np.int64)
    for c in range(SIGMA):
        cum = np.concatenate(([0], np.cumsum(bwt == c, dtype=np.int64)))
        occ[:, c] = cum[::OCC_STEP]
    return bwt, C, occ


def build_index(fasta: str, out_dir: str, verbose: bool = True) -> dict:
    meta = fasta_to_encoded(fasta, out_dir)
    ref = np.fromfile(os.path.join(out_dir, "ref.bin"), dtype=np.uint8)
    text = np.concatenate((ref + 1, np.zeros(1, np.uint8)))      # symbols 1..5, sentinel 0
    sa = build_suffix_array(text)
    bwt, C, occ = _bwt_tables(text, sa)
    np.save(os.path.join(out_dir, "sa.npy"), sa)
    np.save(os.path.join(out_dir, "bwt.npy"), bwt)
    np.save(os.path.join(out_dir, "C.npy"), C)
    np.save(os.path.join(out_dir, "occ.npy"), occ)
    if verbose:
        print(f"[index] {len(meta['contigs'])} contigs, {meta['total_length']:,} bp -> {out_dir}")
    return meta


class FMIndex:
    """Read-only index; arrays are memory-mapped so many processes share them."""

    def __init__(self, directory: str):
        self.dir = directory
        with open(os.path.join(directory, "meta.json")) as fh:
            meta = json.load(fh)
        self.contigs: List[dict] = meta["contigs"]
        self.total_length: int = meta["total_length"]
        self.ref = np.memmap(os.path.join(directory, "ref.bin"), dtype=np.uint8, mode="r")
        ld = lambda n: np.load(os.path.join(directory, n), mmap_mode="r")
        self.sa, self.bwt, self.C, self.occ = ld("sa.npy"), ld("bwt.npy"), ld("C.npy"), ld("occ.npy")
        self._starts = np.array([c["start"] for c in self.contigs], dtype=np.int64)

    def seed_diagonals(self, read: np.ndarray, k: int, step: int, max_occ: int) -> np.ndarray:
        return _seed(self.bwt, self.occ, self.C, self.sa, read, k, step, max_occ)

    def contig_of(self, gpos: int) -> int:
        i = int(np.searchsorted(self._starts, gpos, side="right")) - 1
        return max(0, i)

    def contig_bounds(self, ci: int) -> Tuple[int, int]:
        c = self.contigs[ci]
        return c["start"], c["start"] + c["length"]

    def to_local(self, gpos: int) -> Tuple[str, int]:
        """global position -> (contig name, 0-based local position)"""
        ci = self.contig_of(gpos)
        return self.contigs[ci]["name"], gpos - self.contigs[ci]["start"]
