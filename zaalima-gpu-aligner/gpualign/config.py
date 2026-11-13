"""Central configuration.  Values that become CUDA compile-time constants
(MAX_QUERY) are read from the environment ONCE at import time."""
from __future__ import annotations
import os
from dataclasses import dataclass

# Longest read the GPU kernel accepts. It sizes per-thread local memory
# (2 x (MAX_QUERY+1) int32), so keep it modest. Override: GPUALIGN_MAX_QUERY=512
MAX_QUERY: int = int(os.environ.get("GPUALIGN_MAX_QUERY", "256"))

# Reference window padding on each side of a seed diagonal (max indel tolerated)
DEFAULT_PAD: int = 20


@dataclass(frozen=True)
class ScoringParams:
    """Affine-gap local alignment scoring (all values positive penalties).
    A gap of length L costs  gap_open + (L-1)*gap_extend."""
    match: int = 2
    mismatch: int = 3
    gap_open: int = 5
    gap_extend: int = 2


@dataclass
class PipelineConfig:
    batch_size: int = 2048          # reads per GPU batch
    seed_k: int = 16                # FM-index seed length
    seed_step: int = 8              # distance between seeds on the read
    max_seed_occ: int = 50          # skip seeds that repeat more often
    min_votes: int = 2              # seeds agreeing on one diagonal
    band: int = 16                  # diagonals within this distance = same locus
    pad: int = DEFAULT_PAD
    min_score_frac: float = 0.5     # min score / perfect score to report
    prefetch: int = 3               # prepared batches waiting for the GPU
    prep_workers: int = 2           # threads doing I/O + seeding
    post_workers: int = 2           # threads doing traceback + variant extraction
    use_mapped_memory: bool = False # CUDA true zero-copy (mapped host memory)
    scoring: ScoringParams = ScoringParams()

    @property
    def window(self) -> int:
        return MAX_QUERY + 2 * self.pad
