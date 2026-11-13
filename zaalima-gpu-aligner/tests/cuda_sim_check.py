"""Run in a subprocess (NUMBA_ENABLE_CUDASIM=1) or on a real GPU host.
Checks CudaBackend (pinned + mapped) == pure-Python baseline."""
import sys
import numpy as np
from gpualign.config import PipelineConfig
from gpualign.align.backends import CudaBackend
from gpualign.align.baseline import smith_waterman_score
from gpualign.seqio.encoding import decode

N, M, PAD = 24, 60, 10
rng = np.random.default_rng(3)
reads = rng.integers(0, 4, (N, M), dtype=np.uint8)
W = M + 2 * PAD
refs = rng.integers(0, 4, (N, W), dtype=np.uint8)
refs[:, PAD:PAD + M] = reads
mut = rng.random((N, M)) < 0.05
refs[:, PAD:PAD + M][mut] = (reads[mut] + 1) % 4
refs[3, 30] = 4  # an N

for mapped in (False, True):
    be = CudaBackend(PipelineConfig().scoring, mapped=mapped)
    slot = be.alloc_slot(0, N, 256 + 40)
    slot.reads[:, :M] = reads; slot.rlen[:] = M
    slot.refs[:, :W] = refs; slot.wlen[:] = W
    be.run(slot, N)
    for t in range(N):
        exp = smith_waterman_score(decode(reads[t]), decode(refs[t]))
        got = tuple(int(x) for x in slot.out[t])
        assert got == exp, (mapped, t, got, exp)
print("CUDA kernel OK (mapped=False/True), device:", be.device_name())
