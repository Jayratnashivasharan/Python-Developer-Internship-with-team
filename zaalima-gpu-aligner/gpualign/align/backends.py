"""Alignment back-ends behind one interface:

    backend.alloc_slot(i, batch, window) -> Slot
    backend.run(slot, n)                  # fills slot.out[:n] = (score, qend, rend)

  cuda   Numba CUDA kernel, pinned memory, async copies on a per-slot stream
  cpu    Numba-JIT multi-core kernel (same maths; automatic fallback)
  python pure-Python baseline (slow; oracle / benchmarking only)
"""
from __future__ import annotations
import time
from typing import List

import numpy as np

from ..config import MAX_QUERY, PipelineConfig, ScoringParams
from ..seqio.encoding import decode
from .baseline import smith_waterman_score
from .pool import Slot


def _cuda_ok() -> bool:
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def available_backends() -> List[str]:
    out = ["python", "cpu"]
    if _cuda_ok():
        out.append("cuda")
    return out


def _host_slot(i, B, W, alloc=np.zeros) -> Slot:
    return Slot(i, alloc((B, MAX_QUERY), np.uint8), alloc((B,), np.int32),
                alloc((B, W), np.uint8), alloc((B,), np.int32), alloc((B, 3), np.int32))


class PythonBackend:
    name = "python"

    def __init__(self, p: ScoringParams):
        self.p = p
        self.kernel_seconds = 0.0

    def alloc_slot(self, i, B, W):
        return _host_slot(i, B, W)

    def run(self, slot: Slot, n: int):
        t0 = time.perf_counter()
        for t in range(n):
            q = decode(slot.reads[t, :slot.rlen[t]])
            r = decode(slot.refs[t, :slot.wlen[t]])
            slot.out[t] = smith_waterman_score(q, r, self.p)
        self.kernel_seconds += time.perf_counter() - t0


class CpuBackend:
    name = "cpu"

    def __init__(self, p: ScoringParams):
        from .cpu_kernels import sw_score_batch
        self._k, self.p = sw_score_batch, p
        self.kernel_seconds = 0.0

    def alloc_slot(self, i, B, W):
        return _host_slot(i, B, W)

    def run(self, slot: Slot, n: int):
        p = self.p
        t0 = time.perf_counter()
        self._k(slot.reads, slot.rlen, slot.refs, slot.wlen, n,
                p.match, p.mismatch, p.gap_open, p.gap_extend, slot.out)
        self.kernel_seconds += time.perf_counter() - t0


class CudaBackend:
    name = "cuda"

    def __init__(self, p: ScoringParams, mapped: bool = False):
        from numba import cuda
        from .cuda_kernels import sw_kernel, THREADS_PER_BLOCK
        self.cuda, self._k, self.tpb = cuda, sw_kernel, THREADS_PER_BLOCK
        self.p, self.mapped = p, mapped
        self.kernel_seconds = 0.0

    def device_name(self) -> str:
        try:
            d = self.cuda.get_current_device()
            n = d.name
            return n.decode() if isinstance(n, bytes) else str(n)
        except Exception:
            return "CUDA simulator"

    def alloc_slot(self, i, B, W):
        c = self.cuda
        # Pinned (page-locked) host memory -> true async DMA; mapped memory ->
        # device reads host RAM directly over PCIe/NVLink (zero-copy).
        host = (lambda shape, dt: c.mapped_array(shape, dt)) if self.mapped \
            else (lambda shape, dt: c.pinned_array(shape, dt))
        s = Slot(i, host((B, MAX_QUERY), np.uint8), host((B,), np.int32),
                 host((B, W), np.uint8), host((B,), np.int32), host((B, 3), np.int32))
        for a in (s.reads, s.rlen, s.refs, s.wlen, s.out):
            a[...] = 0
        s.stream = c.stream()
        if self.mapped:
            s.d_reads, s.d_rlen, s.d_refs, s.d_wlen, s.d_out = s.reads, s.rlen, s.refs, s.wlen, s.out
        else:
            s.d_reads = c.device_array(s.reads.shape, np.uint8)
            s.d_rlen = c.device_array(s.rlen.shape, np.int32)
            s.d_refs = c.device_array(s.refs.shape, np.uint8)
            s.d_wlen = c.device_array(s.wlen.shape, np.int32)
            s.d_out = c.device_array(s.out.shape, np.int32)
        return s

    def run(self, slot: Slot, n: int):
        p, st = self.p, slot.stream
        t0 = time.perf_counter()
        if not self.mapped:                                   # H2D (async, pinned)
            slot.d_reads.copy_to_device(slot.reads, stream=st)
            slot.d_rlen.copy_to_device(slot.rlen, stream=st)
            slot.d_refs.copy_to_device(slot.refs, stream=st)
            slot.d_wlen.copy_to_device(slot.wlen, stream=st)
        blocks = (n + self.tpb - 1) // self.tpb
        self._k[blocks, self.tpb, st](slot.d_reads, slot.d_rlen, slot.d_refs, slot.d_wlen, n,
                                      p.match, p.mismatch, p.gap_open, p.gap_extend, slot.d_out)
        if not self.mapped:                                   # D2H
            slot.d_out.copy_to_host(slot.out, stream=st)
        st.synchronize()
        self.kernel_seconds += time.perf_counter() - t0


def get_backend(name: str, cfg: PipelineConfig):
    name = (name or "auto").lower()
    if name == "auto":
        name = "cuda" if _cuda_ok() else "cpu"
    if name == "cuda":
        if not _cuda_ok():
            raise RuntimeError("CUDA requested but no CUDA device available "
                               "(use --backend cpu, or NUMBA_ENABLE_CUDASIM=1 to simulate)")
        return CudaBackend(cfg.scoring, cfg.use_mapped_memory)
    if name == "cpu":
        return CpuBackend(cfg.scoring)
    if name == "python":
        return PythonBackend(cfg.scoring)
    raise ValueError(f"unknown backend {name!r}")
