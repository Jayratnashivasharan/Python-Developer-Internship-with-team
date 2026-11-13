"""Fixed-size pool of reusable batch buffers ("slots").

Why: cudaMallocHost / cudaMalloc are expensive and pinned memory is a scarce
OS resource. We allocate N slots ONCE; the pipeline recycles them forever, so
steady-state runs perform zero allocations and zero extra host-side copies:
the I/O stage writes encoded reads *directly* into the page-locked buffer that
the DMA engine then copies to the device.
"""
from __future__ import annotations
import queue
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Slot:
    id: int
    reads: np.ndarray      # (B, MAX_QUERY) uint8
    rlen: np.ndarray       # (B,) int32
    refs: np.ndarray       # (B, WINDOW)    uint8
    wlen: np.ndarray       # (B,) int32
    out: np.ndarray        # (B, 3) int32   score, qend, rend
    # CUDA-only handles (None on CPU backends)
    stream: Any = None
    d_reads: Any = None
    d_rlen: Any = None
    d_refs: Any = None
    d_wlen: Any = None
    d_out: Any = None
    meta: Any = field(default=None)   # pipeline bookkeeping for the batch in flight


class SlotPool:
    def __init__(self, backend, n_slots: int, batch_size: int, window: int):
        self._q: "queue.Queue[Slot]" = queue.Queue()
        self.n_slots = n_slots
        self.batch_size = batch_size
        for i in range(n_slots):
            self._q.put(backend.alloc_slot(i, batch_size, window))

    def acquire(self, timeout: Optional[float] = None) -> Slot:
        return self._q.get(timeout=timeout)

    def release(self, slot: Slot) -> None:
        slot.meta = None
        self._q.put(slot)

    def free(self) -> int:
        return self._q.qsize()
