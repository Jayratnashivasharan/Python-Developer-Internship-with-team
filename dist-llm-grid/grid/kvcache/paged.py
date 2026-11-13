"""Paged KV-cache block manager (vLLM "PagedAttention" idea, pure Python).

Why this prevents fragmentation
-------------------------------
The cache is one big pool of fixed-size blocks (``block_size`` tokens each). A sequence
owns a *list* of blocks that need not be contiguous. Because every block has the same
size, any free block can serve any sequence: external fragmentation is impossible and
internal waste is bounded to < 1 block per sequence.

The allocator lives on the master (it is the only component that knows which requests
are alive). Workers only hold the tensors and are told the block table on every step,
so no cache metadata has to be synchronised between nodes.

All public coroutines are guarded by an ``asyncio.Lock`` so the scheduler, request
cancellation and the gateway can call into it concurrently without races.
"""
from __future__ import annotations

import asyncio
from collections import deque


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks, self.block_size = num_blocks, block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._owned: dict[int, list[int]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ queries
    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def usage(self) -> float:
        return 1.0 - len(self._free) / self.num_blocks

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= len(self._free)

    # ---------------------------------------------------------------- mutations
    async def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Reserve enough blocks for ``num_tokens`` (prompt + max new tokens)."""
        need = self.blocks_needed(num_tokens)
        async with self._lock:
            if seq_id in self._owned:
                raise ValueError(f"sequence {seq_id} already has blocks")
            if need > len(self._free):
                raise OutOfBlocks(f"need {need} blocks, {len(self._free)} free")
            blocks = [self._free.popleft() for _ in range(need)]
            self._owned[seq_id] = blocks
            return blocks

    async def free(self, seq_id: int) -> None:
        async with self._lock:
            blocks = self._owned.pop(seq_id, None)
            if blocks:
                self._free.extend(blocks)

    def table(self, seq_id: int) -> list[int]:
        return self._owned[seq_id]
