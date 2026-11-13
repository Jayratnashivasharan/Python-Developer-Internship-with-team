import asyncio

import pytest
import torch

from grid.kvcache.paged import BlockAllocator, OutOfBlocks
from tests.helpers import logits_for, make_runner, seq


def test_incremental_decode_matches_full_prefill(tiny_model):
    """Prefill(prompt) + decode(token) through paged blocks == one big prefill."""
    toks = [256, 10, 20, 30, 40, 50, 60]
    full = logits_for(make_runner(tiny_model), [seq(toks, 0, [3, 1])])

    r = make_runner(tiny_model)
    logits_for(r, [seq(toks[:-1], 0, [3, 1])])
    inc = logits_for(r, [seq(toks[-1:], len(toks) - 1, [3, 1])])
    assert torch.allclose(full, inc, atol=1e-4)


def test_ragged_batch_matches_individual_runs(tiny_model):
    """Mixed prefill+decode in a single step (continuous batching) gives the same logits."""
    a, b = [256, 1, 2, 3, 4, 5], [256, 9, 8, 7]
    ref_a = logits_for(make_runner(tiny_model), [seq(a, 0, [0, 1])])
    ref_b = logits_for(make_runner(tiny_model), [seq(b, 0, [2])])

    r = make_runner(tiny_model)
    logits_for(r, [seq(a[:-1], 0, [0, 1])])                                   # a already running
    out = logits_for(r, [seq(a[-1:], len(a) - 1, [0, 1]), seq(b, 0, [2])])    # a decodes while b prefills
    assert torch.allclose(out[0], ref_a[0], atol=1e-4)
    assert torch.allclose(out[1], ref_b[0], atol=1e-4)


def test_allocator_no_fragmentation():
    async def run():
        al = BlockAllocator(num_blocks=8, block_size=4)
        a = await al.allocate(1, 12)   # 3 blocks
        b = await al.allocate(2, 8)    # 2 blocks
        c = await al.allocate(3, 12)   # 3 blocks -> full
        assert al.free_blocks == 0
        with pytest.raises(OutOfBlocks):
            await al.allocate(4, 4)
        await al.free(2)               # holes in the middle...
        d = await al.allocate(5, 8)    # ...are reused immediately
        assert sorted(d) == sorted(b)
        await al.free(1); await al.free(3); await al.free(5)
        assert al.free_blocks == 8 and not (set(a) & set(c))

    asyncio.run(run())
