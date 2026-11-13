"""Continuous batching behaviour, tested against fake in-process workers."""
import asyncio
from types import SimpleNamespace

from grid.config import Settings
from grid.master.engine import Engine, SamplingParams


class FakeClient:
    address = "fake:0"

    def __init__(self):
        self.batches = []

    async def wait_ready(self, t):
        return SimpleNamespace(rank=0, backend="fake", device="cpu")

    async def step(self, req, timeout):
        self.batches.append([(s.seq_id, len(s.token_ids), s.start_pos) for s in req.seqs])
        await asyncio.sleep(0.005)
        return SimpleNamespace(next_tokens=[7] * len(req.seqs), compute_ms=1.0, comm_ms=0.5)

    async def close(self):
        pass


def make_engine(**kw):
    e = Engine(Settings(workers=["x"], num_blocks=64, block_size=4, **kw), eos_id=999)
    e.clients = [FakeClient()]
    return e


def test_new_request_joins_running_batch():
    async def run():
        e = make_engine(max_batch_seqs=4)
        await e.start()
        long = e.submit([1, 2, 3], SamplingParams(max_new_tokens=30))
        await asyncio.sleep(0.05)                      # `long` is mid-generation
        short = e.submit([4, 5], SamplingParams(max_new_tokens=3))
        got = [t async for t in e.stream(short)]
        assert got == [7, 7, 7] and long.finish_reason is None   # short finished while long still runs
        _ = [t async for t in e.stream(long)]
        mixed = [b for b in e.clients[0].batches if len(b) == 2]
        assert mixed, "requests were never batched together"
        assert e.alloc.free_blocks == 64               # every KV block returned
        await e.stop()

    asyncio.run(run())


def test_abort_frees_blocks():
    async def run():
        e = make_engine(max_batch_seqs=2)
        await e.start()
        r = e.submit([1, 2, 3], SamplingParams(max_new_tokens=200))
        await asyncio.sleep(0.03)
        e.abort(r)
        await asyncio.sleep(0.05)
        assert r.finish_reason == "aborted" and e.alloc.free_blocks == 64
        await e.stop()

    asyncio.run(run())
