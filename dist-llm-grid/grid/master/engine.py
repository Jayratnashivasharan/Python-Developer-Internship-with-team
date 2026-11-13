"""Master engine: continuous (iteration-level) batching over a tensor-parallel worker group.

One *iteration* = one forward pass that produces exactly one new token for every active
sequence. Before every iteration the scheduler:

  1. drops finished / cancelled sequences and returns their KV blocks,
  2. admits waiting requests (FCFS) while batch slots, the per-step prefill-token budget
     and free KV blocks allow - their prompts are prefilled *in the same step* in which
     the already-running sequences decode,
  3. sends the ragged batch to all workers and distributes the sampled tokens.

New requests therefore never wait for older sequences to finish (no static batches).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from grid.config import Settings
from grid.kvcache.paged import BlockAllocator
from grid.master.metrics import Metrics
from grid.master.worker_client import WorkerClient
from grid.proto import inference_pb2 as pb

log = logging.getLogger("grid.engine")


class QueueFull(Exception):
    pass


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    stop_on_eos: bool = True


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    params: SamplingParams
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)   # ("token", id) | ("done", reason) | ("error", msg)
    generated: list[int] = field(default_factory=list)
    arrival: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finish_reason: str | None = None
    aborted: bool = False
    blocks: list[int] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float | None:
        return None if self.first_token_at is None else (self.first_token_at - self.arrival) * 1000


class Engine:
    def __init__(self, settings: Settings, eos_id: int, vocab_size: int | None = None, max_len: int | None = None):
        self.s, self.eos_id, self.vocab_size, self.max_len = settings, eos_id, vocab_size, max_len
        self.clients = [WorkerClient(a, r) for r, a in enumerate(settings.workers)]
        self.alloc = BlockAllocator(settings.num_blocks, settings.block_size)
        self.metrics = Metrics()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._ids = itertools.count(1)
        self._step_ids = itertools.count(1)
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.ready = False
        self.worker_info: list[dict] = []

    # ------------------------------------------------------------------ lifecycle
    async def start(self, wait_s: float = 120) -> None:
        infos = await asyncio.gather(*(c.wait_ready(wait_s) for c in self.clients))
        log.info("cluster ready: %s", [(i.rank, i.backend, i.device) for i in infos])
        self.worker_info = [{"rank": i.rank, "address": c.address, "backend": i.backend, "device": i.device}
                            for i, c in zip(infos, self.clients)]
        self.ready = True
        self._task = asyncio.create_task(self._loop(), name="engine-loop")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for r in list(self.running) + list(self.waiting):
            await self._finish(r, "shutdown")
        await asyncio.gather(*(c.close() for c in self.clients))

    # --------------------------------------------------------------------- client API
    def submit(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        if not prompt_ids:
            raise ValueError("empty prompt")
        total = len(prompt_ids) + params.max_new_tokens
        if self.max_len and total > self.max_len:
            raise ValueError(f"prompt + max_new_tokens = {total} exceeds model context {self.max_len}")
        if self.alloc.blocks_needed(total) > self.alloc.num_blocks:
            raise ValueError(f"request needs more KV blocks than the whole cache ({self.alloc.num_blocks})")
        if len(self.waiting) >= self.s.max_queue:
            raise QueueFull("server busy, try again later")
        req = Request(next(self._ids), list(prompt_ids), params)
        self.metrics.requests_total += 1
        self.waiting.append(req)
        self._wake.set()
        return req

    def abort(self, req: Request) -> None:
        if req.finish_reason is None:
            req.aborted = True
            self._wake.set()

    async def stream(self, req: Request):
        """Async generator of generated token ids; always releases the request on exit."""
        try:
            while True:
                kind, val = await req.queue.get()
                if kind == "token":
                    yield val
                elif kind == "done":
                    return
                else:
                    raise RuntimeError(val)
        finally:
            self.abort(req)

    def stats(self) -> dict:
        return self.metrics.snapshot(len(self.running), len(self.waiting), self.alloc.usage)

    # ------------------------------------------------------------------- scheduling
    async def _finish(self, req: Request, reason: str) -> None:
        if req.finish_reason is not None:
            return
        req.finish_reason = reason
        await self.alloc.free(req.id)
        if reason == "error":
            self.metrics.requests_failed += 1
        elif reason in ("stop", "length"):
            self.metrics.requests_finished += 1
        if reason in ("stop", "length", "aborted", "shutdown"):
            req.queue.put_nowait(("done", reason))

    async def _schedule(self) -> tuple[list[Request], list[Request]]:
        # 1. retire cancelled sequences
        for r in [r for r in self.running if r.aborted]:
            self.running.remove(r)
            await self._finish(r, "aborted")
        # 2. admit new requests (FCFS, bounded by slots / prefill budget / KV blocks)
        new: list[Request] = []
        budget = self.s.max_prefill_tokens
        while self.waiting and len(self.running) + len(new) < self.s.max_batch_seqs:
            r = self.waiting[0]
            if r.aborted:
                self.waiting.popleft()
                await self._finish(r, "aborted")
                continue
            if new and len(r.prompt_ids) > budget:
                break                                                    # keep step latency bounded
            if not self.alloc.can_allocate(len(r.prompt_ids) + r.params.max_new_tokens):
                break                                                    # wait for blocks to free up
            self.waiting.popleft()
            r.blocks = await self.alloc.allocate(r.id, len(r.prompt_ids) + r.params.max_new_tokens)
            budget -= len(r.prompt_ids)
            new.append(r)
        return list(self.running), new

    def _build_request(self, decode: list[Request], new: list[Request]) -> pb.StepRequest:
        seqs = []
        for r in decode:
            seqs.append(pb.SeqStep(seq_id=r.id, token_ids=[r.generated[-1]],
                                   start_pos=len(r.prompt_ids) + len(r.generated) - 1, block_table=r.blocks,
                                   temperature=r.params.temperature, top_k=r.params.top_k))
        for r in new:
            seqs.append(pb.SeqStep(seq_id=r.id, token_ids=r.prompt_ids, start_pos=0, block_table=r.blocks,
                                   temperature=r.params.temperature, top_k=r.params.top_k))
        return pb.StepRequest(step_id=next(self._step_ids), seqs=seqs)

    async def _loop(self) -> None:
        while True:
            if not self.waiting and not self.running:
                self._wake.clear()
                await self._wake.wait()
            decode, new = await self._schedule()
            batch = decode + new
            if not batch:
                await asyncio.sleep(0.005)       # head request is waiting for KV blocks
                continue
            req = self._build_request(decode, new)
            t0 = time.perf_counter()
            try:
                replies = await asyncio.gather(*(c.step(req, self.s.step_timeout_s) for c in self.clients))
            except Exception as e:                # worker crash / timeout: fail the batch, keep serving
                log.error("step %d failed: %r", req.step_id, e)
                for r in batch:
                    r.queue.put_nowait(("error", f"worker failure: {e!r}"))
                    await self._finish(r, "error")
                self.running = [r for r in self.running if r.finish_reason is None]
                continue
            self.metrics.record_step((time.perf_counter() - t0) * 1000, replies, len(batch))
            tokens = replies[0].next_tokens
            now = time.perf_counter()
            self.running.extend(new)
            for r, tok in zip(batch, tokens):
                if r.aborted:
                    continue
                if r.first_token_at is None:
                    r.first_token_at = now
                    self.metrics.ttft_ms.append(r.ttft_ms)
                if tok == self.eos_id and r.params.stop_on_eos:
                    self.running.remove(r)
                    await self._finish(r, "stop")
                    continue
                r.generated.append(tok)
                self.metrics.tokens_generated += 1
                r.queue.put_nowait(("token", tok))
                if len(r.generated) >= r.params.max_new_tokens:
                    self.running.remove(r)
                    await self._finish(r, "length")
            await asyncio.sleep(0)                # let the gateway flush tokens between iterations
