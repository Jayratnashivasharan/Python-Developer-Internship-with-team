"""Orchestrator. Three overlapping stages connected by bounded queues:

   [prep threads: mmap I/O + FM seeding + fill pinned slot]
        -> bounded queue ->
   [main thread: H2D copy, CUDA kernel, D2H copy (async stream per slot)]
        -> thread pool ->
   [post threads: traceback + variant events]  -> main thread merges results

While the GPU works on batch k, CPU threads are already reading/seeding batch
k+1..k+prefetch and tracing back batch k-1 - I/O and compute overlap.
"""
from __future__ import annotations
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..align.backends import get_backend
from ..align.pool import SlotPool
from ..config import PipelineConfig
from ..index.fm import FMIndex
from ..seqio.fastq import iter_fastq
from ..variants.caller import VariantAccumulator
from ..variants.report import write_report, evaluate_against_truth
from ..variants.vcf import write_vcf
from .postprocess import postprocess
from .prepare import prepare_batch
from .stats import Stats

_DONE = object()


def run_alignment(index_dir: str, fastq: str, out_dir: str, backend: str = "auto",
                  cfg: Optional[PipelineConfig] = None, stats: Optional[Stats] = None,
                  truth_vcf: Optional[str] = None, write_alignments: bool = True,
                  call_params: Optional[dict] = None) -> Stats:
    cfg = cfg or PipelineConfig()
    os.makedirs(out_dir, exist_ok=True)
    index = FMIndex(index_dir)
    be = get_backend(backend, cfg)
    stats = stats or Stats()
    stats.backend = be.name
    stats.device = be.device_name() if hasattr(be, "device_name") else ""
    stats.t0 = __import__("time").perf_counter()

    n_slots = cfg.prefetch + cfg.prep_workers + cfg.post_workers + 2
    pool = SlotPool(be, n_slots, cfg.batch_size, cfg.window)
    acc = VariantAccumulator(index.total_length, out_dir)
    prep_pool = ThreadPoolExecutor(cfg.prep_workers, thread_name_prefix="prep")
    post_pool = ThreadPoolExecutor(cfg.post_workers, thread_name_prefix="post")
    fut_q: "queue.Queue" = queue.Queue(maxsize=cfg.prefetch)
    error = []

    def producer():
        try:
            for batch in iter_fastq(fastq, cfg.batch_size):
                fut_q.put(prep_pool.submit(prepare_batch, batch, index, pool, cfg, stats))
        except Exception as e:          # surface reader errors in main thread
            error.append(e)
        finally:
            fut_q.put(_DONE)

    threading.Thread(target=producer, daemon=True, name="reader").start()

    aln_fh = open(os.path.join(out_dir, "alignments.tsv"), "w") if write_alignments else None
    if aln_fh:
        aln_fh.write("#read\tcontig\tpos\tstrand\tscore\tcigar\n")
    pending = deque()

    def merge(fut):
        r = fut.result()
        if aln_fh:
            aln_fh.writelines(r.lines)
        for rs, re_, ev in r.alignments:
            acc.add_alignment(rs, re_, ev)
        stats.add(reads_aligned=r.aligned, reads_lowscore=r.lowscore,
                  gpu_cpu_mismatch=r.mismatch, variant_events=r.n_events)

    def post_and_release(slot):
        try:
            return postprocess(slot, index, cfg)
        finally:
            pool.release(slot)

    try:
        while True:
            stats.queue_depth = fut_q.qsize()
            item = fut_q.get()
            if item is _DONE:
                break
            slot = item.result()
            n = slot.meta[0].n
            if n:
                be.run(slot, n)
                stats.set(kernel_seconds=be.kernel_seconds)
            stats.add(batches=1)
            pending.append(post_pool.submit(post_and_release, slot))
            while pending and pending[0].done():
                merge(pending.popleft())
        while pending:
            merge(pending.popleft())
        if error:
            raise error[0]
    finally:
        prep_pool.shutdown(wait=True)
        post_pool.shutdown(wait=True)
        if aln_fh:
            aln_fh.close()

    stats.finish()
    variants = acc.call(**(call_params or {}))
    vcf = os.path.join(out_dir, "variants.vcf")
    write_vcf(vcf, variants, index)
    truth_eval = evaluate_against_truth(vcf, truth_vcf) if truth_vcf else None
    write_report(os.path.join(out_dir, "report.md"), os.path.join(out_dir, "report.json"),
                 stats.as_dict(), variants, truth_eval)
    stats.truth_eval = truth_eval
    stats.n_variants = len(variants)
    return stats
