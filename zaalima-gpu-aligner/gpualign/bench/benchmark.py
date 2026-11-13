"""Benchmarks:
 1. kernel micro-benchmark: alignments/s for python vs cpu(numba) vs cuda, with
    an accuracy check (all back-ends must give identical results);
 2. end-to-end reads/s of this pipeline vs external tools (bwa, bowtie2) when
    they are installed. External tools are never faked: if missing, reported
    as 'not installed'."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import time
from typing import Dict

import numpy as np

from ..align.backends import available_backends, get_backend
from ..config import MAX_QUERY, PipelineConfig
from ..simulate import write_dataset


def _make_pairs(n, read_len, pad, rng):
    reads = rng.integers(0, 4, size=(n, read_len), dtype=np.uint8)
    W = read_len + 2 * pad
    refs = rng.integers(0, 4, size=(n, W), dtype=np.uint8)
    refs[:, pad:pad + read_len] = reads                       # embed true locus
    mut = rng.random((n, read_len)) < 0.02                    # 2 % substitutions
    refs[:, pad:pad + read_len][mut] = (reads[mut] + 1) % 4
    return reads, refs


def bench_kernels(n_pairs=4096, read_len=100, seed=0, python_pairs=100) -> Dict:
    cfg = PipelineConfig()
    rng = np.random.default_rng(seed)
    reads, refs = _make_pairs(n_pairs, read_len, cfg.pad, rng)
    W = refs.shape[1]
    results, ref_out = {}, None
    for name in available_backends():
        be = get_backend(name, cfg)
        n = min(n_pairs, python_pairs) if name == "python" else n_pairs
        B = min(n_pairs, cfg.batch_size)
        slot = be.alloc_slot(0, B, W)
        done, out, t_total = 0, [], 0.0
        # warm-up (JIT compile) on a tiny batch
        slot.reads[:8, :read_len] = reads[:8]; slot.rlen[:8] = read_len
        slot.refs[:8, :W] = refs[:8]; slot.wlen[:8] = W
        be.run(slot, 8); be.kernel_seconds = 0.0
        while done < n:
            k = min(B, n - done)
            slot.reads[:k, :read_len] = reads[done:done + k]; slot.rlen[:k] = read_len
            slot.refs[:k, :W] = refs[done:done + k]; slot.wlen[:k] = W
            t0 = time.perf_counter(); be.run(slot, k); t_total += time.perf_counter() - t0
            out.append(slot.out[:k].copy()); done += k
        out = np.concatenate(out)
        if name == "cpu":
            ref_out = out
        results[name] = {"pairs": n, "seconds": round(t_total, 4),
                         "alignments_per_sec": round(n / t_total, 1),
                         "cells_per_sec": round(n * read_len * W / t_total, 0), "_out": out}
    for name, r in results.items():
        o = r.pop("_out")
        if ref_out is not None:
            r["identical_to_cpu"] = bool(np.array_equal(o, ref_out[:len(o)]))
    return results


def _timed(cmd, **kw):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=kw.get("stdout", subprocess.DEVNULL),
                   stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0


def bench_end_to_end(work_dir: str, genome_size=500_000, coverage=20, threads=None, backend="auto") -> Dict:
    from ..index import build_index
    from ..pipeline import run_alignment
    threads = threads or os.cpu_count() or 1
    d = write_dataset(work_dir, genome_size=genome_size, coverage=coverage)
    n = d["n_reads"]
    res: Dict[str, dict] = {}
    idx = os.path.join(work_dir, "idx")
    t0 = time.perf_counter(); build_index(d["fasta"], idx, verbose=False)
    res["gpualign_index_seconds"] = round(time.perf_counter() - t0, 2)
    run_alignment(idx, d["fastq"], os.path.join(work_dir, "warm"), backend=backend)  # JIT warm-up
    st = run_alignment(idx, d["fastq"], os.path.join(work_dir, "out"), backend=backend,
                       truth_vcf=d["truth"])
    res["gpualign"] = {"backend": st.backend, "device": st.device, "reads": n,
                       "seconds": round(st.elapsed, 2), "reads_per_sec": round(n / st.elapsed, 1),
                       "variant_accuracy": st.truth_eval["all"] if st.truth_eval else None}
    if shutil.which("bwa"):
        _timed(["bwa", "index", d["fasta"]])
        sam = os.path.join(work_dir, "bwa.sam")
        with open(sam, "w") as fh:
            t = _timed(["bwa", "mem", "-t", str(threads), d["fasta"], d["fastq"]], stdout=fh)
        res["bwa_mem"] = {"reads": n, "seconds": round(t, 2), "reads_per_sec": round(n / t, 1), "threads": threads}
    else:
        res["bwa_mem"] = "not installed"
    if shutil.which("bowtie2") and shutil.which("bowtie2-build"):
        pre = os.path.join(work_dir, "bt2")
        _timed(["bowtie2-build", "-q", d["fasta"], pre])
        t = _timed(["bowtie2", "-x", pre, "-U", d["fastq"], "-S", os.path.join(work_dir, "bt2.sam"),
                    "-p", str(threads), "--quiet"])
        res["bowtie2"] = {"reads": n, "seconds": round(t, 2), "reads_per_sec": round(n / t, 1), "threads": threads}
    else:
        res["bowtie2"] = "not installed"
    return res


def run_benchmarks(work_dir: str, genome_size=500_000, coverage=20, backend="auto") -> str:
    os.makedirs(work_dir, exist_ok=True)
    kern = bench_kernels()
    e2e = bench_end_to_end(work_dir, genome_size, coverage, backend=backend)
    with open(os.path.join(work_dir, "benchmark.json"), "w") as fh:
        json.dump({"kernels": kern, "end_to_end": e2e, "max_query": MAX_QUERY}, fh, indent=2)
    md = ["# Benchmark report", "", f"Host CPUs: {os.cpu_count()}  |  backends: {', '.join(available_backends())}", "",
          "## 1. Smith-Waterman kernel throughput (100 bp read x 140 bp window)", "",
          "| backend | pairs | seconds | alignments/s | cells/s | identical to cpu |", "|---|---|---|---|---|---|"]
    for k, v in kern.items():
        md.append(f"| {k} | {v['pairs']} | {v['seconds']} | {v['alignments_per_sec']:,} | "
                  f"{v['cells_per_sec']:,.0f} | {v.get('identical_to_cpu')} |")
    md += ["", f"## 2. End-to-end (genome {genome_size:,} bp, {e2e['gpualign']['reads']:,} reads)", "",
           "| tool | seconds | reads/s | note |", "|---|---|---|---|"]
    g = e2e["gpualign"]
    md.append(f"| gpualign ({g['backend']}) | {g['seconds']} | {g['reads_per_sec']:,} | align + traceback + VCF |")
    for name in ("bwa_mem", "bowtie2"):
        v = e2e[name]
        md.append(f"| {name} | - | - | not installed |" if isinstance(v, str)
                  else f"| {name} | {v['seconds']} | {v['reads_per_sec']:,} | {v['threads']} threads, index excluded |")
    md += ["", "> Caveat: small synthetic genome; external tools are highly optimised and mostly "
           "process-startup bound at this size. Re-run with larger `--genome-size` on a GPU host "
           "for meaningful comparison."]
    path = os.path.join(work_dir, "benchmark_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    return path
