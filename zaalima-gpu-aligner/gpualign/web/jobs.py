"""Background job manager for the web UI.

* Jobs run one at a time on a single worker thread (Numba's parallel runtime
  must not be entered from two threads at once, and one job already uses all
  cores / the GPU).
* Every job lives in  <data_dir>/jobs/<id>/  with a job.json that survives
  server restarts. Jobs interrupted by a restart are marked failed.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

ACTIVE = ("queued", "running")
RESULT_FILES = [
    ("variants.vcf", "Variants (VCF)"),
    ("alignments.tsv", "Alignments (TSV)"),
    ("report.md", "Run report (Markdown)"),
    ("report.json", "Run report (JSON)"),
]


class Job:
    def __init__(self, job_id: str, kind: str, name: str, params: dict, directory: str):
        self.id, self.kind, self.name = job_id, kind, name
        self.params, self.dir = params, directory
        self.status = "queued"
        self.stage = "queued"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[dict] = None
        self.history: List[list] = []        # [elapsed_s, reads_total, percent_of_file]
        self.stats = None                    # live pipeline Stats (not persisted)

    # ------------------------------------------------------------ views
    def _progress(self) -> float:
        if self.status == "done":
            return 100.0
        if self.status == "queued":
            return 0.0
        st = self.stage
        if self.kind == "benchmark":
            return 50.0 if self.status == "running" else 0.0
        if st == "simulating":
            return 3.0
        if st == "indexing":
            return 8.0
        if st in ("aligning", "calling"):
            s = self.stats
            if s is not None and st == "aligning":
                return 12.0 + 80.0 * min(1.0, s.bytes_done / max(s.bytes_total, 1))
            return 95.0
        return 0.0

    def _live(self) -> Optional[dict]:
        s = self.stats
        if s is None or self.status != "running":
            return None
        d = s.as_dict()
        d.update(queue_depth=s.queue_depth, variant_events=s.variant_events,
                 percent=round(100.0 * s.bytes_done / max(s.bytes_total, 1), 1))
        return d

    def public(self, with_history: bool = True) -> dict:
        stage = self.stage
        if stage == "aligning" and self.stats is not None and self.stats.t_end:
            stage = "calling"
        d = {"id": self.id, "kind": self.kind, "name": self.name, "status": self.status,
             "stage": stage, "created": self.created, "started": self.started,
             "finished": self.finished, "error": self.error, "params": self.params,
             "result": self.result, "progress": round(self._progress(), 1), "live": self._live()}
        if self.status == "running" and self.started:
            d["elapsed"] = round(time.time() - self.started, 1)
        elif self.started and self.finished:
            d["elapsed"] = round(self.finished - self.started, 1)
        else:
            d["elapsed"] = 0
        if with_history:
            d["history"] = self.history
        return d

    def to_json(self) -> dict:
        d = self.public()
        h = self.history
        if len(h) > 300:                      # downsample for storage
            step = len(h) / 300.0
            h = [h[int(i * step)] for i in range(300)]
        d.update(history=h, progress=100.0 if self.status == "done" else d["progress"],
                 live=None, stage=self.stage)
        return d


class JobManager:
    def __init__(self, data_dir: str):
        self.root = os.path.join(data_dir, "jobs")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()
        self.jobs: Dict[str, Job] = {}
        self._exec = ThreadPoolExecutor(1, thread_name_prefix="job")
        self._load()

    # ------------------------------------------------------- persistence
    def _load(self):
        for jid in os.listdir(self.root):
            path = os.path.join(self.root, jid, "job.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as fh:
                    d = json.load(fh)
                j = Job(d["id"], d["kind"], d.get("name", d["id"]), d.get("params", {}),
                        os.path.join(self.root, jid))
                j.status, j.stage = d["status"], d.get("stage", "done")
                j.created, j.started, j.finished = d["created"], d.get("started"), d.get("finished")
                j.error, j.result, j.history = d.get("error"), d.get("result"), d.get("history", [])
                if j.status in ACTIVE:
                    j.status, j.error = "failed", "Interrupted by a server restart."
                    j.finished = j.finished or time.time()
                self.jobs[j.id] = j
            except Exception:
                continue          # ignore corrupt job folders

    def _save(self, job: Job):
        path = os.path.join(job.dir, "job.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(job.to_json(), fh)
            os.replace(tmp, path)
        except OSError:
            pass

    # --------------------------------------------------------- public API
    def new_dir(self) -> tuple:
        jid = uuid.uuid4().hex[:12]
        d = os.path.join(self.root, jid)
        os.makedirs(d)
        return jid, d

    def submit(self, jid: str, directory: str, kind: str, name: str, params: dict) -> Job:
        job = Job(jid, kind, name, params, directory)
        with self._lock:
            self.jobs[jid] = job
            self._save(job)
        self._exec.submit(self._execute, job)
        return job

    def get(self, jid: str) -> Optional[Job]:
        return self.jobs.get(jid)

    def list(self) -> List[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def delete(self, jid: str) -> bool:
        with self._lock:
            job = self.jobs.get(jid)
            if job is None or job.status in ACTIVE:
                return False
            del self.jobs[jid]
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def counts(self) -> dict:
        c = {"total": 0, "running": 0, "queued": 0, "done": 0, "failed": 0}
        for j in self.jobs.values():
            c["total"] += 1
            c[j.status] = c.get(j.status, 0) + 1
        return c

    def warmup(self):
        """JIT-compile Numba kernels in the worker thread so the first real job is fast."""
        def _w():
            try:
                import numpy as np
                from ..align.cpu_kernels import sw_score_batch, sw_traceback, extract_events
                q = np.array([0, 1, 2, 3, 0, 1, 2, 3, 1, 1], np.uint8)
                r = np.concatenate((q, q))
                reads = np.zeros((2, 256), np.uint8); reads[:, :10] = q
                refs = np.zeros((2, 296), np.uint8); refs[:, :20] = r
                rl = np.full(2, 10, np.int32); wl = np.full(2, 20, np.int32)
                out = np.zeros((2, 3), np.int32)
                sw_score_batch(reads, rl, refs, wl, 2, 2, 3, 5, 2, out)
                _, qs, qe, rs, re_, ops = sw_traceback(q, r, 2, 3, 5, 2)
                extract_events(ops, q, qs, r, rs)
                from ..index import fm
                if fm.SEED_ENGINE == "numba":      # compile with the read-only types memmaps have
                    from ..index._fm_numba import seed_diagonals
                    bwt = np.array([1, 2, 3, 4, 0], np.uint8)
                    occ = np.zeros((1, 6), np.int64)
                    Cc = np.zeros(6, np.int64)
                    sa = np.arange(5, dtype=np.int32)
                    for a in (bwt, occ, Cc, sa):
                        a.flags.writeable = False
                    seed_diagonals(bwt, occ, Cc, sa, np.zeros(20, np.uint8), 16, 8, 50)
            except Exception:
                pass
        self._exec.submit(_w)

    # ------------------------------------------------------------ worker
    def _execute(self, job: Job):
        with self._lock:
            job.status, job.started = "running", time.time()
            self._save(job)
        try:
            if job.kind == "align":
                job.result = self._run_align(job)
            elif job.kind == "benchmark":
                job.result = self._run_benchmark(job)
            else:
                raise ValueError(f"unknown job kind {job.kind!r}")
            job.status, job.stage = "done", "done"
        except Exception as e:                           # noqa: BLE001 - report to the UI
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            try:
                with open(os.path.join(job.dir, "error.log"), "w") as fh:
                    fh.write(traceback.format_exc())
            except OSError:
                pass
        finally:
            job.finished = time.time()
            job.stats = None
            self._save(job)

    def _sampler(self, job: Job, stop: threading.Event):
        while not stop.wait(0.5):
            s = job.stats
            if s is not None:
                job.history.append([round(s.elapsed, 2), s.reads_total,
                                    round(100.0 * s.bytes_done / max(s.bytes_total, 1), 1)])
                if len(job.history) > 4000:
                    job.history = job.history[::2]

    def _run_align(self, job: Job) -> dict:
        from ..config import PipelineConfig
        from ..index import build_index
        from ..pipeline import Stats, run_alignment
        from ..simulate import write_dataset

        p, d = job.params, job.dir
        data_dir = os.path.join(d, "data")
        if p["source"] == "demo":
            job.stage = "simulating"
            info = write_dataset(data_dir, genome_size=p["genome_size"], n_contigs=p["contigs"],
                                 coverage=p["coverage"], read_len=p["read_len"], seed=p["seed"])
            fasta, fastq, truth = info["fasta"], info["fastq"], info["truth"]
        else:
            fasta, fastq = os.path.join(data_dir, p["reference_file"]), os.path.join(data_dir, p["reads_file"])
            truth = os.path.join(data_dir, p["truth_file"]) if p.get("truth_file") else None

        job.stage = "indexing"
        idx_dir = os.path.join(d, "index")
        meta = build_index(fasta, idx_dir, verbose=False)
        if meta["total_length"] == 0:
            raise ValueError("The reference FASTA contains no sequence.")

        job.stage = "aligning"
        cfg = PipelineConfig(batch_size=p["batch_size"], prep_workers=p["threads"],
                             post_workers=p["threads"], seed_k=p["seed_k"],
                             min_score_frac=p["min_score_frac"], use_mapped_memory=p["mapped"])
        stats = Stats()
        job.stats = stats
        stop = threading.Event()
        t = threading.Thread(target=self._sampler, args=(job, stop), daemon=True)
        t.start()
        try:
            out = os.path.join(d, "results")
            st = run_alignment(idx_dir, fastq, out, p["backend"], cfg, stats, truth,
                               write_alignments=True,
                               call_params={"min_alt": p["min_alt"], "min_af": p["min_af"],
                                            "min_depth": p["min_depth"]})
        finally:
            stop.set()
            t.join(timeout=2)
        if job.stats is not None:                      # final history point
            job.history.append([round(st.elapsed, 2), st.reads_total, 100.0])

        with open(os.path.join(out, "report.json")) as fh:
            report = json.load(fh)
        final = st.as_dict()
        shutil.rmtree(idx_dir, ignore_errors=True)       # index is large and re-buildable
        try:
            os.remove(os.path.join(out, "depth_diff.npy"))
        except OSError:
            pass
        return {"stats": final, "n_variants": st.n_variants, "variant_summary": report["variants"],
                "truth_eval": report.get("truth_evaluation"),
                "contigs": [{"name": c["name"], "length": c["length"]} for c in meta["contigs"]],
                "files": [{"name": n, "label": l} for n, l in RESULT_FILES
                          if os.path.isfile(os.path.join(out, n))]}

    def _run_benchmark(self, job: Job) -> dict:
        from ..bench.benchmark import run_benchmarks
        p = job.params
        job.stage = "benchmarking"
        run_benchmarks(job.dir, p["genome_size"], p["coverage"], p["backend"])
        with open(os.path.join(job.dir, "benchmark.json")) as fh:
            data = json.load(fh)
        for sub in ("data", "idx", "out", "warm"):        # keep only the reports
            shutil.rmtree(os.path.join(job.dir, sub), ignore_errors=True)
        for f in os.listdir(job.dir):
            if f.endswith((".sam", ".fa", ".fq", ".vcf", ".bt2")) or ".fa." in f:
                try:
                    os.remove(os.path.join(job.dir, f))
                except OSError:
                    pass
        return {"kernels": data["kernels"], "end_to_end": data["end_to_end"],
                "files": [{"name": "benchmark_report.md", "label": "Benchmark report (Markdown)"},
                          {"name": "benchmark.json", "label": "Benchmark data (JSON)"}]}
