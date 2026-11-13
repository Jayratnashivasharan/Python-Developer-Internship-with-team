"""Command line interface:  python -m gpualign <command> --help"""
from __future__ import annotations
import argparse
import os
import sys

from .config import MAX_QUERY, PipelineConfig


def _cmd_info(a):
    from .align.backends import available_backends
    from .index.fm import SEED_ENGINE
    print(f"gpualign 1.0.0 | MAX_QUERY={MAX_QUERY} | CPUs={os.cpu_count()}")
    print("backends  :", ", ".join(available_backends()))
    print("FM seeding:", {"cython": "Cython (compiled)", "numba": "Numba JIT (run `make build` for Cython)",
                          "python": "pure Python fallback (slow)"}[SEED_ENGINE])
    try:
        from numba import cuda
        if cuda.is_available():
            for d in cuda.list_devices():
                print("gpu       :", d.name.decode() if isinstance(d.name, bytes) else d.name)
        else:
            print("gpu       : none detected")
    except Exception as e:
        print("gpu       : unavailable:", e)


def _cmd_simulate(a):
    from .simulate import write_dataset
    d = write_dataset(a.out, a.genome_size, a.contigs, a.coverage, a.read_len, a.seed)
    print(d)


def _cmd_index(a):
    from .index import build_index
    build_index(a.fasta, a.out)


def _cmd_align(a):
    from .pipeline import run_alignment, Stats
    from .ui import Dashboard
    cfg = PipelineConfig(batch_size=a.batch_size, prep_workers=a.threads, post_workers=a.threads,
                         use_mapped_memory=a.mapped, seed_k=a.seed_k, min_score_frac=a.min_score_frac)
    stats = Stats()
    with Dashboard(stats, enabled=not a.no_dashboard and sys.stdout.isatty()):
        stats = run_alignment(a.index, a.reads, a.out, a.backend, cfg, stats, a.truth,
                              write_alignments=not a.no_tsv,
                              call_params={"min_alt": a.min_alt, "min_af": a.min_af, "min_depth": a.min_depth})
    print(f"done: {stats.reads_total:,} reads in {stats.elapsed:.1f}s "
          f"({stats.reads_per_sec:,.0f} reads/s, backend={stats.backend}); "
          f"{stats.n_variants} variants -> {os.path.join(a.out, 'variants.vcf')}")
    if stats.truth_eval:
        print("truth eval:", stats.truth_eval["all"])


def _cmd_benchmark(a):
    from .bench.benchmark import run_benchmarks
    print("report:", run_benchmarks(a.out, a.genome_size, a.coverage, a.backend))


def _cmd_web(a):
    from .web import serve
    serve(a.host, a.port, a.data_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gpualign", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show backends / GPU / build status").set_defaults(f=_cmd_info)

    p = sub.add_parser("simulate", help="generate synthetic reference, reads and truth VCF")
    p.add_argument("--out", required=True); p.add_argument("--genome-size", type=int, default=200_000)
    p.add_argument("--contigs", type=int, default=2); p.add_argument("--coverage", type=float, default=30)
    p.add_argument("--read-len", type=int, default=100); p.add_argument("--seed", type=int, default=42)
    p.set_defaults(f=_cmd_simulate)

    p = sub.add_parser("index", help="build FM-index (BWT) from a FASTA")
    p.add_argument("fasta"); p.add_argument("--out", required=True); p.set_defaults(f=_cmd_index)

    p = sub.add_parser("align", help="align FASTQ reads, call variants, write VCF + report")
    p.add_argument("--index", required=True); p.add_argument("--reads", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu", "python"])
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=2, help="prep and post worker threads each")
    p.add_argument("--mapped", action="store_true", help="CUDA mapped (zero-copy) host memory")
    p.add_argument("--seed-k", type=int, default=16)
    p.add_argument("--min-score-frac", type=float, default=0.5)
    p.add_argument("--min-alt", type=int, default=3); p.add_argument("--min-af", type=float, default=0.25)
    p.add_argument("--min-depth", type=int, default=4)
    p.add_argument("--truth", help="truth VCF (simulated data) for precision/recall")
    p.add_argument("--no-dashboard", action="store_true"); p.add_argument("--no-tsv", action="store_true")
    p.set_defaults(f=_cmd_align)

    p = sub.add_parser("benchmark", help="kernel + end-to-end benchmark, writes benchmark_report.md")
    p.add_argument("--out", default="bench_out"); p.add_argument("--genome-size", type=int, default=500_000)
    p.add_argument("--coverage", type=float, default=20)
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu"])
    p.set_defaults(f=_cmd_benchmark)

    p = sub.add_parser("web", help="start the web application")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", default=None, help="where jobs/results are stored (default ./webdata)")
    p.set_defaults(f=_cmd_web)

    a = ap.parse_args(argv)
    a.f(a)
    return 0
