# gpualign - High-Throughput GPU Genomic Sequence Aligner

A DNA read aligner that runs **Smith-Waterman (affine gap) on the GPU directly
from Python** (Numba CUDA), with a Cython **BWT/FM-index pre-filter**,
**memory-mapped streaming** of FASTA/FASTQ, a **multi-threaded pipeline** that
overlaps I/O with GPU work, a **Rich live dashboard**, **VCF variant reports**
and a **benchmark harness**.

> Built for the *Zaalima Development Pvt. Ltd.* internship project. Every
> accelerated path is tested to reproduce a pure-Python baseline exactly.

## Web application (start here)

```bash
./start.sh            # Linux / macOS   (Windows: double-click start.bat)
# or manually:
pip install -r requirements.txt
python run_web.py     # -> http://localhost:8000
```

`start.sh` creates a virtual environment, installs dependencies, tries the optional
Cython build (the app works without it) and launches the server. Use
`HOST=0.0.0.0 PORT=8080 ./start.sh` to expose it on your network.

**What you can do in the browser**

| Page | Function |
|---|---|
| Dashboard | system status (GPU/CPU backend, FM-index engine), recent jobs, **one-click quick demo** |
| New alignment | simulated demo data *or* upload your own reference FASTA + reads FASTQ(.gz) (+ optional truth VCF); advanced tuning options |
| Job page | live progress bar, throughput chart, pipeline counters, then results: accuracy vs truth, variant table (filter/search/paging), alignment preview, downloads (VCF, TSV, report) |
| Jobs | history of all runs (persisted across restarts), delete old runs |
| Benchmark | kernel (Python vs CPU vs GPU) and end-to-end benchmark, BWA/Bowtie2 if installed |

Jobs run one at a time in a background queue; data lives in `./webdata/`
(change with `--data-dir` or `GPUALIGN_DATA_DIR`). Limits via environment variables:
`GPUALIGN_MAX_UPLOAD_MB` (default 4096), `GPUALIGN_MAX_REF_MB` (default 200).
The UI is fully responsive (phone, tablet, desktop) and supports dark mode; it uses no
external CDN, so it works offline.

## Quick start (command line)

```bash
make install        # pip deps + compile Cython extension
make web            # start the web application
make test           # 24+ tests (kernel equivalence also runs on CUDA simulator)
make demo           # simulate -> index -> align -> VCF -> accuracy report
make info           # which backends / GPU were detected
```

Manual use:

```bash
python -m gpualign index   genome.fa --out idx/
python -m gpualign align   --index idx/ --reads reads.fq --out results/            # auto: cuda -> cpu
python -m gpualign align   --index idx/ --reads reads.fq --out results/ --backend cuda --batch-size 16384
python -m gpualign benchmark --out bench_out
```

Outputs in `results/`: `variants.vcf`, `alignments.tsv`, `report.md`, `report.json`.

## Backends

| `--backend` | Engine | Use |
|---|---|---|
| `cuda` | Numba CUDA kernel, pinned memory, async streams | production (needs NVIDIA GPU) |
| `cpu`  | Numba-JIT multi-core, same maths | automatic fallback / CI |
| `python` | pure Python | accuracy oracle, benchmarks |
| `auto` (default) | cuda if available, else cpu | |

No GPU? `NUMBA_ENABLE_CUDASIM=1` runs the *same CUDA kernel* on the CPU
simulator (slow, for testing only).

## Project layout

```
gpualign/
  seqio/      mmap FASTA/FASTQ chunked streaming + encoding
  index/      suffix array, FM-index, Cython seeding (_fmcore.pyx) + Python fallback
  align/      baseline.py (oracle) | cpu_kernels.py | cuda_kernels.py | backends.py | pool.py
  pipeline/   prepare (I/O+seed) -> GPU -> postprocess (traceback) ; engine.py ; stats.py
  variants/   pileup caller, VCF writer, report/truth evaluation
  ui/         Rich dashboard
  bench/      benchmark harness       simulate.py  synthetic data + truth VCF
  web/        Flask app: app.py (routes+API) jobs.py (queue) templates/ static/
  cli.py      `python -m gpualign ...`        run_web.py / start.sh / start.bat  launchers
tests/        io, index, kernel equivalence (baseline==numba==CUDA), pipeline e2e, web app
docs/         ARCHITECTURE.md  HARDWARE_OPTIMIZATION.md  PROJECT_PLAN.md
```

## How it works (30-second version)
1. Reads stream from disk via `mmap` in batches.
2. Each read is seeded against the FM-index (Cython); reads with no consistent
   locus are dropped; the rest get a ~140 bp reference window.
3. Windows + reads are written straight into **pinned memory pool slots**; the GPU
   aligns thousands of pairs in parallel (score + end coordinates).
4. CPU threads trace back only the surviving reads, **re-check the GPU score**,
   and extract SNP/indel events.
5. Events are piled up, called, and written as VCF + a Markdown/JSON report.

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Validation results (this repo, synthetic data, CPU backend)
* 200 kb genome, 30x (`make demo`): **386/386 truth variants recovered, 0 false positives**
  (SNP + indel), 0 GPU/CPU score disagreements.
* Kernel equivalence: pure Python == numba traceback == numba batch == CUDA kernel (simulator).
* Mapping position accuracy > 99 %; unrelated reads are filtered before the GPU.

## Honest notes
* Developed without a physical GPU: CUDA code is verified on Numba's simulator
  against the baseline. **Run `make test` and `make bench` on your GPU machine**
  to confirm and to get real throughput numbers (the benchmark never fabricates
  results; bwa/bowtie2 rows show "not installed" unless present).
* Index construction is in-RAM (practical to a few hundred Mbp); read streaming
  is unbounded. See [`docs/HARDWARE_OPTIMIZATION.md`](docs/HARDWARE_OPTIMIZATION.md)
  for all limits and tuning advice.
