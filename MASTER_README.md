# Master README — GPU & Distributed Computing Projects

This repository collection contains two independent high-performance computing projects:

1. **Distributed Memory Grid for Real-Time LLM Inference** — a distributed LLM inference engine built with PyTorch, gRPC, asyncio, tensor parallelism, paged KV caching, continuous batching, and WebSocket streaming.
2. **gpualign — High-Throughput GPU Genomic Sequence Aligner** — a DNA read aligner using Smith–Waterman alignment, Numba CUDA acceleration, BWT/FM-index pre-filtering, streaming sequence input, and VCF variant reporting.

The projects solve different problems and have separate dependencies, launch commands, tests, and directory structures. Follow the setup instructions for the project you want to run.

---

# 1. Distributed Memory Grid for Real-Time LLM Inference

## Overview

A from-scratch distributed inference engine in **PyTorch + gRPC + asyncio** that shards LLM weights across several modest nodes (tensor parallelism), manages GPU/CPU memory with a **paged KV cache**, and serves multiple users concurrently with **continuous batching**. Generated tokens are streamed token-by-token over **WebSockets**.

## Feature map

| Requirement / feature | Location |
|---|---|
| Master–worker topology, asyncio + gRPC | `grid/master/`, `grid/worker/`, `grid/proto/inference.proto` |
| safetensors and memory-mapped, shard-only loading | `grid/weights/loader.py` |
| Tensor parallelism (heads + FFN) with `torch.distributed` | `grid/model/parallel.py`, `grid/model/llama.py` |
| Async KV-cache manager (no fragmentation) | `grid/kvcache/paged.py` |
| Continuous batching (iteration-level) | `grid/master/engine.py` |
| NCCL (automatic) / Gloo backend | `grid/worker/server.py::init_distributed` |
| FastAPI + WebSocket streaming gateway | `grid/gateway/app.py` |
| Docker / Swarm / Kubernetes with node affinity | `deploy/` |
| Load testing (TTFT, inter-node latency) | `scripts/load_test.py`, `/metrics` |
| Web application (chat, live dashboard, load test) | `grid/gateway/static/index.html` |

## Quick start (CPU only)

No GPU is required for the demo.

```bash
pip install -r requirements.txt
python run.py
```

The launcher starts two workers and the gateway, then opens `http://localhost:8000`. The first start creates a small demo model automatically. Press `Ctrl+C` to stop the processes.

### Options

```bash
python run.py --tp 4 --port 9000 --no-browser
```

To use a real model:

```bash
python run.py --model-dir /path/to/Llama --tokenizer /path/to/Llama
```

See `docs/DEPLOYMENT.md` for deployment and real-model details.

## Web application

Open `http://localhost:8000`.

| Tab | What it does |
|---|---|
| **Playground** | Chat with the cluster; tokens stream live over WebSocket; temperature, top-k, and output-length controls; Stop button |
| **Dashboard** | Live tokens/s, running/waiting requests, TTFT, step time, all-reduce time, gRPC overhead, KV-cache usage, and charts |
| **Load test** | Sends concurrent requests from the browser and reports TTFT p50/p95/p99, throughput, and latency |
| **Cluster & API** | Model details, worker ranks/backends, and API reference |

The interface is responsive (phone, tablet, desktop), supports light and dark themes, and does not require internet access.

**Demo-model note:** the automatically generated demo model has random weights, so its output is not meaningful natural language. It demonstrates the system. Use a real Hugging Face Llama checkpoint for meaningful text.

## Other commands

```bash
python scripts/ws_client.py "Hello"
python scripts/load_test.py --concurrency 32 --requests 128
python -m pytest tests -q
```

The README for this project reports 8 tests.

## API

- `WS /ws/generate` — send a JSON request such as `{"prompt": "...", "max_new_tokens": 64, "temperature": 0.7, "top_k": 40}`. The server sends `{"type":"token","text":...}` frames followed by a `{"type":"done",...}` frame.
- `POST /v1/generate` — JSON generation endpoint.
- `POST /v1/generate/stream` — server-sent events (SSE) streaming endpoint.
- `GET /health` — health endpoint.
- `GET /metrics` — metrics endpoint.

## Project structure

```text
grid/
  config.py            env-based settings shared by all processes
  proto/               gRPC contract + generated stubs (make proto)
  model/               Llama architecture, tensor-parallel layers
  weights/             safetensors shard loader, tiny-model generator
  kvcache/             paged block allocator
  worker/              gRPC worker + model runner (one TP rank)
  master/              engine (continuous batching), worker client, metrics
  gateway/             FastAPI app (HTTP, SSE, WebSocket) + static/index.html web UI
run.py                 one-command launcher (workers + gateway + UI)
scripts/               load_test, ws_client
tests/                 TP equivalence, paged cache, continuous batching
deploy/                docker, swarm, k8s
docs/                  architecture, deployment, evaluation report
```

**Recommended first read:** `docs/ARCHITECTURE.md` explains the design and its stated limitations.

---

# 2. gpualign — High-Throughput GPU Genomic Sequence Aligner

## Overview

A DNA read aligner that runs **Smith–Waterman alignment with affine gaps on the GPU directly from Python** using Numba CUDA. It includes a Cython **BWT/FM-index pre-filter**, memory-mapped FASTA/FASTQ streaming, a multi-threaded pipeline that overlaps I/O with GPU work, a Rich live dashboard, VCF variant reports, and a benchmark harness.

> Built for the *Zaalima Development Pvt. Ltd.* internship project. The project README states that every accelerated path is tested to reproduce a pure-Python baseline exactly.

## Web application (start here)

### Linux / macOS

```bash
./start.sh
```

### Windows

Double-click `start.bat`.

### Manual launch

```bash
pip install -r requirements.txt
python run_web.py
```

The web application is available at `http://localhost:8000`.

`start.sh` creates a virtual environment, installs dependencies, attempts the optional Cython build (the app works without it), and launches the server.

To expose the server on your network:

```bash
HOST=0.0.0.0 PORT=8080 ./start.sh
```

## Web application features

| Page | Function |
|---|---|
| **Dashboard** | System status (GPU/CPU backend, FM-index engine), recent jobs, and one-click quick demo |
| **New alignment** | Simulated demo data or upload of reference FASTA and reads FASTQ/FASTQ.GZ, with optional truth VCF and advanced tuning options |
| **Job page** | Live progress bar, throughput chart, pipeline counters, accuracy versus truth, variant table with filtering/search/paging, alignment preview, and VCF/TSV/report downloads |
| **Jobs** | Run history persisted across restarts; delete old runs |
| **Benchmark** | Kernel (Python vs CPU vs GPU) and end-to-end benchmarks; BWA/Bowtie2 comparison when installed |

Jobs run one at a time in a background queue. Data is stored in `./webdata/` by default. Change the location with `--data-dir` or `GPUALIGN_DATA_DIR`.

Relevant environment limits:

- `GPUALIGN_MAX_UPLOAD_MB` — default `4096`.
- `GPUALIGN_MAX_REF_MB` — default `200`.

The UI is responsive, supports dark mode, and uses no external CDN, so it works offline.

## Command-line quick start

```bash
make install
make web
make test
make demo
make info
```

| Command | Purpose |
|---|---|
| `make install` | Install Python dependencies and compile the Cython extension |
| `make web` | Start the web application |
| `make test` | Run 24+ tests (kernel equivalence also runs on the CUDA simulator) |
| `make demo` | Simulate data, build an index, align reads, produce VCF, and generate an accuracy report |
| `make info` | Show detected backends and GPU information |

## Manual CLI use

```bash
python -m gpualign index genome.fa --out idx/

python -m gpualign align \
  --index idx/ \
  --reads reads.fq \
  --out results/

python -m gpualign align \
  --index idx/ \
  --reads reads.fq \
  --out results/ \
  --backend cuda \
  --batch-size 16384

python -m gpualign benchmark --out bench_out
```

The default alignment backend is automatic: CUDA when available, otherwise CPU.

## Output files

The project writes the following outputs in `results/`:

- `variants.vcf`
- `alignments.tsv`
- `report.md`
- `report.json`

## Backends

| `--backend` | Engine | Use |
|---|---|---|
| `cuda` | Numba CUDA kernel, pinned memory, asynchronous streams | GPU execution; requires an NVIDIA GPU |
| `cpu` | Numba-JIT multi-core implementation using the same maths | Automatic fallback / CI |
| `python` | Pure Python | Accuracy oracle and benchmarks |
| `auto` (default) | CUDA if available, otherwise CPU | Automatic backend selection |

Without a GPU, set `NUMBA_ENABLE_CUDASIM=1` to run the same CUDA kernel on the CPU simulator. This is slow and intended for testing only.

## Project structure

```text
gpualign/
  seqio/      mmap FASTA/FASTQ chunked streaming + encoding
  index/      suffix array, FM-index, Cython seeding (_fmcore.pyx) + Python fallback
  align/      baseline.py (oracle) | cpu_kernels.py | cuda_kernels.py | backends.py | pool.py
  pipeline/   prepare (I/O+seed) -> GPU -> postprocess (traceback) ; engine.py ; stats.py
  variants/   pileup caller, VCF writer, report/truth evaluation
  ui/         Rich dashboard
  bench/      benchmark harness; simulate.py creates synthetic data + truth VCF
  web/        Flask app: app.py (routes+API), jobs.py (queue), templates/, static/
  cli.py      `python -m gpualign ...`
run_web.py / start.sh / start.bat
tests/        I/O, index, kernel equivalence (baseline==numba==CUDA), pipeline e2e, web app
docs/         ARCHITECTURE.md, HARDWARE_OPTIMIZATION.md, PROJECT_PLAN.md
```

## How it works

1. Reads stream from disk through `mmap` in batches.
2. Each read is seeded against the FM-index (Cython). Reads with no consistent locus are filtered out; the remaining reads receive an approximately 140 bp reference window.
3. Windows and reads are written into pinned-memory pool slots. The GPU aligns thousands of pairs in parallel and returns scores and end coordinates.
4. CPU threads perform traceback only for surviving reads, re-check the GPU score, and extract SNP/indel events.
5. Events are piled up, called, and written as VCF plus Markdown/JSON reports.

See `docs/ARCHITECTURE.md` for details.

## Validation results (as reported in the project README)

The README reports the following results for synthetic data using the CPU backend:

- 200 kb genome, 30x: **386/386 truth variants recovered, 0 false positives** (SNP + indel), with 0 GPU/CPU score disagreements.
- Kernel equivalence: pure Python == Numba traceback == Numba batch == CUDA kernel (simulator).
- Mapping-position accuracy greater than 99%; unrelated reads are filtered before GPU execution.

## Hardware and validation notes

- The project README states that development was performed without a physical GPU. CUDA code was verified on Numba's simulator against the baseline.
- Run `make test` and `make bench` on the target GPU machine to confirm behavior and obtain real throughput measurements.
- The benchmark does not fabricate results; BWA/Bowtie2 rows show “not installed” when those tools are absent.
- Index construction is in RAM and is practical for genomes up to a few hundred Mbp; read streaming is described as unbounded.
- See `docs/HARDWARE_OPTIMIZATION.md` for limits and tuning advice.

---

# 3. Project comparison

| Aspect | Distributed Memory Grid | gpualign |
|---|---|---|
| Main domain | Distributed large-language-model inference | Genomic sequence alignment |
| Main workload | Token generation for concurrent users | DNA read mapping/alignment and variant calling |
| Core algorithms | Tensor parallelism, continuous batching, paged KV cache | FM-index seeding, Smith–Waterman with affine gaps |
| Main acceleration / distribution | Multi-worker distributed execution; NCCL/Gloo | Numba CUDA, CPU fallback, Cython pre-filter |
| Main service/UI | FastAPI, WebSockets, live dashboard | Flask web app, job dashboard, Rich terminal dashboard |
| Main outputs | Streamed generated tokens and inference metrics | VCF, TSV, Markdown and JSON reports |
| Main project entry point | `python run.py` | `./start.sh` or `python run_web.py` |

## Which project should I run?

- Choose **Distributed Memory Grid** to explore distributed LLM serving, tensor-parallel execution, KV-cache management, continuous batching, and token streaming.
- Choose **gpualign** to explore GPU-accelerated bioinformatics, sequence indexing, Smith–Waterman alignment, variant calling, and genomic benchmarking.

---

# 4. Important setup notes

- These are **two separate projects**, not one combined runtime. Run each from its own project directory and use its own `requirements.txt` and setup instructions.
- Both projects expose a web application on port `8000` by default. If you want to run both at the same time, change one application's port to avoid a conflict.
- The demo LLM uses random weights and is intended to demonstrate the inference system, not produce meaningful answers.
- The genomic aligner's reported validation figures are the results stated in its project README; run the tests and benchmarks in your own environment to validate performance on your hardware.

## Suggested reading order

1. Distributed Memory Grid: `docs/ARCHITECTURE.md`, then `docs/DEPLOYMENT.md`.
2. gpualign: `docs/ARCHITECTURE.md`, then `docs/HARDWARE_OPTIMIZATION.md` and `docs/PROJECT_PLAN.md`.

---

## License

No license information was included in the two source README files. Add the appropriate license and attribution details for your projects before publishing or distributing this repository.
