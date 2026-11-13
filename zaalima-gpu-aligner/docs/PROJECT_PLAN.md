# Plan -> implementation map

| Week | Requirement | Implemented in | Verified by |
|---|---|---|---|
| 1 | Chunked, memory-mapped FASTA/FASTQ ingestion | `seqio/fasta.py`, `seqio/fastq.py` | `tests/test_io.py` (1-byte chunks) |
| 1 | Pure-Python Smith-Waterman baseline | `align/baseline.py` | `test_known_alignment` |
| 1 | Separation of Python orchestration from C/CUDA kernel code | package layout: `align/cuda_kernels.py`, `align/cpu_kernels.py`, `index/*.pyx` vs `pipeline/`, `web/` | `docs/ARCHITECTURE.md` |
| + | Web application (Flask) | `gpualign/web/` | `tests/test_web.py` |
| 2 | Numba `@cuda.jit` kernel | `align/cuda_kernels.py` | `cuda_sim_check.py` |
| 2 | Memory pooling + zero-copy transfers | `align/pool.py`, `CudaBackend` (pinned + mapped) | slot reuse in engine |
| 2 | Unit tests: accuracy == baseline | `tests/test_kernel_equivalence.py` | `make test` |
| 3 | Multi-threaded I/O overlapping GPU | `pipeline/engine.py` (`concurrent.futures`) | `test_pipeline.py` |
| 3 | BWT index in Cython as pre-filter | `index/_fmcore.pyx`, `index/fm.py` | `test_cython_equals_python`, `test_unrelated_reads_are_filtered_before_gpu` |
| 3 | Rich progress dashboard | `ui/dashboard.py` | `gpualign align` on a TTY |
| 4 | Benchmark vs Bowtie2 / BWA | `bench/benchmark.py` (`gpualign benchmark`) | `benchmark_report.md` |
| 4 | VCF reports from alignment deviations | `variants/` | truth precision/recall in `report.md` |
| 4 | Documentation | `README.md`, `docs/` | - |

## Review checklists

**Mid-project (end of week 2)**
- [x] Ingestion without memory overflow (mmap + chunking; boundary-safe tests)
- [x] GPU kernels compile and execute (real GPU or Numba CUDA simulator)
- [x] Outputs match baseline exactly (`test_kernel_equivalence.py`)

**Final (end of week 4)**
- [x] End-to-end aligner: `make demo`
- [x] Multithreaded I/O + Cython index integrated
- [x] Benchmark report: `make bench`; external tools reported honestly when absent
