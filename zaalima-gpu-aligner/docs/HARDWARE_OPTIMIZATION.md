# Hardware optimisation techniques & scaling limits

## Techniques implemented

| Technique | Where | Effect |
|---|---|---|
| **mmap + chunked reads** | `seqio/` | RAM is O(chunk), not O(file); OS page cache does read-ahead (`MADV_SEQUENTIAL`). |
| **Seed-and-extend pre-filter** | `index/` (FM-index, Cython, GIL released) | Replaces O(genome) alignment with O(read) lookup + a ~140 bp window; unrelated reads never touch the GPU. |
| **One thread per alignment** | `cuda_kernels.py` | Embarrassingly parallel; no inter-thread sync; thousands of alignments in flight. |
| **Score-only kernel, O(m) state** | `cuda_kernels.py` | Fits per-thread local memory; avoids huge direction matrices. |
| **Pinned (page-locked) host memory** | `CudaBackend.alloc_slot` | Enables true asynchronous DMA, ~2x PCIe bandwidth vs pageable. |
| **Memory pool (`SlotPool`)** | `align/pool.py` | All buffers allocated once; steady state has zero `cudaMalloc/Host` calls. |
| **No staging copy** | `pipeline/prepare.py` | Encoded reads are written *directly* into the pinned buffer the DMA reads. |
| **Mapped memory (`--mapped`)** | `CudaBackend` | True zero-copy: kernel reads host RAM over PCIe, no explicit H2D/D2H. Best when each byte is used ~once; the default explicit-copy path is usually faster for compute-dense kernels. Benchmark both on your hardware. |
| **Stream per slot + multiple slots** | `CudaBackend`, `engine.py` | Copy/compute of different batches can overlap. |
| **Bounded producer/consumer queues** | `engine.py` | Overlaps file I/O, seeding, GPU and traceback; bounded memory (back-pressure). |
| **Numba `nogil=True`** | traceback / events | Post-processing threads run in parallel despite the GIL. |
| **Depth as on-disk memmap** | `variants/caller.py` | Coverage array not limited by RAM. |

## Scaling limits (honest list)

* **Reads (FASTQ)**: streamed - no practical limit; throughput is bound by
  disk, seeding and GPU. gzip input is supported but single-threaded.
* **Reference size**: the suffix array is built in RAM by prefix doubling
  (about 40 bytes/base peak) and stored as int32 (limit ~2.1 Gbp). Good to
  ~100-300 Mbp on a 16 GB machine. For a human genome use an external
  index builder (SA-IS / `libdivsufsort`) and a *sampled* suffix array; the
  `FMIndex` API (`seed_diagonals`) is the only thing the pipeline depends on.
* **Read length**: `GPUALIGN_MAX_QUERY` (default 256) is a compile-time
  constant for per-thread local memory; longer reads are counted as skipped.
  Long reads would need a banded or tiled kernel.
* **Pre-filter sensitivity**: exact k-mer seeds (k=16) miss reads whose every
  seed contains an error; highly repetitive seeds (> `max_seed_occ`) are skipped.
* **Variant calling**: simple pileup with a binomial screening score; indels
  are not left-normalised; no base-quality weighting; haploid-style genotype
  from allele fraction. Intended for validation/demonstration, not clinical use.
* **Single GPU**, single process. Multi-GPU = one process per device on disjoint
  FASTQ shards, then merge VCFs.
* **Tested hardware**: development used the Numba CUDA *simulator* + CPU (no
  physical GPU). Performance numbers on real GPUs must be measured with
  `make bench` on the target machine.

## Tuning guide
`--batch-size` (default 2048): raise until GPU occupancy saturates (8192-32768
on modern cards). `--threads`: set to physical cores / 2. Watch the dashboard:
*Prep queue depth* near 0 => CPU-bound (add threads); near `prefetch` and
*Kernel busy* high => GPU-bound (good).
