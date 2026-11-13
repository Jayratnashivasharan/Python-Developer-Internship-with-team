# Architecture

```
                 reads.fastq (TB-scale, mmap, streamed)                 reference.fa
                        |                                                    |
                        |                                      fasta_to_encoded + SA/BWT build
                        v                                                    v
  +--------------------------------------+                 +------------------------------+
  | STAGE 1  prep threads (ThreadPool)   |  seeds (Cython) |  FM-index  (mmap'd .npy)     |
  |  iter_fastq -> encode -> FM seeding  |<--------------->|  sa / bwt / occ / C / ref.bin |
  |  -> vote on diagonal -> choose strand|                 +------------------------------+
  |  -> write (read, ref window) straight|
  |     into a PINNED slot               |   reads with no consistent seed locus are
  +------------------+-------------------+   FILTERED here and never reach the GPU
                     | bounded queue (prefetch)
                     v
  +--------------------------------------+
  | STAGE 2  main thread: GPU            |  per-slot CUDA stream:
  |  H2D async -> sw_kernel -> D2H async |  H2D -> kernel -> D2H -> sync
  |  1 thread = 1 (read, window) pair    |  Score-only, O(m) local memory
  +------------------+-------------------+  Output: score, query end, ref end
                     v  thread pool
  +--------------------------------------+
  | STAGE 3  post threads (numba nogil)  |  traceback only on the sub-window the GPU
  |  traceback -> CIGAR -> variant events|  located; re-verifies GPU score (self-check)
  +------------------+-------------------+
                     v  main thread merges
        alignments.tsv   depth (memmap)   Counter(variant events)
                     v
              variants.vcf  +  report.md / report.json
```

## Why score-only on the GPU?
A traceback needs an O(m*n) direction matrix per pair (~14 KB for 100x140, x3 for
affine gaps). Thousands of resident threads cannot hold that in fast memory.
Score-only needs two O(m) rows in registers/local memory. Only the ~1 % of work
that survives the filter pays for the CPU traceback, and that traceback also
*verifies* the GPU score - `gpu_cpu_mismatch` in the report must be 0.

## Correctness chain (tested)
`baseline.py` (pure Python, list based)  ==  `cpu_kernels.sw_traceback` (numba)
== `cpu_kernels.sw_score_batch` == `cuda_kernels.sw_kernel` (real GPU or simulator).
Tie-breaking rule everywhere: first strict maximum in row-major order.

## Modules
| Module | Responsibility |
|---|---|
| `seqio/` | mmap FASTA/FASTQ chunked streaming, 2-bit-style byte encoding |
| `index/` | suffix array, BWT/FM-index, Cython backward search, diagonal voting |
| `align/` | baseline oracle, numba CPU kernels, CUDA kernel, backends, slot pool |
| `pipeline/` | prepare / GPU / postprocess stages + engine, thread-safe `Stats` |
| `variants/` | pileup accumulator, VCF writer, report + truth evaluation |
| `ui/` | Rich live dashboard |
| `bench/` | kernel + end-to-end benchmarks (bwa/bowtie2 if installed) |
| `simulate.py` | synthetic genome / reads / truth VCF |
