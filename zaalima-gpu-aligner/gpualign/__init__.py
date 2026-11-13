"""gpualign - GPU-accelerated Smith-Waterman genomic read aligner.

Layers (see docs/ARCHITECTURE.md):
  seqio     memory-mapped FASTA/FASTQ streaming
  index     FM-index (BWT) seeding / pre-filter  (Cython + Python fallback)
  align     Smith-Waterman: pure-Python baseline, Numba-CPU, Numba-CUDA kernels
  pipeline  multi-threaded orchestration (I/O  ->  GPU  ->  post-processing)
  variants  pileup, VCF writer, report
"""
__version__ = "1.0.0"
