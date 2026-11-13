# Benchmark report

Host CPUs: 16  |  backends: python, cpu

## 1. Smith-Waterman kernel throughput (100 bp read x 140 bp window)

| backend | pairs | seconds | alignments/s | cells/s | identical to cpu |
|---|---|---|---|---|---|
| python | 100 | 0.7889 | 126.8 | 1,774,512 | True |
| cpu | 4096 | 0.0527 | 77,764.4 | 1,088,701,945 | True |

## 2. End-to-end (genome 200,000 bp, 20,000 reads)

| tool | seconds | reads/s | note |
|---|---|---|---|
| gpualign (cpu) | 4.37 | 4,575.9 | align + traceback + VCF |
| bwa_mem | - | - | not installed |
| bowtie2 | - | - | not installed |

> Caveat: small synthetic genome; external tools are highly optimised and mostly process-startup bound at this size. Re-run with larger `--genome-size` on a GPU host for meaningful comparison.
