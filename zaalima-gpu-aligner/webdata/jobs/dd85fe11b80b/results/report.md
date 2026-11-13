# gpualign run report

## Run

| metric | value |
|---|---|
| reads_total | 7500 |
| reads_seeded | 7500 |
| reads_filtered | 0 |
| reads_skipped | 0 |
| reads_aligned | 7500 |
| reads_lowscore | 0 |
| gpu_cpu_mismatch | 0 |
| batches | 4 |
| bases | 750000 |
| backend | cpu |
| device |  |
| elapsed_s | 1.531 |
| reads_per_sec | 4898.0 |
| kernel_seconds | 0.114 |

## Variants

| type | count |
|---|---|
| SNP | 91 |
| INS | 1 |
| DEL | 3 |
| total | 95 |

Ti/Tv ratio: 0.542

## Accuracy vs truth set

| class | truth | called | TP | precision | recall |
|---|---|---|---|---|---|
| all | 95 | 95 | 95 | 1.0 | 1.0 |
| SNP | 91 | 91 | 91 | 1.0 | 1.0 |
| INDEL | 4 | 4 | 4 | 1.0 | 1.0 |
