# gpualign run report

## Run

| metric | value |
|---|---|
| reads_total | 20000 |
| reads_seeded | 20000 |
| reads_filtered | 0 |
| reads_skipped | 0 |
| reads_aligned | 20000 |
| reads_lowscore | 0 |
| gpu_cpu_mismatch | 0 |
| batches | 10 |
| bases | 2000000 |
| backend | cpu |
| device |  |
| elapsed_s | 4.132 |
| reads_per_sec | 4840.7 |
| kernel_seconds | 0.328 |

## Variants

| type | count |
|---|---|
| SNP | 173 |
| INS | 12 |
| DEL | 5 |
| total | 190 |

Ti/Tv ratio: 0.491

## Accuracy vs truth set

| class | truth | called | TP | precision | recall |
|---|---|---|---|---|---|
| all | 190 | 190 | 190 | 1.0 | 1.0 |
| SNP | 173 | 173 | 173 | 1.0 | 1.0 |
| INDEL | 17 | 17 | 17 | 1.0 | 1.0 |
