#!/usr/bin/env bash
# End-to-end demo on synthetic data (no downloads needed).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
OUT=${1:-demo_out}
python3 -m gpualign info
python3 -m gpualign simulate --out "$OUT/data" --genome-size 200000 --coverage 30
python3 -m gpualign index "$OUT/data/reference.fa" --out "$OUT/index"
python3 -m gpualign align --index "$OUT/index" --reads "$OUT/data/reads.fq" \
        --out "$OUT/results" --truth "$OUT/data/truth.vcf"
echo; echo "== report =="; cat "$OUT/results/report.md"
