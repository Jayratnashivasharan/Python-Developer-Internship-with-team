"""Synthetic data generator: reference genome, a mutated 'sample' genome with a
KNOWN truth set of SNPs/indels, and simulated FASTQ reads from the sample.
Gives end-to-end validation without downloading real genomes."""
from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np

L = "ACGT"


def _wrap(seq: str, w=60):
    return "\n".join(seq[i:i + w] for i in range(0, len(seq), w))


def make_reference(sizes: List[int], rng) -> Dict[str, str]:
    return {f"chr{i+1}": "".join(rng.choice(list(L), size=s)) for i, s in enumerate(sizes)}


def mutate(ref: Dict[str, str], rng, snp_rate=0.002, indel_rate=0.0003, spacing=60):
    """Return (sample genome, truth list[(contig,pos1,ref,alt)]). Indels are
    chosen to be unambiguous (not shiftable by left-normalisation)."""
    sample, truth = {}, []
    for name, s in ref.items():
        n = len(s)
        n_snp, n_indel = int(n * snp_rate), int(n * indel_rate)
        sites = sorted(rng.choice(np.arange(spacing, n - spacing), size=n_snp + n_indel, replace=False))
        # enforce spacing
        keep, last = [], -10**9
        for p in sites:
            if p - last >= spacing:
                keep.append(int(p)); last = p
        kinds = rng.random(len(keep))
        out, cur, vars_ = [], 0, []
        for p, kd in zip(keep, kinds):
            if kd < n_snp / max(1, n_snp + n_indel):
                alt = rng.choice([b for b in L if b != s[p]])
                out.append(s[cur:p]); out.append(alt); cur = p + 1
                vars_.append((name, p + 1, s[p], alt))
            elif kd < n_snp / max(1, n_snp + n_indel) + 0.5 * (1 - n_snp / max(1, n_snp + n_indel)):
                ln = int(rng.integers(1, 4))
                ins = "".join(rng.choice(list(L), size=ln))
                if ins[-1] == s[p] or ins[0] == s[p + 1]:
                    continue
                out.append(s[cur:p + 1]); out.append(ins); cur = p + 1
                vars_.append((name, p + 1, s[p], s[p] + ins))
            else:
                ln = int(rng.integers(1, 4))
                if s[p] == s[p + ln] or s[p + 1] == s[p + ln + 1]:
                    continue
                out.append(s[cur:p + 1]); cur = p + 1 + ln
                vars_.append((name, p + 1, s[p:p + 1 + ln], s[p]))
        out.append(s[cur:])
        sample[name] = "".join(out)
        truth.extend(vars_)
    return sample, truth


def simulate_reads(sample: Dict[str, str], n_reads: int, read_len: int, rng,
                   err_rate=0.004) -> List[Tuple[str, str]]:
    names = list(sample)
    lens = np.array([len(sample[c]) for c in names], dtype=float)
    ci = rng.choice(len(names), size=n_reads, p=lens / lens.sum())
    reads = []
    comp = str.maketrans("ACGT", "TGCA")
    for idx, c in enumerate(ci):
        g = sample[names[c]]
        pos = int(rng.integers(0, len(g) - read_len))
        r = list(g[pos:pos + read_len])
        errs = rng.random(read_len) < err_rate
        for k in np.nonzero(errs)[0]:
            r[k] = L[(L.index(r[k]) + int(rng.integers(1, 4))) % 4]
        r = "".join(r)
        strand = "+"
        if rng.random() < 0.5:
            r, strand = r.translate(comp)[::-1], "-"
        reads.append((f"r{idx}|{names[c]}|{pos}|{strand}", r))
    return reads


def write_dataset(out_dir: str, genome_size=200_000, n_contigs=2, coverage=30,
                  read_len=100, seed=42, err_rate=0.004):
    """Writes reference.fa, reads.fq, truth.vcf under out_dir; returns paths."""
    from .variants.vcf import write_truth_vcf
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = make_reference([genome_size // n_contigs] * n_contigs, rng)
    sample, truth = mutate(ref, rng)
    n_reads = int(genome_size * coverage / read_len)
    reads = simulate_reads(sample, n_reads, read_len, rng, err_rate)
    fa, fq, vcf = (os.path.join(out_dir, f) for f in ("reference.fa", "reads.fq", "truth.vcf"))
    with open(fa, "w") as fh:
        for k, v in ref.items():
            fh.write(f">{k}\n{_wrap(v)}\n")
    with open(fq, "w") as fh:
        q = "I" * read_len
        for name, s in reads:
            fh.write(f"@{name}\n{s}\n+\n{q}\n")
    write_truth_vcf(vcf, truth, {k: len(v) for k, v in ref.items()})
    return {"fasta": fa, "fastq": fq, "truth": vcf, "n_reads": n_reads, "n_truth": len(truth)}
