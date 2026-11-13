"""Biological-accuracy gate: every accelerated path must reproduce the
pure-Python baseline exactly (score, coordinates, CIGAR ops)."""
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from gpualign.align.baseline import smith_waterman, smith_waterman_score
from gpualign.align.cpu_kernels import sw_traceback, sw_score_batch, extract_events
from gpualign.seqio.encoding import encode

random.seed(7)


def _mut(s):
    s = list(s)
    for _ in range(random.randint(0, 5)):
        k = random.randrange(len(s)); r = random.random()
        if r < .5: s[k] = random.choice("ACGTN")
        elif r < .75: s.insert(k, random.choice("ACGT"))
        else: del s[k]
    return "".join(s)


def _pairs(n):
    out = []
    while len(out) < n:
        ref = "".join(random.choice("ACGT") for _ in range(random.randint(30, 90)))
        a = random.randrange(0, 10)
        q = _mut(ref[a:a + random.randint(20, 60)])
        if q:
            out.append((q, ref))
    return out


def test_known_alignment():
    a = smith_waterman("ACGTACGT", "TTACGTACGTTT")
    assert (a.score, a.cigar, a.rstart, a.rend) == (16, "8M", 2, 10)
    ref = "GATTACAGGCTAACCGTTAGCATCGGATCCATGACGTA"
    q = ref[:18] + "CC" + ref[18:]                    # 2-base insertion in the query
    b = smith_waterman(q, ref)
    assert b.cigar == "18M2I20M" and b.score == 2 * 38 - 7


def test_traceback_equals_baseline():
    for q, r in _pairs(250):
        al = smith_waterman(q, r)
        sc, qs, qe, rs, re_, ops = sw_traceback(encode(q), encode(r), 2, 3, 5, 2)
        assert (al.score, al.qstart, al.qend, al.rstart, al.rend) == (sc, qs, qe, rs, re_)
        assert al.ops == ops.tolist()


def test_score_batch_equals_baseline():
    pairs = _pairs(120)
    B = len(pairs)
    reads = np.zeros((B, 256), np.uint8); refs = np.zeros((B, 296), np.uint8)
    rl = np.zeros(B, np.int32); wl = np.zeros(B, np.int32)
    for i, (q, r) in enumerate(pairs):
        reads[i, :len(q)] = encode(q); rl[i] = len(q)
        refs[i, :len(r)] = encode(r); wl[i] = len(r)
    out = np.zeros((B, 3), np.int32)
    sw_score_batch(reads, rl, refs, wl, B, 2, 3, 5, 2, out)
    for i, (q, r) in enumerate(pairs):
        assert tuple(out[i]) == smith_waterman_score(q, r)


def test_event_extraction():
    ref = "ACGTTGCAAGCTTAGGCTAACGTA"
    q = ref[:10] + "A" + ref[11:]              # substitution at index 10 (G->A? ensure differs)
    if q == ref:
        q = ref[:10] + "T" + ref[11:]
    r, qq = encode(ref), encode(q)
    _, qs, qe, rs, re_, ops = sw_traceback(qq, r, 2, 3, 5, 2)
    kind, rpos, qpos, ln = extract_events(ops, qq, qs, r, rs)
    assert kind.tolist() == [0] and rpos.tolist() == [10]


def test_cuda_kernel_matches_baseline():
    """Real GPU if present, else the Numba CUDA simulator (subprocess so the
    env var is set before numba is first imported)."""
    script = os.path.join(os.path.dirname(__file__), "cuda_sim_check.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), ".."))
    try:
        from numba import cuda
        real = cuda.is_available()
    except Exception:
        real = False
    if not real:
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    r = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CUDA kernel OK" in r.stdout
