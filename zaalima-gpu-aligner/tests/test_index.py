import numpy as np
import pytest
from gpualign.index import suffix_array, fm
from gpualign.index import _fm_py


def _text(n, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.integers(1, 5, n, dtype=np.uint8), np.zeros(1, np.uint8)))


def test_suffix_array_matches_naive():
    t = _text(500, 1)
    sa = suffix_array.build_suffix_array(t)
    naive = sorted(range(len(t)), key=lambda i: t[i:].tobytes())
    assert sa.tolist() == naive


def _tables(t):
    sa = suffix_array.build_suffix_array(t)
    bwt, C, occ = fm._bwt_tables(t, sa)
    return sa, bwt, C, occ


def test_seed_hits_match_naive_search():
    t = _text(3000, 2)
    sa, bwt, C, occ = _tables(t)
    ref = t[:-1] - 1
    read = ref[1000:1060].copy()
    got = sorted(_fm_py.seed_diagonals(bwt, occ, C, sa, read, 16, 8, 50).tolist())
    assert set(got) == {1000}                     # unique random seq -> single diagonal
    assert len(got) == (60 - 16) // 8 + 1


@pytest.mark.skipif(not fm.USING_CYTHON, reason="Cython extension not built")
def test_cython_equals_python():
    from gpualign.index import _fmcore
    t = _text(20000, 5)
    sa, bwt, C, occ = _tables(t)
    ref = t[:-1] - 1
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = int(rng.integers(0, 19800))
        read = ref[p:p + 80].copy()
        read[rng.integers(0, 80)] = 4 if rng.random() < .3 else rng.integers(0, 4)
        a = _fmcore.seed_diagonals(bwt, occ, C, sa.astype(np.int32), read, 12, 6, 30)
        b = _fm_py.seed_diagonals(bwt, occ, C, sa, read, 12, 6, 30)
        assert a.tolist() == b.tolist()


def test_index_on_disk_roundtrip(dataset):
    idx = fm.FMIndex(dataset["index"])
    assert len(idx.contigs) == 2
    name, local = idx.to_local(idx.contigs[1]["start"] + 5)
    assert name == "chr2" and local == 5


def test_numba_equals_python():
    from gpualign.index import _fm_numba
    t = _text(20000, 9)
    sa, bwt, C, occ = _tables(t)
    ref = t[:-1] - 1
    rng = np.random.default_rng(1)
    for _ in range(40):
        p = int(rng.integers(0, 19800))
        read = ref[p:p + 80].copy()
        read[rng.integers(0, 80)] = 4 if rng.random() < .3 else rng.integers(0, 4)
        a = _fm_numba.seed_diagonals(bwt, occ, C, sa.astype(np.int32), read, 12, 6, 30)
        b = _fm_py.seed_diagonals(bwt, occ, C, sa, read, 12, 6, 30)
        assert a.tolist() == b.tolist()
