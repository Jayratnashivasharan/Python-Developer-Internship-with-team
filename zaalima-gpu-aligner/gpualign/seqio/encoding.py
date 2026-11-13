"""Nucleotide encoding: A=0 C=1 G=2 T=3 N(anything else)=4."""
import numpy as np

ALPHABET = "ACGTN"
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i
_DEC = np.frombuffer(b"ACGTN", dtype=np.uint8)


def encode(seq) -> np.ndarray:
    """bytes | str | bytearray -> uint8 array."""
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    return _LUT[np.frombuffer(seq, dtype=np.uint8)]


def decode(arr: np.ndarray) -> str:
    return _DEC[np.minimum(arr, 4)].tobytes().decode("ascii")


def revcomp(arr: np.ndarray) -> np.ndarray:
    return np.where(arr < 4, 3 - arr, 4).astype(np.uint8)[::-1].copy()
