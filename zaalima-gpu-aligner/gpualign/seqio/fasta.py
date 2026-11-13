"""Chunked, memory-mapped FASTA reading.

The file is mapped with mmap (the OS pages it in on demand) and consumed in
`chunk_bytes` slices, so RAM use is O(chunk) regardless of file size.
"""
from __future__ import annotations
import json
import mmap
import os
from typing import Iterator, Tuple, Union

import numpy as np

from .encoding import encode

Event = Tuple[str, Union[str, np.ndarray]]   # ("H", name) | ("S", encoded array)


def _lines_to_events(data: bytes) -> Iterator[Event]:
    seq_parts = []
    for line in data.split(b"\n"):
        if not line:
            continue
        if line[:1] == b">":
            if seq_parts:
                yield "S", encode(b"".join(seq_parts).replace(b"\r", b""))
                seq_parts = []
            parts = line[1:].split()
            yield "H", parts[0].decode() if parts else ""
        else:
            seq_parts.append(line)
    if seq_parts:
        yield "S", encode(b"".join(seq_parts).replace(b"\r", b""))


def iter_fasta_events(path: str, chunk_bytes: int = 64 << 20) -> Iterator[Event]:
    """Yield ("H", name) and ("S", uint8-encoded sequence piece) events.
    Sequence of one record may arrive in several "S" pieces (chunk borders)."""
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        try:
            mm.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
        pos, carry = 0, b""
        while pos < size:
            end = min(pos + chunk_bytes, size)
            data = carry + mm[pos:end]
            pos = end
            if pos < size:                       # keep partial last line for next chunk
                cut = data.rfind(b"\n")
                if cut == -1:
                    carry = data
                    continue
                carry, data = data[cut + 1:], data[:cut + 1]
            else:
                carry = b""
            yield from _lines_to_events(data)


def fasta_to_encoded(fasta: str, out_dir: str, chunk_bytes: int = 64 << 20) -> dict:
    """Stream FASTA -> <out_dir>/ref.bin (uint8 codes, contigs separated by N)
    plus meta.json listing contigs. Never holds the genome in RAM."""
    os.makedirs(out_dir, exist_ok=True)
    contigs, cur = [], None
    offset = 0
    sep = np.array([4], dtype=np.uint8)
    with open(os.path.join(out_dir, "ref.bin"), "wb") as out:
        for kind, val in iter_fasta_events(fasta, chunk_bytes):
            if kind == "H":
                if cur is not None:
                    out.write(sep.tobytes()); offset += 1
                cur = {"name": val, "start": offset, "length": 0}
                contigs.append(cur)
            else:
                if cur is None:
                    raise ValueError("FASTA sequence before first '>' header")
                out.write(val.tobytes())
                offset += len(val)
                cur["length"] += len(val)
    meta = {"contigs": contigs, "total_length": offset}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta
