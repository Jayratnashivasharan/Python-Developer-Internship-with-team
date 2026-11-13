"""Streaming FASTQ reader (memory-mapped for plain files, gzip supported)."""
from __future__ import annotations
import gzip
import mmap
import os
from dataclasses import dataclass, field
from typing import Iterator, List


@dataclass
class ReadBatch:
    names: List[str] = field(default_factory=list)
    seqs: List[bytes] = field(default_factory=list)
    bytes_done: int = 0
    bytes_total: int = 1

    def __len__(self):
        return len(self.names)


def iter_fastq(path: str, batch_size: int = 2048) -> Iterator[ReadBatch]:
    total = os.path.getsize(path)
    if total == 0:
        return
    if path.endswith(".gz"):
        fh = gzip.open(path, "rb")
        readline, tell, closer = fh.readline, (lambda: fh.fileobj.tell()), fh.close
    else:
        raw = open(path, "rb")
        mm = mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            mm.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
        readline, tell = mm.readline, mm.tell
        closer = lambda: (mm.close(), raw.close())
    try:
        batch = ReadBatch(bytes_total=total)
        while True:
            head = readline()
            if not head:
                break
            seq = readline().strip()
            readline()          # '+'
            readline()          # qualities (not needed for alignment scoring)
            hp = head[1:].split()
            batch.names.append(hp[0].decode() if hp else "")
            batch.seqs.append(seq)
            if len(batch) >= batch_size:
                batch.bytes_done = tell()
                yield batch
                batch = ReadBatch(bytes_total=total)
        if len(batch):
            batch.bytes_done = total
            yield batch
    finally:
        closer()


def count_fastq_reads(path: str) -> int:
    return sum(len(b) for b in iter_fastq(path, 100_000))
