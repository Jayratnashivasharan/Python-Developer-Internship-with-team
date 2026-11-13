import numpy as np
from gpualign.seqio import encode, decode, revcomp, iter_fasta_events, fasta_to_encoded, iter_fastq


def test_encode_roundtrip_and_revcomp():
    assert decode(encode("ACGTNacgt")) == "ACGTNACGT"
    assert decode(revcomp(encode("AACGN"))) == "NCGTT"


def _fasta(tmp_path):
    p = tmp_path / "x.fa"
    p.write_text(">a desc\nACGT\nACGT\n>b\nGGGG\nCC\n>c\nTTTT\n")
    return str(p)


def _collect(path, chunk):
    seqs, cur = {}, None
    for k, v in iter_fasta_events(path, chunk):
        if k == "H":
            cur = v; seqs[cur] = ""
        else:
            seqs[cur] += decode(v)
    return seqs


def test_fasta_chunking_is_boundary_safe(tmp_path):
    p = _fasta(tmp_path)
    expect = {"a": "ACGTACGT", "b": "GGGGCC", "c": "TTTT"}
    for chunk in (1, 3, 7, 16, 1000):          # tiny chunks split lines & headers
        assert _collect(p, chunk) == expect


def test_fasta_to_encoded_layout(tmp_path):
    meta = fasta_to_encoded(_fasta(tmp_path), str(tmp_path / "o"), chunk_bytes=5)
    ref = np.fromfile(tmp_path / "o" / "ref.bin", dtype=np.uint8)
    assert [c["start"] for c in meta["contigs"]] == [0, 9, 16]
    assert ref[8] == 4 and ref[15] == 4 and len(ref) == 20


def test_fastq_batches(tmp_path):
    p = tmp_path / "r.fq"
    p.write_text("".join(f"@r{i} x\nACGT\n+\nIIII\n" for i in range(10)))
    batches = list(iter_fastq(str(p), 4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert batches[0].names[0] == "r0" and batches[-1].bytes_done == p.stat().st_size
