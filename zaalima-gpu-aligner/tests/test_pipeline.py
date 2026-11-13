import os
from gpualign.config import PipelineConfig
from gpualign.pipeline import run_alignment
from gpualign.variants.report import evaluate_against_truth


def test_end_to_end_variant_recovery(dataset, tmp_path):
    out = str(tmp_path / "out")
    st = run_alignment(dataset["index"], dataset["fastq"], out, backend="cpu",
                       cfg=PipelineConfig(batch_size=512), truth_vcf=dataset["truth"])
    assert st.reads_total == dataset["n_reads"]
    assert st.reads_aligned / st.reads_total > 0.98
    assert st.gpu_cpu_mismatch == 0
    ev = st.truth_eval["all"]
    assert ev["recall"] >= 0.95 and ev["precision"] >= 0.95, ev
    for f in ("variants.vcf", "alignments.tsv", "report.md", "report.json"):
        assert os.path.getsize(os.path.join(out, f)) > 0


def test_mapping_position_accuracy(dataset, tmp_path):
    out = str(tmp_path / "o2")
    run_alignment(dataset["index"], dataset["fastq"], out, backend="cpu")
    ok = tot = 0
    for line in open(os.path.join(out, "alignments.tsv")):
        if line.startswith("#"):
            continue
        name, contig, pos, strand, *_ = line.split("\t")
        _, tc, tp, ts = name.split("|")
        tot += 1
        ok += (tc == contig and abs(int(tp) - int(pos)) <= 30 and ts == strand)
    assert ok / tot > 0.99


def test_unrelated_reads_are_filtered_before_gpu(dataset, tmp_path):
    import random
    random.seed(0)
    fq = tmp_path / "junk.fq"
    fq.write_text("".join(f"@j{i}\n{''.join(random.choice('ACGT') for _ in range(100))}\n+\n{'I'*100}\n"
                          for i in range(300)))
    st = run_alignment(dataset["index"], str(fq), str(tmp_path / "o3"), backend="cpu")
    assert st.reads_filtered >= 295 and st.reads_aligned <= 5
