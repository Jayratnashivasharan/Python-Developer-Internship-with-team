import io
import json
import os
import time

import pytest

from gpualign.web import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(str(tmp_path_factory.mktemp("web")), warmup=False)
    app.testing = True
    return app.test_client()


def wait_done(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/api/jobs/{jid}").get_json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(0.3)
    raise AssertionError("job timed out")


def test_pages_render(client):
    for url in ("/", "/new", "/jobs", "/benchmark", "/about", "/healthz", "/api/system"):
        r = client.get(url)
        assert r.status_code == 200, url
    assert client.get("/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/static/style.css").status_code == 200


def test_validation_errors(client):
    r = client.post("/api/jobs", data={"source": "demo", "genome_size": "5"})
    assert r.status_code == 400 and any("Genome size" in e for e in r.get_json()["errors"])
    r = client.post("/api/jobs", data={"source": "upload"})
    assert r.status_code == 400 and len(r.get_json()["errors"]) == 2
    r = client.post("/api/jobs", data={"source": "demo", "backend": "tpu"})
    assert r.status_code == 400


def test_demo_job_full_flow(client):
    r = client.post("/api/jobs", data={"source": "demo", "genome_size": "20000", "coverage": "15",
                                       "backend": "cpu", "name": "t<b>x</b>"})
    assert r.status_code == 201
    jid = r.get_json()["id"]
    assert client.get(f"/jobs/{jid}").status_code == 200
    d = wait_done(client, jid)
    assert d["status"] == "done", d.get("error")
    res = d["result"]
    assert res["stats"]["reads_total"] > 0 and res["stats"]["gpu_cpu_mismatch"] == 0
    assert res["truth_eval"]["all"]["recall"] >= 0.9
    assert d["progress"] == 100 and len(d["history"]) >= 1
    v = client.get(f"/api/jobs/{jid}/variants?limit=5").get_json()
    assert v["total"] == res["n_variants"] and len(v["rows"]) == min(5, v["total"])
    snp = client.get(f"/api/jobs/{jid}/variants?type=SNP").get_json()
    assert all(x["type"] == "SNP" for x in snp["rows"])
    assert client.get(f"/api/jobs/{jid}/variants?limit=abc").status_code == 400
    assert len(client.get(f"/api/jobs/{jid}/alignments").get_json()["rows"]) > 0
    for f in ("variants.vcf", "alignments.tsv", "report.md", "report.json"):
        dl = client.get(f"/jobs/{jid}/download/{f}")
        assert dl.status_code == 200 and len(dl.data) > 0
    assert client.get(f"/jobs/{jid}/download/job.json").status_code == 404
    assert client.get(f"/jobs/{jid}/download/..%2Fjob.json").status_code == 404
    # job name must be HTML-escaped in the page
    page = client.get(f"/jobs/{jid}").get_data(as_text=True)
    assert "t<b>x</b>" not in page
    # delete
    assert client.post(f"/jobs/{jid}/delete").status_code == 302
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_upload_job_and_bad_uploads(client, tmp_path):
    from gpualign.simulate import write_dataset
    d = write_dataset(str(tmp_path), genome_size=20000, coverage=15)
    def up(fa, fq, vcf=None):
        data = {"source": "upload", "backend": "cpu",
                "reference": (open(fa, "rb"), "ref.fa"), "reads": (open(fq, "rb"), "reads.fq")}
        if vcf:
            data["truth"] = (open(vcf, "rb"), "truth.vcf")
        return client.post("/api/jobs", data=data, content_type="multipart/form-data")
    r = up(d["fasta"], d["fastq"], d["truth"])
    assert r.status_code == 201
    done = wait_done(client, r.get_json()["id"])
    assert done["status"] == "done", done.get("error")
    assert done["result"]["truth_eval"]["all"]["recall"] >= 0.9
    # swapped files are rejected before any job is created
    n = len(client.get("/api/jobs").get_json())
    r = up(d["fastq"], d["fasta"])
    assert r.status_code == 400 and len(r.get_json()["errors"]) >= 1
    assert len(client.get("/api/jobs").get_json()) == n


def test_empty_reads_and_no_hits_finish_cleanly(client, tmp_path):
    fa = tmp_path / "r.fa"; fa.write_text(">c\n" + "ACGTTGCA" * 200 + "\n")
    fq = tmp_path / "r.fq"; fq.write_text("@a\n" + "ACGGTCAAGTCTTTGACCAGATTTT" + "\n+\n" + "I" * 25 + "\n")
    r = client.post("/api/jobs", data={"source": "upload", "backend": "cpu",
                    "reference": (open(fa, "rb"), "r.fa"), "reads": (open(fq, "rb"), "r.fq")},
                    content_type="multipart/form-data")
    d = wait_done(client, r.get_json()["id"])
    assert d["status"] == "done", d.get("error")


def test_failed_job_reports_error(client, tmp_path):
    fa = tmp_path / "e.fa"; fa.write_text(">only_header\n")
    fq = tmp_path / "e.fq"; fq.write_text("@a\nACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIII\n")
    r = client.post("/api/jobs", data={"source": "upload", "backend": "cpu",
                    "reference": (open(fa, "rb"), "e.fa"), "reads": (open(fq, "rb"), "e.fq")},
                    content_type="multipart/form-data")
    d = wait_done(client, r.get_json()["id"])
    assert d["status"] == "failed" and "no sequence" in d["error"]
    assert client.get(f"/jobs/{d['id']}/log").status_code == 200
    assert client.get(f"/jobs/{d['id']}").status_code == 200


def test_benchmark_job(client):
    r = client.post("/api/benchmark", data={"genome_size": "20000", "coverage": "4", "backend": "cpu"})
    assert r.status_code == 201
    d = wait_done(client, r.get_json()["id"], timeout=240)
    assert d["status"] == "done", d.get("error")
    assert d["result"]["kernels"]["cpu"]["identical_to_cpu"] is True
    assert client.get(f"/jobs/{d['id']}/download/benchmark_report.md").status_code == 200
    assert client.post("/api/benchmark", data={"genome_size": "1"}).status_code == 400
