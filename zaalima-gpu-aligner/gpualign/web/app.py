"""Flask web application for gpualign.

Pages   /  /new  /jobs  /jobs/<id>  /benchmark  /about
API     /api/system  /api/jobs  /api/jobs/<id>  .../variants  .../alignments
"""
from __future__ import annotations

import gzip
import os
import shutil
from datetime import datetime
from typing import List, Optional

from .. import __version__


def _system_info() -> dict:
    from ..align.backends import available_backends
    from ..config import MAX_QUERY
    from ..index.fm import USING_CYTHON, SEED_ENGINE
    gpus: List[str] = []
    try:
        from numba import cuda
        if cuda.is_available():
            for d in cuda.list_devices():
                gpus.append(d.name.decode() if isinstance(d.name, bytes) else str(d.name))
    except Exception:
        pass
    backends = available_backends()
    return {"version": __version__, "backends": backends, "gpus": gpus,
            "default_backend": "cuda" if "cuda" in backends else "cpu",
            "cython": USING_CYTHON, "seed_engine": SEED_ENGINE, "cpus": os.cpu_count() or 1, "max_query": MAX_QUERY,
            "cuda_simulator": os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"}


def create_app(data_dir: Optional[str] = None, warmup: bool = True):
    data_dir = os.path.abspath(data_dir or os.environ.get("GPUALIGN_DATA_DIR")
                               or os.path.join(os.getcwd(), "webdata"))
    os.makedirs(data_dir, exist_ok=True)
    # Must be set before numba compiles anything: keeps the JIT cache writable.
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(data_dir, ".numba_cache"))

    from flask import Flask, jsonify, render_template, request, send_from_directory, abort, url_for
    from .jobs import ACTIVE, JobManager

    app = Flask(__name__)
    max_upload_mb = int(os.environ.get("GPUALIGN_MAX_UPLOAD_MB", "4096"))
    max_ref_mb = int(os.environ.get("GPUALIGN_MAX_REF_MB", "200"))
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    manager = JobManager(data_dir)
    app.extensions["gpualign_jobs"] = manager
    sysinfo = _system_info()
    if warmup:
        manager.warmup()

    @app.template_filter("dt")
    def _dt(ts):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"

    @app.template_filter("dur")
    def _dur(s):
        s = float(s or 0)
        return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60)}s"

    @app.context_processor
    def _ctx():
        return {"sys": sysinfo, "version": __version__}

    # ----------------------------------------------------------- validation
    def _num(form, key, default, lo, hi, cast, label, errors):
        raw = (form.get(key) or "").strip()
        if raw == "":
            return default
        try:
            v = cast(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return default
        if not (lo <= v <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return default
        return v

    def _check_head(path: str, expect: bytes, label: str, errors: list, allow_gz=False):
        try:
            with open(path, "rb") as fh:
                head = fh.read(4)
                if allow_gz and head[:2] == b"\x1f\x8b":
                    with gzip.open(path, "rb") as gz:
                        head = gz.read(4)
            if not head.lstrip()[:1] == expect:
                errors.append(f"{label} does not look like a valid file (expected it to start with "
                              f"'{expect.decode()}').")
        except (OSError, EOFError):
            errors.append(f"{label} could not be read (corrupt or unsupported compression).")

    def _make_align_job(form, files):
        errors: List[str] = []
        source = form.get("source", "demo")
        if source not in ("demo", "upload"):
            errors.append("Unknown data source.")
        backend = form.get("backend", sysinfo["default_backend"])
        if backend not in sysinfo["backends"]:
            errors.append(f"Backend '{backend}' is not available on this machine.")
        p = {"source": source, "backend": backend,
             "batch_size": _num(form, "batch_size", 2048, 64, 32768, int, "Batch size", errors),
             "threads": _num(form, "threads", 2, 1, 16, int, "Worker threads", errors),
             "seed_k": _num(form, "seed_k", 16, 10, 32, int, "Seed length", errors),
             "min_score_frac": _num(form, "min_score_frac", 0.5, 0.1, 1.0, float, "Minimum score fraction", errors),
             "min_alt": _num(form, "min_alt", 3, 1, 100, int, "Minimum supporting reads", errors),
             "min_af": _num(form, "min_af", 0.25, 0.01, 1.0, float, "Minimum allele fraction", errors),
             "min_depth": _num(form, "min_depth", 4, 1, 1000, int, "Minimum depth", errors),
             "mapped": form.get("mapped") == "on" and backend == "cuda"}
        if source == "demo":
            p.update(genome_size=_num(form, "genome_size", 100_000, 10_000, 5_000_000, int, "Genome size", errors),
                     coverage=_num(form, "coverage", 20, 1, 100, float, "Coverage", errors),
                     read_len=_num(form, "read_len", 100, 50, min(250, sysinfo["max_query"]), int, "Read length", errors),
                     contigs=_num(form, "contigs", 2, 1, 10, int, "Contigs", errors),
                     seed=_num(form, "seed", 42, 0, 2**31 - 1, int, "Random seed", errors))
        else:
            for key, label in (("reference", "Reference FASTA"), ("reads", "Reads FASTQ")):
                f = files.get(key)
                if f is None or not f.filename:
                    errors.append(f"{label} is required.")
        if errors:
            return None, errors

        jid, jdir = manager.new_dir()
        try:
            if source == "upload":
                from werkzeug.utils import secure_filename
                ddir = os.path.join(jdir, "data")
                os.makedirs(ddir)
                for key, label, allow_gz in (("reference", "Reference FASTA", False),
                                             ("reads", "Reads FASTQ", True),
                                             ("truth", "Truth VCF", False)):
                    f = files.get(key)
                    if f is None or not f.filename:
                        continue
                    name = secure_filename(f.filename) or key
                    if key == "reference" and name.lower().endswith(".gz"):
                        errors.append("Reference FASTA must be uncompressed (gunzip it first).")
                        continue
                    path = os.path.join(ddir, f"{key}_{name}")
                    f.save(path)
                    p[f"{key}_file"] = f"{key}_{name}"
                    p[f"{key}_name"] = f.filename
                    if key == "reference":
                        if os.path.getsize(path) > max_ref_mb * 1024 * 1024:
                            errors.append(f"Reference is larger than {max_ref_mb} MB "
                                          f"(index is built in RAM; see docs/HARDWARE_OPTIMIZATION.md).")
                        _check_head(path, b">", label, errors)
                    elif key == "reads":
                        _check_head(path, b"@", label, errors, allow_gz=True)
                    else:
                        _check_head(path, b"#", label, errors)
                if errors:
                    raise ValueError("invalid upload")
            name = (form.get("name") or "").strip()[:80] or (
                "Demo run" if source == "demo" else f"Alignment of {p.get('reads_name', 'reads')}")
            job = manager.submit(jid, jdir, "align", name, p)
            return job, []
        except ValueError:
            shutil.rmtree(jdir, ignore_errors=True)
            return None, errors
        except Exception:
            shutil.rmtree(jdir, ignore_errors=True)
            raise

    # ---------------------------------------------------------------- pages
    @app.route("/")
    def dashboard():
        jobs = manager.list()
        return render_template("index.html", page="dashboard", counts=manager.counts(),
                               recent=[j.public(False) for j in jobs[:6]])

    @app.route("/new")
    def new_job():
        return render_template("new_job.html", page="new")

    @app.route("/jobs")
    def jobs_page():
        return render_template("jobs.html", page="jobs",
                               jobs=[j.public(False) for j in manager.list()])

    @app.route("/jobs/<jid>")
    def job_page(jid):
        job = manager.get(jid)
        if job is None:
            abort(404)
        return render_template("job.html", page="jobs", job=job.public())

    @app.route("/benchmark")
    def benchmark_page():
        runs = [j.public(False) for j in manager.list() if j.kind == "benchmark"]
        return render_template("benchmark.html", page="benchmark", runs=runs)

    @app.route("/about")
    def about():
        return render_template("about.html", page="about")

    @app.route("/jobs/<jid>/delete", methods=["POST"])
    def delete_job(jid):
        from flask import redirect
        if not manager.delete(jid):
            abort(409 if manager.get(jid) else 404)
        return redirect(url_for("jobs_page"))

    @app.route("/jobs/<jid>/download/<name>")
    def download(jid, name):
        job = manager.get(jid)
        if job is None or not job.result:
            abort(404)
        allowed = {f["name"] for f in job.result.get("files", [])}
        if name not in allowed:
            abort(404)
        for sub in ("results", ""):
            d = os.path.join(job.dir, sub)
            if os.path.isfile(os.path.join(d, name)):
                return send_from_directory(d, name, as_attachment=True)
        abort(404)

    @app.route("/jobs/<jid>/log")
    def job_log(jid):
        job = manager.get(jid)
        path = os.path.join(job.dir, "error.log") if job else ""
        if not os.path.isfile(path):
            abort(404)
        return send_from_directory(job.dir, "error.log", mimetype="text/plain")

    # ------------------------------------------------------------------ API
    @app.route("/healthz")
    def healthz():
        return jsonify(status="ok", version=__version__)

    @app.route("/api/system")
    def api_system():
        return jsonify({**sysinfo, "jobs": manager.counts()})

    @app.route("/api/jobs", methods=["GET"])
    def api_jobs():
        return jsonify([j.public(False) for j in manager.list()])

    @app.route("/api/jobs", methods=["POST"])
    def api_create_job():
        job, errors = _make_align_job(request.form, request.files)
        if errors:
            return jsonify(errors=errors), 400
        return jsonify(id=job.id, url=url_for("job_page", jid=job.id)), 201

    @app.route("/api/benchmark", methods=["POST"])
    def api_create_benchmark():
        errors: List[str] = []
        backend = request.form.get("backend", "auto")
        if backend not in ("auto", "cpu") + (("cuda",) if "cuda" in sysinfo["backends"] else ()):
            errors.append("Unsupported backend.")
        p = {"backend": backend,
             "genome_size": _num(request.form, "genome_size", 200_000, 20_000, 2_000_000, int, "Genome size", errors),
             "coverage": _num(request.form, "coverage", 10, 2, 50, float, "Coverage", errors)}
        if errors:
            return jsonify(errors=errors), 400
        jid, jdir = manager.new_dir()
        job = manager.submit(jid, jdir, "benchmark", "Benchmark", p)
        return jsonify(id=job.id, url=url_for("job_page", jid=job.id)), 201

    @app.route("/api/jobs/<jid>")
    def api_job(jid):
        job = manager.get(jid)
        if job is None:
            return jsonify(error="job not found"), 404
        return jsonify(job.public())

    @app.route("/api/jobs/<jid>/variants")
    def api_variants(jid):
        job = manager.get(jid)
        path = os.path.join(job.dir, "results", "variants.vcf") if job else ""
        if not os.path.isfile(path):
            return jsonify(error="no variants available"), 404
        try:
            offset = max(0, int(request.args.get("offset", 0)))
            limit = min(200, max(1, int(request.args.get("limit", 25))))
        except ValueError:
            return jsonify(error="bad paging parameters"), 400
        vtype = request.args.get("type", "all")
        query = (request.args.get("q") or "").strip().lower()
        total, rows = 0, []
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 8:
                    continue
                ref, alt = f[3], f[4]
                t = "SNP" if len(ref) == len(alt) else ("INS" if len(alt) > len(ref) else "DEL")
                if vtype != "all" and t != vtype:
                    continue
                if query and query not in f"{f[0]} {f[1]} {ref} {alt}".lower():
                    continue
                if offset <= total < offset + limit:
                    info = dict(kv.split("=", 1) for kv in f[7].split(";") if "=" in kv)
                    rows.append({"chrom": f[0], "pos": int(f[1]), "ref": ref, "alt": alt, "type": t,
                                 "qual": f[5], "dp": info.get("DP"), "ac": info.get("AC"),
                                 "af": info.get("AF"), "gt": f[9].split(":")[0] if len(f) > 9 else ""})
                total += 1
        return jsonify(total=total, offset=offset, limit=limit, rows=rows)

    @app.route("/api/jobs/<jid>/alignments")
    def api_alignments(jid):
        job = manager.get(jid)
        path = os.path.join(job.dir, "results", "alignments.tsv") if job else ""
        if not os.path.isfile(path):
            return jsonify(error="no alignments available"), 404
        rows = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) >= 6:
                    rows.append({"read": f[0], "contig": f[1], "pos": int(f[2]), "strand": f[3],
                                 "score": int(f[4]), "cigar": f[5]})
                if len(rows) >= 50:
                    break
        return jsonify(rows=rows)

    # --------------------------------------------------------------- errors
    def _wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(404)
    def _404(e):
        if _wants_json():
            return jsonify(error="not found"), 404
        return render_template("error.html", code=404, title="Page not found",
                               message="That page or job does not exist."), 404

    @app.errorhandler(409)
    def _409(e):
        return render_template("error.html", code=409, title="Job is still active",
                               message="A queued or running job cannot be deleted yet."), 409

    @app.errorhandler(413)
    def _413(e):
        msg = f"Upload too large (limit {max_upload_mb} MB; raise GPUALIGN_MAX_UPLOAD_MB)."
        if _wants_json():
            return jsonify(errors=[msg]), 413
        return render_template("error.html", code=413, title="Upload too large", message=msg), 413

    @app.errorhandler(500)
    def _500(e):
        if _wants_json():
            return jsonify(errors=["Internal server error."]), 500
        return render_template("error.html", code=500, title="Something went wrong",
                               message="The server hit an unexpected error."), 500

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, data_dir: Optional[str] = None):
    app = create_app(data_dir)
    max_body = app.config["MAX_CONTENT_LENGTH"] + 1024 * 1024
    print(f"gpualign web UI -> http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}  (Ctrl+C to stop)")
    try:
        from waitress import serve as wserve
        wserve(app, host=host, port=port, threads=8, max_request_body_size=max_body,
               channel_timeout=600)
    except ImportError:
        app.run(host=host, port=port, threaded=True)
