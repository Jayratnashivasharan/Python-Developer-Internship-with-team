"""Automated Markdown/JSON report: run stats, variant summary, Ti/Tv, and
(optionally) precision/recall against a truth VCF."""
from __future__ import annotations
import json
from typing import Dict, List, Optional

from .vcf import read_vcf_keys

_TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


def _vtype(ref: str, alt: str) -> str:
    if len(ref) == len(alt) == 1:
        return "SNP"
    return "INS" if len(alt) > len(ref) else "DEL"


def summarize(variants: List[dict]) -> Dict:
    c = {"SNP": 0, "INS": 0, "DEL": 0}
    ti = tv = 0
    for v in variants:
        t = _vtype(v["ref"], v["alt"])
        c[t] += 1
        if t == "SNP":
            if (v["ref"], v["alt"]) in _TRANSITIONS:
                ti += 1
            else:
                tv += 1
    c["total"] = len(variants)
    c["ti_tv"] = round(ti / tv, 3) if tv else None
    return c


def evaluate_against_truth(vcf_path: str, truth_path: str) -> Dict:
    called, truth = read_vcf_keys(vcf_path), read_vcf_keys(truth_path)
    res = {}
    for label, sel in (("all", lambda k: True),
                       ("SNP", lambda k: _vtype(k[2], k[3]) == "SNP"),
                       ("INDEL", lambda k: _vtype(k[2], k[3]) != "SNP")):
        c = {k for k in called if sel(k)}
        t = {k for k in truth if sel(k)}
        tp = len(c & t)
        res[label] = {"truth": len(t), "called": len(c), "tp": tp,
                      "precision": round(tp / len(c), 4) if c else None,
                      "recall": round(tp / len(t), 4) if t else None}
    return res


def write_report(path_md: str, path_json: str, stats: Dict, variants: List[dict],
                 truth_eval: Optional[Dict] = None):
    summ = summarize(variants)
    with open(path_json, "w") as fh:
        json.dump({"run": stats, "variants": summ, "truth_evaluation": truth_eval}, fh, indent=2)
    lines = ["# gpualign run report", "", "## Run", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()]
    lines += ["", "## Variants", "", "| type | count |", "|---|---|"]
    lines += [f"| {k} | {summ[k]} |" for k in ("SNP", "INS", "DEL", "total")]
    lines.append(f"\nTi/Tv ratio: {summ['ti_tv']}")
    if truth_eval:
        lines += ["", "## Accuracy vs truth set", "", "| class | truth | called | TP | precision | recall |",
                  "|---|---|---|---|---|---|"]
        for k, v in truth_eval.items():
            lines.append(f"| {k} | {v['truth']} | {v['called']} | {v['tp']} | {v['precision']} | {v['recall']} |")
    with open(path_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
