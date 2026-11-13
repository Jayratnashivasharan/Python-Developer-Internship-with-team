from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple
import time


def _header(contigs: Dict[str, int], source: str, sample: str = None):
    h = ["##fileformat=VCFv4.2", f"##fileDate={time.strftime('%Y%m%d')}", f"##source={source}"]
    h += [f"##contig=<ID={k},length={v}>" for k, v in contigs.items()]
    return h


def write_vcf(path: str, variants: List[dict], index, sample="SAMPLE"):
    contigs = {c["name"]: c["length"] for c in index.contigs}
    with open(path, "w") as fh:
        for l in _header(contigs, "gpualign-1.0"):
            fh.write(l + "\n")
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth at site">\n'
                 '##INFO=<ID=AC,Number=1,Type=Integer,Description="Reads supporting ALT">\n'
                 '##INFO=<ID=AF,Number=1,Type=Float,Description="ALT read fraction">\n'
                 '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                 '##FORMAT=<ID=AD,Number=1,Type=Integer,Description="ALT read count">\n')
        fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
        for v in variants:
            name, local = index.to_local(v["gpos"])
            fh.write(f"{name}\t{local + 1}\t.\t{v['ref']}\t{v['alt']}\t{v['qual']}\tPASS\t"
                     f"DP={v['dp']};AC={v['ac']};AF={v['af']:.3f}\tGT:AD\t{v['gt']}:{v['ac']}\n")


def write_truth_vcf(path: str, truth: Iterable[Tuple[str, int, str, str]], contigs: Dict[str, int]):
    with open(path, "w") as fh:
        for l in _header(contigs, "gpualign-simulator"):
            fh.write(l + "\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, r, a in truth:
            fh.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t.\n")


def read_vcf_keys(path: str) -> Set[Tuple[str, int, str, str]]:
    keys = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            keys.add((f[0], int(f[1]), f[3], f[4]))
    return keys
