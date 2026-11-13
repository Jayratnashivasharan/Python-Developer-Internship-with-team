"""Shard-aware weight loading from HuggingFace ``safetensors`` checkpoints.

``safetensors.safe_open`` memory-maps the file, and ``get_slice`` lets us read *only the
rows/columns this rank owns*. A node therefore never materialises the full model in RAM,
which is what lets commodity machines host a model larger than any single node.

Sharding rules (Megatron-LM):
    q/k/v/gate/up_proj  -> split rows    (dim 0)  [column-parallel]
    o/down_proj         -> split columns (dim 1)  [row-parallel]
    embeddings, norms, lm_head -> replicated
"""
from __future__ import annotations

import glob
import json
import logging
import os

import torch
from safetensors import safe_open

from grid.model.llama import LlamaTP

log = logging.getLogger("grid.weights")

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

_COL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW = ("o_proj", "down_proj")


def _weight_map(model_dir: str) -> dict[str, str]:
    """tensor name -> safetensors file (supports single-file and sharded checkpoints)."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return {k: os.path.join(model_dir, v) for k, v in json.load(f)["weight_map"].items()}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    out: dict[str, str] = {}
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            out.update({k: fp for k in f.keys()})
    return out


def _read(handles: dict, path: str, name: str, dim: int | None, rank: int, world: int) -> torch.Tensor:
    if path not in handles:
        handles[path] = safe_open(path, framework="pt")  # mmap, nothing read yet
    sl = handles[path].get_slice(name)
    shape = sl.get_shape()
    if dim is None:
        return sl[:] if len(shape) else sl[...]
    size = shape[dim]
    assert size % world == 0, f"{name}: dim {dim} of size {size} not divisible by {world}"
    step = size // world
    lo, hi = rank * step, (rank + 1) * step
    return sl[lo:hi] if dim == 0 else sl[:, lo:hi]


def load_sharded(model: LlamaTP, model_dir: str, device: torch.device, dtype: torch.dtype) -> int:
    """Fill ``model`` with this rank's shard. Returns number of bytes loaded."""
    rank, world = model.tp.rank, model.tp.world_size
    wmap = _weight_map(model_dir)
    handles: dict = {}
    loaded = 0
    for name, param in model.named_parameters():
        src = name
        if src not in wmap and name == "lm_head.weight":
            src = "model.embed_tokens.weight"  # tied embeddings
        if src not in wmap:
            raise KeyError(f"weight '{name}' missing from checkpoint")
        dim = 0 if any(c in name for c in _COL) else 1 if any(r in name for r in _ROW) else None
        t = _read(handles, wmap[src], src, dim, rank, world)
        if tuple(t.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint shard {tuple(t.shape)} != model {tuple(param.shape)}")
        param.data = t.to(device=device, dtype=dtype).contiguous()
        loaded += param.numel() * param.element_size()
    log.info("rank %d/%d loaded %.1f MiB of weights", rank, world, loaded / 2**20)
    return loaded
