"""Checks that sharding across N ranks reproduces the single-device logits (no data loss)."""
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers import logits_for, make_runner, seq

TOKS = [256, 11, 22, 33, 44, 55, 66, 77]


def _worker(rank, world, model_dir, store, out):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    r = make_runner(model_dir, rank, world)
    lg = logits_for(r, [seq(TOKS, 0, [0, 1])])
    if rank == 0:
        torch.save(lg, out)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_tp_equals_single_device(tiny_model, world):
    ref = logits_for(make_runner(tiny_model), [seq(TOKS, 0, [0, 1])])
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.pt")
        mp.spawn(_worker, args=(world, tiny_model, os.path.join(d, "store"), out), nprocs=world, join=True)
        got = torch.load(out)
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_each_rank_loads_only_its_shard(tiny_model):
    full = make_runner(tiny_model, 0, 1).model.model.layers[0].mlp.gate_proj.weight
    shard = make_runner(tiny_model, 1, 2).model.model.layers[0].mlp.gate_proj.weight
    assert shard.shape[0] == full.shape[0] // 2
    assert torch.equal(shard, full[full.shape[0] // 2:])
