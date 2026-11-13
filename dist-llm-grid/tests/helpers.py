"""Shared helpers: run a prompt through a LlamaTP shard with the paged cache."""
from types import SimpleNamespace

import torch

from grid.config import Settings
from grid.model.parallel import TPContext
from grid.worker.runner import ModelRunner

BS, NB = 4, 64


def make_runner(model_dir, rank=0, world=1):
    s = Settings(model_dir=model_dir, block_size=BS, num_blocks=NB, workers=["x"] * world)
    return ModelRunner(s, TPContext(rank=rank, world_size=world), torch.device("cpu"))


def seq(tokens, start, blocks, temp=0.0):
    return SimpleNamespace(token_ids=tokens, start_pos=start, block_table=blocks, temperature=temp, top_k=0)


def logits_for(runner, seqs):
    toks, meta = runner.build_meta(seqs)
    return runner.model(toks, meta, runner.kv)
