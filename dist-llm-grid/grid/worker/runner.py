"""ModelRunner: owns one tensor-parallel shard, its KV-cache tensors and sampling."""
from __future__ import annotations

import logging
import time

import torch

from grid.config import Settings
from grid.model.config import ModelConfig
from grid.model.llama import BatchMeta, LlamaTP
from grid.model.parallel import TPContext
from grid.weights.loader import DTYPES, load_sharded

log = logging.getLogger("grid.runner")


def sample(logits: torch.Tensor, temperature: list[float], top_k: list[int]) -> list[int]:
    out = []
    for i in range(logits.shape[0]):
        lg = logits[i]
        if temperature[i] <= 0:
            out.append(int(torch.argmax(lg)))
            continue
        lg = lg / temperature[i]
        if top_k[i] > 0:
            kth = torch.topk(lg, min(top_k[i], lg.numel())).values[-1]
            lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
        out.append(int(torch.multinomial(torch.softmax(lg, -1), 1)))
    return out


class ModelRunner:
    def __init__(self, settings: Settings, tp: TPContext, device: torch.device):
        self.s, self.tp, self.device = settings, tp, device
        self.dtype = DTYPES[settings.dtype]
        cfg = ModelConfig.from_dir(settings.model_dir)
        self.model = LlamaTP(cfg, tp).to(self.dtype)
        load_sharded(self.model, settings.model_dir, device, self.dtype)
        self.model.to(device).eval()
        shape = self.model.kv_shape(settings.num_blocks, settings.block_size)
        self.kv = [
            (torch.zeros(shape, dtype=self.dtype, device=device), torch.zeros(shape, dtype=self.dtype, device=device))
            for _ in range(cfg.num_hidden_layers)
        ]
        kv_mb = 2 * cfg.num_hidden_layers * torch.Size(shape).numel() * self.kv[0][0].element_size() / 2**20
        log.info("rank %d KV cache: %d blocks x %d tokens = %.1f MiB", tp.rank, settings.num_blocks,
                 settings.block_size, kv_mb)

    def build_meta(self, seqs) -> tuple[torch.Tensor, BatchMeta]:
        bs, dev = self.s.block_size, self.device
        tokens, positions, write_slots, new_lens, ctx_slots = [], [], [], [], []
        ar = torch.arange(bs)
        for s in seqs:
            n = len(s.token_ids)
            bt = torch.tensor(list(s.block_table), dtype=torch.long)
            ctx_len = s.start_pos + n
            flat = (bt[:, None] * bs + ar[None, :]).reshape(-1)[:ctx_len]   # slots of the whole context
            tokens.extend(s.token_ids)
            positions.extend(range(s.start_pos, ctx_len))
            write_slots.append(flat[s.start_pos:ctx_len])
            ctx_slots.append(flat.to(dev))
            new_lens.append(n)
        meta = BatchMeta(
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            write_slots=torch.cat(write_slots).to(dev),
            new_lens=new_lens,
            ctx_slots=ctx_slots,
        )
        return torch.tensor(tokens, dtype=torch.long, device=dev), meta

    def step(self, req) -> tuple[list[int], float, float]:
        t0 = time.perf_counter()
        self.tp.comm_seconds = 0.0
        tokens, meta = self.build_meta(req.seqs)
        logits = self.model(tokens, meta, self.kv)
        # Every rank holds identical logits after the final all-reduce; only rank 0 needs to sample.
        nxt: list[int] = []
        if self.tp.rank == 0:
            nxt = sample(logits.cpu(), [s.temperature for s in req.seqs], [s.top_k for s in req.seqs])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return nxt, (time.perf_counter() - t0) * 1000.0, self.tp.pop_comm_ms()
