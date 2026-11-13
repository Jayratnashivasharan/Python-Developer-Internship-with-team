"""Tensor-parallel primitives (Megatron-LM style).

* ``ColumnParallelLinear``  - output features are split across ranks; no communication.
* ``RowParallelLinear``     - input features are split; partial results are summed with
                              an all-reduce.

For one transformer block this costs exactly **two all-reduces** (after the attention
output projection and after the MLP down projection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class TPContext:
    rank: int = 0
    world_size: int = 1
    group: object | None = None
    comm_seconds: float = field(default=0.0)  # accumulated time inside collectives
    sync_for_timing: bool = False  # CUDA: synchronise so timings are real

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return x
        t0 = time.perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        if self.sync_for_timing and x.is_cuda:
            torch.cuda.synchronize()
        self.comm_seconds += time.perf_counter() - t0
        return x

    def pop_comm_ms(self) -> float:
        ms = self.comm_seconds * 1000.0
        self.comm_seconds = 0.0
        return ms


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp: TPContext):
        super().__init__()
        assert out_features % tp.world_size == 0
        self.weight = nn.Parameter(torch.empty(out_features // tp.world_size, in_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp: TPContext):
        super().__init__()
        assert in_features % tp.world_size == 0
        self.tp = tp
        self.weight = nn.Parameter(torch.empty(out_features, in_features // tp.world_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tp.all_reduce(torch.nn.functional.linear(x, self.weight))
