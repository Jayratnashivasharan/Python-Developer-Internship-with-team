"""Central runtime configuration, read from environment variables.

Every process (gateway/master and workers) builds the same ``Settings`` object so
that KV-cache geometry and the worker list can never drift apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # --- model / weights -------------------------------------------------
    model_dir: str = field(default_factory=lambda: _env("GRID_MODEL_DIR", "models/tiny-llama"))
    tokenizer: str = field(default_factory=lambda: _env("GRID_TOKENIZER", "byte"))  # "byte" or HF path/name
    dtype: str = field(default_factory=lambda: _env("GRID_DTYPE", "float32"))

    # --- cluster ---------------------------------------------------------
    workers: list[str] = field(
        default_factory=lambda: [w for w in _env("GRID_WORKERS", "localhost:50051,localhost:50052").split(",") if w]
    )
    dist_backend: str = field(default_factory=lambda: _env("GRID_DIST_BACKEND", "auto"))  # auto|nccl|gloo
    master_addr: str = field(default_factory=lambda: _env("MASTER_ADDR", "127.0.0.1"))
    master_port: int = field(default_factory=lambda: int(_env("MASTER_PORT", "29500")))
    grpc_port: int = field(default_factory=lambda: int(_env("GRID_GRPC_PORT", "50051")))
    step_timeout_s: float = field(default_factory=lambda: float(_env("GRID_STEP_TIMEOUT", "60")))

    # --- paged KV cache (must be identical on master and workers) ---------
    block_size: int = field(default_factory=lambda: int(_env("GRID_BLOCK_SIZE", "16")))
    num_blocks: int = field(default_factory=lambda: int(_env("GRID_NUM_BLOCKS", "512")))

    # --- continuous batching ----------------------------------------------
    max_batch_seqs: int = field(default_factory=lambda: int(_env("GRID_MAX_BATCH_SEQS", "16")))
    max_prefill_tokens: int = field(default_factory=lambda: int(_env("GRID_MAX_PREFILL_TOKENS", "512")))
    max_queue: int = field(default_factory=lambda: int(_env("GRID_MAX_QUEUE", "256")))

    @property
    def world_size(self) -> int:
        return len(self.workers)
