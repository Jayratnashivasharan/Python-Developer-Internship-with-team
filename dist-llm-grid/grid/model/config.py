from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Llama-family hyper-parameters (field names follow HuggingFace config.json)."""

    vocab_size: int = 259
    hidden_size: int = 128
    intermediate_size: int = 352
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 256
    eos_token_id: int = 257
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate_tp(self, world_size: int) -> None:
        for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
            v = getattr(self, name)
            if v % world_size:
                raise ValueError(f"{name}={v} is not divisible by tensor-parallel size {world_size}")

    @classmethod
    def from_dir(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            raw = json.load(f)
        known = cls.__dataclass_fields__.keys()
        kwargs = {k: v for k, v in raw.items() if k in known}
        if isinstance(kwargs.get("eos_token_id"), list):
            kwargs["eos_token_id"] = kwargs["eos_token_id"][0]
        cfg = cls(**kwargs)
        if "num_key_value_heads" not in raw:
            cfg.num_key_value_heads = cfg.num_attention_heads
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({**self.__dict__, "model_type": "llama"}, f, indent=2)
