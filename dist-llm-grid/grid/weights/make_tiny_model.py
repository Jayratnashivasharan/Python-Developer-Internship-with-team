"""Create a small randomly-initialised Llama checkpoint (safetensors + config.json).

Used for development, CI and demos so nothing has to be downloaded. The model produces
gibberish text - it exists to exercise the *systems* (sharding, KV cache, batching).
Point GRID_MODEL_DIR at a real HuggingFace Llama directory for meaningful output.

    python -m grid.weights.make_tiny_model --out models/tiny-llama
"""
from __future__ import annotations

import argparse
import os

import torch
from safetensors.torch import save_file

from grid.model.config import ModelConfig


def make(out: str, cfg: ModelConfig, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    h, i, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim

    def w(*shape, std=0.05):
        return torch.randn(*shape, generator=g) * std

    t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": w(cfg.vocab_size, h, std=0.5),
        "model.norm.weight": torch.ones(h),
        "lm_head.weight": w(cfg.vocab_size, h, std=0.5),
    }
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}."
        t[p + "self_attn.q_proj.weight"] = w(cfg.num_attention_heads * hd, h)
        t[p + "self_attn.k_proj.weight"] = w(cfg.num_key_value_heads * hd, h)
        t[p + "self_attn.v_proj.weight"] = w(cfg.num_key_value_heads * hd, h)
        t[p + "self_attn.o_proj.weight"] = w(h, cfg.num_attention_heads * hd)
        t[p + "mlp.gate_proj.weight"] = w(i, h)
        t[p + "mlp.up_proj.weight"] = w(i, h)
        t[p + "mlp.down_proj.weight"] = w(h, i)
        t[p + "input_layernorm.weight"] = torch.ones(h)
        t[p + "post_attention_layernorm.weight"] = torch.ones(h)
    os.makedirs(out, exist_ok=True)
    save_file(t, os.path.join(out, "model.safetensors"))
    cfg.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/tiny-llama")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=352)
    a = ap.parse_args()
    cfg = ModelConfig(hidden_size=a.hidden, num_hidden_layers=a.layers, num_attention_heads=a.heads,
                      num_key_value_heads=a.kv_heads, intermediate_size=a.ffn)
    print("wrote", make(a.out, cfg))
