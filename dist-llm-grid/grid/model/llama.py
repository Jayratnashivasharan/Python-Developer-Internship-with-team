"""Llama-architecture decoder with tensor parallelism and a paged KV cache.

The forward pass works on a *flattened ragged batch*: tokens of all sequences in the
step are concatenated into one [T, hidden] tensor (prefill chunks and single decode
tokens mix freely - this is what makes iteration-level / continuous batching possible).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from grid.model.config import ModelConfig
from grid.model.parallel import ColumnParallelLinear, RowParallelLinear, TPContext


@dataclass
class BatchMeta:
    """Everything the model needs to know about the ragged batch of one step."""

    positions: torch.Tensor          # [T] absolute position of every token
    write_slots: torch.Tensor        # [T] flat KV slot where each token's K/V is stored
    new_lens: list[int]              # tokens contributed by each sequence this step
    ctx_slots: list[torch.Tensor]    # per sequence: flat KV slots of its *entire* context


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[:, None, :].to(q.dtype), emb.sin()[:, None, :].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.hd = cfg.head_dim
        self.n_heads = cfg.num_attention_heads // tp.world_size
        self.n_kv = cfg.num_key_value_heads // tp.world_size
        self.groups = self.n_heads // self.n_kv
        h = cfg.hidden_size
        self.q_proj = ColumnParallelLinear(h, cfg.num_attention_heads * self.hd, tp)
        self.k_proj = ColumnParallelLinear(h, cfg.num_key_value_heads * self.hd, tp)
        self.v_proj = ColumnParallelLinear(h, cfg.num_key_value_heads * self.hd, tp)
        self.o_proj = RowParallelLinear(cfg.num_attention_heads * self.hd, h, tp)
        self.rotary = Rotary(self.hd, cfg.rope_theta)

    def forward(self, x, meta: BatchMeta, k_cache: torch.Tensor, v_cache: torch.Tensor):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_heads, self.hd)
        k = self.k_proj(x).view(T, self.n_kv, self.hd)
        v = self.v_proj(x).view(T, self.n_kv, self.hd)
        q, k = self.rotary(q, k, meta.positions)

        # k_cache / v_cache: [num_blocks * block_size, n_kv, hd]  (paged storage, flat view)
        k_cache[meta.write_slots] = k.to(k_cache.dtype)
        v_cache[meta.write_slots] = v.to(v_cache.dtype)

        outs, off = [], 0
        for n_new, slots in zip(meta.new_lens, meta.ctx_slots):
            qi = q[off:off + n_new].transpose(0, 1)                       # [H, n, hd]
            ki = k_cache[slots].transpose(0, 1)                           # [Hkv, L, hd]
            vi = v_cache[slots].transpose(0, 1)
            if self.groups > 1:
                ki = ki.repeat_interleave(self.groups, dim=0)
                vi = vi.repeat_interleave(self.groups, dim=0)
            L = ki.shape[1]
            if n_new == 1:
                mask = None  # a decode token may attend to the whole context
            else:
                q_pos = torch.arange(L - n_new, L, device=x.device)[:, None]
                mask = torch.arange(L, device=x.device)[None, :] <= q_pos  # causal, aligned to the end
            o = F.scaled_dot_product_attention(qi.to(ki.dtype), ki, vi, attn_mask=mask)
            outs.append(o.transpose(0, 1).reshape(n_new, self.n_heads * self.hd))
            off += n_new
        return self.o_proj(torch.cat(outs, dim=0).to(x.dtype))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, tp)
        self.up_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, tp)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, tp)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.self_attn = Attention(cfg, tp)
        self.mlp = MLP(cfg, tp)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, meta, k_cache, v_cache):
        x = x + self.self_attn(self.input_layernorm(x), meta, k_cache, v_cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_tokens.weight.requires_grad_(False)
        self.layers = nn.ModuleList([Block(cfg, tp) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LlamaTP(nn.Module):
    """Parameter names mirror HuggingFace Llama so real checkpoints load unchanged."""

    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        cfg.validate_tp(tp.world_size)
        self.cfg, self.tp = cfg, tp
        self.model = _Backbone(cfg, tp)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight.requires_grad_(False)

    def kv_shape(self, num_blocks: int, block_size: int):
        return (num_blocks * block_size, self.cfg.num_key_value_heads // self.tp.world_size, self.cfg.head_dim)

    @torch.inference_mode()
    def forward(self, token_ids, meta: BatchMeta, kv_cache: list[tuple[torch.Tensor, torch.Tensor]]):
        """Returns logits [num_seqs, vocab] for the last token of every sequence."""
        x = self.model.embed_tokens(token_ids)
        for layer, (kc, vc) in zip(self.model.layers, kv_cache):
            x = layer(x, meta, kc, vc)
        last = torch.cumsum(torch.tensor(meta.new_lens, device=x.device), 0) - 1
        x = self.model.norm(x[last])
        return self.lm_head(x).float()
