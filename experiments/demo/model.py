"""Standalone GPT model for the demo — dense weights only, inference only.

Mirrors the architecture in experiments/train_weightparam.py but strips out
QAT, distributed training, and structured-weight variants we don't need to
load post-training checkpoints (which are all dense).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StructuredLinear(nn.Module):
    """Dense-only linear, parameter-name-compatible with the training script."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert not bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        return freqs.cos()[None, None, :, :].to(dtype), freqs.sin()[None, None, :, :].to(dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Manual softmax attention — returns (output, attn_weights).
    attn_weights has shape (B, num_heads, T, T) for visualization."""
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 pos_enc: str = "partial", rope_dims: int = 32):
        super().__init__()
        if dim % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("invalid head config")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = StructuredLinear(dim, dim)
        self.c_k = StructuredLinear(dim, kv_dim)
        self.c_v = StructuredLinear(dim, kv_dim)
        self.proj = StructuredLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.pos_enc = pos_enc

        if pos_enc == "rope":
            self.rope_dims = self.head_dim
            self.rotary = Rotary(self.head_dim, base=rope_base)
        elif pos_enc == "partial":
            self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim // 2
            self.rotary = Rotary(self.rope_dims, base=rope_base)
        else:
            self.rope_dims = 0
            self.rotary = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        if self.pos_enc == "rope":
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        elif self.pos_enc == "partial":
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            rd = self.rope_dims
            q = torch.cat([apply_rotary_emb(q[..., :rd], cos, sin), q[..., rd:]], dim=-1)
            k = torch.cat([apply_rotary_emb(k[..., :rd], cos, sin), k[..., rd:]], dim=-1)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # Expand kv-heads to match q-heads (GQA/MQA).
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal
        attn = F.softmax(scores, dim=-1)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), attn


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = StructuredLinear(dim, hidden)
        self.proj = StructuredLinear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 pos_enc: str = "partial", rope_dims: int = 32):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        pos_enc=pos_enc, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, attn_weights = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x, attn_weights


class GPT(nn.Module):
    def __init__(self,
                 vocab_size: int = 1024,
                 num_layers: int = 9,
                 model_dim: int = 512,
                 num_heads: int = 8,
                 num_kv_heads: int = 4,
                 mlp_mult: int = 2,
                 tie_embeddings: bool = True,
                 logit_softcap: float = 30.0,
                 rope_base: float = 10000.0,
                 qk_gain_init: float = 1.5,
                 pos_enc: str = "partial",
                 rope_dims: int = 32,
                 max_seq_len: int = 1024):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.pos_enc = pos_enc
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim) if pos_enc == "learned" else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  pos_enc=pos_enc, rope_dims=rope_dims)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else StructuredLinear(model_dim, vocab_size)

    def forward_with_intermediates(self, input_ids: Tensor):
        """Returns (logits, attn_per_layer, hidden_norms_per_layer).
        - logits:       (B, T, vocab_size)
        - attn_per_layer:    list of (B, num_heads, T, T), length num_layers
        - hidden_norms:      list of (B, T), length num_layers (L2 norm per token)
        """
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            seqlen = input_ids.size(-1)
            positions = torch.arange(seqlen, device=input_ids.device)
            x = x + self.pos_emb(positions)[None, :, :]
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        attn_per_layer: list[Tensor] = []
        hidden_norms: list[Tensor] = []
        skips = []
        for i in range(self.num_encoder_layers):
            x, attn = self.blocks[i](x, x0)
            attn_per_layer.append(attn)
            hidden_norms.append(x.float().norm(dim=-1))
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, attn = self.blocks[self.num_encoder_layers + i](x, x0)
            attn_per_layer.append(attn)
            hidden_norms.append(x.float().norm(dim=-1))

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        # Match the training-time soft-cap so logits are comparable to val-time outputs.
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits, attn_per_layer, hidden_norms

    def forward(self, input_ids: Tensor) -> Tensor:
        logits, _, _ = self.forward_with_intermediates(input_ids)
        return logits
