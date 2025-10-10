# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs import Config
cfg = Config()

# ------------------------
# RMSNorm
# ------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: (B, T, D)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.scale

# ------------------------
# Rotary embeddings helpers (RoPE)
# ------------------------
def fixed_pos_embedding(dim, max_seq_len=2048, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).type_as(inv_freq)
    freqs = torch.einsum("i , j -> i j", t, inv_freq)  # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, dim)
    sin = emb.sin()[None, :, None, :]  # (1, seq, 1, dim)
    cos = emb.cos()[None, :, None, :]
    return sin, cos

def apply_rotary(x, sin, cos):
    # x: (..., dim) where dim is even
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)

# ------------------------
# Multi-Head Causal Attention with RoPE
# ------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_dim=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, sin, cos, attention_mask=None, past_kv=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # apply rotary to first rotary_dim of each head
        if self.rotary_dim > 0:
            q_rot = q[..., :self.rotary_dim]
            k_rot = k[..., :self.rotary_dim]
            q_rest = q[..., self.rotary_dim:]
            k_rest = k[..., self.rotary_dim:]
            q_rot = apply_rotary(q_rot, sin[:, :T, :, :][:, :T, :, :], cos[:, :T, :, :][:, :T, :, :])
            k_rot = apply_rotary(k_rot, sin[:, :T, :, :][:, :T, :, :], cos[:, :T, :, :][:, :T, :, :])
            q = torch.cat([q_rot, q_rest], dim=-1)
            k = torch.cat([k_rot, k_rest], dim=-1)

        # causal attention
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        # mask future tokens
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.contiguous().view(B, T, D)
        return self.out_proj(out)

# ------------------------
# SwiGLU MLP
# ------------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)
    def forward(self, x):
        a = self.w1(x)
        b = self.w2(x)
        return self.w3(a * F.silu(b))  # silu == swish

# ------------------------
# Transformer block
# ------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_multiplier=cfg.ffn_multiplier, rotary_dim=None):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rotary_dim=rotary_dim)
        self.rms2 = RMSNorm(d_model)
        hidden_dim = int(d_model * ffn_multiplier)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x, sin, cos):
        x = x + self.attn(self.rms1(x), sin, cos)
        x = x + self.mlp(self.rms2(x))
        return x

# ------------------------
# Full Model
# ------------------------
class MiniLLaMA(nn.Module):
    def __init__(self, vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, block_size=cfg.block_size, rotary_dim=cfg.rotary_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, cfg.ffn_multiplier, rotary_dim=rotary_dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.block_size = block_size
        self.max_seq = block_size

        # precompute sin/cos for RoPE up to block_size
        sin, cos = fixed_pos_embedding(rotary_dim, max_seq_len=block_size)
        # save as buffers for use in forward (sin, cos shapes: (1, seq, 1, dim))
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

    def forward(self, input_ids):
        # input_ids: (B, T)
        x = self.token_emb(input_ids)  # (B,T,D)
        sin = self.sin.to(x.dtype)
        cos = self.cos.to(x.dtype)
        for layer in self.layers:
            x = layer(x, sin, cos)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
