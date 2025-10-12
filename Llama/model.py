# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

cfg = Config()


# =====================================================
# RMSNorm
# =====================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize by root mean square
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.scale


# =====================================================
# Rotary embeddings (RoPE)
# =====================================================
def fixed_pos_embedding(dim, max_seq_len=2048, base=10000):
    """
    Compute rotary position embeddings (sin, cos) for the first `dim` dimensions.
    Returns:
        sin, cos of shape (1, max_seq_len, 1, dim/2)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i , j -> i j", t, inv_freq)  # (seq, dim/2)
    sin = freqs.sin()[None, :, None, :]  # (1, seq, 1, dim/2)
    cos = freqs.cos()[None, :, None, :]
    return sin, cos


def apply_rotary(x, sin, cos):
    """
    Apply rotary position embedding to tensor x.
    x: (..., dim), sin/cos: broadcastable to x[..., :dim/2]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # rotate pairs (x_even, x_odd)
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)


# =====================================================
# Multi-Head Causal Self-Attention with RoPE
# =====================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_dim=None, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        assert self.rotary_dim <= self.head_dim, "rotary_dim must be <= head_dim"
        assert self.rotary_dim % 2 == 0, "rotary_dim must be even"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute causal mask (for efficiency)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=True)

    def forward(self, x, sin, cos):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)

        # Apply rotary embeddings to the first rotary_dim dims
        if self.rotary_dim > 0:
            sin = sin[:, :T, :, :]
            cos = cos[:, :T, :, :]
            q_rot, q_rest = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
            k_rot, k_rest = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
            q_rot = apply_rotary(q_rot, sin, cos)
            k_rot = apply_rotary(k_rot, sin, cos)
            q = torch.cat([q_rot, q_rest], dim=-1)
            k = torch.cat([k_rot, k_rest], dim=-1)

        # Scaled dot-product attention
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)

        # Apply causal mask
        mask = self.causal_mask[:, :, :T, :T].to(x.device)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.contiguous().view(B, T, D)
        return self.out_proj(out)


# =====================================================
# SwiGLU MLP
# =====================================================
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


# =====================================================
# Transformer Block
# =====================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_multiplier=cfg.ffn_multiplier, rotary_dim=None, max_seq_len=2048):
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rotary_dim=rotary_dim, max_seq_len=max_seq_len)
        self.rms2 = RMSNorm(d_model)
        hidden_dim = int(d_model * ffn_multiplier)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x, sin, cos):
        x = x + self.attn(self.rms1(x), sin, cos)
        x = x + self.mlp(self.rms2(x))
        return x


# =====================================================
# Full Mini-LLaMA Model with proper initialization
# =====================================================
class MiniLLaMA(nn.Module):
    def __init__(self, vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                 block_size=cfg.block_size, rotary_dim=cfg.rotary_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        # -----------------------------
        # Embedding layer
        # -----------------------------
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # -----------------------------
        # Transformer layers
        # -----------------------------
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, cfg.ffn_multiplier, rotary_dim=rotary_dim, max_seq_len=block_size)
            for _ in range(n_layers)
        ])

        # -----------------------------
        # Final normalization and output head
        # -----------------------------
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # -----------------------------
        # Precompute RoPE sin/cos
        # -----------------------------
        sin, cos = fixed_pos_embedding(rotary_dim, max_seq_len=block_size)
        self.register_buffer("sin", sin, persistent=True)
        self.register_buffer("cos", cos, persistent=True)

        # -----------------------------
        # Apply LLaMA-style initialization
        # -----------------------------
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        input_ids: (B, T)
        Returns logits of shape (B, T, vocab_size)
        """
        B, T = input_ids.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")

        x = self.token_emb(input_ids)  # (B, T, D)
        sin = self.sin.to(device=x.device, dtype=x.dtype)
        cos = self.cos.to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, sin, cos)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

