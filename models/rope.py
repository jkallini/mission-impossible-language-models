# rope.py
# Author: Róbert Csordás

import torch
from typing import Tuple, Optional


class RotaryPosEncoding(torch.nn.Module):
    # RoPE based on: https://www.kaggle.com/code/aeryss/rotary-postitional-encoding-rope-pytorch
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("RoPE can only be used with an even number of dimensions")

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rot(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, seq_dim: int, offset: int) -> torch.Tensor:
        sin = sin.narrow(seq_dim, offset, x.shape[seq_dim])
        cos = cos.narrow(seq_dim, offset, x.shape[seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                             seq_dim: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(k, sin, cos, seq_dim, 0)

    def get(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        enable_cache = t is None

        if (not enable_cache) or (seq_len > self.seq_len_cached):
            self.seq_len_cached = seq_len
            if t is None:
                t = torch.arange(x.shape[self.seq_dim], device=x.device)

            t = t.type_as(self.inv_freq)

            freqs = torch.einsum("...i,j->...ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            # support batch.
            tgt_shape[0] = -1

            cos = emb.cos().view(*tgt_shape)
            sin = emb.sin().view(*tgt_shape)

            if enable_cache:
                self.cos_cached = cos
                self.sin_cached = sin
            else:
                return sin, cos

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k, t)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)