# Copyright (c) Sophont, Inc
#
# This source code is licensed under the Apache License, Version 2.0

"""
DDPM decoder components for diffusion-based masked autoencoder.

NoiseSchedule: cosine beta schedule with forward/reverse diffusion
TimestepEmbedding: sinusoidal timestep encoding
DDPMDecoder: lightweight transformer decoder conditioned on encoder latents
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .modules import Block, LayerNorm


class NoiseSchedule(nn.Module):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""

    def __init__(self, T: int = 1000, s: float = 0.008):
        super().__init__()
        self.T = T

        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]

        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = betas.clamp(max=0.999)
        alphas = 1 - betas

        alphas_cumprod = alphas_cumprod[1:]  # drop the t=0 entry, now length T

        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # posterior q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef1 = (alphas.sqrt() * (1 - alphas_cumprod_prev)) / (1 - alphas_cumprod)
        posterior_mean_coef2 = (alphas_cumprod_prev.sqrt() * betas) / (1 - alphas_cumprod)

        for name, val in [
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_cumprod", alphas_cumprod),
            ("sqrt_alphas_cumprod", sqrt_alphas_cumprod),
            ("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod),
            ("posterior_variance", posterior_variance),
            ("posterior_mean_coef1", posterior_mean_coef1),
            ("posterior_mean_coef2", posterior_mean_coef2),
        ]:
            self.register_buffer(name, val.float())

    def _gather(self, buf: Tensor, t: Tensor) -> Tensor:
        """Index buffer by t and reshape for broadcasting over [B, Q, P]."""
        return buf.gather(0, t).view(-1, 1, 1)

    def q_sample(
        self,
        x0: Float[Tensor, "B Q P"],
        t: Int[Tensor, "B"],
        noise: Float[Tensor, "B Q P"],
    ) -> Float[Tensor, "B Q P"]:
        """Forward diffusion: add noise to x0 at timestep t."""
        a = self._gather(self.sqrt_alphas_cumprod, t)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return a * x0 + b * noise

    def p_sample(
        self,
        x_t: Float[Tensor, "B Q P"],
        t: Int[Tensor, "B"],
        noise_pred: Float[Tensor, "B Q P"],
    ) -> Float[Tensor, "B Q P"]:
        """Single reverse step: predict x_{t-1} from x_t."""
        alpha = self._gather(self.alphas, t)
        beta = self._gather(self.betas, t)
        sqrt_one_minus_acp = self._gather(self.sqrt_one_minus_alphas_cumprod, t)

        # predicted mean
        mean = (x_t - beta / sqrt_one_minus_acp * noise_pred) / alpha.sqrt()

        # add noise for t > 0
        var = self._gather(self.posterior_variance, t)
        noise = torch.randn_like(x_t)
        # mask noise at t=0
        nonzero_mask = (t > 0).float().view(-1, 1, 1)
        return mean + nonzero_mask * var.sqrt() * noise


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by 2-layer MLP."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, t: Int[Tensor, "B"]) -> Float[Tensor, "B D"]:
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class DDPMDecoder(nn.Module):
    """
    Lightweight transformer decoder for DDPM-style denoising.

    Supports two attention modes:
    - self-attention (cross_attn=False): concatenate encoder context with noisy patches
    - cross-attention (cross_attn=True): noisy patches attend to encoder context
    """

    def __init__(
        self,
        pos_embed: nn.Module,
        patch_dim: int,
        context_dim: int,
        depth: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.cross_attn = cross_attn

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.pos_embed = pos_embed
        self.time_embed = TimestepEmbedding(embed_dim)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                mlp_ratio=mlp_ratio,
                context_dim=embed_dim if cross_attn else None,
            )
            for _ in range(depth)
        ])

        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)

    def extra_repr(self):
        return f"cross_attn={self.cross_attn}"

    def forward(
        self,
        x_t: Float[Tensor, "B Q P"],
        encoder_latents: Float[Tensor, "B L D"],
        t: Int[Tensor, "B"],
        pred_ids: Int[Tensor, "B Q"],
        visible_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B Q P"]:
        ctx = self.context_proj(encoder_latents)  # [B, L, embed_dim]
        x = self.patch_proj(x_t)  # [B, Q, embed_dim]
        x = self.pos_embed(x, pos_ids=pred_ids)  # add positional encoding
        x = x + self.time_embed(t).unsqueeze(1)  # broadcast timestep embedding

        if self.cross_attn:
            for block in self.blocks:
                x = block(x, context=ctx)
        else:
            L = ctx.shape[1]
            x = torch.cat([ctx, x], dim=1)  # [B, L+Q, embed_dim]
            for block in self.blocks:
                x = block(x)
            x = x[:, L:]  # slice out Q noisy-patch tokens

        x = self.norm(x)
        return self.head(x)  # [B, Q, patch_dim]
