# Copyright (c) Sophont, Inc
#
# This source code is licensed under the Apache License, Version 2.0

"""
DDPM decoder components for diffusion-based masked autoencoder.

NoiseSchedule: cosine beta schedule with forward/reverse diffusion
TimestepEmbedding: sinusoidal timestep encoding
AdaLNBlock: Transformer block with Adaptive Layer Normalization
DDPMDecoder: lightweight transformer decoder conditioned on encoder latents
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

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


class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN-Zero)."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=bias, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-initialize the modulation for identity mapping at initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Unsqueeze for sequence broadcasting: [B, D] -> [B, 1, D]
        shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

        # Attention block
        attn_input = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa * attn_out

        # MLP block
        mlp_input = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(mlp_input)
        return x


class FinalAdaLN(nn.Module):
    """Final AdaLN layer before the projection head."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DDPMDecoder(nn.Module):
    """
    Lightweight transformer decoder for DiffMAE-style denoising.
    Uses self-attention over combined visible context and noisy patches,
    conditioned on the diffusion timestep via AdaLN.
    """

    def __init__(
        self,
        pos_embed: nn.Module,
        patch_dim: int,
        context_dim: int,
        depth: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: int | float = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        cross_attn: bool = False,
    ):
        super().__init__()
        if cross_attn:
            raise NotImplementedError("cross_attn=True not implemented yet in DDPMDecoder")
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.pos_embed = pos_embed
        self.time_embed = TimestepEmbedding(embed_dim)

        # PyTorch MHA has a single bias flag; combine qkv_bias and proj_bias
        bias = qkv_bias and proj_bias

        self.blocks = nn.ModuleList([
            AdaLNBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                bias=bias,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalAdaLN(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(
        self,
        x_t: Float[Tensor, "B Q P"],
        encoder_latents: Float[Tensor, "B L D"],
        t: Int[Tensor, "B"],
        pred_ids: Int[Tensor, "B Q"],
        visible_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B Q P"]:
        
        # 1. Prepare unmasked context (with spatial awareness)
        ctx = self.context_proj(encoder_latents)  # [B, L, embed_dim]
        if visible_ids is not None:
            ctx = self.pos_embed(ctx, pos_ids=visible_ids)

        # 2. Prepare noisy patches (with spatial awareness)
        x = self.patch_proj(x_t)  # [B, Q, embed_dim]
        x = self.pos_embed(x, pos_ids=pred_ids)

        # 3. Combine sequence
        L = ctx.shape[1]
        x = torch.cat([ctx, x], dim=1)  # [B, L+Q, embed_dim]

        # 4. Process through AdaLN blocks
        t_emb = self.time_embed(t)  # [B, embed_dim]
        for block in self.blocks:
            x = block(x, t_emb)

        # 5. Final norm and slice
        x = self.final_layer(x, t_emb)
        x = x[:, L:]  # Slice out the Q noisy-patch tokens
        
        return self.head(x)  # [B, Q, patch_dim]
