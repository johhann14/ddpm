#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-12 23:01:17
Author: joh
"""

import torch.nn as nn
import torch

from ddpm.models.modules import AttenionFiLMResUNet
import torch.nn.functional as F
from einops import rearrange
nn_cfg = {

}


class DiffusionModel(nn.Module):
    def __init__(self, T, in_channels, in_size):
        super().__init__()
        self.T = T
        self.network = AttenionFiLMResUNet(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            initial_channels=64,
                            base_channels=64,
                            dim_mults=(1, 2, 4, 8),
                            groups=8
        )


        betas = self.linear_beta_schedule(self.T)
        alphas = self.get_alphas(betas)
        alphas_prod = self.get_alphas_prod(alphas)

        # move along device and for save
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("alphas_prod", alphas_prod.to(torch.float32))

    @staticmethod
    def linear_beta_schedule(timesteps):

        scale = 1000 / timesteps

        # values in ddpm paper
        beta_start = 1e-4 * scale
        beta_end = 0.02 * scale

        return torch.linspace(start=beta_start, end=beta_end, steps=timesteps, dtype=torch.float64)

    @staticmethod
    def get_alphas(betas):
        return 1.0 - betas

    @staticmethod
    def get_alphas_prod(alphas):
        return torch.cumprod(alphas, dim=0)


    def forward(self, x: torch.Tensor):
        # pred: B, C, H, W 
        B, C, H, W = x.shape
        t = torch.randint(low=0, high=self.T, size=(B,), device=x.device)


        # images should be in [-1, 1] in the ddpm paper
        noise = torch.randn(size=(B, C, H, W), device=x.device)

        alpha_t = self.alphas_prod[t] # (B,)
        alpha_t = rearrange(alpha_t, "b -> b 1 1 1")
        xt = (alpha_t ** .5) * x + ((1 - alpha_t) ** .5) * noise 


        noise_preds = self.network(xt, t)   

        loss = F.mse_loss(noise_preds, noise)

        return loss


    @torch.no_grad
    def sampling(self, x, t):
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = 0

        
        predicted_noise = self.network(x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long))
        noise_scale = (1 - self.alphas[t]) / ((1 - self.alphas_prod[t]) ** 0.5)
        x = (1 / self.alphas[t]**0.5) * (x - noise_scale * predicted_noise) + (self.betas[t] ** 0.5) * z

        return x