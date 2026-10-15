#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-13 22:09:34
Author: joh
"""

import torch
import pytest
from ddpm.models.models import DiffusionModel


@pytest.fixture
def diffusion_model():
    return DiffusionModel(T=1000, in_channels=3, in_size=(32,32))

def test_init_DiffusionModel(diffusion_model):
    m = diffusion_model
    assert m.betas.shape[0] == m.T
    assert m.alphas.shape[0] == m.T
    assert m.alphas_prod.shape[0] == m.T


def test_forward_DiffusionModel(diffusion_model):
    x = torch.randn(size=(16, 3, 32, 32))

    loss = diffusion_model(x)
    assert loss.shape == ()



def test_sampling_DiffusionModel(diffusion_model):
    x = torch.randn(size=(16, 3, 32, 32))
    for t in [1, diffusion_model.T // 2, diffusion_model.T - 1]:
        x_denoised = diffusion_model.sampling(x, t)

        assert x_denoised.shape == x.shape