#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-14 23:15:12
Author: joh
"""

import torch 
import pytest
from omegaconf import OmegaConf
from ddpm.models.models import DiffusionModel

@pytest.fixture
def diffusion_model():
    return DiffusionModel(T=1000, in_channels=3, in_size=(32,32))


@pytest.fixture
def config():
    cfg = OmegaConf.load("/home/joh/Sync/pro/projects/ddpm/configs/base_config.yaml")
    return cfg

@pytest.fixture
def data_config(config):
    return config.data
