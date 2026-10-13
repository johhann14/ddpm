#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-12 23:01:17
Author: joh
"""

import torch.nn as nn
import torch


nn_cfg = {

}


class DiffusionModel(nn.Module):
    def __init__(self, ):
        self.T = None
        self.betas = None
        self.network = None
