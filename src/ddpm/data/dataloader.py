#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-15 00:16:16
Author: joh
"""

from torch.utils.data import DataLoader
from ddpm.data.dataset import DatasetWrapper
import torch

def get_loaders(root_dir, dataset_name, batch_size, num_workers, seed):
    torch.manual_seed(seed)

    tr_dataset = DatasetWrapper(root_dir=root_dir, dataset_name=dataset_name, train_bool=True)
    val_dataset = DatasetWrapper(root_dir=root_dir, dataset_name=dataset_name, train_bool=False)

    tr_loader = DataLoader(
                        dataset=tr_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
    )
    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
    )    

    return tr_loader, val_loader
