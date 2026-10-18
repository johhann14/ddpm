#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-15 00:16:16
Author: joh
"""

from torch.utils.data import DataLoader, Subset
from ddpm.data.dataset import DatasetWrapper
import torch


def get_loaders(dataset_name, batch_size, num_workers, seed, max_samples=None):
    torch.manual_seed(seed)

    tr_dataset = DatasetWrapper(dataset_name=dataset_name, train_bool=True)
    val_dataset = DatasetWrapper(dataset_name=dataset_name, train_bool=False)

    if max_samples is not None:
        tr_dataset = Subset(tr_dataset, range(min(max_samples, len(tr_dataset))))
        val_dataset = Subset(val_dataset, range(min(max_samples, len(val_dataset))))

        
    tr_loader = DataLoader(
                        dataset=tr_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(),
    )    

    return tr_loader, val_loader
