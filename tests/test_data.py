#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-15 18:10:28
Author: joh
"""

import pytest

from ddpm.data.dataset import DatasetWrapper
from ddpm.data.dataloader import get_loaders
def test_dataset_init(data_config):
    dataset = DatasetWrapper(
        root_dir=data_config.root_dir,
        dataset_name=data_config.dataset_name,
        train_bool=True,
    )

    assert dataset is not None
    assert len(dataset) != 0
    assert iter(dataset) is not None



def test_get_loaders(data_config):
    tr_loader, val_loader = get_loaders(
                                root_dir=data_config.root_dir,
                                dataset_name=data_config.dataset_name,
                                batch_size=data_config.batch_size,
                                num_workers=data_config.num_workers,
                                seed=data_config.seed
    )
    assert tr_loader is not None
    assert val_loader is not None