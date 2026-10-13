#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-11 22:12:37
Author: joh
"""

from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import Dataset
import torch
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"



dataset_cfg = {
    "CIFAR10": {
        "cl": datasets.CIFAR10,
        "img_size": 32,
        "num_channels": 3, 
        "padding": None
    }
}

class DatasetWrapper(Dataset):
    def __init__(self, dataset_name: str, train_bool: bool):


        # if self.dataset[dataset_name]['padding']:
            
        #     #@todo
        
        t = transforms.Compose([transforms.ToTensor()]) # transforms.toTensor already range from 0 to 1
        self.dataset = dataset_cfg[dataset_name]['cl'](
                            root=DATA_DIR,
                            train=train_bool,
                            transform=t,
                            target_transform=None,
                            download=True
        )


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        dataset_item = self.dataset[index][0] 
        return dataset_item





if __name__ == "__main__":

    
    dataset_name = "CIFAR10"

    dataset = DatasetWrapper(dataset_name=dataset_name, train_bool=True)
    print(f"DatasetWrapper ok")
    print(vars(dataset))