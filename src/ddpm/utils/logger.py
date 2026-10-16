#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-15 23:49:07
Author: joh
"""

from datetime import datetime
from pathlib import Path
import os 
from collections import defaultdict
from loguru import logger

class CustomLogger:
# https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/logging/nnunet_logger.py
    def __init__(
            self,
            root_dir: str,
            logging: bool,
            dataset_name: str,
            experiment_name: str

    ):

        
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(root_dir) / f"{dataset_name}" / f"{experiment_name}_{stamp}"

        os.makedirs(f"{self.run_dir}/plots")
        os.makedirs(f"{self.run_dir}/checkpoints")
        os.makedirs(f"{self.run_dir}/samples")


        self.trace = defaultdict(list)
        self.logger = logger
        logger.add(sink=self.run_dir / f"trace.log")

    def log_key_value(self, key, value):
        self.trace[key].append(value)


    def get_value(self, key, epoch):
        return self.trace[key][epoch]
    

    def plot_png(self):
        print(self.trace)
        epoch = min([len(l) for l in self.trace.values()])
        import matplotlib.pyplot as plt
        print(f"epoch : {epoch}")
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

        # axes = axes.flatten()


        axes.plot(list(range(epoch)), self.trace["tr_loss"], color="b", ls="-", label='tr_loss')
        axes.plot(list(range(epoch)), self.trace["val_loss"], color="g", ls="-", label='val_loss')
        axes.set_ylabel(f"Loss")
        axes.set_xlabel(f"Epoch")
        axes.legend()

        plt.savefig(fname=self.run_dir / f"plots/progress.png")
        plt.close(fig)
if __name__ == "__main__":

    run_dir = Path(__file__).parent.parent.parent.parent / f"experiments"

    logger = CustomLogger(root_dir=run_dir, logging=True, dataset_name="CIFAR10",
                    experiment_name="test")
