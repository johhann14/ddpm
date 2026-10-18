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
import mlflow
import matplotlib
matplotlib.use("Agg") # no GUI
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]

    

def _flatten(d, prefix=""):
    out = {}
    for k,v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v

    return out
class CustomLogger:
# https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/logging/nnunet_logger.py
    def __init__(
            self,
            logging: bool,
            use_mlflow: bool,
            dataset_name: str,
            run_name: str,

    ):

        root_dir = Path(__file__).parent.parent.parent.parent / f"experiments"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = root_dir / f"{dataset_name}" / f"{run_name}_{stamp}"

        

        self.logging = logging
        self.trace = defaultdict(list)
        self.logger = logger

        self.use_mlflow = use_mlflow

        # if use_mlflow:
        #     self.mlflow_setup()
        self.dataset_name = dataset_name

    def mlflow_setup(self):
        mlflow.set_tracking_uri(uri=f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment(f"{self.dataset_name}_ddpm")
        mlflow.start_run(run_name=self.run_dir.name)

    def log_key_value(self, key, value, epoch):
        self.trace[key].append(value)
        if self.use_mlflow:
            if key not in ["epoch_start_time", "epoch_end_time"]:
                mlflow.log_metric(key, value, epoch)

    def get_value(self, key, epoch):
        return self.trace[key][epoch]
    

    def plot_png(self):
        # print(self.trace)
        epoch = min([len(l) for l in self.trace.values()])
     
        # print(f"epoch : {epoch}")
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

        # axes = axes.flatten()


        axes.plot(list(range(epoch)), self.trace["tr_loss"], color="b", ls="-", label='tr_loss')
        axes.plot(list(range(epoch)), self.trace["val_loss"], color="g", ls="-", label='val_loss')
        axes.set_ylabel(f"Loss")
        axes.set_xlabel(f"Epoch")
        axes.legend()

        fig.savefig(fname=self.run_dir / f"plots/progress.png")
        plt.close()


    def save_checkpoint(self, model_state_dict, optimizer_state_dict, ema_state_dict, epoch, mode):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        file_path = f'{self.run_dir}/checkpoints/{mode}_model.pt'
        tmp_path = f'{file_path}.tmp'
        torch.save(checkpoint, f=tmp_path)
        os.replace(src=tmp_path, dst=file_path)

        if self.use_mlflow:
            mlflow.log_artifact(file_path, artifact_path="checkpoints")

    # def log_model(self, model, registered_name=None):
    #     if not self.use_mlflow:
    #         return
    #     mlflow.pytorch.log_model(
    #                 pytorch_model=model,
    #                 name="model",
    #                 registered_model_name=registered_name
    #     )

    def log_params(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_unwrap = _flatten(cfg_dict)

        OmegaConf.save(cfg, f=f'{self.run_dir}/config.yaml', resolve=True)
        if self.use_mlflow:
            mlflow.log_params(cfg_unwrap)

    def __enter__(self):
        if self.logging:
            os.makedirs(f"{self.run_dir}/plots")
            os.makedirs(f"{self.run_dir}/checkpoints")
            os.makedirs(f"{self.run_dir}/samples")
            logger.add(sink=self.run_dir / f"trace.log")

        if self.use_mlflow:
            self.mlflow_setup()
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            status = "FINISHED"
        elif exc_type is KeyboardInterrupt:
            status = "KILLED"
            logger.opt(exception=(exc_type, exc_val, tb)).error("Killed by me")

        else:
            status = "FAILED"
            logger.opt(exception=(exc_type, exc_val, tb)).error("Training crashed")
        self.close(status)
        return False                             


    def close(self, status="FINISHED"):
        if self.use_mlflow:
            mlflow.end_run(status=status)



