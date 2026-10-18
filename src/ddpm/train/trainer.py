#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creation date: 2026-10-14 22:45:50
Author: joh
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from ddpm.models.models import DiffusionModel
from ddpm.utils.logger import CustomLogger
from pathlib import Path
import time
import argparse

from itertools import islice

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
# class ModelConfig:

#     def __init__(
#             self,
#             diffusion_steps,
#             in_channels,
#             in_size,
#             initial_channels,
#     ):

#         self.diffusion_steps = diffusion_steps
#         self.in_channels = in_channels
#         self.in_size = in_size
#         self.initial_channels = initial_channels


# class TrainerConfig:

#     def __init__(
#             self,
#             batch_size,
#             lr,
#             adam_betas,
#             num_epochs,
#             device,
#             model_config
#     ):

#         self.batch_size = batch_size
#         self.lr = lr
#         self.adam_betas = tuple(adam_betas) if isinstance(adam_betas, list) else adam_betas
#         self.num_epochs = num_epochs
#         self.device = device
#         self.model_config = ModelConfig(**model_config)


# class DataConfig:

#     def __init__(
#             self,
#             dataset_name,
#             img_size,
#             num_channels,
#             num_workers
#     ):
#         self.dataset_name = dataset_name
#         self.img_size = tuple(img_size) if isinstance(img_size, list) else img_size
#         self.num_channels = num_channels
#         self.num_workers = num_workers


        
class Trainer:

    def __init__(
            self,
            diffusion_model: nn.Module,
            num_epochs,
            batch_size,
            lr,
            adam_betas,
            device,
            custom_loger: CustomLogger,
    ):
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        self.model = diffusion_model
        self.model.to(self.device)

        self.ema = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.adam_betas = adam_betas
        self.optimizer = self.configure_optimizers()
        
        self.logger = custom_loger
        
        self.step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
        )
        return optimizer

        
    

    def train_one_epoch(self, loader):
        self.model.train()
        loss_acc = 0
        n= 0


        pbar = tqdm(
            iterable=loader,
            total=len(loader),
            desc=f"Train [{self.step+1}/{self.num_epochs}]",
            colour="blue"
        )
        for x in pbar:
            x = x.to(self.device)
            # print(next(self.model.paramaers()).device)
            # print(x.device)
            loss = self.model(x)
            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )
            
            self.optimizer.step()
            self.ema.update_parameters(self.model)

            loss_acc += loss.item()
            n+=1
            pbar.set_postfix({"Loss": {loss_acc/n}})
        return {
                    "tr_loss": loss_acc/n
        }
    
    @torch.no_grad
    def validation_one_epoch(self, loader):
        self.model.eval() 
        loss_acc = 0
        n = 0

        pbar = tqdm(
            iterable=loader,
            total=len(loader),
            desc=f"Validation [{self.step+1}/{self.num_epochs}]",
            colour="green"
        )
        for x in pbar:
            x = x.to(self.device)
            loss = self.model(x)
            loss_acc += loss.item()
            n+=1
            pbar.set_postfix({"Loss": {loss_acc/n}})

        return {
                    "val_loss": loss_acc/n
        }

    def on_epoch_start(self):
        logger.info(f"Epoch [{self.step+1}/{self.num_epochs}]...")
        self.logger.log_key_value('epoch_start_time', time.perf_counter(), self.step)



    def on_epoch_end(self):
        self.logger.log_key_value('epoch_end_time', time.perf_counter(), self.step)
        epoch_duration = self.logger.get_value('epoch_end_time', epoch=self.step) - self.logger.get_value('epoch_start_time', epoch=self.step)
        self.logger.log_key_value(key="epoch_duration", value=epoch_duration, epoch=self.step)
        
        tr_loss = self.logger.get_value('tr_loss', epoch=self.step)
        val_loss = self.logger.get_value('val_loss', epoch=self.step)

        logger.info(f"Train Loss: {tr_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"took: {epoch_duration:.4f} s")

        self.logger.plot_png()

        if (self.step+1) % 2 == 0:
            self.logger.save_checkpoint(
                        model_state_dict=self.model.state_dict(),
                        ema_state_dict=self.ema.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        epoch=self.step,
                        mode='latest'
            )
        self.step+=1

    def on_train_epoch_start(self):
        pass


    def on_train_epoch_end(self, train_outputs: dict):

        for k, v in train_outputs.items():
            self.logger.log_key_value(k, v, self.step)


    def on_validation_epoch_end(self, validation_outputs: dict):
        for k,v in validation_outputs.items():
            self.logger.log_key_value(k, v, self.step)


    def on_train_start(self):
        logger.info(f"Running trainning")

    def on_train_end(self):
        self.logger.save_checkpoint(
                    model_state_dict=self.model.state_dict(),
                    ema_state_dict=self.ema.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    epoch=self.step,
                    mode='latest'
        )
    def train(self, train_loader, val_loader):


        for epoch in range(self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            # logger.info(f"Epoch [{epoch+1}/{self.num_epochs}]")
            train_outputs = self.train_one_epoch(train_loader)
            self.on_train_epoch_end(train_outputs=train_outputs)
            val_outputs = self.validation_one_epoch(val_loader)
            self.on_validation_epoch_end(validation_outputs=val_outputs)

            self.on_epoch_end()

        self.on_train_end()
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == "__main__":
    from ddpm.data.dataloader import get_loaders
    from pathlib import Path

    parser = argparse.ArgumentParser(description='DDPM Training')
    parser.add_argument("--cfg_path", type=str, help='Path to the config.yaml file')

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_path)


    experiment_cfg = cfg.experiment
    data_cfg = cfg.data
    train_cfg = cfg.training
    model_cfg = cfg.model
    logger_cfg = cfg.logger

    with CustomLogger(
            logging=logger_cfg.logging,
            dataset_name=logger_cfg.dataset_name,
            run_name=logger_cfg.run_name,
            use_mlflow=logger_cfg.use_mlflow
        ) as lg:
        
        # Model init
        model = DiffusionModel(
                    T=model_cfg.diffusion_steps, 
                    in_channels=model_cfg.in_channels, 
                    in_size=tuple(model_cfg.in_size)
        )

        # Trainer init
        trainer = Trainer(
                        diffusion_model=model,
                        num_epochs=train_cfg.num_epochs,
                        batch_size=train_cfg.batch_size,
                        lr=train_cfg.lr,
                        adam_betas=tuple(train_cfg.adam_betas),
                        device=train_cfg.device,
                        custom_loger=lg
        )

        tr_loader, val_loader = get_loaders(dataset_name=data_cfg.dataset_name,
                                            batch_size=train_cfg.batch_size,
                                            num_workers=data_cfg.num_workers,
                                            seed=experiment_cfg.seed,
                                            max_samples=data_cfg.max_samples                
                            )

        lg.log_params(cfg)
        trainer.train(train_loader=tr_loader, val_loader=val_loader)

    