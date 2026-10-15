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


class Trainer:

    def __init__(
            self,
            diffusion_model,
            diffusion_steps,
            num_epochs,
            batch_size,
            lr,
            adam_betas,
            device
    ):

        self.model = diffusion_model
        self.diffusion_steps = diffusion_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.adam_betas = adam_betas
        self.optimizer = self.configure_optimizers()
        self.device = device
        
        
        
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

        pbar = tqdm(
            iterable=loader,
            desc=f"Train Epoch [{self.step+1}/{self.num_epochs}]",
            colour="blue"
        )
        for i, x in enumerate(pbar):
            x = x.to(self.device)
            loss = self.model(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_acc += loss.item()
            pbar.set_postfix({"Loss": {loss_acc/(i+1)}})
        return loss_acc/len(loader)

    
    @torch.no_grad
    def validation_one_epoch(self, loader):
        self.model.eval() # dropout and batchnorm etc
        loss_acc = 0
        for x in loader:
            x = x.to(self.device)
            loss = self.model(x)
            loss_acc += loss
    
        return loss_acc/len(loader)


    def train(self, train_loader, val_loader):
        logger.info(f"Training...")
        for epoch in range(self.num_epochs):
            # logger.info(f"Epoch [{epoch+1}/{self.num_epochs}]")
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validation_one_epoch(val_loader)
            logger.info(f"\tTrain loss: {train_loss}:.4f")
            logger.info(f"\Val loss: {val_loss}")

    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == "__main__":

    from ddpm.data.dataloader import get_loaders

    model = DiffusionModel(T=1000, in_channels=3, in_size=(32,32))
    cfg = OmegaConf.load(f"/home/joh/Sync/pro/projects/ddpm/configs/base_config.yaml")
    print(cfg)
    trainer = Trainer(
                    diffusion_model=model,
                    diffusion_steps=1000,
                    num_epochs=cfg.training.num_epochs,
                    batch_size=cfg.training.batch_size,
                    lr=cfg.training.lr,
                    adam_betas=tuple(cfg.training.adam_betas),
                    device=cfg.training.device
    )
    print(f"ok")
    print(trainer.count_parameters())

    tr_loader, val_loader = get_loaders(**cfg.data)
    print(f"OKOK")

    trainer.train(train_loader=tr_loader, val_loader=val_loader)

    