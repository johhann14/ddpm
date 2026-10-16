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

class Trainer:

    def __init__(
            self,
            diffusion_model: nn.Module,
            diffusion_steps,
            num_epochs,
            batch_size,
            lr,
            adam_betas,
            device
    ):

        self.model = diffusion_model
        self.model.to(device)
        self.diffusion_steps = diffusion_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.adam_betas = adam_betas
        self.optimizer = self.configure_optimizers()
        self.device = device
        
        self.logger = CustomLogger(
            root_dir=Path(__file__).parent.parent.parent.parent / f"experiments",
            logging=True,
            dataset_name="CIFAR10",
            experiment_name="test"
        )
        
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
            desc=f"Train [{self.step+1}/{self.num_epochs}]",
            colour="blue"
        )
        for i, x in list(enumerate(pbar))[:100]:
            x = x.to(self.device)
            # print(next(self.model.paramaers()).device)
            # print(x.device)
            loss = self.model(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_acc += loss.item()
            pbar.set_postfix({"Loss": {loss_acc/(i+1)}})
        return {
                    "tr_loss": loss_acc/len(loader)
        }
    
    @torch.no_grad
    def validation_one_epoch(self, loader):
        self.model.eval() 
        loss_acc = 0

        pbar = tqdm(
        iterable=loader,
        desc=f"Val [{self.step+1}/{self.num_epochs}]",
        colour="green"
    )
        for i, x in list(enumerate(pbar))[:100]:
            x = x.to(self.device)
            loss = self.model(x)
            loss_acc += loss.item()
            pbar.set_postfix({"Loss": {loss_acc/(i+1)}})

        return {
                    "val_loss": loss_acc/len(loader)
        }

    def on_epoch_start(self):
        logger.info(f"Epoch [{self.step+1}/{self.num_epochs}]...")
        self.logger.log_key_value('epoch_start_time', time.perf_counter())


    def save_checkpoint(self, mode: str):
        logger.debug(f"enter")
        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optimizer.state_dict()
        checkpoint = {
            "epoch": self.step,
            "model_state_dict": model_state_dict,
            "optimize_state_dict": optim_state_dict,
        }
        torch.save(checkpoint, f=f'{self.logger.run_dir}/checkpoints/{mode}_model.pt')


    def on_epoch_end(self):
        self.logger.log_key_value('epoch_end_time', time.perf_counter())
        epoch_duration = self.logger.get_value('epoch_end_time', epoch=self.step) - self.logger.get_value('epoch_start_time', epoch=self.step)
        
        tr_loss = self.logger.get_value('tr_loss', epoch=self.step)
        val_loss = self.logger.get_value('val_loss', epoch=self.step)

        logger.info(f"Train Loss: {tr_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"took: {epoch_duration:.4f} s")

        self.logger.plot_png()

        if self.step % 9 == 0:
            self.save_checkpoint(mode='latest')
        self.step+=1

    def on_train_epoch_start(self):
        pass


    def on_train_epoch_end(self, train_outputs: dict):

        for k, v in train_outputs.items():
            self.logger.log_key_value(k, v)


    def on_validation_epoch_end(self, validation_outputs: dict):
        for k,v in validation_outputs.items():
            self.logger.log_key_value(k, v)


    def on_train_start(self):
        logger.info(f"Running trainning")


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
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == "__main__":

    from ddpm.data.dataloader import get_loaders
    from pathlib import Path
    model = DiffusionModel(T=1000, in_channels=3, in_size=(32,32))
    cfg_path = Path(__file__).parent.parent.parent.parent / f"configs/gpu_base_config.yaml"
    cfg = OmegaConf.load(cfg_path)
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

    