
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import os.path as osp
import sys
from sklearn.metrics import mean_squared_error
import cv2
import numpy as np
from loss.SSIMLoss import SSIMLoss

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.CL1 import L1_Charbonnier_loss 

from loss.Perceptual import *

from argparse import Namespace
from myutils.dataloader import UIEBD_Dataset
from pytorch_lightning import seed_everything
from myutils.imgqual_utils import PSNR,SSIM

import os

import wandb
#Set seed
seed = 42 #Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import WandbLogger
from archs.FIVE_APLUS import FIVE_APLUSNet
logger = WandbLogger(project="underwater_image_enhancement",
                     name = "FIVEAPLUS_Net",
                     log_model=True)



os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
            
        self.train_datasets = self.params.train_datasets
        self.train_batchsize = self.params.train_bs
        self.validation_datasets = self.params.val_datasets
        self.val_batchsize = self.params.val_bs
            #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers


        self.loss_f = L1_Charbonnier_loss()
        self.ssim_loss = SSIMLoss()
        self.loss_per = PerceptualLoss()
        self.model = FIVE_APLUSNet()
        self.save_hyperparameters()

    def forward(self,x_1 ):
        pred,out_head = self.model(x_1)
        return pred,out_head

        
    def configure_optimizers(self):
            # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr,betas=[0.9,0.999])#,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr,max_lr=1.2*self.initlr,cycle_momentum=False)

    
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
            # REQUIRED
        x ,y,id= batch
        y_hat,out_head= self.forward(x)
    
        loss_d = self.loss_f(y_hat,y)
        loss = loss_d +0.2*self.loss_per(y_hat,y)+0.5*self.ssim_loss(y_hat,y)+0.2*self.loss_per(out_head,y)
        self.log('train_loss', loss,sync_dist=True)

        return {'loss': loss}
    
        

    def validation_step(self, batch, batch_idx):
            # OPTIONAL
        x ,y,id= batch
        y_pred,out_head = self.forward(x) 
        
        loss = self.loss_f(y_pred,y) +0.2*self.loss_per(y_pred,y)+0.5*self.ssim_loss(y_pred,y)+0.2*self.loss_per(out_head,y)

        psnr = PSNR(y_pred,y)
        ssim = SSIM(y_pred,y)
        self.log('val_loss', loss,sync_dist=True)

        self.log('psnr', psnr,sync_dist=True)
        self.log('ssim', ssim,sync_dist=True)

        self.trainer.checkpoint_callback.best_model_score #save the best score model

        if batch_idx==0:
            self.logger.experiment.log({
                    "raw_image":[wandb.Image(x[i].cpu(),caption="raw_image") for i in range(1)],
                    "gt":[wandb.Image(y[i].cpu(),caption="gt") for i in range(1)],
                    "enhancement_pred":[wandb.Image(y_pred[i].cpu(),caption="enhancement_pred") for i in range(1)]
                                            })
                                                    

        return {'val_loss': loss, 'psnr': psnr,'ssim':ssim}
        


    def train_dataloader(self):
            # REQUIRED
        train_set = UIEBD_Dataset(self.train_datasets,train=True,size=self.crop_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
        
    def val_dataloader(self):
        val_set = UIEBD_Dataset(self.validation_datasets,train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=True, num_workers=self.num_workers)

        return val_loader

def main():
    RESUME =False
    resume_checkpoint_path = r'/root/Project/underwater_image_enhancement////'
    
    args = {
    'epochs':400,
    #datasetsw
    'train_datasets':r'/root/autodl-tmp/UIEB/train',
  
    'val_datasets':r'/root/autodl-tmp/UIEB/test',
    #bs
    'train_bs':100,
    #'train_bs':4,
    'val_bs':32,
    'initlr':0.0004,
    'weight_decay':0.01,
    'crop_size':256,
    'num_workers':4,
    #Net
    'model_blocks':5,
    'chns':64
    }

    hparams = Namespace(**args)
   
    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    filename='B_16_32_64-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=3,
    mode = "max",
    save_last=True
    )

    if RESUME:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            resume_from_checkpoint = resume_checkpoint_path,
            gpus= [0],
            logger=logger,
            accelerator='cuda',
            callbacks = [checkpoint_callback],
            gradient_clip_val=0.5, gradient_clip_algorithm="value",
        ) 
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            gpus= [0],
            logger=logger,
            accelerator='cuda',
            callbacks = [checkpoint_callback],
            gradient_clip_val=0.5, gradient_clip_algorithm="value",
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0
        )  

    trainer.fit(model)
    

if __name__ == '__main__':
	#your code
    main()





