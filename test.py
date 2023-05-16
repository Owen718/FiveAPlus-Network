import argparse
import os
from os.path import exists, join as join_paths
from unicodedata import name
import torch
import numpy as np
from torchvision.transforms import functional as FF
import warnings
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
from myutils.dataloader import UIEBD_Dataset
from archs.NEW_ARCH import FIVE_APLUSNet
from PIL import Image
warnings.filterwarnings("ignore")
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=15, help='number of cpu threads to use during batch generation')
parser.add_argument('--dataset_type', type=str, default='uieb') 
parser.add_argument('--dataset', type=str, default='/root/autodl-tmp/UIEB/test', help='path of CSD dataset') 
parser.add_argument('--savepath', type=str, default='/root/Project/OUT/UIEB', help='path of output image') 
parser.add_argument('--model_path', type=str, default='model/FAPlusNet-alpha-0.1.pth', help='path of SnowFormer checkpoint') 
opt = parser.parse_args()

val_set = UIEBD_Dataset(opt.dataset,train=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)


netG_1 = FIVE_APLUSNet().cuda()

if __name__ == '__main__':   

    ssims = []
    psnrs = []
    # rmses = []
    
    g1ckpt1 = opt.model_path
    ckpt = torch.load(g1ckpt1)
    netG_1.load_state_dict(ckpt)
    netG_1.eval()
   
    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(val_loader),total=len(val_loader))
    name = 0
    for idx,(raw,gt,id) in loop:
        
        with torch.no_grad():
                
                raw = raw.cuda()
                gt = gt.cuda()
                enhancement_img,enhancementhead= netG_1(raw)

                save_image(enhancement_img,os.path.join(savepath_dataset,'%s'%(id)),normalize=False)
                