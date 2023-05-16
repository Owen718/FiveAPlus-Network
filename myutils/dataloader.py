from distutils.command.clean import clean
from doctest import FAIL_FAST
from locale import normalize
import os
import sys
from tkinter import W
from unicodedata import name

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import cv2
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision
import os,sys
sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
random.seed(1143)


def ShiftPaddingTo(img,size=512):
    w,h = img.size
    w_new = size
    h_new = size
    p = Image.new('RGB', (w_new,h_new), (0, 0, 0))
    p.paste(img, (w_new-w, h_new-h, w_new, h_new))
    p.paste(img, (0, h_new-h, w, h_new))
    p.paste(img, (w_new-w, 0, w_new, h))
    p.paste(img, (0, 0, w, h))
    return p

def ShiftPaddingTo_h(img):
    w,h = img.size
    w_new = w
    h_new = h+1
    p = Image.new('RGB', (w_new,h_new), (0, 0, 0))
    p.paste(img, (w_new-w, h_new-h, w_new, h_new))
    p.paste(img, (0, h_new-h, w, h_new))
    p.paste(img, (w_new-w, 0, w_new, h))
    p.paste(img, (0, 0, w, h))
    return p

def ShiftPaddingTo_w(img):
    w,h = img.size
    w_new = w+1
    h_new = h
    p = Image.new('RGB', (w_new,h_new), (0, 0, 0))
    p.paste(img, (w_new-w, h_new-h, w_new, h_new))
    p.paste(img, (0, h_new-h, w, h_new))
    p.paste(img, (w_new-w, 0, w_new, h))
    p.paste(img, (0, 0, w, h))
    return p

class UIEBD_Dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.png'):
        super(UIEBD_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'input'))
        self.haze_imgs=[os.path.join(path,'input',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'target')
        # self.padding = torch.nn.ReflectionPad2d(8)
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
      
        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))


        if self.train:
            haze = ShiftPaddingTo(haze,size=self.size)
            clear = ShiftPaddingTo(clear,size=self.size)
            if not isinstance(self.size,str):
                i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
                haze=FF.crop(haze,i,j,h,w)
                clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB"),clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
           
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        #推理时关闭
        # else:
        #     # data_1=data
        #     # data_2=FF.rotate(data,90)
        #     # data_3=FF.rotate(data,180)
        #     # data_4=FF.rotate(data,270)
            
        #     data = data.resize((256,256))
        #     target = target.resize((256,256))
            
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)
