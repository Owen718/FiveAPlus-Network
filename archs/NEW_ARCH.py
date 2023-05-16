import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
        	
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),  
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

#Multi-Scale Pyramid Module
class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        self.relu=nn.ReLU(inplace=True)

        self.tanh=nn.Tanh()
        self.refine2= nn.Conv2d(16, 16, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(16+3, 16, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest


    def forward(self, x):
        dehaze = self.relu((self.refine2(x)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 128)

        x102 = F.avg_pool2d(dehaze, 64)

        x103 = F.avg_pool2d(dehaze, 32)

        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze


class SFDIM(nn.Module):
    def __init__(self, n_feats):   
        super().__init__()
        # i_feats =n_feats*2
        
        self.Conv1 =nn.Sequential(
            nn.Conv2d(n_feats,2*n_feats,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(2*n_feats,n_feats,1,1,0)) 
        self.Conv1_1 =nn.Sequential(
            nn.Conv2d(n_feats,2*n_feats,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(2*n_feats,n_feats,1,1,0)) 
    
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.FF = FreBlock()
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
    def forward(self, x,y):     
        b,c,H,W=x.shape 
        a = 0.1
        mix = x+y
        mix_mag,mix_pha =  self.FF(mix)
        #Ghost Expand      
        mix_mag = self.Conv1(mix_mag)
        mix_pha = self.Conv1_1(mix_pha)

        real_main = mix_mag * torch.cos(mix_pha)
        imag_main = mix_mag * torch.sin(mix_pha)
        x_out_main = torch.complex(real_main, imag_main)
        x_out_main = torch.abs(torch.fft.irfft2(x_out_main, s=(H, W), norm='backward'))+1e-8

        return   self.Conv2(a*x_out_main+(1-a)*mix)

class FreBlock(nn.Module):
    def __init__(self):
        super(FreBlock, self).__init__()
        

    def forward(self,x):
        x = x+1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)

        return mag,pha

#Multi-branch Color Enhancement Modul
class MCEM(nn.Module):
    def __init__(self, in_channels, channels):
        super(MCEM, self).__init__()
        self.conv_first_r = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_g = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_r = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_g = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_b = nn.InstanceNorm2d(channels//2, affine=True)
        
        self.conv_out_r = nn.Conv2d( channels//2,in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_g = nn.Conv2d( channels//2, in_channels//4,kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d( channels//2,in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        
        x1,x2, x3,x4= torch.chunk(x, 4, dim=1)
        
        x_1 = self.conv_first_r(x1)
        x_2 = self.conv_first_g(x2)
        x_3 = self.conv_first_b(x3)
        
        out_instance_r = self.instance_r(x_1)
        out_instance_g = self.instance_g(x_2)
        out_instance_b = self.instance_b(x_3)

        out_instance_r=self.conv_out_r(out_instance_r)
        out_instance_g=self.conv_out_g(out_instance_g)
        out_instance_b=self.conv_out_b(out_instance_b)

        mix = out_instance_r+out_instance_g+out_instance_b+x4
        
        out_instance= torch.cat((out_instance_r, out_instance_g,out_instance_b,mix),dim=1)

        return out_instance

class MCEM_2(nn.Module):
    def __init__(self, in_channels, channels):
        super(MCEM_2, self).__init__()
        self.conv_first_r = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_g = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_r = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_g = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_b = nn.InstanceNorm2d(channels//2, affine=True)
        
        self.conv_out_r = nn.Conv2d( channels//2,in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_g = nn.Conv2d( channels//2, in_channels//4,kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d( channels//2,in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        
        x1,x2, x3,x4= torch.chunk(x, 4, dim=1)
        
        x_1 = self.conv_first_r(x1)
        x_2 = self.conv_first_g(x2)
        x_3 = self.conv_first_b(x3)
        
        out_instance_r = self.instance_r(x_1)
        out_instance_g = self.instance_g(x_2)
        out_instance_b = self.instance_b(x_3)

        out_instance_r=self.conv_out_r(out_instance_r)
        out_instance_g=self.conv_out_g(out_instance_g)
        out_instance_b=self.conv_out_b(out_instance_b)

        mix = out_instance_r+out_instance_g+out_instance_b+x4
        
        out_instance= torch.cat((out_instance_r, out_instance_g,out_instance_b,mix),dim=1)
        # out_instance = self.act(self.conv2(out_instance))

        return out_instance
# MAIN-Net
class FIVE_APLUSNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=16):
        super(FIVE_APLUSNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc
        self.pyramid_enhance = Enhance()
        # self.encoder = Condition()
        self.color_cer_1 = MCEM(base_nf,base_nf*2)
        self.color_cer_2 = MCEM_2(base_nf,base_nf*2)
       
        self.fusion_mixer=SFDIM(base_nf)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.stage2 = PALayer(base_nf)
        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        # cond = self.cond_net(x)

        out = self.conv1(x)
        out_1=self.color_cer_1(out)
        out_2=self.pyramid_enhance(out)
        mix_out = self.fusion_mixer(out_1,out_2)

        out_stage2 = self.act(mix_out)
        out_stage2_head = self.conv4(out_stage2)

        out_stage2 = self.conv2(out_stage2)
        out_stage2=self.color_cer_2(out_stage2)
        out = self.stage2(out_stage2)
        out = self.act(out)

        out = self.conv3(out)

        return out,out_stage2_head