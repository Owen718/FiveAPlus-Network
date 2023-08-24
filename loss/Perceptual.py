import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


# --- Perceptual loss network  --- #
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).cuda().eval()
        #vgg_model.load_state_dict(torch.load(r'/root/autodl-tmp/MPUNet/loss/vgg16-397923af.pth'))
        vgg_model = vgg_model.features[:16]
        # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
          #  '15': "relu3_3"
        }

    def output_features(self, x):
        # x=x.cuda().half()
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)




class CRPerceptual(nn.Module):
    def __init__(self):
        super(CRPerceptual, self).__init__()
        self.FtsExtractor = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2),
            nn.Tanh(),
            nn.Conv2d(64,64,kernel_size=7,stride=2)
        )
        self.FtsExtractor.requires_grad = False

        self.L1 = nn.L1Loss()
    def forward(self,x, P,N):
        fs_an = self.FtsExtractor(x)
        fs_P = self.FtsExtractor(P)
        fs_N = self.FtsExtractor(N)
        loss = self.L1(fs_an,fs_P) / self.L1(fs_an,fs_N)
        return loss
         


# CR = CRPerceptual().cuda()
# input1 = torch.ones((4,3,256, 256)).cuda()
# loss = CR(input1,input1,input1)
# print('')