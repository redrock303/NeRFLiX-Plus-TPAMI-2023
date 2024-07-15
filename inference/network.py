import warnings
warnings.filterwarnings("ignore")
import sys 

 
sys.path.append('/newdata/kunzhou/project/rendering/nerflix_sr/')
import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F


import torch
import torch.nn as nn

import torchvision 
import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F

from utils import common, model_opr
# from utils.rife_flow.rife_flow import IFNet

from utils.core import imresize

from utils.spynet import flow_viz
from utils.spynet.spynet import *

# from libs.DCNv2.dcn_v2 import * # for 2080
from libs.dcnv2.dcn_v2 import * # for 3090 



class FlowGuidedDCN(DCNv2):
    '''Use other features to generate offsets and masks'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=8):
        super(FlowGuidedDCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels+2, channels_, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea,flow):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        out = self.conv_offset_mask(torch.cat([fea,flow],1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset = offset + flow.flip(1).repeat(1,offset.size()[1]//2,1,1)
        # offset_mean = torch.mean(torch.abs(offset))
        # if offset_mean > 100:
        #     logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding,
                           self.dilation, self.deformable_groups)

class Dense_Warping(nn.Module):
    def __init__(self, nf=64):
        super(Dense_Warping, self).__init__()
        self.conv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.flow_dcn = FlowGuidedDCN(nf,nf,3,1,1)
    def forward(self,nbr,ref,optical_flow):
        if optical_flow.shape[2] != nbr.shape[2]:
            scale = float(nbr.shape[2] / optical_flow.shape[2])
            optical_flow = scale * torch.nn.functional.interpolate(optical_flow,size = (nbr.size()[-2],nbr.size()[-1]),mode='bilinear',align_corners=False)

        # print(optical_flow.shape,nbr.shape)

        nbr_warp = backwarp(nbr,optical_flow)
        offset = torch.cat([nbr_warp,ref],1)
        offset = F.relu(self.conv1(offset))
        return self.flow_dcn(nbr,offset,optical_flow)
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, 90, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(90, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=False)
        out = self.conv2(out)
        return identity + out
class FeatureEncoder(torch.nn.Module):
    def __init__(self,inc=3, nf=64, N_RB=5):
        super(FeatureEncoder, self).__init__()
        RB = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_pre = torch.nn.Sequential(
            nn.Conv2d(inc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(32, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            # make_layer(RB,2),
        ) 

        self.conv_first = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,2),
        ) 
        self.down_scale1 = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,2),
        )
        self.down_scale2 = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,2),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x,prior=None):
        if prior is not None:
            x = torch.cat([x,prior],1)

        x = self.conv_pre(x)
        fea_d0 = self.lrelu(self.conv_first(x))
        
        fea_d1 = self.down_scale1(fea_d0)
        fea_d2 = self.down_scale2(fea_d1)

        return [fea_d0,fea_d1,fea_d2]

class SimpleNet(nn.Module):
    def __init__(self, config=None,nf=128, front_RB=8, back_RB=15, nbr=2, groups=8):

        super(SimpleNet, self).__init__()
        self.nbr = nbr
        self.flownet = Network()

        self.fea_extract = FeatureEncoder(3,nf=nf)
        self.denoise_extract = FeatureEncoder(3+2,nf=nf)

        self.lrx4_dw = Dense_Warping(nf=nf)

        self.fuse3D = nn.Conv3d(nf, nf, kernel_size=(3,3,3), stride= 1,padding= 1, bias=True)
        self.fuse = nn.Conv2d(nf*(self.nbr+1), nf , 3, 1, 1, bias=True)

        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon = make_layer(RB_f, 10)

        self.conv_x4 = nn.Conv2d(nf, 3 , 3, 1, 1, bias=True) # supervison

        self.lrx2_dw = Dense_Warping(nf=nf)
        self.fuse3D_d1 = nn.Conv3d(nf, nf, kernel_size=(3,3,3), stride= 1,padding= 1, bias=True)
        self.fuse_d1 = nn.Conv2d(nf*(self.nbr+1), nf , 3, 1, 1, bias=True)
        self.recon_d1 = make_layer(RB_f, 5)
        self.conv_x2 = nn.Conv2d(nf, 3 , 3, 1, 1, bias=True) # supervison

        self.lrx1_dw = Dense_Warping(nf=nf)
        self.fuse3D_d0 = nn.Conv3d(nf, nf, kernel_size=(3,3,3), stride= 1,padding= 1, bias=True)
        self.fuse_d0 = nn.Conv2d(nf*(self.nbr+1), nf , 3, 1, 1, bias=True)
        self.recon_d0 = make_layer(RB_f, 5)

        # self.conv_x1 = nn.Conv2d(nf, 3 , 3, 1, 1, bias=True) # supervison

        

        # self.cs_fuse = nn.Conv2d(nf*2, nf , 3, 1, 1, bias=True)
        # self.hr_fuse = nn.Conv2d(nf*self.nbr, nf , 3, 1, 1, bias=True)

        self.up_conv1 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        # self.up_conv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.out_conv = nn.Conv2d(64, 3, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)


        # self.guided_filter = DeepConvGuidedFilter()

        # self.degradeNet = RCABUnet(inc = 5,nfeat = 48) # for data degradation only 

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        

        

        # model_opr.load_model(self ,  '/dataset/kunzhou/move_project/vfi/logs/fast_ivm_sim/simulationv2mix/models/iteration100000G.pth', strict=True, cpu=True)

        # no_grads = [self.flownet]
        # for module in no_grads:
        #     for param in module.parameters():
        #         param.requires_grad = False 
    def inference(self,x,noise_prior,scale = 4):
        b,n,c,h,w = x.size()

        x_nbr = []
        for i in range(n):
            if i!=1:
                x_nbr.append(x[:,i].contiguous())
        
        x_nbr = torch.stack(x_nbr,1)
        x_nbr = x_nbr.view(-1,c,h,w)

        # test 
        x0,x1,x2 = x[:,0].contiguous(),x[:,1].contiguous(),x[:,2].contiguous()

        noise_prior = noise_prior.view(b,2,1,1)
        
        
        fea_r_d2,fea_r_d4,fea_r_d8  = self.fea_extract(x_nbr)
        # print(x1.shape,fea_r_d8.shape)
        fea_r_d2 = fea_r_d2.view(b,n-1,-1,h//2,w//2)
        fea_r_d4 = fea_r_d4.view(b,n-1,-1,h//4,w//4)
        fea_r_d8k = fea_r_d8.view(b,n-1,-1,h//8,w//8)
        fea_l_d0,fea_l_d1,fea_l_d2 = fea_r_d2[:,0],fea_r_d4[:,0],fea_r_d8k[:,0]
        fea_r_d0,fea_r_d1,fea_r_d2 = fea_r_d2[:,1],fea_r_d4[:,1],fea_r_d8k[:,1]

        # fea_l_d0,fea_l_d1,fea_l_d2 =  self.fea_extract(x0)
        # fea_r_d0,fea_r_d1,fea_r_d2  = self.fea_extract(x2)

        fea_m_d0,fea_m_d1,fea_m_d2 =  self.denoise_extract(x1,noise_prior[:,:2].clone().repeat(1,1,h,w))
        
        # if scale != 4:
        #     inv_scale = 1.0 / scale
        #     x1_d4 = imresize(x1,inv_scale)
        #     x0_d4,x2_d4 = imresize(x0,inv_scale),imresize(x2,inv_scale)
        #     flow_x01 = self.flownet(x1_d4,x0_d4)
        #     flow_x21 = self.flownet(x1_d4,x2_d4)

        inv_scale = 1.0 / scale
        x1_lr = imresize(x1,inv_scale)
        nbr_lr = imresize(x_nbr,inv_scale)
        nbr_flow_lr = self.flownet(x1_lr.repeat(n-1,1,1,1),nbr_lr)

        # print('---------scale-----------',scale,x1_lr.shape)

        
        nbr_flow_lr_sep = nbr_flow_lr.view(b,n-1,-1,h//scale,w//scale)
        flow_x01_lr,flow_x21_lr = nbr_flow_lr_sep[:,0],nbr_flow_lr_sep[:,1]
        # print('nbr_flow_lr',nbr_flow_lr.shape,flow_x01_lr.shape,flow_x21_lr.shape)
        # input('cc')

        if scale !=8:
            nbr_flow_x8 =  float(scale)/8 * torch.nn.functional.interpolate(nbr_flow_lr,size = (h//8,w//8),mode='bilinear',align_corners=False)
            warp_d2 = self.lrx4_dw(fea_r_d8,fea_m_d2.repeat(n-1,1,1,1),nbr_flow_x8)
        else:
            warp_d2 = self.lrx4_dw(fea_r_d8,fea_m_d2.repeat(n-1,1,1,1),nbr_flow_lr)

        warp_d2 = warp_d2.view(b,n-1,-1,h//8,w//8)            # 1/8
        warp_l_d2,warp_r_d2 = warp_d2[:,0],warp_d2[:,1]
        warp_d2 = torch.stack([warp_l_d2,fea_m_d2,warp_r_d2],2)
        warp_d2 = self.lrelu(self.fuse3D(warp_d2))

        warp_d2 = warp_d2.view(b,-1,h//8,w//8)
        warp_d2 = self.lrelu(self.fuse(warp_d2))

        warp_d2 = self.recon(warp_d2)
        if scale == 8:
            # print(self.conv_x4(warp_d2).shape,x1_lr.shape,scale)
            out_x4 = self.conv_x4(warp_d2) +  x1_lr #imresize(x1_d4,0.5)
        else:
            out_x4 = self.conv_x4(warp_d2) +  imresize(x1,0.125)
        # print('out_x4',out_x4.shape)
        # input('cc x4')

        warp_d2 = torch.nn.functional.interpolate(warp_d2,size = (h//4,w//4),mode='bilinear',align_corners=False)
        if scale == 4:
            warp_l_d1 = self.lrx2_dw(fea_l_d1,warp_d2,flow_x01_lr)
            warp_r_d1 = self.lrx2_dw(fea_r_d1,warp_d2,flow_x21_lr)
        else:
            relative_scale = float(scale)/4
            flow_x01_x2 = relative_scale * torch.nn.functional.interpolate(flow_x01_lr,size = (h//4,w//4),mode='bilinear',align_corners=False)
            flow_x21_x2 = relative_scale * torch.nn.functional.interpolate(flow_x21_lr,size = (h//4,w//4),mode='bilinear',align_corners=False)
            warp_l_d1 = self.lrx2_dw(fea_l_d1,warp_d2,flow_x01_x2)
            warp_r_d1 = self.lrx2_dw(fea_r_d1,warp_d2,flow_x21_x2)

        warp_d1 = torch.stack([warp_l_d1,fea_m_d1,warp_r_d1],2)
        warp_d1 = self.lrelu(self.fuse3D_d1(warp_d1))
        # print('warp_d1',warp_d1.shape)
        warp_d1 = warp_d1.view(b,-1,h//4,w//4)             # 1/4
        # print('warp_d1',warp_d1.shape,self.fuse_d1)
        warp_d1 = self.lrelu(self.fuse_d1(warp_d1))
        warp_d1 = self.recon_d1(warp_d1)

        out_x2 = self.conv_x2(warp_d2) + torch.nn.functional.interpolate(out_x4,size = (h//4,w//4),mode='bilinear',align_corners=False)
        # print('out_x2',out_x2.shape)
        # input('cc x2')


        warp_d1 = torch.nn.functional.interpolate(warp_d1,size = (h//2,w//2),mode='bilinear',align_corners=False)

        if scale == 2:
            warp_l_d0 = self.lrx1_dw(fea_l_d0,warp_d1,flow_x01_lr)
            warp_r_d0 = self.lrx1_dw(fea_r_d0,warp_d1,flow_x21_lr)
        else:
            relative_scale = float(scale)/2
            flow_x01_x2 = relative_scale * torch.nn.functional.interpolate(flow_x01_lr,size = (h//2,w//2),mode='bilinear',align_corners=False)
            flow_x21_x2 = relative_scale * torch.nn.functional.interpolate(flow_x21_lr,size = (h//2,w//2),mode='bilinear',align_corners=False)
            warp_l_d0 = self.lrx1_dw(fea_l_d0,warp_d1,flow_x01_x2)
            warp_r_d0 = self.lrx1_dw(fea_r_d0,warp_d1,flow_x21_x2)

        warp_d0 = torch.stack([warp_l_d0,fea_m_d0,warp_r_d0],2)
        warp_d0 = self.lrelu(self.fuse3D_d0(warp_d0))
        warp_d0 = warp_d0.view(b,-1,h//2,w//2)              # 1/2
        warp_d0 = self.lrelu(self.fuse_d0(warp_d0))
        warp_d0 = self.recon_d0(warp_d0)
        # print('warp_d0',warp_d0.shape)

        # out = self.conv_x1(warp_d0) +  torch.nn.functional.interpolate(out_x2,size = (h,w),mode='bilinear',align_corners=False)
        

        hr_fea = self.lrelu(self.ps(self.up_conv1(warp_d0)))
        out = self.out_conv(self.lrelu(self.hr_conv(hr_fea))) + torch.nn.functional.interpolate(out_x2,size = (h,w),mode='bilinear',align_corners=False)
        # print('out',out.shape)
        # input('cc')
        return flow_x01_x2,flow_x21_x2,out
    def forward(self,x,noise_prior,gt=None,is_train = True):
        
        return self.inference(x,noise_prior)
        
        b,n,c,h,w = x.size()

        noise_prior = noise_prior.view(b,4,1,1)
        x0,x1,x2 = x[:,0].contiguous(),x[:,1].contiguous(),x[:,2].contiguous()

        fake_x1 = x1.clone()
        
        
        gen_fake = self.degradeNet(torch.cat([sim_x,noise_prior[:,2:4].clone().repeat(1,1,h,w)],1))
        # x1_ff = gen_fake[mask<0.5].clone()
        fake_x1[mask<0.5] = gen_fake[mask<0.5].clone()

        # x1_r = x1[mask>0.5].clone()
        # x1_rf = gen_fake[mask>0.5].clone()
        

        
        # print(gen_fake.shape,x0.shape,x1.shape,x2.shape)

        # print(fake_v1.shape,x0.shape,x1.shape,x2.shape)
        
        
        
        fea_l_d0,fea_l_d1,fea_l_d2 =  self.fea_extract(x0)
        fea_r_d0,fea_r_d1,fea_r_d2  = self.fea_extract(x2)
        fea_m_d0,fea_m_d1,fea_m_d2 =  self.denoise_extract(fake_x1,noise_prior[:,:2].clone().repeat(1,1,h,w))
        
        x1_d4 = imresize(fake_x1,0.25)
        x0_d4,x2_d4 = imresize(x0,0.25),imresize(x2,0.25)
        flow_x01 = self.flownet(x1_d4,x0_d4)
        flow_x21 = self.flownet(x1_d4,x2_d4)

        flow_x01_x8 = 0.5 * torch.nn.functional.interpolate(flow_x01,scale_factor=0.5,mode='bilinear',align_corners=False)
        flow_x21_x8 = 0.5 * torch.nn.functional.interpolate(flow_x21,scale_factor=0.5,mode='bilinear',align_corners=False)

        # print(flow_x01.shape,fea_m_d2.shape)
        


        warp_l_d2 = self.lrx4_dw(fea_l_d2,fea_m_d2,flow_x01_x8)
        warp_r_d2 = self.lrx4_dw(fea_l_d2,fea_m_d2,flow_x21_x8)
        warp_d2 = torch.stack([warp_l_d2,fea_m_d2,warp_r_d2],2)
        warp_d2 = self.lrelu(self.fuse3D(warp_d2))

        warp_d2 = warp_d2.view(b,-1,h//8,w//8)
        warp_d2 = self.lrelu(self.fuse(warp_d2))

        warp_d2 = self.recon(warp_d2)

        out_x4 = self.conv_x4(warp_d2) +  imresize(x1_d4,0.5)
        

        warp_d2 = torch.nn.functional.interpolate(warp_d2,size = (h//4,w//4),mode='bilinear',align_corners=False)
        flow_x01_x2 = 2.0 * torch.nn.functional.interpolate(flow_x01,scale_factor=2,mode='bilinear',align_corners=False)
        flow_x21_x2 = 2.0 * torch.nn.functional.interpolate(flow_x21,scale_factor=2,mode='bilinear',align_corners=False)


        warp_l_d1 = self.lrx2_dw(fea_l_d1,warp_d2,flow_x01)
        warp_r_d1 = self.lrx2_dw(fea_r_d1,warp_d2,flow_x21)
        warp_d1 = torch.stack([warp_l_d1,fea_m_d1,warp_r_d1],2)
        warp_d1 = self.lrelu(self.fuse3D_d1(warp_d1))
        # print('warp_d1',warp_d1.shape)
        warp_d1 = warp_d1.view(b,-1,h//4,w//4)
        # print('warp_d1',warp_d1.shape,self.fuse_d1)
        warp_d1 = self.lrelu(self.fuse_d1(warp_d1))
        warp_d1 = self.recon_d1(warp_d1)

        out_x2 = self.conv_x2(warp_d2) +  torch.nn.functional.interpolate(out_x4,size = (h//4,w//4),mode='bilinear',align_corners=False)

        warp_d1 = torch.nn.functional.interpolate(warp_d1,size = (h//2,w//2),mode='bilinear',align_corners=False)

        warp_l_d0 = self.lrx1_dw(fea_l_d0,warp_d1,flow_x01_x2)
        warp_r_d0 = self.lrx1_dw(fea_r_d0,warp_d1,flow_x21_x2)
        warp_d0 = torch.stack([warp_l_d0,fea_m_d0,warp_r_d0],2)
        warp_d0 = self.lrelu(self.fuse3D_d0(warp_d0))
        warp_d0 = warp_d0.view(b,-1,h//2,w//2)
        warp_d0 = self.lrelu(self.fuse_d0(warp_d0))
        warp_d0 = self.recon_d0(warp_d0)
        # print('warp_d0',warp_d0.shape)

        # out = self.conv_x1(warp_d0) +  torch.nn.functional.interpolate(out_x2,size = (h,w),mode='bilinear',align_corners=False)
        

        hr_fea = self.lrelu(self.ps(self.up_conv1(warp_d0)))
        out = self.out_conv(self.lrelu(self.hr_conv(hr_fea))) + torch.nn.functional.interpolate(out_x2,size = (h,w),mode='bilinear',align_corners=False)


        if gt is not None:
            return out_x4,out_x2,out,gen_fake
        # #     # loss = ((out - gt)**2).sum()
        #     loss = self.charloss(out,gt) 
        #     loss_x4 = 0.1*self.charloss(out_x4,imresize(gt,0.125))
        #     loss_x2 = 0.1*self.charloss(out_x2,imresize(gt,0.25))
        #     return dict(loss = loss,loss_x4=loss_x4,loss_x2=loss_x2 )
        return x1,out


