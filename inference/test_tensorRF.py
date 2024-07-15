import argparse
import os
import os.path as osp
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import sys

sys.path.append('/dataset/kunzhou/project/package_l')


from config import config
from utils import common, model_opr


from network import SimpleNet as SIVSR

import torchvision

from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr
import json 
from utils.core import imresize
import numpy as np
import cv2
device = torch.device('cuda')
import glob

def imgtotensor(img):
    img = img.transpose(2,0,1)
    lr_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
    return lr_tensor
def forward(i0,i1,i2,prior,model):
    return model( torch.stack([i0,i1,i2],1),prior)[-1]
def forward_x4(i0,i1,i2,prior,model):
    result_f = forward(i0,i1,i2,prior,model)

    result = forward(torch.flip(i0,(-1,)),torch.flip(i1,(-1,)),torch.flip(i2,(-1,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-1,))

    result = forward(torch.flip(i0,(-2,)),torch.flip(i1,(-2,)),torch.flip(i2,(-2,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,))

    result = forward(torch.flip(i0,(-2,-1)),torch.flip(i1,(-2,-1)),torch.flip(i2,(-2,-1)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,-1))

    
    return 0.25 * result_f

model = SIVSR(config).to(device)
print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

model_opr.load_model(model, '/dataset/kunzhou/project/rendering/nerflix_sr/exp/baseline_x4/nerflix_plus.pt', strict=False, cpu=True)
# model_opr.load_model(model,  '/dataset/kunzhou/move_project/vfi/logs/fast_ivm_flow/GammaLightGAIndoor10/models/iteration200000.pth', strict=False, cpu=True)
# 


all_object_file = './result_fast/all.txt'
if os.path.exists(all_object_file) is False:
    with open(all_object_file,'w') as f:
        f.write('-------------result_tensorf---------------- \n')

hold_every = 8

settings = {'flower':[40,0.2],'fern':[30,0.4],'room':[40,0.5],'horns':[30,0.5],'fortress':[90,0.1],'trex':[80,0.1],'orchids':[80,0.7],'leaves':[30,0.2]}
object_list = ['fern','flower','room','horns','fortress','trex','orchids','leaves']
# object_list = object_list[:1]
metrics_dict = {}
for object_name in object_list:
    metrics_dict[object_name] = []
for object_name in object_list :
    root_dir = '/dataset/kunzhou/project/rendering/nerf_data/nerf_llff_data/{}'.format(object_name)
    # print(root_dir)
    result_path = './result_fast/{}'.format(object_name)
    if os.path.exists(result_path) is False:
        os.mkdir(result_path)

    json_file = os.path.join(root_dir,'img_pairs.json')
    with open(json_file,'r') as f:
        img_pair_dict = json.load(f)
    print('img_pair_dict',img_pair_dict)
    image_paths = sorted(glob.glob(os.path.join(root_dir, 'images_4/*')))
   
    i_test = np.arange(0, len(image_paths), hold_every) 
    all_rgbimgs = []

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        img = cv2.imread(image_path)
        all_rgbimgs.append([img,image_path])

    psnr_l = []
    ssim_l = []
    psnr_init_l = []
    ssim_init_l = []
    for i in i_test:
        gt_img,gt_url = all_rgbimgs[i]
        img_name = gt_url.split('/')[-1]
        if i == 0:
            i1,i2 = 1,2 
        elif i == len(image_paths) -1:
            i1,i2 = len(image_paths) -2,len(image_paths) -3
        else:
            i1 = i - 1
            i2 = i + 1

        i1,i2 = img_pair_dict[str(i)][:2]
        # print(i,i1,i2)
        # if i not in [4*8,5*8,6*8]:
            # i1,i2 = img_pair_dict[str(i)]
        # print(i,i1,i2)
        # print(i,i1,i2,img_pair_dict[str(i)])

        _ref = os.path.join('/dataset/kunzhou/project/rendering/TensoRF-main/result/llff/',\
                object_name,img_name)

        # _ref = os.path.join('/newdata/kunzhou/project/rendering/svox2-master/opt/ckpt/{}/test_renders/'.format(object_name),\
        #         img_name)

        print('_ref',_ref)
        img1 = cv2.imread(_ref)
        H,W = img1.shape[:2]
        H,W = int(H//8)*8,(W//8)*8
        img1 = img1[:H,:W]

        img0 = all_rgbimgs[i1][0][:H,:W]
        img2=  all_rgbimgs[i2][0][:H,:W]

        gt_img = gt_img[:H,:W]

        tensor1 = imgtotensor(img1)
        tensor0 = imgtotensor(img0)
        tensor2 = imgtotensor(img2)


        if object_name not in settings:
            jpeg_quality = 50 *0.01
            noisy = 0.5 #* 0.02  * 90
        else:
            jpeg_quality,noisy = settings[object_name]
            jpeg_quality = jpeg_quality * 0.01

        prior = np.array([jpeg_quality,noisy])
        prior = torch.from_numpy(prior).to(tensor1.device).float()

        with torch.no_grad():
            # nbr_img = torch.stack([tensor0,tensor1,tensor2],1)
            # sr_norm,sr_vsr = model(nbr_img,prior)
            sr_vsr = forward_x4(tensor0,tensor1,tensor2,prior,model)
            # sr_vsr_half1 = sr_vsr_half1.clamp(0,1)

            # sr_vsr_half2 = forward_x4(tensor2,tensor1,tensor0,prior,model)
            # sr_vsr_half2 = sr_vsr_half2.clamp(0,1)

            # sr_vsr = 0.5*sr_vsr_half1 + 0.5 * sr_vsr_half2

            output_new = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)
            output_new = np.transpose(output_new,(1,2,0))

            output_old = tensor1.detach().cpu().numpy()[0].astype(np.float32)
            output_old = np.transpose(output_old,(1,2,0))

        psnr_init = calculate_psnr(output_old*255.0, gt_img)
        psnr_pred = calculate_psnr(output_new*255.0, gt_img)


        ssim_pred = calculate_ssim(output_new*255.0, gt_img)
        ssim_init = calculate_ssim(output_old*255.0, gt_img)

        ipath = os.path.join(result_path,img_name)


        psnr_init_l.append(psnr_init)
        ssim_init_l.append(ssim_init)

        psnr_l.append(psnr_pred)
        ssim_l.append(ssim_pred)

        print('ipath',ipath,psnr_init,ssim_init)
        print('ipath',ipath,psnr_pred,ssim_pred)
        cv2.imwrite(ipath,output_new*255 )

        # print('ipath',ipath,psnr_pred,ssim_pred)
        # input('cc')
    avg_psnr_init = sum(psnr_init_l) / len(psnr_init_l)
    avg_ssim_init = sum(ssim_init_l) / len(ssim_init_l)

    avg_psnr = sum(psnr_l) / len(psnr_l)
    avg_ssim = sum(ssim_l) / len(ssim_l)

    print(object_name,'init',avg_psnr_init,avg_ssim_init)
    print(object_name,'refine',avg_psnr,avg_ssim)


    with open(os.path.join(result_path,'metrics.txt'),'a') as f:
        f.write('q:{}n{}\n'.format(jpeg_quality * 100,noisy/(0.02  * 90)))
        f.write('init {}/{} \n'.format(avg_psnr_init,avg_ssim_init))
        f.write('refine {}/{} \n '.format(avg_psnr,avg_ssim))
    metrics_dict[object_name].append([avg_psnr_init,avg_ssim_init,avg_psnr,avg_ssim])

avg_psnr_objects = []
avg_ssim_objects = []
for object_name in metrics_dict:
    print('---------------------------')
    print(object_name,metrics_dict[object_name])
    p0,s0,p1,s1 = metrics_dict[object_name][0]
    avg_psnr_objects.append([p0,p1])
    avg_ssim_objects.append([s0,s1])
    with open(all_object_file,'a') as f:
        f.write('object {} init metrics {}/{} refine metrics {}/{} \n'.format(object_name,p0,s0,p1,s1))

avg_psnr_objects = np.array(avg_psnr_objects).reshape(-1,2)
avg_ssim_objects = np.array(avg_ssim_objects).reshape(-1,2)
print('init',avg_psnr_objects[:,0].mean(),avg_ssim_objects[:,0].mean())
print('refine',avg_psnr_objects[:,1].mean(),avg_ssim_objects[:,1].mean())
with open(all_object_file,'a') as f:
    f.write(' init metrics {}/{} refine metrics {}/{} \n'.format(avg_psnr_objects[:,0].mean(),avg_ssim_objects[:,0].mean(),avg_psnr_objects[:,1].mean(),avg_ssim_objects[:,1].mean()))

        
avg_psnr_objects = np.array(avg_psnr_objects).reshape(-1,2)
avg_ssim_objects = np.array(avg_ssim_objects).reshape(-1,2)
print('init',avg_psnr_objects[:,0].mean(),avg_ssim_objects[:,0].mean())
print('refine',avg_psnr_objects[:,1].mean(),avg_ssim_objects[:,1].mean())