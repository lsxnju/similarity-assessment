import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from i3d import I3D
import numpy as np
from matplotlib import pyplot as plt
from grad_cam import GradCam
from utils import get_video_frame
import time 
import cv2
def show_cam_on_image(img, mask, i, imdir):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img.permute(1,2,0))
    cam = cam / np.max(cam)
    if not os.path.exists(imdir):
        os.makedirs(imdir) 
    cv2.imwrite(imdir+"/cam_"+str(i).zfill(6)+'.jpg', np.uint8(255 * cam))


if __name__ == '__main__':
    import os
    # datasets
    from gluoncv.data import HMDB51
    import mxnet.gluon.data
    from mxnet.gluon.data.vision import transforms
    from gluoncv.data.transforms import video

    transform_train = transforms.Compose([
        video.VideoCenterCrop(size=224),
        video.VideoToTensor()
    ])
    
    train_dataset = HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_train_split_1_rawframes.txt',train=True, new_length=79, transform=transform_train)
    train_data = mxnet.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset= HMDB51(root='/data/luyi/lsx/datasets/hmdb51/rawframes',setting='/data/luyi/lsx/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt',train=False, new_length=79, transform=transform_train)
    test_data = mxnet.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    # class_list
    classes_list = os.listdir('/data/luyi/lsx/datasets/hmdb51/rawframes')
    classes_list.sort()

    rgb_weights_path='/data/luyi/lsx/i3d/model_weight/model_rgb.pth'#/net/net3.pth'

    # rgb
    print("rgb")
    i3d_rgb = torch.load('/data/luyi/lsx/i3d/net/net3.pth')
    i3d_rgb.requires_grad_(True)
    print(i3d_rgb.mixed_5c._modules.keys())
    

    grad_cam = GradCam(model=i3d_rgb, feature_module=i3d_rgb.mixed_5c, target_layer_names=["branch_0"])

    batch=0
    size=len(test_data)
    print(size)
    # assert 1==2
    loss_sum=0
    correct_sum=0
 
    for x,y in test_data:
        rgb_sample = x.asnumpy().squeeze(0)
        x=torch.Tensor(rgb_sample)
        print(torch.Tensor(rgb_sample).shape)
        start_time = time.time()

        
        pred, out_logit = i3d_rgb(x.cuda())
    
        mask = grad_cam(x, None)
        imgs=sorted(os.listdir('/data/luyi/lsx/i3d/gradcam'))
        imdir='/data/luyi/lsx/i3d/gradcam'.split('/')[-1]+'_cams'

        imgs = x.permute(0,2,1,3,4).squeeze()
        print(imgs.shape)
        for i, fimg in enumerate(imgs):
            k=i*len(mask)//len(imgs)
            img = fimg

            show_cam_on_image(img, mask[k], i, imdir)
        batch =batch+1
        if batch==1: 
            break


