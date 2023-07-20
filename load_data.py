import sys
sys.path.append(r'G:\My Drive\23-similarity\similarity-assessment\i3d')


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
from i3d import I3D
import numpy as np
from matplotlib import pyplot as plt
from videoset_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os


videos_root=r'G:\My Drive\23-similarity\data_preprocess'
annotation_file=r'G:\My Drive\23-similarity\data_preprocess\label.txt'
dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template='img_{:05d}.jpg',
        transform=None,
        test_mode=False
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for epoch in range(10):
    for video_batch, labels in dataloader:
        print(labels)