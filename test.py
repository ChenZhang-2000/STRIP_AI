import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from data.preprocess import get_gradient, white_balance
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage, Resize, Grayscale
vipshome = r'C:\Users\ChenZhang\AppData\Local\Programs\vips-dev-8.13\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
import imagesize

from data.preprocess import img_preprocess


img_size = 512
downsampling = {"Max": nn.AdaptiveMaxPool2d((img_size, img_size)),
                "Avg": nn.AdaptiveAvgPool2d((img_size, img_size)),
                "Res": Resize((img_size, img_size))}

target = fr"E:\Datasets\STRIP_AI\raw\train\1f9d4f_0.tif"

with torch.no_grad():
    img_preprocess(target, fr"C:\Users\ChenZhang\Desktop\data",
                   fr"b01110_", downsampling)

# print(get_gradient(torch.rand(3,256,256)).shape)
