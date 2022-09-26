import os
from pathlib import Path
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
from model.networks.classifier import ResNet


directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg'
to_directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg_gd_7'


for idx, patient_id in enumerate(os.listdir(to_directory)):
    if Path(to_directory + f"\\{patient_id}").is_dir():
        for file_name in os.listdir(to_directory + f"\\{patient_id}"):
            images = torch.load(to_directory + f"\\{patient_id}\\{file_name}")
            # print(file_name)
            for i, image in enumerate(images):
                out = ResNet(image)
                if out.isnan().any():
                    print(file_name, i)
                    print()




