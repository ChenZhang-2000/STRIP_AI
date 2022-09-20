import os
import re
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import imgaug as ia
from imgaug import augmenters as iaa

from dataset import load_image
from preprocess import get_gradient

directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg'
to_directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg_gd_7'


for idx, patient_id in enumerate(os.listdir(directory)):
    Path(f"{to_directory}\\{patient_id}").mkdir(parents=True, exist_ok=True)
    img_data = []
    file_names = []
    last_img_num = -1
    for file_name in os.listdir(directory + f"\\{patient_id}"):
        image = 1 - torch.load(directory + f"\\{patient_id}\\{file_name}")[:3]
        groups = re.match(r"(\d{1,})_(?:\d{1,})(?:\.gd)", file_name).groups()
        image_num = int(groups[0])
        file_names.append(file_name)
        gd = get_gradient(image.cuda(), device='cuda:0').detach()
        if image_num != last_img_num:
            img_data.append([gd[:7]])
        else:
            img_data[image_num].append(gd[:7])
        last_img_num = image_num

    index_data = []
    out_data = []
    index = 0
    for images in img_data:
        image_stack = torch.stack(images)
        index += image_stack.shape[0]
        index_data.append(index)
        out_data.append(image_stack)
    out_tensor = torch.cat(out_data)

    index_tensors = []
    last_index = 0
    for index in index_data:
        idx_t = torch.zeros(out_tensor.shape[0], dtype=bool)
        idx_t[last_index:index] = True
        last_index = index
        index_tensors.append(idx_t)
    index_tensor = torch.stack(index_tensors)

    torch.save((out_tensor, index_tensor), f"{to_directory}\\{patient_id}\\gradient.gd")
