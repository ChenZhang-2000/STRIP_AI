import os
import re
from multiprocessing import Pool, Manager
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import imgaug as ia
from imgaug import augmenters as iaa

from dataset import load_image
from preprocess import get_gradient


ia.seed(42)
rotate = iaa.Affine(rotate=(-180, 180))
# noise = iaa.AdditiveGaussianNoise(scale=(0, 0.2))
crop = iaa.Crop(percent=(0, 0.1))

transform = iaa.Sequential([rotate, crop])

KEEP_DIM = (0, 7)


def augmentation(directory, to_directory, patient_id, times_count):
    img_data = []
    file_names = []
    last_img_num = -1
    for file_name in os.listdir(directory + f"\\{patient_id}"):
        image = 1 - torch.load(directory + f"\\{patient_id}\\{file_name}")[:3]
        groups = re.match(r"(\d{1,})_(?:\d{1,})(?:\.gd)", file_name).groups()
        image_num = int(groups[0])
        file_names.append(file_name)
        np_image = image.transpose(0, 1).transpose(1, 2).numpy()
        aug_image = transform(image=np_image)
        gd = get_gradient(torch.from_numpy(aug_image).transpose(1, 2).transpose(0, 1).cuda(), device='cuda:0').detach()
        if image_num != last_img_num:
            img_data.append([gd[KEEP_DIM[0]:KEEP_DIM[1]]])
        else:
            img_data[image_num].append(gd[KEEP_DIM[0]:KEEP_DIM[1]])
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

    Path(f"{to_directory}\\{patient_id}_{times_count}").mkdir(parents=True, exist_ok=True)
    torch.save((out_tensor, index_tensor), f"{to_directory}\\{patient_id}_{times_count}\\gradient.gd")


if __name__ == "__main__":
    # dataset = GradientDataset(rf'E:\Datasets\STRIP_AI\processed\Avg')
    directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg'
    to_directory = rf'E:\Datasets\STRIP_AI\new_processed\Avg_Aug_7'

    patient_ids = [patient_id for patient_id in os.listdir(directory)]

    info = pd.read_csv(rf"E:\Datasets\STRIP_AI\raw\train.csv")
    df = info[['patient_id', 'label']].drop_duplicates(subset=['patient_id'])

    ce_df = df.loc[df['label'] == "CE"]
    ce_df = ce_df.loc[[(i in patient_ids) for i in ce_df['patient_id']]]
    laa_df = df.loc[df['label'] == "LAA"]
    laa_df = laa_df.loc[[(i in patient_ids) for i in laa_df['patient_id']]]

    draw_ce = np.random.choice(ce_df['patient_id'], len(ce_df)*7)
    draw_laa = np.random.choice(laa_df['patient_id'], len(laa_df)*7)
    sample_size = len(ce_df)*7 + len(laa_df)*7

    draw_all = np.concatenate((draw_ce, draw_laa))
    # print(draw_all.shape)

    times_count_dict = defaultdict(lambda: 1)

    pool = Pool(8)
    info = []
    for patient_id in draw_all:
        info.append((directory, to_directory, patient_id, times_count_dict[patient_id]))
        times_count_dict[patient_id] += 1

    pool.starmap(augmentation, info)

    pool.close()
    pool.join()

    # for i, patient_id in enumerate(os.listdir(to_directory)):
    #     if i >= 9:
    #         break

    #     images = []
    #     for file_name in os.listdir(to_directory + f"\\{patient_id}"):
    #         image = torch.load(to_directory + f"\\{patient_id}\\{file_name}")[:3]
    #         # print(image.dtype)
    #         groups = re.match(r"(\d{1,})_(?:\d{1,})(?:\.gd)", file_name).groups()
    #         np_image = image.transpose(0, 1).transpose(1, 2).numpy()
    #         images.append(np_image)
    #         # print(np_image.dtype)

    #     ia.imshow(np.hstack(images))
