import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn
from torchvision.transforms import ToPILImage, Resize, Grayscale
import imgaug as ia

vipshome = r'C:\Users\ChenZhang\AppData\Local\Programs\vips-dev-8.13\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
import imagesize


sqrt2 = math.sqrt(2)

X_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]]])
Y_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]])
PXPY_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -sqrt2 / 2, 0.], [0., 0., sqrt2 / 2]]]])
PXNY_GD_KERNEL = torch.tensor([[[[0., 0., sqrt2 / 2], [0., -sqrt2 / 2, 0.], [0., 0., 0.]]]])
NXPY_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -sqrt2 / 2, 0.], [sqrt2 / 2, 0., 0.]]]])
NXNY_GD_KERNEL = torch.tensor([[[[sqrt2 / 2, 0., 0.], [0., -sqrt2 / 2, 0.], [0., 0., 0.]]]])


def resize(x, device):
    c, w, h = x.shape[0], x.shape[1], x.shape[2]
    d = x.shape[1] - x.shape[2]
    if d > 0:
        return torch.cat([torch.ones(c, w, d//2, device=device), x, torch.ones(c, w, d-d//2, device=device)], dim=2)
    elif d < 0:
        return torch.cat([torch.ones(c, -d//2, h, device=device), x, torch.ones(c, -d+d//2, h, device=device)], dim=1)
    else:
        return x


def variance_mask(img, device):
    _, h, w = img.shape
    cm = torch.ones(w, dtype=bool, device=device)
    rm = torch.ones(h, dtype=bool, device=device)
    for c in range(3):
        channel = img[c, :, :]
        col_var = torch.std(channel, dim=0) ** 2
        row_var = torch.std(channel, dim=1) ** 2

        col_var_max = torch.max(col_var)
        row_var_max = torch.max(row_var)

        cm *= (col_var < col_var_max/7)
        rm *= (row_var < row_var_max/7)
    return cm, rm


def split_img(img, device):
    _, h, w = img.shape
    col_mask, row_mask = variance_mask(img, device)

    col_start_mask = torch.cat([col_mask[0:1] == 1, (col_mask[1:] == 1) * (col_mask[:-1] == 0)])
    row_start_mask = torch.cat([row_mask[0:1] == 1, (row_mask[1:] == 1) * (row_mask[:-1] == 0)])
    col_end_mask = torch.cat([(col_mask[1:] == 0) * (col_mask[:-1] == 1), col_mask[-1:] == 1])
    row_end_mask = torch.cat([(row_mask[1:] == 0) * (row_mask[:-1] == 1), row_mask[-1:] == 1])

    col_start_idx = torch.arange(w)[col_start_mask]
    row_start_idx = torch.arange(h)[row_start_mask]
    col_end_idx = torch.arange(w)[col_end_mask]
    row_end_idx = torch.arange(h)[row_end_mask]

    col_empty_len = col_end_idx - col_start_idx + 1
    row_empty_len = row_end_idx - row_start_idx + 1

    block_col_mask = (col_empty_len > w / 8) * (col_start_idx != 0) * (col_end_idx != w - 1)
    block_row_mask = (row_empty_len > h / 8) * (row_start_idx != 0) * (row_end_idx != h - 1)

    block_col_start = torch.cat([torch.tensor([0]), col_end_idx[block_col_mask]])
    block_row_start = torch.cat([torch.tensor([0]), row_end_idx[block_row_mask]])
    block_col_end = torch.cat([col_start_idx[block_col_mask], torch.tensor([w - 1])])
    block_row_end = torch.cat([row_start_idx[block_row_mask], torch.tensor([h - 1])])

    blocks = []
    empty = True
    for i in range(len(block_col_start)):
        for j in range(len(block_row_start)):
            block = img[:, block_row_start[j]:block_row_end[j],
                        block_col_start[i]:block_col_end[i]]
            keep, out_img = rm_bg(block, device)
            empty = empty and not keep
            if keep:
                blocks.append(out_img.detach())
    if empty:
        blocks.append(img.detach())

    return blocks


def rm_bg(img, device):
    _, h, w = img.shape
    col_mask, row_mask = variance_mask(img, device)

    col_mask = col_mask.reshape(1, w).repeat(h, 1)
    row_mask = row_mask.reshape(h, 1).repeat(1, w)
    block_bg_mask = (row_mask + col_mask).float()

    point_tl = block_bg_mask.flatten().argmin()
    point_br = block_bg_mask.flip(0, 1).flatten().argmin()

    pos_tl = (torch.div(point_tl, w, rounding_mode='floor'), point_tl % w)
    pos_br = (h-torch.div(point_br, w, rounding_mode='floor'), w-point_br % w)

    keep = torch.std(img) >= 0.1
    # print(torch.std(img))

    return keep, img[:, pos_tl[0]:pos_br[0], pos_tl[1]:pos_br[1]].detach()


def recursively_split(image, device):
    blocks = split_img(image, device)
    if len(blocks) == 1:
        return blocks
    else:
        out = []
        for block in blocks:
            out += split_img(block, device)
        return out


def get_gradient(image):
    _, h, w = image.shape
    gray_image = Grayscale()(image).reshape(1, 1, h, w)
    image = image.reshape(1, 3, h, w)

    pxzy = nn.functional.conv2d(gray_image, X_GD_KERNEL, stride=1, padding=1)
    zxpy = nn.functional.conv2d(gray_image, Y_GD_KERNEL, stride=1, padding=1)
    pxpy = nn.functional.conv2d(gray_image, PXPY_GD_KERNEL, stride=1, padding=1)
    pxny = nn.functional.conv2d(gray_image, PXNY_GD_KERNEL, stride=1, padding=1)
    # nxpy = nn.functional.conv2d(gray_image, NXPY_GD_KERNEL, stride=1, padding=1)
    # nxny = nn.functional.conv2d(gray_image, NXNY_GD_KERNEL, stride=1, padding=1)
    # print(gray_image.shape)

    return torch.cat([image, pxzy, zxpy, pxpy, pxny], dim=1).reshape(7, h, w).detach()


def white_balance(image, device):
    r_bg_color = torch.bincount(image[0].flatten()).argmax().detach()
    g_bg_color = torch.bincount(image[1].flatten()).argmax().detach()
    b_bg_color = torch.bincount(image[2].flatten()).argmax().detach()

    balanced = torch.zeros(*image.shape, device=device)
    balanced[0] = image[0] / r_bg_color
    balanced[1] = image[1] / g_bg_color
    balanced[2] = image[2] / b_bg_color

    return balanced.detach()


def img_preprocess(in_dir, out_dir, file_name, device, downsampling_methods, train=True):
    w, h = imagesize.get(in_dir)
    w, h = w // 5, h // 5
    out = pyvips.Image.thumbnail(in_dir, w, height=h)

    array = out.numpy()
    t = torch.from_numpy(array).transpose(1, 2).transpose(0, 1).to(device)
    image = white_balance(t, device).detach()

    # print(image.device)
    blocks = recursively_split(image, device)

    patient_id, image_num = file_name.split('\\')

    if train:
        for i, block in enumerate(blocks):
            block = get_gradient(1-block)
            padded = resize(block, device)
            for method in downsampling_methods:
                im = downsampling_methods[method](padded).detach().cpu()
                torch.save(im, fr"{out_dir}\{method}\{patient_id}\{image_num}_{i}.gd")
    else:
        method = downsampling_methods[0]
        img_data = [[] for _ in downsampling_methods]
        last_img_num = -1
        for i, block in enumerate(blocks):
            block = get_gradient(1-block)
            padded = resize(block, device)
            im = downsampling_methods[method](padded).detach().cpu()
            if image_num != last_img_num:
                img_data.append([im])
            else:
                img_data[image_num].append(im)
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

        torch.save((out_tensor, index_tensor), fr"{out_dir}\{method}\{patient_id}.gd")

if __name__ == '__main__':
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\reduced\CE'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\reduced\CE', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\reduced\LAA'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\reduced\LAA', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\processed\CE'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\processed\CE', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\processed\LAA'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\processed\LAA', f))

    DEVICE = 'cuda:0'

    X_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]]]).to(DEVICE)
    Y_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]]).to(DEVICE)
    PXPY_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -sqrt2/2, 0.], [0., 0., sqrt2/2]]]]).to(DEVICE)
    PXNY_GD_KERNEL = torch.tensor([[[[0., 0., sqrt2/2], [0., -sqrt2/2, 0.], [0., 0., 0.]]]]).to(DEVICE)
    NXPY_GD_KERNEL = torch.tensor([[[[0., 0., 0.], [0., -sqrt2/2, 0.], [sqrt2/2, 0., 0.]]]]).to(DEVICE)
    NXNY_GD_KERNEL = torch.tensor([[[[sqrt2/2, 0., 0.], [0., -sqrt2/2, 0.], [0., 0., 0.]]]]).to(DEVICE)

    info = pd.read_csv(rf"E:\Datasets\STRIP_AI\raw\train.csv")

    df = info[['patient_id', 'label']].drop_duplicates(subset=['patient_id'])
    # print(df)
    df.to_csv(fr"E:\Datasets\STRIP_AI\processed\info.csv")

    img_size = 512
    downsampling = {"Max": nn.AdaptiveMaxPool2d((img_size, img_size)),
                    "Avg": nn.AdaptiveAvgPool2d((img_size, img_size)),
                    "Res": Resize((img_size, img_size))}

    for idx in range(len(info)):
        # image_id = info['image_id'][idx]
        label = info['label'][idx]
        patient_id = info['patient_id'][idx]
        img_num = info['image_num'][idx]

        filename = rf'E:\Datasets\STRIP_AI\raw\train\{patient_id}_{img_num}.tif'
        print(f"Image: {patient_id}_{img_num}")
        print(f"Start Time: {time.ctime()}")
        print()
        Path(f"E:\\Datasets\\STRIP_AI\\processed\\Max\\{patient_id}").mkdir(parents=True, exist_ok=True)
        Path(f"E:\\Datasets\\STRIP_AI\\processed\\Avg\\{patient_id}").mkdir(parents=True, exist_ok=True)
        Path(f"E:\\Datasets\\STRIP_AI\\processed\\Res\\{patient_id}").mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            try:
                img_preprocess(filename, fr"E:\Datasets\STRIP_AI\processed", fr"{patient_id}\{img_num}",
                               downsampling_methods=downsampling, device=DEVICE)
            except RuntimeError:
                X_GD_KERNEL = X_GD_KERNEL.to('cpu')
                Y_GD_KERNEL = Y_GD_KERNEL.to('cpu')
                PXPY_GD_KERNEL = PXPY_GD_KERNEL.to('cpu')
                PXNY_GD_KERNEL = PXNY_GD_KERNEL.to('cpu')
                NXPY_GD_KERNEL = NXPY_GD_KERNEL.to('cpu')
                NXNY_GD_KERNEL = NXNY_GD_KERNEL.to('cpu')
                img_preprocess(filename, fr"E:\Datasets\STRIP_AI\processed", fr"{patient_id}\{img_num}",
                               downsampling_methods=downsampling, device='cpu')

                X_GD_KERNEL = X_GD_KERNEL.to(DEVICE)
                Y_GD_KERNEL = Y_GD_KERNEL.to(DEVICE)
                PXPY_GD_KERNEL = PXPY_GD_KERNEL.to(DEVICE)
                PXNY_GD_KERNEL = PXNY_GD_KERNEL.to(DEVICE)
                NXPY_GD_KERNEL = NXPY_GD_KERNEL.to(DEVICE)
                NXNY_GD_KERNEL = NXNY_GD_KERNEL.to(DEVICE)

        torch.cuda.empty_cache()
