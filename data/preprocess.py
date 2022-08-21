import os
import pandas as pd
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage, Resize
vipshome = r'C:\Users\ChenZhang\AppData\Local\Programs\vips-dev-8.13\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
import imagesize


def variance_mask(img):
    _, h, w = img.shape
    cm = torch.ones(w, dtype=bool)
    rm = torch.ones(h, dtype=bool)
    for c in range(3):
        channel = img[c, :, :]
        col_var = torch.std(channel, dim=0) ** 2
        row_var = torch.std(channel, dim=1) ** 2

        col_var_max = torch.max(col_var)
        row_var_max = torch.max(row_var)

        cm *= (col_var < col_var_max/7)
        rm *= (row_var < row_var_max/7)
    return cm, rm


def split_img(img):
    _, h, w = img.shape
    col_mask, row_mask = variance_mask(img)

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
            keep, out_img = rm_bg(block)
            empty = empty and not keep
            if keep:
                blocks.append(out_img)
    if empty:
        blocks.append(img)

    return blocks


def rm_bg(img):
    _, h, w = img.shape
    col_mask, row_mask = variance_mask(img)

    col_mask = col_mask.reshape(1, w).repeat(h, 1)
    row_mask = row_mask.reshape(h, 1).repeat(1, w)
    block_bg_mask = (row_mask + col_mask).float()

    point_tl = block_bg_mask.flatten().argmin()
    point_br = block_bg_mask.flip(0, 1).flatten().argmin()

    pos_tl = (torch.div(point_tl, w, rounding_mode='floor'), point_tl % w)
    pos_br = (h-torch.div(point_br, w, rounding_mode='floor'), w-point_br % w)

    keep = torch.std(img) >= 0.1
    # print(torch.std(img))

    return keep, img[:, pos_tl[0]:pos_br[0], pos_tl[1]:pos_br[1]]


def recursively_split(image):
    blocks = split_img(image)
    if len(blocks) == 1:
        return blocks
    else:
        out = []
        for block in blocks:
            out += split_img(block)
        return out



def img_preprocess(in_dir, out_dir, img_size=256):
    w, h = imagesize.get(in_dir)
    w, h = w // 5, h // 5
    out = pyvips.Image.thumbnail(in_dir, w, height=h)

    array = out.numpy()
    t = torch.from_numpy(array).transpose(1, 2).transpose(0, 1)
    # print(t.shape)
    image = t.float()/255

    blocks = recursively_split(image)

    for i, block in enumerate(blocks):
        # print(block.transpose(0, 1).transpose(1, 2).numpy())\
        im = ToPILImage()(Resize([img_size, img_size])(block))
        im.save(f"{out_dir}_{i}.png")


def to_square(x):
    w, h = x.shape[1], x.shape[2]
    d = x.shape[1] - x.shape[2]
    if d > 0:
        return torch.cat([torch.zeros(3, w, d//2), x, torch.zeros(3, w, d-d//2)], dim=2)
    elif d < 0:
        return torch.cat([torch.zeros(3, -d//2, h), x, torch.zeros(3, -d+d//2, h)], dim=1)
    else:
        return x


if __name__ == '__main__':
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\reduced\CE'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\reduced\CE', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\reduced\LAA'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\reduced\LAA', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\processed\CE'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\processed\CE', f))
    # for f in os.listdir(r'E:\Datasets\STRIP_AI\processed\LAA'):
    #     os.remove(os.path.join(r'E:\Datasets\STRIP_AI\processed\LAA', f))

    info = pd.read_csv(rf"E:\Datasets\STRIP_AI\raw\train.csv")

    df = info[['patient_id', 'label']].drop_duplicates(subset=['patient_id'])
    # print(df)
    df.to_csv(fr"E:\Datasets\STRIP_AI\processed\info.csv")

    for idx in range(len(info)):
        # image_id = info['image_id'][idx]
        label = info['label'][idx]
        patient_id = info['patient_id'][idx]
        img_num = info['image_num'][idx]

        filename = rf'E:\Datasets\STRIP_AI\raw\train\{patient_id}_{img_num}.tif'
        print(filename)
        Path(f"E:\\Datasets\\STRIP_AI\\processed\\{patient_id}").mkdir(parents=True, exist_ok=True)

        img_preprocess(filename, f"E:\\Datasets\\STRIP_AI\\processed\\{patient_id}\\{img_num}")
        img_preprocess(filename, f"E:\\Datasets\\STRIP_AI\\processed\\{patient_id}\\{img_num}")
        # break
        # print(img.shape)
        # Save with LZW compression
        # Path(f"E:\\Datasets\\STRIP_AI\\png_raw\\train\\{label}\\{patient_id}").mkdir(parents=True, exist_ok=True)
        # out.tiffsave(rf'E:\Datasets\STRIP_AI\png_raw\train\{label}\{patient_id}\{img_num}.tif', tile=True, compression='lzw')

