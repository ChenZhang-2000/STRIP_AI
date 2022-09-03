import os
import shutil
import re
import time
from pathlib import Path
from functools import reduce

import PIL.Image as Image
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split


def collate_fn(l):
    return list(map(lambda x: (x[0][0].cuda(), x[0][1].cuda()), l)), list(map(lambda x: x[1].cuda(), l))


def load_image(directory, patient_id):
    img_data = []
    last_img_num = -1
    for file_name in os.listdir(directory + f"\\{patient_id}"):
        image = torch.load(directory + f"\\{patient_id}\\{file_name}")[:5]
        groups = re.match(r"(\d{1,})_(?:\d{1,})(?:\.gd)", file_name).groups()
        image_num = groups[0]
        image_num = int(image_num)
        if image_num != last_img_num:
            img_data.append([image])
        else:
            img_data[image_num].append(image)
        last_img_num = image_num
    index_data = []
    out_data = []
    index = 0
    # print(img_data)
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

    return out_tensor, index_tensor


class GradientDataset:
    def __init__(self, directory, list_id=None, label=None, store=False, keep_dim=(0, 5)):
        self.directory = directory
        self.data = []
        self._store = store
        if list_id is None:
            list_dir = os.listdir(directory)
        elif isinstance(list_id, list):
            list_dir = list_id
        else:
            raise

        if label is None:
            self.label = [None for _ in list_id]
        else:
            self.label = label

        for i, patient_id in enumerate(list_dir):
            if store:
                self.data.append(torch.load(directory + f"\\{patient_id}\\gradient.gd"))
            else:
                self.data.append(directory + f"\\{patient_id}\\gradient.gd")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._store:
            return self.data[idx], self.label[idx]
        else:
            # print(self.data[idx])
            file_path = self.data[idx]
            data = torch.load(file_path)
            return data, self.label[idx]

    def __add__(self, other):
        if isinstance(other, GradientDataset):
            self.data += other.data
            return self
        else:
            raise TypeError()


class StripDataset:
    def __init__(self, base_directory, aug_set, test_size, random_state=42, dataset_class=GradientDataset, store=False,
                 test_set=True, test_direc=fr"Avg_gd", keep_dim=(0, 5)):
        self.test_set = test_set
        directory = fr"{base_directory}\{aug_set}"
        test_dir = fr"{base_directory}\{test_direc}"

        dim = 2
        labels_map = {'CE': 0, "LAA": 1}
        label_info = pd.read_csv(directory + "\\info.csv", index_col=0)

        patient_ids = label_info["patient_id"].to_list()
        labels = label_info["label"].to_list()

        if random_state is None:
            train_id, valid_id, train_label, valid_label = train_test_split(patient_ids, labels, test_size=test_size)
        else:
            train_id, valid_id, train_label, valid_label = train_test_split(patient_ids, labels, test_size=test_size, random_state=random_state)

        train_id_aug = []
        valid_id_aug = []

        train_label_aug = []
        valid_label_aug = []

        for patient_id_num in os.listdir(directory):
            if not Path(fr"{directory}\{patient_id_num}").is_file():
                patient_id = patient_id_num[:6]
                if patient_id in train_id:
                    train_id_aug.append(patient_id_num)
                    target = label_info.loc[label_info['patient_id'] == patient_id]
                    label = target['label'].iloc[0]
                    train_label_aug.append(one_hot(torch.tensor(labels_map[label]), dim))
                else:
                    valid_id_aug.append(patient_id_num)
                    target = label_info.loc[label_info['patient_id'] == patient_id]
                    label = target['label'].iloc[0]
                    valid_label_aug.append(one_hot(torch.tensor(labels_map[label]), dim))

        self.split = [[train_id, train_label],
                      [train_id_aug, train_label_aug],
                      [valid_id, valid_label],
                      [valid_id_aug, valid_label_aug]]

        self.train = dataset_class(directory, train_id_aug, train_label_aug, store=store)
        # print(f"Train Set Size: {len(self.train)}")
        if test_set:
            self.test = dataset_class(test_dir, valid_id,
                                      list(map(lambda t: one_hot(torch.tensor(labels_map[t]), dim), valid_label)),
                                      store=False, keep_dim=keep_dim)
            self.valid = dataset_class(directory, valid_id_aug, valid_label_aug, store=False, keep_dim=keep_dim)
            # print(f"Valid Set Size: {len(self.train)}")
            # print(f"Test Set Size: {len(self.train)}")
        else:
            self.test = None
            self.valid = dataset_class(directory, valid_id_aug, valid_label_aug, store=False, keep_dim=keep_dim)

    def save_split(self, time_code):
        torch.save(self.split, rf"runs\{time_code}\split.data")


if __name__ == "__main__":
    dataset = StripDataset(r'E:\Datasets\STRIP_AI\processed', test_size=0.2)
    # print(dataset[0])
    loader = DataLoader(dataset.train, collate_fn=collate_fn, batch_size=80, shuffle=True)
    t = time.time()
    for i, (data, label) in enumerate(loader):
        print(time.time()-t)
        t = time.time()
        # print(label)
        # print(data[0][1])
        # print(label[0])
        if i == 1:
            break
