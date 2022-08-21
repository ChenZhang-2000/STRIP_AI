import os
import re
import time
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


class Dataset:
    def __init__(self, directory, list_id=None, label=None):
        self.data = []
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
            img_data = []
            last_img_num = -1
            for file_name in os.listdir(directory+f"\\{patient_id}"):
                image = ToTensor()(Image.open(directory+f"\\{patient_id}\\{file_name}"))
                groups = re.match(r"(\d{1,})_(?:\d{1,})(?:\.png)", file_name).groups()
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

            self.data.append((out_tensor, index_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __add__(self, other):
        if isinstance(other, Dataset):
            self.data += other.data
            return self
        else:
            raise TypeError()


class StripDataset:
    def __init__(self, directory, test_size, random_state=42):
        dim = 2
        labels_map = {'CE': 0, "LAA": 1}
        label_info = pd.read_csv(directory + "\\info.csv", index_col=0)

        patient_ids = []
        labels=[]
        for _, (label, patient_id) in label_info.iterrows():
            # print(label)
            # print(patient_id)
            patient_ids.append(patient_id)
            labels.append(one_hot(torch.tensor(labels_map[label]), dim))
        if random_state is None:
            train_id, valid_id, train_label, valid_label = train_test_split(patient_ids, labels, test_size=test_size)
        else:
            train_id, valid_id, train_label, valid_label = train_test_split(patient_ids, labels, test_size=test_size, random_state=random_state)

        self.train = Dataset(directory, train_id, train_label)
        self.valid = Dataset(directory, valid_id, valid_label)


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
