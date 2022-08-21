import time

from data.dataset import StripDataset, collate_fn
from model.networks.classifier import Classifier

from torch.utils.data import DataLoader


if __name__ == "__main__":
    dataset = StripDataset(r'E:\Datasets\STRIP_AI\processed', test_size=0.2)
    # print(dataset[0])
    loader = DataLoader(dataset.train, collate_fn=collate_fn, batch_size=80, shuffle=True)
    c = Classifier().cuda()
    t = time.time()
    for i, (data, label) in enumerate(loader):
        print(time.time()-t)
        t = time.time()
        c(data)
        # print(label)
        # print(data[0][1])
        # print(label[0])
        if i == 1:
            break