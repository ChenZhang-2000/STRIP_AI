import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Accuracy, AUROC, Precision
from torch.utils.tensorboard import SummaryWriter

from data.dataset import collate_fn


class Trainer:
    def __init__(self, classifier, num_workers, in_channel, backbone_layers, attention_layers, img_size, bs, max_epoch,
                 data, lr, optimizer, loss, scheduler, scheduler_params,  device='cuda:0', weights_directory=r'runs',
                 selection_metric='AUROC', random_seed=42):

        self.num_workers = num_workers
        self.random_seed = random_seed
        self.device = device
        current_time = time.localtime()
        self.time_code = "".join([str(current_time.tm_year), str(current_time.tm_mon).zfill(2),
                                  str(current_time.tm_mday).zfill(2), str(current_time.tm_hour).zfill(2),
                                  str(current_time.tm_min).zfill(2)])

        self.in_channel = in_channel
        self.classifier = classifier(img_size, backbone_layers, attention_layers, in_channel).cuda(device)
        self.data = data
        self.optimizer = optimizer(self.classifier.parameters(), lr=lr)
        self.scheduler = scheduler(self.optimizer, **scheduler_params)
        self.loss = loss

        self.epoch_loss = 0
        self.img_size = img_size
        self.bs = bs
        self.max_epoch = max_epoch
        self.lr = lr

        self.metrics = [Accuracy(), AUROC(num_classes=2), Precision()]
        self.weights_directory = weights_directory
        self.selection_metric = selection_metric

        self.highest_epoch = 0
        self.highest_metric = 0
        self.writer = SummaryWriter(log_dir=rf"runs\{self.time_code}_{random_seed}")

        Path(fr"runs\{self.time_code}_{random_seed}\weights").mkdir(parents=True, exist_ok=True)
        self.data.save_split(fr"{self.time_code}_{random_seed}")

    def train(self):
        loader = DataLoader(self.data.train, collate_fn=collate_fn, batch_size=self.bs, shuffle=True,
                            num_workers=self.num_workers)
        print("Training")
        for epoch in range(self.max_epoch):
            t = time.time()
            self.epoch_loss = 0
            for i, (data, label) in enumerate(loader):
                torch.cuda.synchronize()

                bs = len(data)
                y = torch.argmax(torch.stack(label), dim=1)

                probs = self.classifier(data)

                loss = self.loss(probs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.epoch_loss += loss.detach().cpu().item() * bs

                gc.collect()
                del data
                del y
                del probs
                del loss
                gc.collect()

                # torch.cuda.empty_cache()

            self.scheduler.step()
            torch.cuda.empty_cache()

            train_loss, valid_loss, metrics = self.evaluate()
            self.writer.add_scalar("Training Loss", train_loss, epoch)
            self.writer.add_scalar("Validation Loss", valid_loss, epoch)
            self.writer.flush()
            print(f"Epoch {epoch+1}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"  Epoch Running Time: {time.time()-t}")
            print(f"  Training Loss: {train_loss}")
            print(f"  Validation Loss: {valid_loss}")
            for metric_name, value in metrics:
                if (metric_name == self.selection_metric) and (value > self.highest_metric):
                    self.highest_metric = value
                    self.save(epoch)
                    self.highest_epoch = epoch
                elif (metric_name == self.selection_metric) and (epoch - self.highest_epoch >= 10):
                    self.highest_metric = value
                    self.save(epoch)
                    self.highest_epoch = epoch
                self.writer.add_scalar(f"{metric_name}", value, epoch)
                print(f"  {metric_name}: {value}")
            print()
            torch.cuda.empty_cache()

    def evaluate(self):
        if self.data.test:
            loader = DataLoader(self.data.test, collate_fn=collate_fn, batch_size=self.bs, shuffle=True,
                                num_workers=self.num_workers)
            valid_size = len(self.data.test)
            # print(self.data.test[0])
        else:
            loader = DataLoader(self.data.valid, collate_fn=collate_fn, batch_size=self.bs, shuffle=True,
                                num_workers=self.num_workers)
            valid_size = len(self.data.valid)
        validation_loss = 0
        prediction = []
        true_label = []
        for i, (data, label) in enumerate(loader):
            # print(torch.cuda.memory_allocated())
            bs = len(data)
            # print(torch.cuda.memory_allocated())
            probs = self.classifier(data).detach()
            # print(torch.cuda.memory_allocated())
            y = torch.argmax(torch.stack(label), dim=1)
             #print(torch.cuda.memory_allocated())
            prediction.append(probs.detach().cpu())
            # print(torch.cuda.memory_allocated())
            true_label.append(y.detach().cpu())
            # print(torch.cuda.memory_allocated())
            loss = self.loss(probs, y).detach().cpu().item()
            # print(torch.cuda.memory_allocated())
            validation_loss += loss * bs
            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated())
            # print()

            gc.collect()
            del data
            del y
            del probs
            del loss
            gc.collect()

        self.epoch_loss /= len(self.data.train)
        validation_loss /= valid_size

        prediction = torch.cat(prediction)
        true_label = torch.cat(true_label)
        result = []
        for metric in self.metrics:
            result.append((metric.__class__.__name__, metric(prediction, true_label)))

        gc.collect()
        del prediction
        del true_label
        gc.collect()

        return self.epoch_loss, validation_loss, result

    def save(self, epoch):
        directory = fr"{self.weights_directory}\{self.time_code}_{self.random_seed}\weights"
        torch.save(self.classifier.state_dict(), fr"{directory}\{epoch}.pth")

