import os
import shutil
import time
import random
import re

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from data.dataset import StripDataset, collate_fn, GradientDataset
from model.trainer import Trainer
from model.networks.register import CLASSIFIER_FACTORY, SCHEDULER_FACTORY, OPTIMIZER_FACTORY, LOSS_FACTORY


def read_config(file_directory, file_name):
    config = yaml.load(open(f"{file_directory}\\{file_name}.yaml"), Loader)
    config['global'] = yaml.load(open(f"{file_directory}\\global.yaml"), Loader)
    scheduler_config = yaml.load(open(f"{file_directory}\\scheduler\\{config['module']['scheduler']}.yaml"), Loader)
    config['module']['scheduler'] = SCHEDULER_FACTORY[scheduler_config['scheduler']]
    config['module']['scheduler_params'] = scheduler_config['params']
    config['module']['classifier'] = CLASSIFIER_FACTORY[config['module']['classifier']]
    if config['module']['pretrained']:
        config['module']['backbone'] = CLASSIFIER_FACTORY[config['module']['backbone']]
    config['module']['optimizer'] = OPTIMIZER_FACTORY[config['module']['optimizer']]
    config['module']['loss'] = LOSS_FACTORY[config['module']['loss']]
    if config['seed'] == -1:
        config['seed'] = random.randint(0, 10000)
    return config


def setup_seed(seed=3407):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == "__main__":
    for config_file in ['version_0']:
        # pretrain_resnet152_7
        # pretrain_resnet101_7
        # pretrain_resnet50_7
        # version_0
        config = read_config("configs", config_file)
        # print(config['module']['pretrained'], config['module']['backbone'])
        setup_seed(config['seed'])

        dataset = StripDataset(rf"{config['global']['data_dir']}", config['dataset'], test_size=config['test_size'],
                               store=config['store'], random_state=config['seed'], test_set=True,
                               keep_dim=config['in_dimensions'], test_direc=config['test_set'])
        # print(config['seed'])
        if config['module']['pretrained']:
            pretrianed_name = re.match(r"(\d+_\d+)(?:\\weights\\.+)", config['module']['pretrained']).groups()[0]
            dataset.read_split(pretrianed_name)

        trainer = Trainer(data=dataset, num_workers=config['num_workers'],
                          classifier=config['module']['classifier'],
                          optimizer=config['module']['optimizer'],
                          scheduler=config['module']['scheduler'],
                          scheduler_params=config['module']['scheduler_params'],
                          loss=config['module']['loss'],
                          in_channel=config['in_dimensions'][1]-config['in_dimensions'][0],
                          backbone_layers=config['backbone_layers'],
                          attention_layers=config['attention_layers'],
                          img_size=config['img_size'], bs=config['bs'],
                          max_epoch=config['max_epoch'], lr=config['lr'],
                          random_seed=config['seed'],
                          pretrained=config['module']['pretrained'], backbone=config['module']['backbone'])
        time_code = trainer.time_code
        shutil.copyfile(fr"configs\{config_file}.yaml", fr"runs\{time_code}_{config['seed']}\config.yaml")
        trainer.train()
