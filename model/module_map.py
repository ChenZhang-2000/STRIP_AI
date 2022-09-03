from torch.nn.functional import cross_entropy
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR

OPTIMIZER = {"Adam": Adam,
             "SGD": SGD}

LOSS = {'CrossEntropy': cross_entropy}

SCHEDULAR = {"Exponential": ExponentialLR}