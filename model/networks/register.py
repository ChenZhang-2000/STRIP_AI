from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, SGD
from torch.nn.functional import cross_entropy


CLASSIFIER_FACTORY = {}
SCHEDULER_FACTORY = {"Exponential": ExponentialLR}
OPTIMIZER_FACTORY = {'Adam': Adam, 'SGD': SGD}
LOSS_FACTORY = {'CrossEntropy': cross_entropy}


def register_classifier(cls):
    cls_name = cls.name

    def register(cls):
        CLASSIFIER_FACTORY[cls_name] = cls

    return register(cls)


def register_scheduler(cls):
    cls_name = cls.name

    def register(cls):
        SCHEDULER_FACTORY[cls_name] = cls

    return register(cls)
