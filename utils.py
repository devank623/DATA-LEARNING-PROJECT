import torch.nn as nn


class Average(object):
    def __init__(self):
        self.sum = 0.
        self.n = 0

    def update(self, elem):
        self.sum += elem
        self.n += 1

    def mean(self):
        return self.sum / self.n


# Custom weights initialization called on both generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
