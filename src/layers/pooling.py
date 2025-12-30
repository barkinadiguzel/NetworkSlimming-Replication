import torch.nn as nn

def max_pool(kernel=2, stride=2):
    return nn.MaxPool2d(kernel, stride)

def avg_pool(kernel=2, stride=2):
    return nn.AvgPool2d(kernel, stride)
