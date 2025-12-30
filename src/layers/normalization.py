import torch.nn as nn

def get_bn(num_features):
    return nn.BatchNorm2d(num_features)
