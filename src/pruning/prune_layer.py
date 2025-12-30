import torch

def prune_channels(model, prune_ratio=0.3):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            weight = m.weight.data.abs().clone()
            threshold = torch.quantile(weight, prune_ratio)
            mask = weight > threshold
            m.weight.data.mul_(mask)
