import torch
import torch.nn as nn

class CrossEntropyL1Loss(nn.Module):
    def __init__(self, lambda_l1=1e-4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_l1 = lambda_l1

    def forward(self, outputs, targets, model=None):
        loss = self.ce(outputs, targets)
        if model:
            l1 = 0
            for m in model.modules():
                if hasattr(m, 'weight') and isinstance(m, nn.BatchNorm2d):
                    l1 += m.weight.abs().sum()
            loss += self.lambda_l1 * l1
        return loss
