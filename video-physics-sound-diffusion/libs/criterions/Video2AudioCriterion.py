import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, loss):
        scalar_stats = {}
        recon_loss = loss
        scalar_stats['recon_loss'] = recon_loss
        return scalar_stats