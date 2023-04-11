import torch
import torch.nn as nn
import auraloss
import torch.nn.functional as F
class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mrft_loss = auraloss.freq.STFTLoss(fft_size=2048, hop_size=128, w_sc=1, w_log_mag=1, w_lin_mag=1)

    def forward(self, gt_audio, pred_audio_, gt_fea, pred_fea):
        scalar_stats = {}
        mrft_loss = self.mrft_loss(pred_audio_, gt_audio)
        scalar_stats['mrft_loss'] = mrft_loss
        percept_loss = F.l1_loss(pred_fea, gt_fea)
        scalar_stats['percept_loss'] = percept_loss
        return scalar_stats