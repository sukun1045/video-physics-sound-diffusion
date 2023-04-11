import math
import numpy as np
import torch
import torch.nn as nn
from libs.models.sound_physics_residual_model import ImpactSound as base_model
class Renderer(nn.Module):
    def __init__(self,
                 model,
                 is_train=True):
        super().__init__()
        self.model = torch.nn.DataParallel(model).cuda()
        self.is_train = is_train

    def render(self, gt_spec, pred_wav):
        pred_audios = self.model(gt_spec, pred_wav)
        return pred_audios


def build_render(cp_path):
    print(cp_path)
    model = base_model()
    render_config = {
        'model': model
    }
    render = Renderer(**render_config)
    return render