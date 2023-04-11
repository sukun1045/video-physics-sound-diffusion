import math
import numpy as np
import torch
import torch.nn as nn
from libs.models.great_hits_spec_diff_params_latent_video_fea_model import Unet3D as base_model
from libs.models.great_hits_spec_diff_params_latent_video_fea_model import GaussianDiffusion

class Renderer(nn.Module):
    def __init__(self, model,
                 is_train=True):
        super().__init__()
        self.model = torch.nn.DataParallel(model).cuda()
        self.diffusion = torch.nn.DataParallel(GaussianDiffusion(self.model)).cuda()
        self.is_train = is_train

    def render(self, f, p, t, nw, nt, video_fea, spec):
        cond = [f.float(), p.float(), t.float(), nw.float(), nt.float()]
        loss = self.diffusion(spec.float(), cond=cond, video_fea=video_fea.float())
        return loss

    def sample(self, f, p, t, nw, nt, video_fea):
        cond = [f.float(), p.float(), t.float(), nw.float(), nt.float()]
        pred_spec = self.diffusion.module.sample(cond=cond, video_fea=video_fea.float())
        return pred_spec

    def query_sample(self, latent, video_fea):
        pred_spec = self.diffusion.module.query_sample(cond=latent.float(), video_fea=video_fea.float())
        return pred_spec

    def extract_params_latents(self, f, p, t, nw, nt):
        latents = self.diffusion.module.params_layer(f.float(), p.float(), t.float(), nw.float(), nt.float())
        return latents

def build_render(cfg):
    model = base_model(
        dim=64,
        dim_mults=(1, 2, 4, 8),
    )
    render_config = {
        'model': model
    }
    render = Renderer(**render_config)
    return render