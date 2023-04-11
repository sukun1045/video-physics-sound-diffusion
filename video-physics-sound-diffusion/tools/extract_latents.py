# this script use diffusion model to generate spectrogram
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
from importlib import import_module as impm
import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torchaudio
import torch
import pickle
import torch.nn.functional as F
import _init_paths
from configs import cfg
from configs import update_config

from libs.datasets.GreatHitsDiffDataset import GreatHitsDataset
from libs.utils import misc
import librosa
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Acoustic')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/base_config.yaml',
        required=True,
        type=str)
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument(
        '--dist-url',
        dest='dist_url',
        default='tcp://10.5.38.36:23456',
        type=str,
        help='url used to set up distributed training')
    parser.add_argument(
        '--world-size',
        dest='world_size',
        default=1,
        type=int,
        help='number of nodes for distributed training')
    parser.add_argument(
        '--rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args
args = parse_args()

update_config(cfg, args)
ngpus_per_node = torch.cuda.device_count()

# torch seed
seed = cfg.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

split = 'test'
eval_dataset = GreatHitsDataset(data_root=cfg.dataset.data_root, split=split)

if cfg.device == 'cuda':
    torch.cuda.set_device(0)
device = torch.device(cfg.device)
model = getattr(impm(cfg.render.file), 'build_render')(cfg)
model = torch.nn.DataParallel(model).to(device)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=None,
    drop_last=cfg.dataset.test.drop_last,
    num_workers=cfg.workers,
    sampler=None
)

# model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


resume_path = cfg.render.resume_path
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path, map_location='cpu')
    # resume
    model.module.load_state_dict(checkpoint, strict=True)
    # if 'state_dict' in checkpoint:
    #     model.module.load_state_dict(checkpoint['state_dict'], strict=True)
    #     print('hello', resume_path)
    #     logging.info(f'==> model loaded from {resume_path} \n')

if not os.path.exists('results'):
    os.makedirs('results')
criterion = getattr(impm(cfg.train.criterion_file), 'Criterion')(cfg)
# build trainer
Trainer = getattr(impm(cfg.train.file), 'Trainer')(
    cfg,
    model,
    criterion=criterion,
    optimizer=None,
    lr_scheduler=None,
    logger=None,
    log_dir=cfg.log_dir,
    performance_indicator=cfg.pi,
    last_iter=-1,
    rank=0,
    device=device,
)
save_root = '/data/great_hits/hits_ten_mat_spec_diff_params_latent_video_fea_latents'
save_dict = {}
os.makedirs(save_root, exist_ok=True)
Trainer.render.eval()
with torch.no_grad():
    for i, val_data in enumerate(eval_loader):
        raw_data = eval_dataset.data[i]
        fn = raw_data['fn']
        tmp = fn.split('_')[0]
        select_tmp = f'{tmp}'
        print(select_tmp)
        print(fn)
        val_data = Trainer._read_inputs(val_data)

        f, p, t, nw, nt, video_fea, spec = val_data

        latents = Trainer.render.module.extract_params_latents(f, p, t, nw, nt)

        np_latents = latents.squeeze().cpu().numpy()

        print('check shape')
        print(np_latents.shape)
        print('---------------------------')
        save_dict[fn] = np_latents

with open(f'{save_root}/{split}.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle)