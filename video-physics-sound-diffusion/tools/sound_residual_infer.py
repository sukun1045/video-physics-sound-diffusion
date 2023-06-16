from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
from importlib import import_module as impm
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import pickle

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.GreatHitsSoundResidualDataset import GreatHitsDataset
from libs.utils import misc

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
p_mae = 0
t_mae = 0
count = 0

save_root = '/data/great_hits/hits_mode_data_ten_physics_residual_params'
os.makedirs(save_root, exist_ok=True)
Trainer.render.eval()
save_dict = {}
with torch.no_grad():
    for i, val_data in enumerate(eval_loader):
        val_data = Trainer._read_inputs(val_data)
        gt_audios, pred_audios, gt_spec = val_data
        audio_fea, res_audio_fea = Trainer.render.module.model.module.audio_enc(gt_spec.float().transpose(1, 2))
        audio_fea = F.adaptive_avg_pool1d(audio_fea.transpose(1, 2), 1).squeeze(2)
        noise_weights = torch.sigmoid(Trainer.render.module.model.module.noise_proj(audio_fea))
        noise_t = (1e-5 + torch.sigmoid(Trainer.render.module.model.module.noise_t_proj(audio_fea)) * 0.5)
        np_noise_weights = noise_weights.squeeze(0).cpu().numpy()
        np_noise_t = noise_t.squeeze(0).cpu().numpy()
        data_file = eval_dataset.data[i]
        with open(data_file, 'rb') as handle:
            raw_data = pickle.load(handle)
        fn = raw_data['fn']
        f = raw_data['f']
        p = raw_data['p']
        t = raw_data['t']
        save_dict[fn] = {'f':f, 'p':p, 't':t, 'noise_weights':np_noise_weights, 'noise_t':np_noise_t}
        print(fn)
        print('---------------------------')

with open(f'{save_root}/{split}.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle)




