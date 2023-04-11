from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import argparse
from importlib import import_module as impm
import numpy as np
import os
import random

import torch
import torch.distributed as dist

torch.autograd.set_detect_anomaly(True)
import _init_paths
from configs import cfg
from configs import update_config

from libs.datasets.GreatHitsDiffDataset import GreatHitsDataset

from libs.utils import misc
from libs.utils.lr_scheduler import ExponentialLR
from libs.utils.utils import create_logger
from libs.utils.utils import load_checkpoint


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


def get_ip(ip_addr):
    ip_list = ip_addr.split('-')[2:6]
    for i in range(4):
        if ip_list[i][0] == '[':
            ip_list[i] = ip_list[i][1:].split(',')[0]
    # TODO random ip
    return f'tcp://{ip_list[0]}.{ip_list[1]}.{ip_list[2]}.{ip_list[3]}:23456'


def main_per_worker():
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

    if 'SLURM_PROCID' in os.environ.keys():
        proc_rank = int(os.environ['SLURM_PROCID'])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1

    args.distributed = (args.world_size > 1 or args.distributed)

    # create logger
    if proc_rank == 0:
        logger, output_dir = create_logger(cfg, proc_rank)

    train_dataset = GreatHitsDataset(data_root=cfg.dataset.data_root, split='train')
    eval_dataset = GreatHitsDataset(data_root=cfg.dataset.data_root, split='test')
    print(f'The length of train dataset is {len(train_dataset)}')
    print(f'The length of eval dataset is {len(eval_dataset)}')

    # distribution
    if args.distributed:
        # dist_url = get_ip(os.environ['SLURM_STEP_NODELIST'])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        if proc_rank == 0:
            logger.info(
                # f'Init process group: dist_url: {dist_url},  '
                f'world_size: {args.world_size}, '
                f'proc_rank: {proc_rank}, '
                f'local_rank:{local_rank}'
            )
        dist.init_process_group(
            backend=cfg.dist_backend,
            # init_method=dist_url,
            world_size=args.world_size,
            rank=proc_rank
        )
        # torch.distributed.barrier()

        torch.cuda.set_device(local_rank)
        device = torch.device(cfg.device)
        # TODO build render
        model = getattr(impm(cfg.render.file), 'build_render')(cfg)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
        batch_size = cfg.dataset.img_num_per_gpu
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        assert proc_rank == 0, ('proc_rank != 0, it will influence '
                                'the evaluation procedure')
        device = 'cuda'
        model = getattr(impm(cfg.render.file), 'build_render')(cfg)
        train_sampler = None
        if ngpus_per_node == 0:
            batch_size = cfg.dataset.img_num_per_gpu
        else:
            batch_size = cfg.dataset.img_num_per_gpu * ngpus_per_node

    eval_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        drop_last=cfg.dataset.train.drop_last,
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=(eval_sampler is None),
        drop_last=cfg.dataset.test.drop_last,
        num_workers=cfg.workers,
        sampler=eval_sampler
    )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if p.requires_grad]},
    ]
    # TODO become step based, cosine, sine
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    lr_scheduler = ExponentialLR(optimizer, decay_epochs=cfg.train.decay_epochs,
                                 gamma=cfg.train.gamma)
    model, optimizer, lr_scheduler, last_iter = load_checkpoint(cfg, model,
                                                                optimizer, lr_scheduler, device)

    criterion = getattr(impm(cfg.train.criterion_file), 'Criterion')(cfg)

    # build trainer
    Trainer = getattr(impm(cfg.train.file), 'Trainer')(
        cfg,
        model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        log_dir=cfg.log_dir,
        last_iter=last_iter,
        rank=proc_rank,
        device=device,
    )

    print('start training...')
    while True:
        Trainer.train(train_loader, eval_loader)


if __name__ == '__main__':
    main_per_worker()