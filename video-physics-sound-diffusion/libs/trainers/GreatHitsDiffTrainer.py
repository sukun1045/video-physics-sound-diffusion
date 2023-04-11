import cv2
import datetime
import logging
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from matplotlib.lines import Line2D
import torch
from torch import autograd
from tensorboardX import SummaryWriter
import libs.utils.misc as utils
from libs.utils.utils import save_checkpoint

import librosa, librosa.display

def data_loop(data_loader):
    """
    Loop an iterable infinitely
    """
    while True:
        for x in iter(data_loader):
            yield x


# TODO logging the info
class Trainer(object):
    def __init__(self,
                 cfg,
                 render,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 logger,
                 log_dir,
                 performance_indicator='mse',
                 last_iter=-1,
                 rank=0,
                 device='cuda'):
        self.cfg = cfg
        self.render = render
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.logger = logger
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.cfg.output_dir)
            self.epoch = last_iter + 1
        self.PI = performance_indicator
        self.rank = rank
        self.best_performance = 1e6
        self.is_best = False
        self.max_epoch = self.cfg.train.max_epoch
        self.model_name = self.cfg.render.file
        self.device = device
        self.iter_count = 0
        if self.optimizer is not None and rank == 0:
            self.writer = SummaryWriter(self.log_dir, comment=f'_rank{rank}')
            logging.info(f"max epochs = {self.max_epoch} ")

    def _read_inputs(self, batch):
        for k in range(len(batch)):
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            if isinstance(batch[k], dict):
                batch[k] = {key: value.to(self.device) for key, value in batch[k].items()}
            else:
                batch[k] = batch[k].to(self.device)
            # print(batch[k].device)
        return batch

    def _forward(self, data):
        f, p, t, nw , nt, video_fea, gt_spec = data
        loss = self.render.render(f, p, t, nw, nt, video_fea, gt_spec)
        final_loss = self.criterion(loss)
        return final_loss

    def train(self, train_loader, eval_loader):
        start_time = time.time()
        # self.render.module.model.train()
        self.render.model.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = self.cfg.train.print_freq
        eval_data_iter = data_loop(eval_loader)
        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(0)
        for data in metric_logger.log_every(train_loader, print_freq, header, self.logger):
            data = self._read_inputs(data)
            loss_dict = self._forward(data)
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values()).item()
            if not math.isfinite(loss_value):
                # print("Loss is {}, stopping training".format(loss_value))
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            self.iter_count += 1
            # quick val
            if self.rank == 0 and self.iter_count % self.cfg.train.valiter_interval == 0:
                # evaluation
                if self.cfg.train.val_when_train:
                    performance = self.quick_val(eval_data_iter)
                    self.writer.add_scalar(self.PI, performance, self.iter_count)
                    logging.info('Now: {} is {:.4f}'.format(self.PI, performance))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        if self.rank == 0:
            for (key, val) in log_stats.items():
                self.writer.add_scalar(key, val, log_stats['iter'])
        self.lr_scheduler.step()

        # save checkpoint
        if self.rank == 0 and self.epoch > 0 and self.epoch % self.cfg.train.save_interval == 0:
            # evaluation TODO val all the val
            if self.cfg.train.val_when_train:
                recon_loss = self.all_val(eval_loader)
                performance = recon_loss
                self.writer.add_scalar(self.PI, performance, self.iter_count)
                if performance < self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'epoch:{self.epoch} recon_loss:{performance}')
                logging.info(f'Now: best recon_loss is {self.best_performance}')
            else:
                performance = -1

            # save checkpoint
            try:
                print('try getting state dict correct')
                # state_dict = self.render.module.state_dict()  # remove prefix of multi GPUs
                state_dict = self.render.state_dict()  # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.render.state_dict()

            if self.rank == 0:
                if self.cfg.train.save_every_checkpoint:
                    filename = f"{self.epoch}.pth"
                else:
                    filename = "latest.pth"
                save_dir = os.path.join(self.log_dir, self.cfg.output_dir)
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        f'performance/{self.PI}': performance,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                    },
                    self.is_best,
                    save_dir,
                    filename=f'{filename}'
                )
                # remove previous pretrained model if the number of models is too big
                pths = [
                    int(pth.split('.')[0]) for pth in os.listdir(save_dir)
                    if pth != 'latest.pth' and pth != 'model_best.pth'
                ]
                if len(pths) > 20:
                    os.system('rm {}'.format(
                        os.path.join(save_dir, '{}.pth'.format(min(pths)))))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Training time {}'.format(total_time_str))
        self.epoch += 1

    def all_val(self, eval_loader):
        recon_loss = 0
        count = 0
        self.render.eval()
        with torch.no_grad():
            for val_data in eval_loader:
                val_data = self._read_inputs(val_data)
                f, p, t, nw, nt, video_fea, gt_spec = val_data
                loss = self.render.render(f, p, t, nw, nt, video_fea, gt_spec)
                loss_dict = self.criterion(loss)
                recon_loss += loss_dict['recon_loss'].item()
                count += 1
        self.render.model.train()
        self.criterion.train()
        return recon_loss/count

    def quick_val(self, eval_data_iter):
        self.render.eval()
        self.criterion.eval()
        val_stats = {}
        with torch.no_grad():
            val_data = next(eval_data_iter)
            val_data = self._read_inputs(val_data)
            f, p, t, nw , nt, video_fea, gt_spec = val_data
            loss = self.render.render(f, p, t, nw , nt, video_fea, gt_spec)
            B = f.shape[0]
            loss_dict = self.criterion(loss)
            loss_stats = utils.reduce_dict(loss_dict)
            for k, v in loss_stats.items():
                val_stats.setdefault(k, 0)
                val_stats[k] += v
        # save metrics and loss
        log_stats = {**{f'eval_{k}': v for k, v in val_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        for (key, val) in log_stats.items():
            self.writer.add_scalar(key, val, log_stats['iter'])
        recon_loss = val_stats['recon_loss']
        msg = 'recon_loss: {:.4f}'.format(recon_loss)
        self.logger.info(msg)
        self.render.model.train()
        self.criterion.train()
        return val_stats[self.PI]

    @staticmethod
    def process_img(gt_spec, pred_spec):
        # TODO save pred_ir in a meaningful way for visualization check!
        # pred_ir: `(n_bins, n_samples_each_bin)`
        sr = 44100
        fig = plt.figure()
        ax_gt_audios_spec = fig.add_subplot(2, 1, 1)
        gt_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(gt_spec), ref=np.max), sr=sr, hop_length=256,x_axis='time',
                                               y_axis='log', cmap='viridis', ax=ax_gt_audios_spec)

        ax_pred_audios_spec = fig.add_subplot(2, 1, 2)
        pred_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(pred_spec), ref=np.max), sr=sr, hop_length=256, x_axis='time',
                                 y_axis='log', cmap='viridis',ax=ax_pred_audios_spec)
        fig.colorbar(gt_spec_img, ax=[ax_gt_audios_spec, ax_pred_audios_spec])
        # fig.tight_layout()
        # new
        fig.savefig('great_hits_end2end_val_plot.jpg')
        plt.close(fig)
        return {'fig_plot': 0}
        # return {'fig_plot': fig}

    def evaluate(self, eval_loader, result_path, is_vis=False):
        pass

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n) and ("mass_estimator" not in n):
                # print(n)
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().numpy())
                max_grads.append(p.grad.abs().max().cpu().numpy())
        fig = plt.figure(figsize=(12, 12))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        fig.savefig('simple_audio_gradients.jpg')
        plt.close(fig)