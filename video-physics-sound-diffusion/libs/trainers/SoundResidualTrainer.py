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
                 performance_indicator='mrft_loss',
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
        return batch

    def _forward(self, data):
        gt_audios, pred_audios, gt_spec = data
        pred_audios_ = self.render.render(gt_spec.float(), pred_audios.float())
        loss = self.criterion( gt_audios.float(), pred_audios_)
        return loss

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
                mrft_loss = self.all_val(eval_loader)
                performance = mrft_loss
                self.writer.add_scalar(self.PI, performance, self.iter_count)
                if performance < self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'epoch:{self.epoch} mrft error:{mrft_loss}')
                logging.info(f'Now: best mrft error is {self.best_performance}')
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
        mrft_loss = 0
        count = 0
        self.render.eval()
        with torch.no_grad():
            for val_data in eval_loader:
                val_data = self._read_inputs(val_data)
                gt_audios, pred_audios, gt_spec = val_data
                pred_audios_ = self.render.render(gt_spec.float(), pred_audios.float())
                loss_dict = self.criterion(gt_audios.float(), pred_audios_)
                mrft_loss += loss_dict['mrft_loss'].item()
                count += 1
        self.render.model.train()
        self.criterion.train()
        return mrft_loss/count

    def quick_val(self, eval_data_iter):
        self.render.eval()
        self.criterion.eval()
        val_stats = {}
        plot_stats = {}
        with torch.no_grad():
            val_data = next(eval_data_iter)
            val_data = self._read_inputs(val_data)
            gt_audios, pred_audios, gt_spec = val_data

            pred_audios_ = self.render.render(gt_spec.float(), pred_audios.float())
            B = gt_audios.shape[0]
            idx = np.random.choice(range(B))
            plot_stat = self.process_img(gt_audios[idx], pred_audios_[idx])
            # plot_stats.update(plot_stat)
            loss_dict = self.criterion(gt_audios, pred_audios_)
            loss_stats = utils.reduce_dict(loss_dict)
            for k, v in loss_stats.items():
                val_stats.setdefault(k, 0)
                val_stats[k] += v

        # save metrics and loss
        log_stats = {**{f'eval_{k}': v for k, v in val_stats.items()},
                     'epoch': self.epoch, 'iter': self.iter_count}
        for (key, val) in log_stats.items():
            self.writer.add_scalar(key, val, log_stats['iter'])
        mrft_loss = val_stats['mrft_loss']
        msg = 'mrft_loss: {:.4f}'.format(mrft_loss)
        self.logger.info(msg)
        self.render.model.train()
        self.criterion.train()
        return val_stats['mrft_loss']

    @staticmethod
    def process_img(gt_audios, pred_audios):
        # TODO save pred_ir in a meaningful way for visualization check!
        # pred_ir: `(n_bins, n_samples_each_bin)`
        sr = 44100
        gt_audios_plot = gt_audios.reshape(-1).data.cpu().numpy()
        pred_audios_plot = pred_audios.reshape(-1).data.cpu().numpy()
        min_val = np.minimum(np.min(gt_audios_plot), np.min(pred_audios_plot))
        max_val = np.maximum(np.max(gt_audios_plot), np.max(pred_audios_plot))
        gt_spec = librosa.stft(gt_audios_plot, n_fft=2048, hop_length=256)
        pred_spec = librosa.stft(pred_audios_plot, n_fft=2048, hop_length=256)
        fig = plt.figure()
        ax_gen = fig.add_subplot(3, 1, 1)
        ax_gen.plot(gt_audios_plot, color='green')
        ax_gen.plot(pred_audios_plot, color='red', alpha=0.7)
        ax_gen.set_ylim(min_val, max_val)
        ax_gt_audios_spec = fig.add_subplot(3, 1, 2)
        gt_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(gt_spec), ref=np.max), sr=sr, hop_length=256,x_axis='time',
                                               y_axis='log', cmap='viridis', ax=ax_gt_audios_spec)
        ax_pred_audios_spec = fig.add_subplot(3, 1, 3)
        pred_spec_img = librosa.display.specshow(librosa.amplitude_to_db(abs(pred_spec), ref=np.max), sr=sr, hop_length=256, x_axis='time',
                                 y_axis='log', cmap='viridis',ax=ax_pred_audios_spec)
        fig.colorbar(gt_spec_img, ax=[ax_gt_audios_spec, ax_pred_audios_spec])
        # fig.tight_layout()
        # new
        fig.savefig('great_hits_noise_val_plot.jpg')
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