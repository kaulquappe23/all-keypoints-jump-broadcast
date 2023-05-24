# -*- coding: utf-8 -*-
"""
Created on 20.04.23

"""
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import pytz
import torch
from torch.utils.tensorboard import SummaryWriter


def set_deterministic(all=False):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    if all:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TrainUtils:
    """
    Capsules all utilities used during training, such as logger, weights dir, tensorboard logging, train steps
    """
    def __init__(self, cfg, phase='train', tb_prefix=None):
        self.time_str = datetime.now().astimezone(pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d-%H-%M')
        self.train_steps = 0
        self.root_output_dir = Path(cfg.LOGS)
        self.phase = phase
        self.cfg = cfg
        self.tensorboard_dir = None
        self.writer_dict = None
        self.weights_dir = None
        self.tb_prefix = tb_prefix

        if not self.root_output_dir.exists() and hasattr(cfg, "RANK") and cfg.RANK == 0:
            print('=> creating {}'.format(self.root_output_dir))
            self.root_output_dir.mkdir()
        else:
            while not self.root_output_dir.exists():
                print('=> wait for {} created'.format(self.root_output_dir))
                time.sleep(30)

    def setup_tensorboard_dir(self, prefix=None):
        if prefix is not None:
            self.tb_prefix = prefix
        dataset = self.cfg.NAME
        dataset = dataset.replace(':', '_')
        tb_name = "{}_{}_{}".format(dataset, self.tb_prefix, self.time_str) if self.tb_prefix is not None else "{}_{}".format(dataset, self.time_str)
        tensorboard_log_dir = self.root_output_dir / dataset / "tensorboard" / tb_name
        print('\ncreating => creating {}\n'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = tensorboard_log_dir
        return tensorboard_log_dir

    def setup_weights_dir(self):
        dataset = self.cfg.NAME
        dataset = dataset.replace(':', '_')
        weights_log_dir = self.root_output_dir / dataset / "weights" / self.time_str
        print('\ncreating => creating {}\n'.format(weights_log_dir))
        weights_log_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = weights_log_dir
        return weights_log_dir

    def setup_prediction_dir(self):
        dataset = self.cfg.NAME
        dataset = dataset.replace(':', '_')
        pred_log_dir = self.root_output_dir / dataset / "eval"
        print('\ncreating => creating {}\n'.format(pred_log_dir))
        pred_log_dir.mkdir(parents=True, exist_ok=True)
        return pred_log_dir

    def get_writer_dict(self):
        if self.writer_dict is None:
            if self.tensorboard_dir is None:
                self.setup_tensorboard_dir()
            if self.weights_dir is None:
                self.setup_weights_dir()
            self.writer_dict = {
                'writer': SummaryWriter(log_dir=self.tensorboard_dir),
                'ckpt_dir': self.weights_dir,
                'time_str': self.time_str
            }

        self.writer_dict['train_global_steps'] = self.train_steps
        return self.writer_dict

    def write_value(self, step, title, value):
        if self.tensorboard_dir is None:
            self.setup_tensorboard_dir()
        with open(os.path.join(self.tensorboard_dir, "validation.csv"), "a") as file:
            file.write("{:7d}, {}, {}, {:.6f}\n".format(step, datetime.now().astimezone(pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d-%H-%M-%S'), title, value))

    def get_weights_dir(self):
        if self.weights_dir is None:
            self.setup_weights_dir()
        return self.weights_dir

    def next_step(self):
        self.train_steps = self.train_steps + 1

    def write_tensorboard(self, title, value):
        writer = self.get_writer_dict()['writer']
        writer.add_scalar(
            title,
            value,
            self.train_steps
        )


def save_checkpoint(states, weights_dir, time_str, step, save=True):
    torch.save(states, os.path.join(weights_dir, '{}_checkpoint.pth.tar'.format(time_str)))

    if save:
        torch.save(
            states,
            os.path.join(weights_dir, 'step_{:07d}.pth.tar'.format(step))
        )


class BestWeightsOrganizer:

    def __init__(self, weights_dir, metric_high_better, time_str="current", keep_num_best_weights=10, keep_last_steps=10):
        self.keep_num_best_weights = keep_num_best_weights
        self.keep_last_steps = keep_last_steps
        self.time_str = time_str
        self.weights_dir = weights_dir
        self.score_dicts = [{} for _ in range(len(metric_high_better))]
        self.last_steps = []
        self.metric_high_better = metric_high_better

    def save_weights(self, metrics, states, steps):

        if len(metrics) != len(self.score_dicts):
            print("-----------------------------------------------------\nDefine for each metric if a higher score is better or not, use config.METRIC_HIGH_BETTER\n-----------------------------------------------------")

        save_new_weights = False
        weights_to_delete = []

        if len(self.last_steps) < self.keep_last_steps:  # we don't have enough values yet
            save_new_weights = True
            self.last_steps.append(steps)
        else:
            min_steps = min(self.last_steps)
            if steps > min_steps:  # new weights are more recent (usual case)
                save_new_weights = True
                self.last_steps.append(steps)
                self.last_steps.remove(min_steps)
                weights_to_delete.append(min_steps)

        if len(self.score_dicts[0]) < self.keep_num_best_weights:  # we don't have enough values yet
            save_new_weights = True
            for metric, score_dict in zip(metrics, self.score_dicts):
                score_dict[metric] = steps
        else:
            for high_is_better, metric, score_dict in zip(self.metric_high_better, metrics, self.score_dicts):
                new_best = False
                if high_is_better:  # higher values for metric are better
                    current_worst = min(list(score_dict.keys()))
                    if metric > current_worst:
                        new_best = True
                else:  # lower values for metric are better
                    current_worst = max(list(score_dict.keys()))
                    if metric < current_worst:
                        new_best = True

                if new_best:  # new score is under keep_num_best_weights best scores
                    save_new_weights = True
                    score_dict[metric] = steps
                    weights_to_delete.append(score_dict.pop(current_worst))

        if save_new_weights:
            states['best_weights_organizer'] = self
            save_checkpoint(states=states, weights_dir=self.weights_dir, time_str=self.time_str, step=steps)

            weights_to_delete = list(set(weights_to_delete))

            all_steps = self.last_steps.copy()
            for score_dict in self.score_dicts:
                all_steps.extend(score_dict.values())
            all_steps = list(set(all_steps))
            for step in all_steps:  # don't delete weights that are removed by one algo, but included by another one
                if step in weights_to_delete:
                    weights_to_delete.pop(weights_to_delete.index(step))

            for step in weights_to_delete:
                delete_filename = os.path.join(self.weights_dir, 'step_{:07d}.pth.tar'.format(step))
                os.remove(delete_filename)
