# -*- coding: utf-8 -*-
"""
Created on 21.01.21

"""
import copy
import sys
from typing import Optional

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from datasets.general.arbitrary_keypoints_data_wrapper import ArbitraryKeypointsDataWrapper
from datasets.general.dataset_utils import get_dataloader
from metrics.pck import eval_pck_array
from metrics.thickness_metric import thickness_percentage_differences, thickness_metric
from transformer.data.transformer_dataset import TransformerDatasetWrapper, TransformerDataset
from transformer.execution.inference import inference_data_loader
from transformer.execution.joints_mse_loss import JointsMSELoss
from transformer.execution.pre_postprocessing import additional_vectors_to_list
from transformer.experiments.transformer_config import TransformerConfig
from transformer.model import token_pose
from utils.config_utils import check_arbitrary_keypoints
from utils.training_utils import TrainUtils, BestWeightsOrganizer


class TokenPoseTraining:
    """
    Routine for training of TokenPose-like models with single person datasets.
    """

    def __init__(self, cfg: TransformerConfig, tb_prefix=""):

        self.name = "" if tb_prefix is None or len(tb_prefix) == 0 else tb_prefix

        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Data loading code
        train_dataset = TransformerDataset(
                self.cfg, params={"subset": cfg.TRAIN_SUBSET, "arbitrary_points": check_arbitrary_keypoints(cfg)}
                )
        self.train_loader = get_dataloader(
                train_dataset, is_train=True, batch_size=cfg.SINGLE_GPU_BATCH_SIZE, num_workers=cfg.WORKERS
                )
        self.train_loader_iter = iter(self.train_loader)
        self.val_dataset = TransformerDataset(
                self.cfg, params={
                        "subset": cfg.VAL_SUBSET, "arbitrary_points": check_arbitrary_keypoints(cfg)
                        }
                )
        self.val_loader = get_dataloader(
                self.val_dataset, is_train=False, batch_size=cfg.SINGLE_GPU_BATCH_SIZE, num_workers=cfg.WORKERS
                )
        self.val_loader_iter = iter(self.val_loader)

        self.model = token_pose.get_token_pose_net(self.cfg, load_weights_from_config=True)
        self.model = self.model.to(self.device)

        self.ema_rate = 0.99
        self.ema_model = token_pose.get_token_pose_net(self.cfg, load_weights_from_config=False)
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.ema_model = self.ema_model.to(self.device)

        if getattr(self.cfg, "FREEZE_BACKBONE", False):
            self.model.freeze_backbone()

        # define loss function, optimizer
        self.loss_function = JointsMSELoss(use_target_weight=True).to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.LR)

        if hasattr(self.cfg, "BODYPART_ORDER"):
            self.name += self.cfg.BODYPART_ORDER.head_strategy()
        self.name += "_" + self.model.description

        self.train_utils = TrainUtils(self.cfg, tb_prefix="TF" + self.name)
        self.name += "_" + self.train_utils.time_str

        self.best_weights_organizer = BestWeightsOrganizer(
                weights_dir=self.train_utils.get_weights_dir(),
                time_str=self.train_utils.time_str,
                metric_high_better=cfg.METRICS_HIGH_IS_BETTER * 2,  # for normal and ema model
                keep_num_best_weights=10,
                keep_last_steps=5
                )

    def run(self):

        for _ in tqdm(range(self.cfg.NUM_STEPS), desc=self.name):  # we count gradient update steps, not epochs
            self.train()

        self.train_utils.get_writer_dict()["writer"].close()

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        num_steps = self.cfg.NUM_STEPS
        self.cfg = checkpoint["config"]
        self.cfg.NUM_STEPS = num_steps
        self.model = token_pose.get_token_pose_net(self.cfg, load_weights_from_config=False)
        self.model = self.model.to(self.device)
        self.ema_model = token_pose.get_token_pose_net(self.cfg, load_weights_from_config=False)
        self.ema_model = self.ema_model.to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_utils.train_steps = checkpoint["epoch"]
        self.cfg.NUM_STEPS = self.cfg.NUM_STEPS - checkpoint["epoch"]
        if 'train_utils' in checkpoint:
            self.train_utils = checkpoint['train_utils']
        print("-------------- RESUMING FROM {} --------------------".format(checkpoint_path))
        self.log_and_save_weights()

    def _update_ema_model_parameters(self):
        """
        We train with an exponential moving average, as it usually improves the performance and has less variance.
        :return:
        """
        with torch.no_grad():
            ema_r = self.ema_rate
            for param_s, param_t in zip(self.model.parameters(), self.ema_model.parameters()):
                param_t.data = param_t.data * ema_r + param_s.data * (1. - ema_r)
            for buffer_s, buffer_t in zip(self.model.buffers(), self.ema_model.buffers()):
                buffer_t.data = buffer_t.data * ema_r + buffer_s.data * (1. - ema_r)

    def train(self):

        self.model.train()

        try:
            # Samples the batch
            (images, targets, target_weight, meta) = next(self.train_loader_iter)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.train_loader_iter = iter(self.train_loader)
            (images, targets, target_weight, meta) = next(self.train_loader_iter)

        additional_vectors = meta["representations"] if "representations" in meta else None
        additional_vectors_list = additional_vectors_to_list(additional_vectors, device=self.device)
        # compute output
        outputs = self.model(images.to(self.device), additional_vectors=additional_vectors_list)

        targets = targets.to(device=self.device, non_blocking=True)
        heatmaps_loss = self.loss_function(outputs, targets, target_weight.cuda())

        # compute gradient and do update step
        self.optimizer.zero_grad()
        heatmaps_loss.backward()
        self.optimizer.step()

        self._update_ema_model_parameters()
        self.train_utils.next_step()
        self.log_and_save_weights(heatmaps_loss.item())

    def log_and_save_weights(self, loss=None):
        """
        Logging, validation and checkpoint saving
        Only checkpoints that are recent or have a better metric are kept. In the config, it needs to be defined if better
        means higher or lower metric.
        :param loss: log loss if it is not None
        :return:
        """
        if self.train_utils.train_steps % self.cfg.TENSORBOARD_FREQ == 0 and loss is not None:
            self.train_utils.write_tensorboard("loss", loss)

        metric = None

        if self.cfg.VAL_STEPS is not None and self.train_utils.train_steps % self.cfg.VAL_STEPS == 0:

            pf = self.train_loader.dataset.postfix()
            postfix = "{}_{}".format(pf, len(self.val_loader.dataset)) if len(pf) > 0 else "{}".format(
                    len(self.val_loader.dataset)
                    )
            val_params = {'postfix': postfix}

            if sys.gettrace() is None:
                try:
                    # PCK validation and logging
                    metric = do_validation(
                            self.cfg, self.model, self.val_loader, self.val_dataset, self.train_utils, ema_model=self.ema_model,
                            val_params=val_params
                            )
                except BaseException as e:
                    print("Exception caught during validation: {}: {}".format(type(e), e))
            else:
                metric = do_validation(
                        self.cfg, self.model, self.val_loader, self.val_dataset, self.train_utils, ema_model=self.ema_model,
                        val_params=val_params
                        )
            print("Continue training " + self.name)

        if metric is not None:
            train_util_copy = copy.copy(self.train_utils)
            train_util_copy.writer_dict = None
            states = {
                    'epoch':       self.train_utils.train_steps,
                    'state_dict':  self.model.state_dict(),
                    'optimizer':   self.optimizer.state_dict(),
                    'config':      self.cfg,
                    'ema_model':   self.ema_model.state_dict(),
                    'train_utils': train_util_copy,
                    }
            self.best_weights_organizer.save_weights(metrics=metric, states=states, steps=self.train_utils.train_steps)


def do_validation(
        cfg: TransformerConfig, model, val_data_loader, val_dataset, train_utils: Optional[TrainUtils], val_params=None,
        ema_model=None
        ):
    """
    Validates and logs to tensorboard.
    :param cfg: configuration file
    :param model: model to validate (should be called twice for normal and ema model)
    :param val_data_loader: data loader for validation
    :param val_dataset: dataset for validation
    :param train_utils: TrainUtils object for saving to tensorboard or None
    :param val_params: additional parameters for validation, e.g. postfix
    :param ema_model: ema model for validation or None
    """

    metric_results = []
    val_params = {} if val_params is None else val_params

    metric_res = validation_single_model(cfg, val_data_loader, val_dataset, model, train_utils, val_params)
    metric_results.extend(metric_res)

    if ema_model is not None:
        val_params["postfix"] = "ema" if "postfix" not in val_params else "ema_" + val_params["postfix"]
        metric_res = validation_single_model(cfg, val_data_loader, val_dataset, ema_model, train_utils, val_params)
        metric_results.extend(metric_res)
    return metric_results


def validation_single_model(
        cfg, val_data_loader, val_dataset: TransformerDatasetWrapper, model, train_utils: Optional[TrainUtils], val_params
        ):
    """
    Validate a single model and log to tensorboard, if desired
    :param cfg: configuration file
    :param val_data_loader: data loader for validation
    :param val_dataset: dataset for validation
    :param model: model to validate
    :param train_utils: TrainUtils object for tensorboard logging, can be None if not desired
    :param val_params: additional parameters for validation, e.g. postfix
    :return: List of metric results, starting with full PCK, then PCK on standard points, MTE and PCT
    """
    preds, _, image_ids, additional_output = inference_data_loader(val_data_loader, val_dataset, model, use_flip=False)

    annos = additional_output["original_annotations"]
    pck_thresholds = [0.1, 0.05, 0.2]
    metric_results = eval_pck_array(preds, annos, cfg.JOINT_ORDER, pck_thresholds=pck_thresholds)
    postfix = "" if not "postfix" in val_params else val_params["postfix"]
    tb_string = "/" + postfix if len(postfix) > 0 else postfix
    print("Current PCK@{} {} with all joints: {:.4f}".format(pck_thresholds[0], postfix, metric_results[0]))
    if train_utils is not None:
        for pck, threshold in zip(metric_results, pck_thresholds):
            train_utils.write_tensorboard("pck_full_{}".format(threshold) + tb_string, pck)
    std_pck_results = eval_pck_array(
            preds, annos, cfg.JOINT_ORDER, pck_thresholds=pck_thresholds,
            use_indices=[i for i in range(cfg.JOINT_ORDER.get_num_joints())]
            )
    print("Current PCK@{} {} only standard keypoints: {:.4f}".format(pck_thresholds[0], postfix, std_pck_results[0]))
    if train_utils is not None:
        for pck, threshold in zip(std_pck_results, pck_thresholds):
            train_utils.write_tensorboard("pck_{}".format(threshold) + tb_string, pck)
    metric_results.extend(std_pck_results)

    if isinstance(val_dataset.dataset, ArbitraryKeypointsDataWrapper):
        bbox_dict = val_dataset.dataset.bboxes
        fixed_annos = annos[:, :val_dataset.dataset.joint_order.get_num_joints()]
        annos = annos[:, val_dataset.dataset.joint_order.get_num_joints():]
        preds = preds[:, val_dataset.dataset.joint_order.get_num_joints():]
        thickness_vector = None if "thickness_vector" not in additional_output else additional_output["thickness_vector"]
        thickness_res = thickness_percentage_differences(
                preds, annos, fixed_annos, val_dataset.dataset.bodypart_order, val_dataset.dataset.load_bodypart_mask, bbox_dict,
                image_ids, thickness_vector,
                anchors=getattr(val_dataset.dataset, "anchors", {}),
                return_all_distances=True
                )

        mean, std, pct = thickness_metric(thickness_res[0])
        metric_results.extend([mean, pct])
        train_utils.write_tensorboard("MTE" + tb_string, mean)
        train_utils.write_tensorboard("PCT" + tb_string, pct)
        print("Current {} PCT : {:.4f}".format(postfix, pct))

        if isinstance(
                val_dataset.dataset, ArbitraryKeypointsDataWrapper
                ) and val_dataset.dataset.arbitrary_keypoint_mode == ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS:
            num_bodyparts = len(val_dataset.dataset.bodypart_order.get_used_bodypart_triples())
            distances = thickness_res[0]
            assert len(distances) % num_bodyparts == 0
            distances = np.asarray(distances)
            distances = distances.reshape(len(annos), num_bodyparts, -1)
            names = val_dataset.dataset.bodypart_order.names()
            names = [names[bodypart[-1]] for bodypart in val_dataset.dataset.bodypart_order.get_used_bodypart_triples()]
            for i in range(num_bodyparts):
                mean, std, pct = thickness_metric(distances[:, i])
                train_utils.write_tensorboard("MTE_per_joint" + tb_string + "/{}".format(names[i]), mean)
                train_utils.write_tensorboard("PCT_per_joint" + tb_string + "/{}".format(names[i]), pct)

    return metric_results
