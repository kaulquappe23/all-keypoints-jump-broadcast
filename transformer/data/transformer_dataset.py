# -*- coding: utf-8 -*-
"""
Created on 17.04.23

"""
import random

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from datasets.general.dataset_utils import get_dataset_from_config
from datasets.general.dataset_wrapper import DataWrapper
from transformer.execution.pre_postprocessing import get_affine_transform, affine_transform, flip_additional_vectors, \
    fliplr_joints
from utils.general_utils import get_dict


class TransformerDatasetWrapper(Dataset):
    """
    Wrapper for the transformer dataset
    """

    def __init__(self, cfg, params=None):

        self.aspect_ratio = 192.0 / 256
        self.pixel_std = 200

        self.name = cfg.NAME
        self.cfg = cfg
        self.dataset: DataWrapper = get_dict(params, 'dataset')

        self.num_joints = cfg.JOINT_ORDER.get_num_joints()
        self.image_size = cfg.INPUT_SIZE
        self.heatmap_size = cfg.OUTPUT_SIZE[0]

        self.transform = None if 'transforms' not in params else params['transforms']
        if self.transform is None:
            self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.is_train = False if 'is_train' not in params else params['is_train']

        self.augment = get_dict(params, "augment", True)

    def postfix(self):
        return self.dataset.postfix

    def __len__(self):
        return len(self.dataset.image_ids)

    def __getitem__(self, idx):
        image_id = self.dataset.image_ids[idx]

        num_points_total = self.dataset.get_num_keypoints()
        joints = self.dataset.load_keypoints(image_id)
        model_representations = None

        if isinstance(joints, tuple):  # we have annotations and representation vectors to process
            joints, representations = joints
            if len(joints.shape) == 3:
                joints = joints[0]  # the transformer can only deal with a single person per image
            model_representations = {}
            num_pad = num_points_total - joints.shape[0]
            joints = np.pad(joints, ((0, num_pad), (0, 0)))
            if "keypoints" in representations or "kp_vector" in representations:
                keypoint_vectors = representations["keypoints"] if "keypoints" in representations else representations[
                    "kp_vector"]
                keypoint_vectors = np.pad(keypoint_vectors, ((0, num_pad), (0, 0)))
                model_representations["keypoint_vector"] = keypoint_vectors
            if "norm_pose" in representations:
                norm_pose_vectors = representations["norm_pose"]
                norm_pose_vectors = np.pad(norm_pose_vectors, ((0, num_pad), (0, 0)))
                model_representations["norm_pose_vector"] = norm_pose_vectors
            if "thickness" in representations:
                thickness_vectors = representations["thickness"]
                thickness_vectors = np.pad(thickness_vectors, ((0, num_pad), (0, 0)))
                model_representations["thickness_vector"] = thickness_vectors
            if "angle" in representations:
                angle_vectors = representations["angle"]
                angle_vectors = np.pad(angle_vectors, ((0, num_pad), (0, 0)))
                model_representations["angle_vector"] = angle_vectors
        else:
            joints = joints[0]  # the transformer can only deal with a single person per image

        img = self.dataset.load_image(image_id)
        bbox = self.dataset.load_bbox(image_id)

        height, width = img.shape[:2]

        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if not x2 >= x1 and y2 >= y1:
            raise RuntimeError
        clean_box = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        c, s = self._box2cs(clean_box[:4])
        r = 0

        # augmentation
        if self.is_train and self.augment:
            if np.random.rand() < self.cfg.HALF_BODY_AUG and np.sum(joints[:, 2]) > self.cfg.NUM_JOINTS_HALF_BODY:
                c_half_body, s_half_body = self.half_body_transform(joints)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.cfg.MAX_SCALE
            rf = self.cfg.MAX_ROTATION
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if random.random() < self.cfg.USE_FLIP:
                img = img[:, ::-1, :]
                if model_representations is None:
                    joints = fliplr_joints(joints, img.shape[1], self.cfg.FLIP_PAIRS)
                else:
                    joints = fliplr_joints(joints, img.shape[1],
                                           [])  # just calculate new coordinates but do not switch left and right
                    model_representations = flip_additional_vectors(model_representations,
                                                                    flip_pairs=self.dataset.joint_order.flip_pairs())

                c[0] = img.shape[1] - c[0] - 1

        joints_heatmap = joints.copy()
        orig_joints = joints.copy()
        trans = get_affine_transform(c, s, r, self.image_size)
        trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
                img,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

        for i in range(joints.shape[0]):
            if joints[i, 2] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

        input = self.transform(input)

        target, target_weight = self.generate_target(joints_heatmap)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
                'image_id':                    image_id,
                'center':                      c,
                'scale':                       s,
                'rotation':                    r,
                'transformed_annotations':     joints,
                'non_transformed_annotations': orig_joints,
                'image_width':                 img.shape[1],
                }

        if representations is not None:
            meta['representations'] = model_representations

        return input, target, target_weight, meta

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w - 1) * 0.5
        center[1] = y + (h - 1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
                [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def half_body_transform(self, joints):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints[joint_id][2] > 0:
                if joint_id in self.cfg.UPPER_BODY:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0] + 1
        h = right_bottom[1] - left_top[1] + 1

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
                [
                        w * 1.0 / self.pixel_std,
                        h * 1.0 / self.pixel_std
                        ],
                dtype=np.float32
                )

        scale = scale * 1.5

        return center, scale

    def generate_target(self, joints):
        num_joints = joints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, 2]

        target = np.zeros((num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.cfg.SIGMA * 3

        for joint_id in range(num_joints):
            target_weight[joint_id] = \
                self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)

            if target_weight[joint_id] == 0:
                continue

            mu_x = joints[joint_id][0]
            mu_y = joints[joint_id][1]

            x = np.arange(0, self.heatmap_size[0], 1, np.float32)
            y = np.arange(0, self.heatmap_size[1], 1, np.float32)
            y = y[:, np.newaxis]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.cfg.SIGMA ** 2))

        return target, target_weight

    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


class TransformerDataset(TransformerDatasetWrapper):
    """
    The real Transformer dataset, loading a dataset from the config
    """

    def __init__(self, cfg, params):
        subset = params["subset"]
        dataset = get_dataset_from_config(cfg, subset, params)
        params_superclass = {
                "dataset":  dataset,
                "is_train": "train" in subset,
                "augment":  True if "train" in subset else False,
                }
        super().__init__(cfg, params_superclass)
