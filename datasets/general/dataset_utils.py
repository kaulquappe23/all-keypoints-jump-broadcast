# -*- coding: utf-8 -*-
"""
Created on 20.04.23

"""

from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import DataLoader

from datasets.general.arbitrary_keypoints_data_wrapper import ArbitraryKeypointsDataWrapper
from datasets.jump.jump_data import JumpArbitraryKeypoints
from transformer.model.token_embedding import TokenEmbedding
from utils.general_utils import get_dict


def get_dataset_class_and_params(name, params):
    if name == "yt_jump" and get_dict(params, "arbitrary_points", False):
        return JumpArbitraryKeypoints, {"is_youtube": True}
    else:
        raise ValueError("Unknown dataset name: {} with params".format(name, params))


def get_dataset_from_config(cfg, subset, params):
    arbitrary_keypoints_mode = ArbitraryKeypointsDataWrapper.GENERATE_POINTS if "train" in subset else \
        ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS
    num_annotations_to_use = None if not "val" in subset else cfg.NUM_VAL_IMAGES
    dataset_class, additional_params = get_dataset_class_and_params(cfg.NAME, params)
    if cfg.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.KEYPOINT_VECTOR_WITHOUT_THICKNESS or \
            cfg.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS or \
            cfg.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE:
        representation_type = ArbitraryKeypointsDataWrapper.KEYPOINT_VECTOR
    elif cfg.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.NORM_POSE:
        representation_type = ArbitraryKeypointsDataWrapper.NORM_POSE
    else:
        raise ValueError("Unknown representation type: {}".format(cfg.TOKEN_EMBEDDING["representation_type"]))
    dataset = dataset_class(
            subset=subset,
            params=additional_params,
            bodypart_order=cfg.BODYPART_ORDER,
            representation_type=representation_type,
            arbitrary_keypoint_mode=arbitrary_keypoints_mode,
            test_points=(2, 25),
            num_points_to_generate=cfg.GENERATE_KEYPOINTS,
            num_annotations_to_use=num_annotations_to_use
            )
    return dataset


def get_dataloader(dataset, is_train, batch_size=1, num_workers=1):
    """
    Get dataloader for given dataset
    :param is_train: shuffle and drop last are set to True if is_train is True
    :param dataset:
    :param batch_size:
    :param num_workers:
    :return:
    """
    dataloader_class = MultiEpochsDataLoader if is_train else DataLoader
    data_loader = dataloader_class(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=is_train
            )
    return data_loader
