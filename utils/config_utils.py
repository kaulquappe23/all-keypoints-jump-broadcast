# -*- coding: utf-8 -*-
"""
Created on 13.04.23

"""
from transformer.experiments.jump.yt_jump_segmentation_config import YTJumpSegmentationTransformerConfig


def get_transformer_config_by_name(config_name):
    registered_configs = {
            "yt_jump-seg": YTJumpSegmentationTransformerConfig,
            }
    return registered_configs[config_name]()


def check_arbitrary_keypoints(cfg):
    if not hasattr(cfg, "GENERATE_KEYPOINTS"):
        return False
    if cfg.GENERATE_KEYPOINTS is None:
        return False
    return True
