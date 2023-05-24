# -*- coding: utf-8 -*-
"""
Created on 23.10.20

"""

from datasets.jump.jump_bodypart_order import JumpHeadAngleBodypartOrder, JumpHeadEndpointBodypartOrder
from datasets.jump.jump_joint_order import JumpJointOrder
from paths import YTJumpLoc
from transformer.experiments.transformer_config import TransformerConfig
from transformer.model.token_embedding import TokenEmbedding


class YTJumpSegmentationTransformerConfig(TransformerConfig):

    def __init__(self):

        super().__init__()

        self.NAME = "yt_jump"
        self.ROOT = YTJumpLoc.base_path
        self.ANNOTATIONS_DIR = YTJumpLoc.annotation_dir

        self.NUM_VAL_IMAGES = None
        self.VAL_STEPS = 5000
        # Define if a metric score during validation is better if its higher or lower
        # PCK at 0.05, 0.1 and 0.2 for standard points and all points and MTE and PCT
        self.METRICS_HIGH_IS_BETTER = [True, True, True] * 2 + [False, True]

        self.NUM_STEPS = 300000

        self.PRETRAINED_FILE = YTJumpLoc.pretrained_model

        self.JOINT_ORDER = JumpJointOrder

        # select the number of keypoints to generate for each image
        self.GENERATE_KEYPOINTS = (5, 50)

        # select the token embedding type, see class TokenEmbedding for all options
        self.TOKEN_EMBEDDING = {
                "representation_type": TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE,
                "num_layers":          2,
                }

        if self.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE or \
                self.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.NORM_POSE:
            self.BODYPART_ORDER = JumpHeadAngleBodypartOrder
        else:
            self.BODYPART_ORDER = JumpHeadEndpointBodypartOrder

        self.check_config()

    def check_config(self):
        super().check_config()
        assert hasattr(self, "TOKEN_EMBEDDING")
        assert hasattr(self, "BODYPART_ORDER")
        if self.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE or \
                self.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.NORM_POSE:
            assert self.BODYPART_ORDER == JumpHeadAngleBodypartOrder
        else:
            assert self.BODYPART_ORDER == JumpHeadEndpointBodypartOrder
        if self.NUM_VAL_IMAGES is not None and self.NUM_VAL_IMAGES < 100:
            print(
                    "------------------------------------------------------------------------\n"
                    "WARNING: VALIDATION STEPS ARE REALLY LOW!\n"
                    "------------------------------------------------------------------------"
                    )
