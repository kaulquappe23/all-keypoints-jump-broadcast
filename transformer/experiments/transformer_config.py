# -*- coding: utf-8 -*-
"""
Created on 26.08.21

"""
import sys

from paths import GeneralLoc


class TransformerConfig:

    def __init__(self):
        super().__init__()

        self.TRAIN_SUBSET = "train"
        self.TEST_SUBSET = "test"
        self.VAL_SUBSET = "val"

        self.SIGMA = 2
        self.OUTPUT_SIZE = [(48, 64)]
        self.INPUT_SIZE = (192, 256)
        self.PATCH_SIZE = (4, 3)

        self.SINGLE_GPU_BATCH_SIZE = 32
        if sys.gettrace() is None:
            self.WORKERS = 32
        else:
            self.WORKERS = 0

        self.METRICS_HIGH_IS_BETTER = [True, True, True]  # Standard metric is PCK at thresholds 0.05, 0.1, 0.2

        # DATA AUGMENTATION
        self.MAX_ROTATION = 45
        self.MAX_SCALE = 0.35
        self.USE_FLIP = 0.5
        self.HALF_BODY_AUG = 0.0
        self.COLOR_JITTER = 0.2

        self.SCALE_TYPE = "short"

        self.EMBED_SIZE = 192
        self.CNN = "hrnet_stage3"
        self.POS_ENCODING = "sine"
        self.LR = 1e-3

        self.NUM_VAL_IMAGES = None
        self.VAL_STEPS = 5000
        self.TEST_POINTS = (2, 25)

        # include keypoint tokens for keys and values in the attention
        self.KEYPOINT_TOKEN_ATTENTION = False

        self.NUM_STEPS = None
        self.TENSORBOARD_FREQ = 1000
        self.LOGS = GeneralLoc.log_path

        self.JOINT_ORDER = None

    def check_config(self):
        assert hasattr(self, "NAME"), "Please specify a name for the experiment"
        assert self.JOINT_ORDER is not None, "Please specify a joint order"
        assert hasattr(self, "NUM_STEPS"), "Please specify the training duration"
