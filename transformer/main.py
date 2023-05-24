# -*- coding: utf-8 -*-
"""
Created on 03.08.22

"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train HPE network')

    parser.add_argument('--gpu',
                        default="0",
                        help='gpu id for multiprocessing training',
                        type=str)

    parser.add_argument('--cfg',
                        help='experiment configuration file name',
                        required=False,
                        default="yt_jump-seg",
                        type=str)

    parser.add_argument('--tb',
                        help='tensorboard string',
                        required=False,
                        type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    from utils import training_utils
    from utils.config_utils import get_transformer_config_by_name

    training_utils.set_deterministic()

    if args.tb is None:
        args.tb = ""

    from transformer.execution.train import TokenPoseTraining

    assert args.cfg is not None, "Config file --cfg is required"
    config = get_transformer_config_by_name(args.cfg)

    params = None

    train = TokenPoseTraining(config, tb_prefix=args.tb)
    if hasattr(config, "RESUME"):
        train.resume(config.RESUME)
    train.run()
