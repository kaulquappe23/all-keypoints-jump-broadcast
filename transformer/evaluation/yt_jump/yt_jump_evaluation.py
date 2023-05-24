# -*- coding: utf-8 -*-
"""
Created on 14.04.23

"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HPE network')

    parser.add_argument('--gpu',
                        default="0",
                        help='gpu id for evaluation',
                        type=str)

    parser.add_argument('--bs',
                        help='batch size during evaluation',
                        required=False,
                        default=200,
                        type=int)

    parser.add_argument('--subset',
                        help='subset for evaluation',
                        required=False,
                        default="val",
                        type=str)

    args = parser.parse_args()

    return args


def head_endpoint_2_layer(inference_again=False, distances_again=False, batch_size=200, subset="val"):
    weights_file = "transformer/pretrained_weights/jump_broadcast_head_endpoint_2L.pth.tar"
    print(os.path.abspath(weights_file))
    title = "head endpoint 2 layer embedding"

    general_eval_pipeline(inference_again, distances_again, weights_file, title, config=None, model_name="ema_model",
                          subset=subset, batch_size=batch_size)


def head_angle_2_layer(inference_again=False, distances_again=False, batch_size=200, subset="val"):
    weights_file = "transformer/pretrained_weights/jump_broadcast_head_angle_2L.pth.tar"
    title = "head angle 2 layer embedding"

    general_eval_pipeline(inference_again, distances_again, weights_file, title, config=None, model_name="ema_model",
                          subset=subset, batch_size=batch_size)


def norm_pose_4_layer(inference_again=False, distances_again=False, batch_size=200, subset="val"):
    weights_file = "transformer/pretrained_weights/jump_broadcast_norm_pose.pth.tar"
    title = "norm pose: head angle 4 layer embedding"

    general_eval_pipeline(inference_again, distances_again, weights_file, title, config=None, model_name="ema_model",
                          subset=subset, batch_size=batch_size)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from transformer.evaluation.evaluation import general_eval_pipeline

    subset = args.subset
    batch_size = args.bs

    head_endpoint_2_layer(True, True, batch_size=batch_size, subset=subset)
    head_angle_2_layer(True, True, batch_size=batch_size, subset=subset)
    norm_pose_4_layer(True, True, batch_size=batch_size, subset=subset)
