# -*- coding: utf-8 -*-
"""
Created on 24.08.22

"""
import argparse
import os

from datasets.general.csv_annotation_utils import read_csv_bboxes
from datasets.jump.jump_data import YTJumpCSVInformation, setup_jump_norm_pose
from paths import YTJumpLoc
from transformer.visualization.streamlit_model_visualization import streamlit_show_model_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        default="0",
                        help='gpu id to use',
                        type=str)

    parser.add_argument('--weights',
                        required=True,
                        help='weights file to use',
                        type=str)

    parser.add_argument('--model_name',
                        required=False,
                        help='name of model in weights file',
                        default="ema_model",
                        type=str)

    parser.add_argument('--subset',
                        help='subset for evaluation',
                        required=False,
                        default="val",
                        type=str)

    args = parser.parse_args()

    return args


def streamlit_jump_broadcast_results(weights_file, model_name="ema_model", subset="val", gpu_id="0"):
    config_name = "yt_jump-seg"
    csv_info = YTJumpCSVInformation
    seg_boxes = YTJumpLoc.segmentation_bbox_path
    bboxes = read_csv_bboxes(seg_boxes, header=True)
    save_path = os.path.join(YTJumpLoc.base_path, "visualizations")
    norm_pose_settings_func = setup_jump_norm_pose
    streamlit_show_model_results(config_name, weights_file, subset, csv_info, bboxes, save_path, norm_pose_settings_func,
                                 gpu_id=gpu_id, model_name=model_name)


if __name__ == '__main__':
    args = parse_args()
    streamlit_jump_broadcast_results(args.weights, args.model_name, subset=args.subset, gpu_id=args.gpu)
