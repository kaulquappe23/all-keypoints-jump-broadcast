# -*- coding: utf-8 -*-
"""
Created on 14.04.23

"""
import os
import time
from typing import Type

import cv2
import numpy as np
import streamlit as st
from pandas import DataFrame
from st_aggrid import GridOptionsBuilder, AgGrid

from datasets.general.csv_annotation_utils import CSVInformation, read_csv_annotations
from transformer.model.token_embedding import TokenEmbedding
from utils.config_utils import get_transformer_config_by_name


def streamlit_show_model_results(
        config_name, weights_file, subset, csv_info: Type[CSVInformation], bboxes, save_path, norm_pose_settings_func,
        model_name=None, gpu_id='0'
        ):
    """
    streamlit interactive visualization for model results
    :param norm_pose_settings_func: function to get norm pose settings, should take body part order as an input
    :param config_name: name of config file
    :param weights_file: weights path
    :param subset: subset of images to visualize
    :param csv_info: CSVInformation object for the dataset
    :param bboxes: bounding boxes for the images, dict with image_id as key and bbox as value
    :param save_path: path to save visualized images
    :param gpu_id: gpu id to run on, should be string and None if running on CPU
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import torch
    from transformer.model import token_pose
    from transformer.model.token_pose import convert_state_dict
    from transformer.visualization.vis_thickness import visualize_thickness_grid, visualize_thickness_outline

    torch.multiprocessing.set_sharing_strategy('file_system')

    st.set_page_config(page_title="Model Results Visualization")

    st.header("Model Results Visualization")

    if gpu_id is not None:
        st.write("Running on GPU: {}".format(gpu_id))
    else:
        st.write("Running on CPU".format(gpu_id))

    st.write("Displaying results based on {}".format(weights_file))

    if "model" not in st.session_state:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_file, map_location=device)
        if "config" in checkpoint:
            st.session_state.config = checkpoint["config"]
        else:
            st.session_state.config = get_transformer_config_by_name(config_name)
        if model_name is not None:
            checkpoint = checkpoint[model_name]
        st.session_state.norm_pose_settings = norm_pose_settings_func(st.session_state.config.BODYPART_ORDER) if \
            st.session_state.config.TOKEN_EMBEDDING["representation_type"] == TokenEmbedding.NORM_POSE else None
        model = token_pose.get_token_pose_net(st.session_state.config, load_weights_from_config=False)
        model.load_state_dict(convert_state_dict(checkpoint, st.session_state.config.TOKEN_EMBEDDING))
        if torch.cuda.is_available():
            model = model.cuda()
        st.session_state.model = model

    if "image_ids" not in st.session_state:
        st.session_state.bboxes = bboxes
        additional_info, keypoints = read_csv_annotations(csv_info.csv_path(subset), csv_info.get_header(),
                                                          st.session_state.config.JOINT_ORDER.get_num_joints())
        image_ids = []
        st.session_state.image_paths = {}
        for info in additional_info:
            image_id = csv_info.image_id_from_info(info)
            image_ids.append(image_id)
            st.session_state.image_paths[image_id] = csv_info.image_path(info, subset)
        image_ids = sorted(image_ids)
        st.session_state.image_ids = image_ids

    style_selection = st.sidebar.radio("Display Style:", ["Fixed Keypoints", "Grid", "Lines"])

    df = DataFrame({'Image IDs': st.session_state.image_ids})

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gridoptions = gb.build()

    with st.sidebar:
        response = AgGrid(df,
                          height=350,
                          width=200,
                          gridOptions=gridoptions,
                          )

        selected = response['selected_rows']

    bps = [bp for bp in list(st.session_state.config.BODYPART_ORDER.get_bodypart_to_keypoint_dict().keys()) if
           bp not in st.session_state.config.BODYPART_ORDER.get_adjacent_bodyparts()]
    bodypart_names = np.asarray(st.session_state.config.BODYPART_ORDER.names())[bps]
    boxes = []
    st.sidebar.write("Bodyparts to draw:")
    for bodypart_name in bodypart_names:
        boxes.append(st.sidebar.checkbox(bodypart_name, value=True))
    used_bodyparts_ = st.session_state.config.BODYPART_ORDER.get_used_bodypart_triples()
    used_bodyparts = []
    i = 0
    for bodypart in used_bodyparts_:
        if bodypart[-1] not in st.session_state.config.BODYPART_ORDER.get_adjacent_bodyparts():
            if boxes[i]:
                used_bodyparts.append(bodypart)
            i += 1

    image_scale = st.sidebar.slider('Image scale:', 0.5, 5.0, 1.5)

    if len(selected) > 0:
        image_id = selected[0]['Image IDs']

        if style_selection == "Grid":
            indices = [idx[-1] for idx in used_bodyparts]
            image = visualize_thickness_grid(st.session_state.image_paths[image_id], st.session_state.bboxes[image_id],
                                             st.session_state.model, st.session_state.config.BODYPART_ORDER,
                                             indices, st.session_state.config.JOINT_ORDER, scale=image_scale,
                                             norm_pose_settings=st.session_state.norm_pose_settings,
                                             convert_bgr=False, blur=None, cutout=(3, 4))
        elif style_selection == "Lines":
            num_points_per_line = st.sidebar.slider('Points per thickness line:', 2, 200, 100)
            num_lines = st.sidebar.slider('Number of thickness lines (one side):', 0, 10, 2)
            circle_size = st.sidebar.slider('Circle size:', 1, 10, 1)
            indices = [idx[-1] for idx in used_bodyparts]
            colors = np.asarray(st.session_state.config.BODYPART_ORDER.get_colors())[indices].tolist()
            image = visualize_thickness_outline(st.session_state.image_paths[image_id], st.session_state.bboxes[image_id],
                                                st.session_state.model, st.session_state.config.BODYPART_ORDER,
                                                indices, st.session_state.config.JOINT_ORDER, scale=image_scale,
                                                norm_pose_settings=st.session_state.norm_pose_settings,
                                                convert_bgr=False, cutout=(3, 4), num_outline_points=num_points_per_line,
                                                num_thickness_lines=num_lines, model_points=None,
                                                circle_size=circle_size, colors=colors, blur=None)
        else:
            image = visualize_thickness_grid(st.session_state.image_paths[image_id], st.session_state.bboxes[image_id],
                                             st.session_state.model, st.session_state.config.BODYPART_ORDER,
                                             [], st.session_state.config.JOINT_ORDER, scale=image_scale,
                                             norm_pose_settings=st.session_state.norm_pose_settings,
                                             convert_bgr=False, blur=None, cutout=(3, 4))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image)

        if st.button("Save"):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_save_path = os.path.join(save_path, image_id.replace(".jpg", "") + "_{}.png".format(time.time()))
            cv2.imwrite(image_save_path, image)
            st.success("Wrote image to " + image_save_path)


    else:
        st.write("Choose an image from the table on the left.")
