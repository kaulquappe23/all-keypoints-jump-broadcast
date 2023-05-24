# -*- coding: utf-8 -*-
"""
Created on 21.09.22

"""
import os

from pandas import DataFrame

from datasets.general.arbitrary_keypoints_data_wrapper import ArbitraryKeypointsDataWrapper
from datasets.general.bodypart_segmentation_utils import calculate_kp, calculate_endpoint_angle_kp, calculate_adjacent_kp
from datasets.jump.jump_bodypart_order import JumpBodypartOrder, JumpHeadEndpointBodypartOrder, JumpHeadAngleBodypartOrder
from datasets.jump.jump_joint_order import JumpJointOrder
from utils.color_utils import adjust_color

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

from datasets.jump.jump_data import JumpArbitraryKeypoints


def vis_lines(
        skeleton_image, keypoints, image_id, bodypart_keypoints, bodypart_index, bbox, segmentation_masks, x1, y1, circle_size,
        thickness
        ):
    kp_vector = np.zeros((500, 20))
    thickness_vector = np.zeros((500, 3))
    angle_vector = np.zeros((500, 1))
    new_annotations = np.zeros((500, 3))
    norm_pose = np.zeros((500, 2))
    bodypart = (bodypart_keypoints[0], bodypart_keypoints[1], bodypart_index)
    count = 0
    for thick in range(5):
        percentage_thickness = (thick / 4 - 0.5) * 2
        for position in range(100):
            percentage_projection = position / 99
            res = st.session_state.jump.calculate_arbitrary_keypoint(percentage_projection, percentage_thickness, bodypart,
                                                                     image_id, bbox, keypoints, segmentation_masks,
                                                                     new_annotations, kp_vector, thickness_vector, norm_pose,
                                                                     angle_vector, count, switch_left_right_if_necessary=True)
            if res:
                if percentage_thickness >= 0:
                    color = adjust_color((255, 0, 0), 1 - np.abs(percentage_thickness))
                else:
                    color = adjust_color((0, 255, 0), 1 - np.abs(percentage_thickness))
                cv2.circle(skeleton_image, center=(int(new_annotations[count, 0]) - x1, int(new_annotations[count, 1]) - y1),
                           radius=circle_size, color=color, thickness=thickness)

                count += 1


def create_vis_image(image_id, bbox, keypoints):
    seg_path = st.session_state.jump.bodypart_masks[image_id]
    segmentation_masks = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED) / 10
    image = st.session_state.jump.load_image(image_id)
    x1, y1, w, h = bbox
    image = image[y1:y1 + h, x1:x1 + w]
    I_vis = image.copy() * 0.7
    MaskBool = np.tile((segmentation_masks == 0)[:, :, np.newaxis], [1, 1, 3])
    #  Replace the visualized mask image with I_vis.
    Mask_vis = np.zeros((segmentation_masks.shape[0], segmentation_masks.shape[1], 3))
    colors = JumpBodypartOrder.get_rgb_colors()
    for i in range(15):
        Mask_vis[segmentation_masks == i] = list(colors[i])
    Mask_vis *= 255
    Mask_vis[MaskBool] = I_vis[MaskBool]
    I_vis = I_vis * 0.3 + Mask_vis * 0.7
    I_vis = np.asarray(I_vis, dtype=int)
    image = I_vis
    skeleton_image = image.astype(np.uint8).copy()
    for i in range(keypoints.shape[0]):
        cv2.circle(skeleton_image, center=(int(keypoints[i, 0]) - x1, int(keypoints[i, 1]) - y1), radius=circle_size,
                   color=(255, 255, 255), thickness=thickness)
    if bodypart_index <= 14:
        mask_ind = np.where(segmentation_masks == bodypart_index)
        used_bodypart_exists = True
        for i in range(2):
            if isinstance(bodypart_keypoints[i], int):
                if keypoints[bodypart_keypoints[i]][2] == 0:
                    used_bodypart_exists = False
                    break
            else:
                bp1, bp2 = bodypart_keypoints[i]
                if keypoints[bp1][2] == 0 or keypoints[bp2][2] == 0:
                    used_bodypart_exists = False
                    break
    else:
        mask_ind = np.where(np.logical_or(segmentation_masks == bodypart1_index, segmentation_masks == bodypart2_index))
        used_bodypart_exists = True
        for i in range(2):
            if isinstance(bodypart1_keypoints[i], int):
                if keypoints[bodypart1_keypoints[i]][2] == 0:
                    used_bodypart_exists = False
                    break
            else:
                bp1, bp2 = bodypart1_keypoints[i]
                if keypoints[bp1][2] == 0 or keypoints[bp2][2] == 0:
                    used_bodypart_exists = False
                    break
        for i in range(2):
            if isinstance(bodypart2_keypoints[i], int):
                if keypoints[bodypart2_keypoints[i]][2] == 0:
                    used_bodypart_exists = False
                    break
            else:
                bp1, bp2 = bodypart2_keypoints[i]
                if keypoints[bp1][2] == 0 or keypoints[bp2][2] == 0:
                    used_bodypart_exists = False
                    break
    line_on_mask_exists = len(mask_ind[0]) > 0
    return skeleton_image, segmentation_masks, used_bodypart_exists, line_on_mask_exists


def vis_normal_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness):
    bodypart = (bodypart_keypoints[0], bodypart_keypoints[1], bodypart_index)
    new_point, vis_res = calculate_kp(keypoints, bodypart, segmentation_masks, percentage_thickness, percentage_projection, bbox,
                                      crop_region=True, projection_point_outside_mask=True, switch_left_right=False)
    if vis_res is not None:
        mask, x_inputs, y_outputs, projection_point, p1, p2 = vis_res

        if x_inputs is not None:
            y_outputs -= y1
            x_inputs -= x1
            skeleton_image[y_outputs, x_inputs] = 255
        if p1 is not None:
            cv2.circle(skeleton_image, center=(int(p1[0] - x1), int(p1[1] - y1)), radius=circle_size, color=(0, 0, 255),
                       thickness=thickness)
            cv2.circle(skeleton_image, center=(int(p2[0] - x1), int(p2[1] - y1)), radius=circle_size, color=(255, 255, 0),
                       thickness=thickness)
        cv2.circle(skeleton_image, center=(int(projection_point[0]) - x1, int(projection_point[1]) - y1), radius=circle_size,
                   color=(0, 255, 0), thickness=thickness)
        if new_point is not None:
            cv2.circle(skeleton_image, center=(int(new_point[0]) - x1, int(new_point[1]) - y1), radius=circle_size,
                       color=(255, 0, 0), thickness=thickness)


def vis_endpoint_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness, keypoints):
    endpoints = st.session_state.jump.endpoint_dict[image_id]
    if bodypart_index in endpoints:
        endpoints = endpoints[bodypart_index]
        endpoint1, endpoint2 = endpoints
        cv2.circle(skeleton_image, center=(int(endpoint1[0]) - x1, int(endpoint1[1]) - y1), radius=circle_size, color=(0, 0, 0),
                   thickness=thickness)
        cv2.circle(skeleton_image, center=(int(endpoint2[0]) - x1, int(endpoint2[1]) - y1), radius=circle_size, color=(0, 0, 0),
                   thickness=thickness)

        bodypart = (keypoints.shape[0], keypoints.shape[0] + 1, bodypart_index)
        keypoints = np.concatenate((keypoints, np.array(endpoints)), axis=0)
        new_point, vis_res = calculate_kp(keypoints, bodypart, segmentation_masks, percentage_thickness, percentage_projection,
                                          bbox,
                                          crop_region=True, projection_point_outside_mask=True)
        if vis_res is not None:
            mask, x_inputs, y_outputs, projection_point, p1, p2 = vis_res

            if x_inputs is not None:
                y_outputs -= y1
                x_inputs -= x1
                skeleton_image[y_outputs, x_inputs] = 255
            if p1 is not None:
                cv2.circle(skeleton_image, center=(int(p1[0] - x1), int(p1[1] - y1)), radius=circle_size, color=(0, 0, 255),
                           thickness=thickness)
                cv2.circle(skeleton_image, center=(int(p2[0] - x1), int(p2[1] - y1)), radius=circle_size, color=(255, 255, 0),
                           thickness=thickness)
            cv2.circle(skeleton_image, center=(int(projection_point[0]) - x1, int(projection_point[1]) - y1), radius=circle_size,
                       color=(0, 255, 0), thickness=thickness)
            if new_point is not None:
                cv2.circle(skeleton_image, center=(int(new_point[0]) - x1, int(new_point[1]) - y1), radius=circle_size,
                           color=(255, 0, 0), thickness=thickness)


def vis_endpoint_angle_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness):
    bodypart = (bodypart_keypoints[0], bodypart_keypoints[1], bodypart_index)
    new_point, vis_res = calculate_endpoint_angle_kp(keypoints, bodypart, segmentation_masks, percentage_thickness,
                                                     percentage_projection, bbox, crop_region=True)

    if vis_res is not None:
        (mask, keypoint, x_res, y_res, border_point, intersection, anchor) = vis_res
        cv2.circle(skeleton_image, center=(int(keypoint[0]) - x1, int(keypoint[1]) - y1), radius=circle_size, color=(0, 255, 0),
                   thickness=thickness)
        cv2.circle(skeleton_image, center=(int(border_point[0]) - x1, int(border_point[1]) - y1), radius=circle_size,
                   color=(0, 255, 255), thickness=thickness)
        cv2.circle(skeleton_image, center=(int(intersection[0]) - x1, int(intersection[1]) - y1), radius=circle_size,
                   color=(255, 255, 0), thickness=thickness)
        cv2.circle(skeleton_image, center=(int(anchor[0]) - x1, int(anchor[1]) - y1), radius=circle_size, color=(0, 0, 255),
                   thickness=thickness)
        if x_res is not None:
            y_res -= y1
            x_res -= x1
            skeleton_image[y_res, x_res] = 255
    if new_point is not None:
        cv2.circle(skeleton_image, center=(int(new_point[0]) - x1, int(new_point[1]) - y1), radius=circle_size, color=(255, 0, 0),
                   thickness=thickness)


def vis_elbow_knee_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness):
    bodypart1 = (bodypart1_keypoints[0], bodypart1_keypoints[1], bodypart1_index)
    bodypart2 = (bodypart2_keypoints[0], bodypart2_keypoints[1], bodypart2_index)

    if bodypart_index in st.session_state.jump.anchors[image_id]:
        anchor_res = st.session_state.jump.anchors[image_id][bodypart_index]
        new_point, percentage, _, vis_res = calculate_adjacent_kp(keypoints, bodypart1, bodypart2, segmentation_masks,
                                                                  percentage_thickness, percentage_projection, bbox,
                                                                  crop_region=True, anchor=anchor_res, switch_left_right=True)
        if vis_res is not None:
            (mask, keypoint, x_res, y_res, border_point, intersection, anchor) = vis_res
            cv2.circle(skeleton_image, center=(int(keypoint[0]) - x1, int(keypoint[1]) - y1), radius=circle_size,
                       color=(255, 255, 0), thickness=thickness)
            cv2.circle(skeleton_image, center=(int(border_point[0]) - x1, int(border_point[1]) - y1), radius=circle_size,
                       color=(0, 255, 0), thickness=thickness)
            cv2.circle(skeleton_image, center=(int(intersection[0]) - x1, int(intersection[1]) - y1), radius=circle_size,
                       color=(255, 0, 255), thickness=thickness)
            cv2.circle(skeleton_image, center=(int(anchor[0]) - x1, int(anchor[1]) - y1), radius=circle_size, color=(255, 155, 0),
                       thickness=thickness)
            if x_res is not None:
                y_res -= y1
                x_res -= x1
                skeleton_image[y_res, x_res] = 255
        if new_point is not None:
            cv2.circle(skeleton_image, center=(int(new_point[0]) - x1, int(new_point[1]) - y1), radius=circle_size,
                       color=(255, 0, 0), thickness=thickness)
        st.write("Anchor: orange, Intermediate point: purple, Border: green, Common Keypoint: yellow, New point: red")


def vis_creation_details(image_id, bbox, segmentation_masks, percentage_projection, percentage_thickness, keypoints):
    kp_vector = np.zeros((1, 20))
    thickness_vector = np.zeros((1, 3))
    angle_vector = np.zeros((1, 1))
    new_annotations = np.zeros((1, 3))
    norm_pose = np.zeros((1, 2))
    bodypart = (bodypart_keypoints[0], bodypart_keypoints[1], bodypart_index)
    res = st.session_state.jump.calculate_arbitrary_keypoint(percentage_projection, percentage_thickness, bodypart, image_id,
                                                             bbox, keypoints, segmentation_masks,
                                                             new_annotations, kp_vector, thickness_vector, norm_pose,
                                                             angle_vector, 0, switch_left_right_if_necessary=True)
    if st.session_state.jump.representation_type == ArbitraryKeypointsDataWrapper.KEYPOINT_VECTOR:
        percentages = ""
        for i in range(kp_vector.shape[1]):
            if kp_vector[0, i] > 0:
                percentages += "{}: {:.2f}, ".format(JumpJointOrder.names()[i], kp_vector[0, i])

        st.write(percentages)
        st.write("Thickness Vector:")
        st.write(thickness_vector[0])
        st.write("Angle Vector:")
        st.write(angle_vector[0])
    else:
        st.write("Norm Pose Coordinates:")
        st.write(norm_pose[0])

        norm_pose_info = st.session_state.jump.norm_pose_settings
        keypoints = norm_pose_info["keypoints"]
        norm_pose_mask = norm_pose_info["bodypart_map"]

        Mask_vis = np.zeros((norm_pose_mask.shape[0], norm_pose_mask.shape[1], 3))
        colors = JumpBodypartOrder.get_rgb_colors()
        for i in range(15):
            Mask_vis[norm_pose_mask == i] = list(colors[i])
        Mask_vis *= 255
        image1 = Mask_vis
        skeleton_image1 = image1.astype(np.uint8).copy()

        color_ = (255, 255, 255)  # (200, 200, 200)
        for i in range(keypoints.shape[0]):
            cv2.circle(skeleton_image1, center=(int(keypoints[i, 0]), int(keypoints[i, 1])), radius=2, color=color_, thickness=2)
        cv2.circle(skeleton_image1,
                   center=(int(norm_pose[0, 0] * norm_pose_mask.shape[1]), int(norm_pose[0, 1] * norm_pose_mask.shape[0])),
                   radius=2, color=(255, 0, 0), thickness=2)

        st.image(skeleton_image1)


if __name__ == "__main__":

    st.set_page_config(page_title="Segmentation Mask Visualization")

    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    st.header("Keypoint Visualization")


    def load_dataset():
        if st.session_state.head_strategy == "Angle" or st.session_state.representation_type == "Normalized Pose":
            bodypart_order = JumpHeadAngleBodypartOrder
        else:
            bodypart_order = JumpHeadEndpointBodypartOrder
        if st.session_state.representation_type == "Normalized Pose":
            representation_type = ArbitraryKeypointsDataWrapper.NORM_POSE
        else:
            representation_type = ArbitraryKeypointsDataWrapper.KEYPOINT_VECTOR
        if st.session_state.visualization == "Validation":
            arbitrary_keypoint_mode = ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS
        else:
            arbitrary_keypoint_mode = ArbitraryKeypointsDataWrapper.GENERATE_POINTS
        params = {}
        params["is_youtube"] = True
        jump = JumpArbitraryKeypoints(subset="val",
                                      params=params,
                                      bodypart_order=bodypart_order,
                                      representation_type=representation_type,
                                      arbitrary_keypoint_mode=arbitrary_keypoint_mode,
                                      test_points=(2, 25),
                                      num_points_to_generate=50)
        st.session_state.jump = jump
        st.session_state.jump.image_ids = sorted(st.session_state.jump.image_ids)


    st.sidebar.radio("Representation", ["Normalized Pose", "Keypoint Vector"], on_change=load_dataset, key="representation_type")
    st.sidebar.radio("Head Strategy", ["Angle", "Extension"], on_change=load_dataset, key="head_strategy",
                     disabled=st.session_state.representation_type == "Normalized Pose")
    st.sidebar.radio('Visualize GT Creation', ["None", "Single Point", "Lines", "Validation"], on_change=load_dataset,
                     key="visualization")

    if "jump" not in st.session_state:
        load_dataset()
    image_ids = st.session_state.jump.image_ids

    df = DataFrame({'Image IDs': image_ids})

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

    bodypart_order = st.session_state.jump.bodypart_order
    bodypart_names = np.asarray(bodypart_order.names())[sorted(list(bodypart_order.get_bodypart_to_keypoint_dict().keys()))]
    bp_kp_dict = bodypart_order.get_bodypart_to_keypoint_dict()

    bodypart_name = st.sidebar.radio("Bodypart", bodypart_names)

    bodypart_index = bodypart_order.names().index(bodypart_name)

    if bodypart_index not in bodypart_order.get_adjacent_bodyparts():
        bodypart_keypoints = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_index]
    else:
        bodypart1_index, bodypart2_index = bp_kp_dict[bodypart_index]
        bodypart1_keypoints = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart1_index]
        bodypart2_keypoints = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart2_index]
        bodypart_keypoints = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_index]

    circle_size = st.sidebar.slider('Circle size:', 1, 5, 2)
    thickness = st.sidebar.slider('Circle thickness:', 1, 10, 2)

    if len(selected) > 0:

        image_id = selected[0]['Image IDs']
        bbox_ = np.asarray(st.session_state.jump.bboxes[image_id], dtype=int)
        x1, y1, w, h = bbox_
        keypoints = st.session_state.jump.annotations[image_id]
        skeleton_image, segmentation_masks, used_bodypart_exists, line_on_mask_exists = create_vis_image(image_id, bbox_,
                                                                                                         keypoints)

        x1_, y1_, x2_, y2_ = bbox_[0], bbox_[1], bbox_[0] + bbox_[2], bbox_[1] + bbox_[3]
        bbox = x1_, y1_, x2_, y2_

        percentage_projection, percentage_thickness = None, None
        visualization = st.session_state.visualization
        if visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists and bodypart_index <= 14 and \
                bodypart_index not in bodypart_order.get_endpoint_bodyparts() and bodypart_index not in \
                bodypart_order.get_endpoint_angle_bodyparts():

            percentage_projection = st.slider('Projection point:', 0.0, 1.0, 0.5)
            percentage_thickness = st.slider('Thickness:', -1.0, 1.0, 0.8)
            vis_normal_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness)

        elif visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists and bodypart_index is not None \
                and bodypart_index in bodypart_order.get_endpoint_bodyparts():
            percentage_projection = st.slider('Projection point:', 0.0, 1.0, 0.5)
            percentage_thickness = st.slider('Thickness:', -1.0, 1.0, 0.8)
            vis_endpoint_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness, keypoints)

        elif visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists and bodypart_index is not None \
                and bodypart_index in bodypart_order.get_endpoint_angle_bodyparts():
            percentage_projection = st.slider('Projection point:', 0.0, 1.0, 0.5)
            percentage_thickness = st.slider('Thickness:', -1.0, 1.0, 0.8)
            vis_endpoint_angle_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness)

        elif visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists and bodypart_index > 14:
            percentage_projection = st.slider('Projection point:', 0.0, 1.0, 0.5)
            percentage_thickness = st.slider('Thickness:', -1.0, 1.0, 0.8)
            vis_elbow_knee_point(skeleton_image, segmentation_masks, x1, y1, percentage_projection, percentage_thickness)

        elif visualization == "Lines" and used_bodypart_exists and line_on_mask_exists:
            vis_lines(skeleton_image, keypoints, image_id, bodypart_keypoints, bodypart_index, bbox, segmentation_masks, x1, y1,
                      circle_size, thickness)
            if bodypart_index > 14:
                vis_lines(skeleton_image, keypoints, image_id, bodypart1_keypoints, bodypart1_index, bbox, segmentation_masks, x1,
                          y1, circle_size, thickness)
                vis_lines(skeleton_image, keypoints, image_id, bodypart2_keypoints, bodypart2_index, bbox, segmentation_masks, x1,
                          y1, circle_size, thickness)
        elif visualization == "Validation":

            points = st.session_state.jump.test_points[image_id][0][0]
            for i in range(points.shape[0]):
                if points[i, 2] > 0:
                    cv2.circle(skeleton_image, center=(int(points[i, 0]) - x1, int(points[i, 1]) - y1), radius=circle_size,
                               color=(255, 255, 255), thickness=thickness)

        vis_creation = False
        if visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists:
            vis_creation = st.checkbox("Show creation details")

        st.image(skeleton_image)

        if vis_creation and visualization == "Single Point" and used_bodypart_exists and line_on_mask_exists:
            vis_creation_details(image_id, bbox, segmentation_masks, percentage_projection, percentage_thickness, keypoints)


    else:
        st.write("Select an image in the table on the left.")
