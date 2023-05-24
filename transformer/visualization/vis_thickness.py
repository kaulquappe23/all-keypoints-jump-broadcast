# -*- coding: utf-8 -*-
"""
Created on 23.02.22

"""
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets.general.arbitrary_keypoints_data_wrapper import get_norm_pose_point
from datasets.general.bodypart_order import BodypartOrder
from datasets.general.joint_order import JointOrder
from transformer.execution.inference import inference_single_image, prepare_single_image
from utils.visualization import cutout_image


def visualize_thickness_grid(
        image_path, bbox, model, bodypart_order: BodypartOrder, bodyparts_to_show, joint_order: JointOrder, scale=1,
        norm_pose_settings=None, convert_bgr=True, blur=None, cutout=(3, 4), use_flip=False
        ):
    """
    Visualize thickness results as a grid overlay on the image
    :param use_flip:
    :param image_path: path to the image
    :param bbox: bbox to crop the person
    :param model: model for inference_data_loader
    :param bodypart_order: BodyPartOrder object matching the dataset
    :param bodyparts_to_show: indices of bodyparts to include in the visualization. Important: only ids of bodyparts, not triples
    :param joint_order: JointOrder object matching the dataset
    :param scale: Scale the image by this factor
    :param norm_pose_settings: settings for norm_pose usage (dict containing keypoints and bodypart map of used norm pose)
    :param convert_bgr: convert image to BGR format, otherwise will be RGB format
    :param blur: if regions of the image should be blurred, this function will be called with the image and the prediction,
    should return a blurred image then
    :param cutout: cutout a region of the full image, this is the width to height ratio that the final image has (default (3, 4))
    :return: visualized image
    """
    image_model, center, inference_scale = prepare_single_image(image_path, bbox=bbox)
    num_keypoints = joint_order.get_num_joints()
    flip_pairs = joint_order.flip_pairs()

    angle_keypoints = bodypart_order.get_endpoint_angle_bodyparts()
    used_bodyparts = [bodypart_order.get_used_bodypart_triples()[i] for i in
                      range(len(bodypart_order.get_used_bodypart_triples())) if
                      bodypart_order.get_used_bodypart_triples()[i][-1] in bodyparts_to_show]
    keypoint_vector = np.zeros((num_keypoints + len(used_bodyparts) * 15, num_keypoints), dtype=np.float32)
    thickness_vector = np.zeros((num_keypoints + len(used_bodyparts) * 15, 3), dtype=np.float32)
    norm_pose_vector = np.zeros((num_keypoints + len(used_bodyparts) * 15 + 1, 2), dtype=np.float32)
    angle_vector = np.zeros((num_keypoints + len(used_bodyparts) * 15, 1), dtype=np.float32)

    if norm_pose_settings is not None:
        norm_pose_keypoints = norm_pose_settings["keypoints"]
        norm_pose_bodyparts = norm_pose_settings["bodypart_map"]
        norm_pose_keypoints = norm_pose_keypoints.astype(np.float32)[:, :2]

        norm_pose_keypoints[:, 0] /= norm_pose_bodyparts.shape[0]
        norm_pose_keypoints[:, 1] /= norm_pose_bodyparts.shape[1]

    for i in range(num_keypoints):
        if norm_pose_settings is None:
            keypoint_vector[i, i] = 1
            thickness_vector[i, 1] = 1
        else:
            norm_pose_vector[i] = norm_pose_keypoints[i, :2]

    vec_ind = num_keypoints
    for bodypart in used_bodyparts:
        for j in range(1, 4):
            if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[0]] = 0.25 * j
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (0.25 * j)
                thickness_vector[vec_ind, 0] = 1
                vec_ind += 1
                keypoint_vector[vec_ind, bodypart[0]] = 0.25 * j
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (0.25 * j)
                thickness_vector[vec_ind, 2] = 1
                vec_ind += 1
            elif norm_pose_settings is None and bodypart[-1] in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[1]] = 1
                angle_vector[vec_ind, 0] = 0.125 * j
                thickness_vector[vec_ind, 2] = 1
                vec_ind += 1
                keypoint_vector[vec_ind, bodypart[1]] = 1
                angle_vector[vec_ind, 0] = 0.5 + 0.125 * j
                thickness_vector[vec_ind, 2] = 1
                vec_ind += 1
            elif bodypart[-1] not in angle_keypoints:
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, -1, 1 - 0.25 * j, 0, bodypart)
                vec_ind += 1
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 1, 1 - 0.25 * j, 0, bodypart)
                vec_ind += 1
            else:
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 1, 0, 0.125 * j, bodypart)
                vec_ind += 1
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 1, 0, 0.5 + 0.125 * j,
                                                                bodypart)
                vec_ind += 1
    for bodypart in used_bodyparts:
        for j in range(1, 4):
            if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[0]] = 0.25 * j
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (0.25 * j)
                thickness_vector[vec_ind, 0] = 0.5
                thickness_vector[vec_ind, 1] = 0.5
                vec_ind += 1
                keypoint_vector[vec_ind, bodypart[0]] = 0.25 * j
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (0.25 * j)
                thickness_vector[vec_ind, 2] = 0.5
                thickness_vector[vec_ind, 1] = 0.5
                vec_ind += 1
            elif norm_pose_settings is None and bodypart[-1] in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[1]] = 1
                angle_vector[vec_ind, 0] = 0.125 * j
                thickness_vector[vec_ind, 2] = 0.5
                thickness_vector[vec_ind, 1] = 0.5
                vec_ind += 1
                keypoint_vector[vec_ind, bodypart[1]] = 1
                angle_vector[vec_ind, 0] = 0.5 + 0.125 * j
                thickness_vector[vec_ind, 2] = 0.5
                thickness_vector[vec_ind, 1] = 0.5
                vec_ind += 1
            elif bodypart[-1] not in angle_keypoints:
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, -0.5, 1 - 0.25 * j, 0,
                                                                bodypart)
                vec_ind += 1
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0.5, 1 - 0.25 * j, 0,
                                                                bodypart)
                vec_ind += 1
            else:
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0.5, 0, 0.125 * j, bodypart)
                vec_ind += 1
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0.5, 0, 0.5 + 0.125 * j,
                                                                bodypart)
                vec_ind += 1
    for bodypart in used_bodyparts:
        for j in range(1, 4):
            if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[0]] = 0.25 * j
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (0.25 * j)
                thickness_vector[vec_ind, 1] = 1
                vec_ind += 1
            elif norm_pose_settings is not None and bodypart[-1] not in angle_keypoints:
                norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0, 1 - 0.25 * j, 0, bodypart)
                vec_ind += 1
        if norm_pose_settings is None and bodypart[-1] in angle_keypoints:
            keypoint_vector[vec_ind, bodypart[1]] = 1
            angle_vector[vec_ind, 0] = 0.5
            thickness_vector[vec_ind, 2] = 1
            vec_ind += 1
            keypoint_vector[vec_ind, bodypart[1]] = 1
            angle_vector[vec_ind, 0] = 1
            thickness_vector[vec_ind, 2] = 1
            vec_ind += 1
        elif norm_pose_settings is not None and bodypart[-1] in angle_keypoints:
            norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 1, 0, 0.5, bodypart)
            vec_ind += 1
            norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 1, 0, 1, bodypart)
            vec_ind += 1
            norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0.5, 0, 0.5, bodypart)
            vec_ind += 1
            norm_pose_vector[vec_ind] = get_norm_pose_point(norm_pose_settings, bodypart_order, 0.5, 0, 1, bodypart)
            vec_ind += 1

    if not norm_pose_settings:
        keypoint_vector = torch.from_numpy(keypoint_vector[None, :])
        thickness_vector = torch.from_numpy(thickness_vector[None, :])
        angle_vector = torch.from_numpy(angle_vector[None, :])
        if torch.cuda.is_available():
            keypoint_vector = keypoint_vector.cuda()
            thickness_vector = thickness_vector.cuda()
            angle_vector = angle_vector.cuda()
    else:
        norm_pose_vector = torch.from_numpy(norm_pose_vector[None, :])
        if torch.cuda.is_available():
            norm_pose_vector = norm_pose_vector.cuda()
        keypoint_vector = norm_pose_vector
        thickness_vector = None
    angle_vector = angle_vector if len(angle_keypoints) > 0 else None

    additional_vectors = {}
    if norm_pose_settings is not None:
        additional_vectors['norm_pose_vector'] = norm_pose_vector
    else:
        additional_vectors['keypoint_vector'] = keypoint_vector
        additional_vectors['thickness_vector'] = thickness_vector
        if angle_vector is not None:
            additional_vectors['angle_vector'] = angle_vector

    heatmap, prediction = inference_single_image(image_model, center[None, :], inference_scale[None, :], model,
                                                 additional_vectors, use_flip=use_flip, flip_pairs=flip_pairs,
                                                 with_confidence=True)

    image = cv2.imread(image_path)
    if blur is not None:
        image = blur(image, prediction[0])
    if convert_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale != 1:
        image = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))), cv2.INTER_CUBIC)
    image = image * 0.5
    image = np.asarray(image, dtype=np.uint8)

    lightblue = swap_colors(np.asarray(matplotlib.colors.to_rgb("dodgerblue")) * 255, swap=not convert_bgr)
    purple = swap_colors(np.asarray(matplotlib.colors.to_rgb("mediumorchid")) * 255, swap=not convert_bgr)
    lightgreen = swap_colors(np.asarray(matplotlib.colors.to_rgb("limegreen")) * 255, swap=not convert_bgr)
    yellow = swap_colors(np.asarray(matplotlib.colors.to_rgb("yellow")) * 255, swap=not convert_bgr)
    orange = swap_colors(np.asarray(matplotlib.colors.to_rgb("orange")) * 255, swap=not convert_bgr)

    threshold = 0.5
    for i in range(prediction.shape[1]):
        if prediction[0, i, 2] < threshold:
            continue
        if i < num_keypoints:
            color = swap_colors((255, 0, 0), swap=not convert_bgr)
        elif i < num_keypoints + len(used_bodyparts) * 6 and i % 2 == 0:
            color = lightgreen
        elif i < num_keypoints + len(used_bodyparts) * 6:
            color = yellow
        elif i < num_keypoints + len(used_bodyparts) * 12 and i % 2 == 0:
            color = lightblue
        elif i < num_keypoints + len(used_bodyparts) * 12:
            color = orange
        else:
            color = purple
        cv2.circle(image, (int(round(prediction[0, i, 0] * scale)), int(round(prediction[0, i, 1] * scale))), 1, color, 2)

    if cutout:
        rel_annos = prediction[prediction[:, :, 2] > threshold]
        if rel_annos.shape[0] > 0:
            image = cutout_image(rel_annos, image, cutout, scale)

    return image


def swap_colors(color, swap=True):
    if swap:
        return color[2], color[1], color[0]
    else:
        return color


def visualize_thickness_outline(
        image_path, bbox, model, bodypart_order: BodypartOrder, bodyparts_to_show, joint_order: JointOrder, scale=1,
        norm_pose_settings=None, convert_bgr=True, num_outline_points=50, num_thickness_lines=4, blur=None,
        cutout=(3, 4), model_points=None, circle_size=1, colors=None, use_flip=False
        ):
    """
    Visualize thickness results as a multiple equally spaced points on equally spaced lines as an overlay on the image
    :param colors: specify the colors to use
    :param circle_size: circle size for the points
    :param model_points: specify maximum number of points to use in each model call, default None means all at once
    :param use_flip: use flip test
    :param image_path: path to the image
    :param bbox: bbox to crop the person
    :param model: model for inference_data_loader
    :param bodypart_order: BodyPartOrder object matching the dataset
    :param bodyparts_to_show: indices of bodyparts to include in the visualization. Important: only ids of bodyparts, not triples
    :param joint_order: JointOrder object matching the dataset
    :param scale: Scale the image by this factor
    :param norm_pose_settings: settings for norm_pose usage (dict containing keypoints and bodypart map of used norm pose)
    :param convert_bgr: convert image to BGR format, otherwise will be RGB format
    :param blur: if regions of the image should be blurred, this function will be called with the image and the image path,
    should return a blurred image then
    :param cutout: cutout a region of the full image, this is the width to height ratio that the final image has (default (3, 4))
    :return: visualized image
    """
    image_model, center, inference_scale = prepare_single_image(image_path, bbox=bbox)

    num_keypoints = joint_order.get_num_joints()
    flip_pairs = joint_order.flip_pairs()

    used_bodyparts = [bodypart_order.get_used_bodypart_triples()[i] for i in
                      range(len(bodypart_order.get_used_bodypart_triples())) if
                      bodypart_order.get_used_bodypart_triples()[i][-1] in bodyparts_to_show]
    bodyparts_with_projection_limits = bodypart_order.get_bodyparts_with_min_max()
    angle_keypoints = bodypart_order.get_endpoint_angle_bodyparts()

    num_points = len(used_bodyparts) * num_outline_points * (num_thickness_lines * 2 + 1)
    keypoint_vector = np.zeros((num_points, num_keypoints), dtype=np.float32)
    thickness_vector = np.zeros((num_points, 3), dtype=np.float32)
    norm_pose_vector = np.zeros((num_points, 2), dtype=np.float32)
    angle_vector = np.zeros((num_points, 1), dtype=np.float32)

    if norm_pose_settings is not None:
        norm_pose_keypoints = norm_pose_settings["keypoints"]
        norm_pose_bodyparts = norm_pose_settings["bodypart_map"]
        norm_pose_keypoints = norm_pose_keypoints.astype(np.float32)[:, :2]

        norm_pose_keypoints[:, 0] /= norm_pose_bodyparts.shape[0]
        norm_pose_keypoints[:, 1] /= norm_pose_bodyparts.shape[1]
    vec_ind = 0
    for bodypart in used_bodyparts:
        # l is the line index
        if bodyparts_with_projection_limits is not None and bodypart[-1] in bodyparts_with_projection_limits:
            mini, maxi = bodyparts_with_projection_limits[bodypart[-1]]
        else:
            mini, maxi = 0, 1
        for l in range(num_thickness_lines):
            # j iterates over the points in each line
            # at first, we create the left line
            for j in range(num_outline_points):
                if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                    keypoint_vector[vec_ind, bodypart[0]] = mini + 1. / (num_outline_points - 1) * j * (maxi - mini)
                    keypoint_vector[vec_ind, bodypart[1]] = 1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini))
                    if bodypart[-1] not in bodypart_order.get_bps_without_central_projection():
                        thickness_vector[vec_ind, 0] = 1 - l / num_thickness_lines
                        thickness_vector[vec_ind, 1] = 1 - thickness_vector[vec_ind, 0]
                    else:
                        thickness_vector[vec_ind, 0] = 1 - (l / (num_thickness_lines * 2))
                        thickness_vector[vec_ind, 2] = 1 - thickness_vector[vec_ind, 0]
                    vec_ind += 1
                elif norm_pose_settings is None and bodypart[-1] in angle_keypoints:
                    keypoint_vector[vec_ind, bodypart[1]] = 1
                    angle_vector[vec_ind, 0] = 0.5 / (num_outline_points) * (j + 1)
                    thickness_vector[vec_ind, 2] = 1 - l / (num_thickness_lines + 1)
                    thickness_vector[vec_ind, 1] = 1 - thickness_vector[vec_ind, 2]
                    vec_ind += 1
                elif norm_pose_settings is not None:
                    if bodypart[-1] not in angle_keypoints:
                        p = get_norm_pose_point(norm_pose_settings, bodypart_order, 1 - l / num_thickness_lines,
                                                1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini)), 0, bodypart)
                    else:
                        p = get_norm_pose_point(norm_pose_settings, bodypart_order, 1 - l / num_thickness_lines, 0,
                                                0.5 / (num_outline_points) * (j + 1), bodypart)
                    if p is not None:
                        norm_pose_vector[vec_ind] = p
                    vec_ind += 1

            # next, the right line
            for j in range(num_outline_points):
                if bodyparts_with_projection_limits is not None and bodypart[-1] in bodyparts_with_projection_limits:
                    mini, maxi = bodyparts_with_projection_limits[bodypart[-1]]
                else:
                    mini, maxi = 0, 1
                if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                    keypoint_vector[vec_ind, bodypart[0]] = mini + 1. / (num_outline_points - 1) * j * (maxi - mini)
                    keypoint_vector[vec_ind, bodypart[1]] = 1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini))
                    if bodypart[-1] not in bodypart_order.get_bps_without_central_projection():
                        thickness_vector[vec_ind, 2] = 1 - l / num_thickness_lines
                        thickness_vector[vec_ind, 1] = 1 - thickness_vector[vec_ind, 2]
                    else:
                        thickness_vector[vec_ind, 2] = 1 - (l / (num_thickness_lines * 2))
                        thickness_vector[vec_ind, 0] = 1 - thickness_vector[vec_ind, 2]
                    vec_ind += 1
                elif norm_pose_settings is None and bodypart[-1] in angle_keypoints:
                    keypoint_vector[vec_ind, bodypart[1]] = 1
                    angle_vector[vec_ind, 0] = 0.5 + 0.5 / (num_outline_points) * (j + 1)
                    thickness_vector[vec_ind, 2] = 1 - l / (num_thickness_lines + 1)
                    thickness_vector[vec_ind, 1] = 1 - thickness_vector[vec_ind, 2]
                    vec_ind += 1
                else:
                    if bodypart[-1] not in angle_keypoints:
                        p = get_norm_pose_point(norm_pose_settings, bodypart_order, - (1 - l / num_thickness_lines),
                                                1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini)), 0, bodypart)
                    else:
                        p = get_norm_pose_point(norm_pose_settings, bodypart_order, - (1 - l / num_thickness_lines), 0,
                                                0.5 + 0.5 / (num_outline_points) * (j + 1), bodypart)
                    if p is not None:
                        norm_pose_vector[vec_ind] = p
                    vec_ind += 1
        # this code produces the central line
        for j in range(num_outline_points):
            if norm_pose_settings is None and bodypart[-1] not in angle_keypoints:
                if bodyparts_with_projection_limits is not None and bodypart[-1] in bodyparts_with_projection_limits:
                    mini, maxi = bodyparts_with_projection_limits[bodypart[-1]]
                else:
                    mini, maxi = 0, 1
                keypoint_vector[vec_ind, bodypart[0]] = mini + 1. / (num_outline_points - 1) * j * (maxi - mini)
                keypoint_vector[vec_ind, bodypart[1]] = 1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini))
                if bodypart[-1] not in bodypart_order.get_bps_without_central_projection():  # just set the middle to 1
                    thickness_vector[vec_ind, 1] = 1
                else:  # left and right to 0.5 if we do not use the middle
                    thickness_vector[vec_ind, 0] = 0.5
                    thickness_vector[vec_ind, 2] = 0.5
                vec_ind += 1
            elif norm_pose_settings is None and bodypart[-1] in angle_keypoints:
                keypoint_vector[vec_ind, bodypart[1]] = 1
                angle_vector[vec_ind, 0] = 1 / (num_outline_points) * (j + 1)
                thickness_vector[vec_ind, 2] = 1 / (num_thickness_lines + 1)
                thickness_vector[vec_ind, 1] = 1 - thickness_vector[vec_ind, 1]
                vec_ind += 1
            else:
                if bodypart[-1] not in angle_keypoints:
                    p = get_norm_pose_point(norm_pose_settings, bodypart_order, 0,
                                            1 - (mini + (1. / (num_outline_points - 1) * j) * (maxi - mini)), 0, bodypart)
                else:
                    p = get_norm_pose_point(norm_pose_settings, bodypart_order, 0, 0, 0.5, bodypart)
                if p is not None:
                    norm_pose_vector[vec_ind] = p
                vec_ind += 1

    if norm_pose_settings is None:
        keypoint_vector = torch.from_numpy(keypoint_vector[None, :])
        thickness_vector = torch.from_numpy(thickness_vector[None, :])
        angle_vector = torch.from_numpy(angle_vector[None, :])
        if torch.cuda.is_available():
            keypoint_vector = keypoint_vector.cuda()
            thickness_vector = thickness_vector.cuda()
            angle_vector = angle_vector.cuda()
    else:
        norm_pose_vector = torch.from_numpy(norm_pose_vector[None, :])
        if torch.cuda.is_available():
            norm_pose_vector = norm_pose_vector.cuda()
        thickness_vector = None
    prediction = []

    if model_points is not None:
        for i in tqdm(range(int(np.ceil(num_points / model_points)))):
            angle_vec = angle_vector[:, i * model_points: min((i + 1) * model_points, num_points)] if len(
                angle_keypoints) > 0 else None
            additional_vectors = {}
            if norm_pose_vector is not None:
                additional_vectors['norm_pose_vector'] = norm_pose_vector[:,
                                                         i * model_points: min((i + 1) * model_points, num_points)]
            else:
                additional_vectors['keypoint_vector'] = keypoint_vector[:,
                                                        i * model_points: min((i + 1) * model_points, num_points)]
                additional_vectors['thickness_vector'] = thickness_vector[:,
                                                         i * model_points: min((i + 1) * model_points, num_points)]
                if angle_vec is not None:
                    additional_vectors['angle_vector'] = angle_vec
            _, part_prediction = inference_single_image(image_model, center[None, :], inference_scale[None, :], model,
                                                        additional_vectors, use_flip=use_flip, flip_pairs=flip_pairs,
                                                        with_confidence=True)
            prediction.append(part_prediction)
        prediction = np.concatenate(prediction[:], axis=1)
    else:
        # all at once
        angle_vec = angle_vector if len(angle_keypoints) > 0 else None
        additional_vectors = {}
        if norm_pose_settings is not None:
            additional_vectors['norm_pose_vector'] = norm_pose_vector
        else:
            additional_vectors['keypoint_vector'] = keypoint_vector
            additional_vectors['thickness_vector'] = thickness_vector
            if angle_vec is not None:
                additional_vectors['angle_vector'] = angle_vec
        _, prediction = inference_single_image(image_model, center[None, :], inference_scale[None, :], model,
                                               additional_vectors, use_flip=use_flip, flip_pairs=flip_pairs, with_confidence=True)

    image = cv2.imread(image_path)
    if blur is not None:
        image = blur(image, image_path)
    if convert_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale != 1:
        image = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))), cv2.INTER_CUBIC)

    if colors is None:
        blue = swap_colors(np.asarray(matplotlib.colors.to_rgb("dodgerblue")) * 255, swap=not convert_bgr)
        darkgreen = swap_colors(np.asarray(matplotlib.colors.to_rgb("green")) * 255, swap=not convert_bgr)
        violet = swap_colors(np.asarray(matplotlib.colors.to_rgb("darkviolet")) * 255, swap=not convert_bgr)
        gold = swap_colors(np.asarray(matplotlib.colors.to_rgb("goldenrod")) * 255, swap=not convert_bgr)
        red = swap_colors((255, 0, 0), swap=not convert_bgr)
        colors = [gold, darkgreen, violet, red, blue] * 5
    else:
        colors = [swap_colors(np.asarray(matplotlib.colors.to_rgb(c)) * 255, swap=not convert_bgr) for c in colors]

    threshold = 0
    pil_image = Image.fromarray(image)
    draw_image = ImageDraw.Draw(pil_image)
    for i in range(prediction.shape[1]):
        if prediction[0, i, 2] < threshold:
            continue

        color_idx = int(i // (num_outline_points * (num_thickness_lines * 2 + 1)))
        try:
            color = colors[color_idx]
        except IndexError:
            print()

        adjust_ratio = i % (num_outline_points * (num_thickness_lines * 2 + 1))
        adjust_ratio = adjust_ratio // num_outline_points // 2
        divisor = 1 if num_thickness_lines == 0 else num_thickness_lines
        color = np.asarray(adjust_color(color, adjust_ratio / divisor), dtype=np.int32)
        color = (color[0], color[1], color[2])

        x, y = int(round(prediction[0, i, 0] * scale)), int(round(prediction[0, i, 1] * scale))
        draw_image.ellipse([x - circle_size, y - circle_size, x + circle_size + 1, y + circle_size + 1], fill=color)

    image = np.array(pil_image)
    if cutout:
        if not norm_pose_settings:
            keypoint_vector = np.eye(num_keypoints, dtype=np.float32)
            thickness_vector = np.zeros((num_keypoints, 3), dtype=np.float32)
            angle_vector = np.zeros((num_keypoints, 1), dtype=np.float32)
            keypoint_vector = torch.from_numpy(keypoint_vector[None, :])
            thickness_vector = torch.from_numpy(thickness_vector[None, :])
            angle_vector = torch.from_numpy(angle_vector[None, :])
            if torch.cuda.is_available():
                keypoint_vector = keypoint_vector.cuda()
                thickness_vector = thickness_vector.cuda()
                angle_vector = angle_vector.cuda() if len(angle_keypoints) > 0 else None

            additional_vectors = {'keypoint_vector': keypoint_vector, 'thickness_vector': thickness_vector}
            if angle_vector is not None:
                additional_vectors['angle_vector'] = angle_vector
        else:
            norm_pose_vector = torch.from_numpy(norm_pose_keypoints[None, :])
            if torch.cuda.is_available():
                norm_pose_vector = norm_pose_vector.cuda()
            additional_vectors = {'norm_pose_vector': norm_pose_vector}
        _, orig_prediction = inference_single_image(image_model, center[None, :], inference_scale[None, :], model,
                                                    additional_vectors, use_flip=use_flip, flip_pairs=flip_pairs,
                                                    with_confidence=True)
        rel_annos = orig_prediction[orig_prediction[:, :, 2] > threshold]
        rel_preds = prediction[prediction[:, :, 2] > threshold]
        rel_annos = np.concatenate([rel_annos, rel_preds], axis=0)

        if rel_annos.shape[0] > 0:
            image = cutout_image(rel_annos, image, cutout, scale)

    return image


def adjust_color(color, ratio):
    if ratio == 0:
        return color
    else:
        color = np.asarray(color)
        white = np.asarray([255, 255, 255])
        diff = white - color
        adjust = diff * ratio
        return color + adjust
