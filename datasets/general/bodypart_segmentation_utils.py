# -*- coding: utf-8 -*-
"""
Created on 22.12.22

"""

import numpy as np
from scipy.spatial import distance_matrix

from datasets.general.keypoint_generator import get_intermediate_point, angle_between_points_oriented, get_correct_border_point, angle_between_points, get_intersection_point_of_vector_defined_lines


def get_endpoint(keypoints, bodypart, mask, bbox):
    """
    Returns the endpoint of a body part, meaning the point where the line through the body part's keypoints intersects with its segmentation mask
    The segmentation mask is only defined for the specified bounding box, while the keypoints are defined for the whole image
    """

    x1, y1, x2, y2 = bbox
    mask_ = np.zeros_like(mask)
    ys, xs = np.where(mask == bodypart[2])
    if len(xs) == 0 or len(ys) == 0:
        return None
    mask_[ys, xs] = 1

    x1_ = max(0, np.min(xs) - 2)
    x2_ = min(np.max(xs) + 2, mask.shape[1])
    y1_ = max(0, np.min(ys) - 2)
    y2_ = min(np.max(ys) + 2, mask.shape[0])
    mask = mask_[y1_:y2_, x1_:x2_]
    x2 = x1 + x2_
    y2 = y1 + y2_
    x1 = x1 + x1_
    y1 = y1 + y1_
    bbox = x1, y1, x2, y2

    if isinstance(bodypart[0], int):
        keypoint1 = keypoints[bodypart[0]]
    else:
        keypoint1 = get_intermediate_point(keypoints[bodypart[0][0]], keypoints[bodypart[0][1]], 0.5)
    if isinstance(bodypart[1], int):
        keypoint2 = keypoints[bodypart[1]]
    else:
        keypoint2 = get_intermediate_point(keypoints[bodypart[1][0]], keypoints[bodypart[1][1]], 0.5)

    # line
    if (keypoint1[0] - keypoint2[0]) != 0:
        m =  (keypoint1[1] - keypoint2[1]) / (keypoint1[0] - keypoint2[0])
    else:
        m = float('inf')
    t = keypoint2[1] - m * keypoint2[0] if m != float('inf') else 0

    x_inputs, y_outputs = generate_segmentation_line(m, t, bbox, keypoint2)

    if len(x_inputs) == 0:
        return None

    x_mask = x_inputs - x1
    y_mask = y_outputs - y1

    mask_out = np.where(mask[y_mask, x_mask])

    if len(mask_out[0]) == 0:
        return None
    p1 = [x_mask[mask_out[0][0]] + x1, y_mask[mask_out[0][0]] + y1, 1]
    p2 = [x_mask[mask_out[0][-1]] + x1, y_mask[mask_out[0][-1]] + y1, 1]

    dist_p1 = np.linalg.norm(keypoint1[:2] - p1[:2])
    dist_p2 = np.linalg.norm(keypoint1[:2] - p2[:2])

    if dist_p1 < dist_p2:
        return p2, p1
    else:
        return p1, p2



def generate_segmentation_line(m, t, bbox, point):
    """
    Generates a line with the specified slope and offset in the specified bounding box.
    The x-coodinate point[0] is used for lines with infinite slope
    :param m: slope
    :param t: offset
    :param bbox:
    :param point: alternative point if the slope is infinite
    :return:
    """

    x1, y1, x2, y2 = bbox

    if np.abs(m) != float('inf') and m != 0:

        if np.abs(m) > 1:

            y_from_x1 = m * x1 + t
            y_from_x2 = m * (x2 - 1) + t
            y_low = y_from_x1 if y_from_x1 < y_from_x2 else y_from_x2
            y_high = y_from_x2 if y_low == y_from_x1 else y_from_x1

            y_min = max(y_low, y1)
            y_max = min(y_high, y2)
            y_min = np.ceil(y_min)
            y_max = np.floor(y_max)

            y_outputs = np.arange(y_min, y_max - 0.5)
            x_inputs = (y_outputs - t) / m
            if len(x_inputs) > 0 and x_inputs[0] > x_inputs[-1]:
                x_inputs = x_inputs[::-1]
                y_outputs = y_outputs[::-1]

        else:
            x_from_y1 = (y1 - t) / m
            x_from_y2 = ((y2 - 1) - t) / m
            x_low = x_from_y1 if x_from_y1 < x_from_y2 else x_from_y2
            x_high = x_from_y2 if x_low == x_from_y1 else x_from_y1

            x_min = max(x_low, x1)
            x_max = min(x_high, x2)
            x_min = np.ceil(x_min)
            x_max = np.floor(x_max)

            x_inputs = np.arange(x_min, x_max - 0.5)
            y_outputs = m * x_inputs + t
        y_outputs = np.asarray(np.round(y_outputs), dtype=int)
        x_inputs = np.asarray(np.round(x_inputs), dtype=int)

        # sometimes we have odd behaviour at the borders, just to be sure
        if x_inputs.shape[0] > 0 and x_inputs[-1] >= x2:
            x_inputs = x_inputs[:-1]
            y_outputs = y_outputs[:-1]
        if x_inputs.shape[0] > 0 and x_inputs[0] < x1:
            x_inputs = x_inputs[1:]
            y_outputs = y_outputs[1:]
        if x_inputs.shape[0] > 0 and (y_outputs[-1] >= y2 or y_outputs[-1] < y1):
            x_inputs = x_inputs[:-1]
            y_outputs = y_outputs[:-1]
        if x_inputs.shape[0] > 0 and (y_outputs[0] >= y2 or y_outputs[0] < y1):
            x_inputs = x_inputs[1:]
            y_outputs = y_outputs[1:]

    elif m == 0: # m is 0, meaning we have a line parallel to the x-axis
        if y1 <= t < y2:
            x_inputs = np.arange(x1, x2)
            y_outputs = np.full(x2 - x1, t, dtype=int)
        else:
            x_inputs = np.zeros(0)
            y_outputs = np.zeros(0)
    else:  # m is infinity, meaning we have a line parallel to the y-axis
        if x1 <= point[0] < x2:
            x_inputs = np.full(y2 - y1, point[0], dtype=int)
            y_outputs = np.arange(y1, y2, dtype=int)
        else:
            x_inputs = np.zeros(0)
            y_outputs = np.zeros(0)

    return x_inputs, y_outputs


def get_mask_line_intersection_points(x_line, y_line, mask, offset_x, offset_y, point=None, direction=None):
    """
    Get the intersection points of a line with a mask. The longest sequence of points on the line that lie within the mask is considered and the first
    and last point of this sequence are returned.
    If the points are desired in a certain direction of a point, the point and direction can be specified, the direction is interpreted as a vector
    """
    if len(x_line) == 0:
        return None, None
    x_mask = x_line - offset_x
    y_mask = y_line - offset_y
    mask_out = np.where(mask[y_mask, x_mask])
    if len(mask_out[0]) == 0:
        return None, None

    # Find the largest sequence with consecutive indices in mask_out
    mask_out = mask_out[0]
    mask_shifted = mask_out[1:]
    differences = mask_shifted - mask_out[:-1]
    indices_of_gaps = np.where(differences != 1)[0]

    if len(indices_of_gaps) == 0:
        p1 = np.asarray([x_mask[mask_out[0]] + offset_x, y_mask[mask_out[0]] + offset_y, 1])
        p2 = np.asarray([x_mask[mask_out[-1]] + offset_x, y_mask[mask_out[-1]] + offset_y, 1])
    else:
        indices_of_gaps = np.concatenate(([-1], indices_of_gaps))
        last = np.asarray([len(mask_out) - 1])
        indices_of_gaps = np.concatenate((indices_of_gaps, last))
        lengths = indices_of_gaps[1:] - indices_of_gaps[:-1]
        if point is None:
            max_length_index = np.argmax(lengths)
            start_index = indices_of_gaps[max_length_index] + 1
            end_index = indices_of_gaps[max_length_index + 1]
            p1 = np.asarray([x_mask[mask_out[start_index]] + offset_x, y_mask[mask_out[start_index]] + offset_y, 1])
            p2 = np.asarray([x_mask[mask_out[end_index]] + offset_x, y_mask[mask_out[end_index]] + offset_y, 1])
        else:
            # at least one intersection point needs to lie in the desired direction
            p1, p2 = None, None
            sorted_lengths = np.sort(lengths)[::-1]
            found = False
            for length in sorted_lengths:
                length_index = np.where(lengths == length)[0][0]
                start_index = int(indices_of_gaps[length_index]) + 1
                end_index = indices_of_gaps[length_index + 1]
                p1 = np.asarray([x_mask[mask_out[start_index]] + offset_x, y_mask[mask_out[start_index]] + offset_y, 1])
                p2 = np.asarray([x_mask[mask_out[end_index]] + offset_x, y_mask[mask_out[end_index]] + offset_y, 1])

                if np.dot(direction[:2], p1[:2] - point[:2]) > 0 and np.linalg.norm(p1 - point) > 5 or np.dot(direction[:2], p2[:2] - point[:2]) > 0 and np.linalg.norm(p2 - point) > 5:
                    found = True
                    break
            if not found:
                return None, None

    return p1, p2


def get_keypoints_from_adjacent_bodyparts(keypoints, bodypart1, bodypart2):
    """
    Body parts 1 and 2 are given as tuples of keypoint indices. One of these indices needs to appear in both body part tuples.
    The coordinates of this keypoint index from the keypoints array is returned
    :param keypoints: list of keypoint coordinates
    :param bodypart1:
    :param bodypart2:
    :return: the common keypoint coordinates, the coordinates of the other keypoint of bodypart1 and the coordinates of the other keypoint of bodypart2
    """
    if bodypart1[0] == bodypart2[0]:
        keypoint = keypoints[bodypart1[0]]
        keypoint1 = keypoints[bodypart1[1]]
        keypoint2 = keypoints[bodypart2[1]]
    elif bodypart1[0] == bodypart2[1]:
        keypoint = keypoints[bodypart1[0]]
        keypoint1 = keypoints[bodypart1[1]]
        keypoint2 = keypoints[bodypart2[0]]
    elif bodypart1[1] == bodypart2[0]:
        keypoint = keypoints[bodypart1[1]]
        keypoint1 = keypoints[bodypart1[0]]
        keypoint2 = keypoints[bodypart2[1]]
    elif bodypart1[1] == bodypart2[1]:
        keypoint = keypoints[bodypart1[1]]
        keypoint1 = keypoints[bodypart1[0]]
        keypoint2 = keypoints[bodypart2[0]]
    else:
        raise RuntimeError("Two keypoints of both bodyparts need to be the same")
    return keypoint, keypoint1, keypoint2


def calculate_orientation_vecs(keypoints, bodypart_upper, bodypart_lower):
    """
    Calculates the (normalized) orientation vectors of the upper and lower body part and the angle between the two body parts
    :param keypoints: list of keypoint coordinates
    :param bodypart_upper: indices of the keypoints of the upper body part
    :param bodypart_lower: indices of the keypoints of the lower body part
    :return: The orientation of the upper body part, the orientation of the lower body part, the oriented angle between the two body parts and the common keypoint
    """
    keypoint, keypoint1, keypoint2 = get_keypoints_from_adjacent_bodyparts(keypoints, bodypart_upper, bodypart_lower)

    upper_vec = np.asarray(keypoint1[:2]) - np.asarray(keypoint[:2])
    upper_vec /= np.linalg.norm(upper_vec)
    lower_vec = np.asarray(keypoint2[:2]) - np.asarray(keypoint[:2])
    lower_vec /= np.linalg.norm(lower_vec)

    angle_between_body_parts = angle_between_points_oriented(keypoint2, keypoint, keypoint1)

    return upper_vec, lower_vec, angle_between_body_parts, keypoint



def calculate_adjacent_kp(keypoints, upper_bodypart, lower_bodypart, masks, percentage_thickness, percentage_angle, bbox, crop_region=False, anchor=None, switch_left_right=False):
    """
    Calculate an arbitrary keypoint on a adjacent body part (elbow, knee)
    :param keypoints: List of standard keypoint annotations
    :param upper_bodypart: The upper body part
    :param lower_bodypart: The lower body part
    :param masks: segmentation masks
    :param bbox: bounding box of the segmentation mask
    :param crop_region: If true, the segmentation mask is cropped to the bounding box of the body part for generation (speeds up generation)
    :param anchor: output of anchor calculation: anchor_point, orth_vec_up, orth_vec_low, percentage_upper, percentage_lower, _
    :param switch_left_right: if left and right switch from lower to upper bodypart, left and right semantic for the lower body part are switched so that the same thickness lies on the same line
    :return:
    """

    invalid_result = None, None, None, None
    x1, y1, x2, y2 = bbox
    mask = np.zeros_like(masks)
    ys1, xs1 = np.where(masks == upper_bodypart[2])
    mask[ys1, xs1] = 1
    ys2, xs2 = np.where(masks == lower_bodypart[2])
    mask[ys2, xs2] = 1

    if crop_region:
        x1_ = max(0, min(np.min(xs1) - 2, np.min(xs2) - 2))
        x2_ = min(max(np.max(xs2) + 2, np.max(xs1) + 2), masks.shape[1])
        y1_ = max(0, min(np.min(ys1) - 2, np.min(ys2) - 2))
        y2_ = min(max(np.max(ys2) + 2, np.max(ys1) + 2), masks.shape[0])
        mask = mask[y1_:y2_, x1_:x2_]
        x2 = x1 + x2_
        y2 = y1 + y2_
        x1 = x1 + x1_
        y1 = y1 + y1_
        bbox = x1, y1, x2, y2

    upper_vec, lower_vec, angle_between_body_parts, keypoint = calculate_orientation_vecs(keypoints, upper_bodypart, lower_bodypart)

    if anchor is not None:
        anchor_point, orth_vec_up, orth_vec_low, percentage_upper, percentage_lower, _ = anchor
    else:
        # we raise an Error because otherwise the runtime is exploding
        raise RuntimeError("Use precalculated anchors!")
        # percentage_upper, percentage_lower = 0, 0
        # anchor_point, orth_vec_up, orth_vec_low = calculate_anchor_point(upper_vec, lower_vec, keypoint, angle_between_body_parts, mask, bbox, x1, y1)

    if anchor_point is None:
        return invalid_result

    is_upper = percentage_angle > 0.5  # we want a point on the upper body part if the angle percentage is > 0.5
    if angle_between_body_parts > 180:  # angles are different depending if the anchor is on the left or right side
        angle_between_orth_vecs = angle_between_points_oriented(orth_vec_low, np.asarray([0, 0]), orth_vec_up)
        angle_orth_vec = angle_between_points_oriented(orth_vec_low, np.asarray([0, 0]), np.asarray([1, 0]))
    else:
        angle_between_orth_vecs = angle_between_points_oriented(orth_vec_up, np.asarray([0, 0]), orth_vec_low)
        angle_orth_vec = angle_between_points_oriented(orth_vec_up, np.asarray([0, 0]), np.asarray([1, 0]))
        percentage_angle = 1 - percentage_angle

    final_angle = angle_orth_vec - percentage_angle * angle_between_orth_vecs
    final_angle_rad = np.deg2rad(final_angle)
    final_vec = np.asarray([np.cos(final_angle_rad), np.sin(final_angle_rad)])

    m_res = final_vec[1] / final_vec[0]
    t_res = anchor_point[1] - m_res * anchor_point[0]
    x_res, y_res = generate_segmentation_line(m_res, t_res, bbox, keypoint)

    if percentage_angle > 0.5 and angle_between_body_parts > 180 or percentage_angle < 0.5 and angle_between_body_parts < 180:
        intermediate_x = (keypoint[1] * upper_vec[0] - keypoint[0] * upper_vec[1] - t_res * upper_vec[0]) / (m_res * upper_vec[0] - upper_vec[1])
    else:
        intermediate_x = (keypoint[1] * lower_vec[0] - keypoint[0] * lower_vec[1] - t_res * lower_vec[0]) / (m_res * lower_vec[0] - lower_vec[1])
    intermediate_y = m_res * intermediate_x + t_res
    intermediate_point = np.asarray([intermediate_x, intermediate_y, 1])

    p1, p2 = get_mask_line_intersection_points(x_res, y_res, mask, x1, y1, point=anchor_point, direction=intermediate_point[:2] - anchor_point[:2])

    if p1 is None or p2 is None:
        return invalid_result

    border_point = get_correct_border_point(anchor_point, intermediate_point, p1, p2)

    if percentage_angle > 0.5 or switch_left_right:
        angle_original = angle_between_points(orth_vec_up, np.asarray([0, 0]), np.asarray([1, 0]))
    else:
        angle_original = angle_between_points(orth_vec_low, np.asarray([0, 0]), np.asarray([1, 0]))
    angle_final = angle_between_points(final_vec, np.asarray([0, 0]), np.asarray([1, 0]))

    point_left = anchor_point if anchor_point[0] < border_point[0] else border_point
    point_right = anchor_point if anchor_point[0] > border_point[0] else border_point

    if anchor_point[0] == border_point[0]:  # we need to ensure that this matches the other creation technique, so lower y corresponds to left point
        angle_final = 90  # necessary as the next if statement should not take effect and normally the angle is close to 90 but not exactly 90
        point_left = anchor_point if anchor_point[1] >= border_point[1] else border_point
        point_right = anchor_point if anchor_point[1] < border_point[1] else border_point

    if angle_final >= 90 > angle_original or angle_final < 90 < angle_original:
        point_left, point_right = point_right, point_left

    if percentage_thickness <= 0:
        final_point = get_intermediate_point(intermediate_point, point_left, np.abs(percentage_thickness))
    else:
        final_point = get_intermediate_point(intermediate_point, point_right, percentage_thickness)

    mask_point = [int(round(final_point[1])) - y1, int(round(final_point[0])) - x1]
    if mask_point[0] < 0 or mask_point[0] >= mask.shape[0] or mask_point[1] < 0 or mask_point[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask
        final_point = None
    elif mask[mask_point[0], mask_point[1]] == 0:
        final_point = None

    if percentage_angle > 0.5:
        percentage = (1 - percentage_upper) + percentage_upper * (1 - percentage_angle) * 2
    else:
        percentage = (1 - percentage_lower) + percentage_lower * percentage_angle * 2

    return final_point, percentage, is_upper, (mask, keypoint, x_res, y_res, border_point, intermediate_point, anchor_point)


def calculate_endpoint_angle_kp(keypoints, bodypart, masks, percentage_thickness, percentage_angle, bbox, crop_region=False):
    """
    Calculates the arbitrary keypoint for endpoints with angle strategy
    :param keypoints: List of annotated standard keypoints
    :param bodypart: Tuple (border kp, inner kp, bodypart id)
    :param masks: Segmentation mask
    :param percentage_thickness:
    :param percentage_angle:
    :param bbox: The bounding box of the segmentation mask
    :param crop_region: If true, the segmentation mask is cropped to the bounding box of the body part for generation (speeds up generation)
    :return:
    """
    x1, y1, x2, y2 = bbox
    mask = np.zeros_like(masks)
    ys, xs = np.where(masks == bodypart[2])
    mask[ys, xs] = 1
    if crop_region:
        x1_ = max(0, np.min(xs) - 2)
        x2_ = min(np.max(xs) + 2, masks.shape[1])
        y1_ = max(0, np.min(ys) - 2)
        y2_ = min(np.max(ys) + 2, masks.shape[0])
        mask = mask[y1_:y2_, x1_:x2_]
        x2 = x1 + x2_
        y2 = y1 + y2_
        x1 = x1 + x1_
        y1 = y1 + y1_
        bbox = x1, y1, x2, y2

    if isinstance(bodypart[0], int):
        keypoint1 = keypoints[bodypart[0]]
    else:
        keypoint1 = get_intermediate_point(keypoints[bodypart[0][0]], keypoints[bodypart[0][1]], 0.5)
    if isinstance(bodypart[1], int):
        keypoint = keypoints[bodypart[1]]
    else:
        keypoint = get_intermediate_point(keypoints[bodypart[1][0]], keypoints[bodypart[1][1]], 0.5)

    orientation_vec = keypoint1 - keypoint
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)

    base_angle = angle_between_points_oriented(orientation_vec, np.asarray([0, 0]), np.asarray([1, 0]))

    final_angle = (base_angle + percentage_angle * 360) % 360
    final_angle_rad = np.deg2rad(final_angle)
    final_vec = np.asarray([np.cos(final_angle_rad), np.sin(final_angle_rad)])

    m_res = final_vec[1] / final_vec[0]
    t_res = keypoint[1] - m_res * keypoint[0]
    x_res, y_res = generate_segmentation_line(m_res, t_res, bbox, keypoint)
    p1, p2 = get_mask_line_intersection_points(x_res, y_res, mask, x1, y1)

    if p1 is None or p2 is None:
        return None, None

    if np.dot(p1[:2] - keypoint[:2], final_vec) > 0:
        border_point = p1
    else:
        border_point = p2

    final_point = get_intermediate_point(keypoint, border_point, np.abs(percentage_thickness))

    mask_point = [int(round(final_point[1])) - y1, int(round(final_point[0])) - x1]
    if mask_point[0] < 0 or mask_point[0] >= mask.shape[0] or mask_point[1] < 0 or mask_point[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask
        final_point = None
    elif mask[mask_point[0], mask_point[1]] == 0:
        final_point = None

    return final_point, (mask, keypoint, x_res, y_res, border_point, p1, p2)


def calculate_kp(keypoints, bodypart, masks, percentage_thickness, percentage_projection, bbox, use_central_projection=True, crop_region=False,
                 projection_point_outside_mask=False, switch_left_right=False):
    """
    Calculates arbitrary keypoints for bodyparts that are not adjacent, and not endpoints
    :param keypoints: List of standard keypoint annotations
    :param bodypart:
    :param masks: segmentation mask
    :param percentage_thickness:
    :param percentage_projection:
    :param bbox: The bounding box of the segmentation mask
    :param use_central_projection: indicates if a central projection point is used, meaning that there is a left point, central point and right point, otherwise only left and right point
    :param crop_region: If true, the segmentation mask is cropped to the bounding box of the body part for generation (speeds up generation)
    :param projection_point_outside_mask: allows the projection point to be outside of the segmentation mask
    :param switch_left_right: if true, the order of the border points is changed
    :return:
    """

    x1, y1, x2, y2 = bbox
    mask = np.zeros_like(masks)
    ys, xs = np.where(masks == bodypart[2])
    mask[ys, xs] = 1
    if crop_region:
        x1_ = max(0, np.min(xs) - 2)
        x2_ = min(np.max(xs) + 2, masks.shape[1])
        y1_ = max(0, np.min(ys) - 2)
        y2_ = min(np.max(ys) + 2, masks.shape[0])
        mask = mask[y1_:y2_, x1_:x2_]
        x2 = x1 + x2_
        y2 = y1 + y2_
        x1 = x1 + x1_
        y1 = y1 + y1_
        bbox = x1, y1, x2, y2

    if isinstance(bodypart[0], int):
        keypoint1 = keypoints[bodypart[0]]
    else:
        keypoint1 = get_intermediate_point(keypoints[bodypart[0][0]], keypoints[bodypart[0][1]], 0.5)
    if isinstance(bodypart[1], int):
        keypoint2 = keypoints[bodypart[1]]
    else:
        keypoint2 = get_intermediate_point(keypoints[bodypart[1][0]], keypoints[bodypart[1][1]], 0.5)

    projection_point = get_intermediate_point(keypoint1, keypoint2, percentage_projection)
    mask_projection = [int(round(projection_point[1])) - y1, int(round(projection_point[0])) - x1]
    if mask_projection[0] < 0 or mask_projection[0] >= mask.shape[0] or mask_projection[1] < 0 or mask_projection[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask area
        return None, None
    if not projection_point_outside_mask and mask[mask_projection[0], mask_projection[1]] == 0:  # projection point should lie in mask
        return None, None
    # orthogonal line
    if (keypoint1[1] - keypoint2[1]) != 0:
        m = - (keypoint1[0] - keypoint2[0]) / (keypoint1[1] - keypoint2[1])
    else:
        m = float('inf')
    t = projection_point[1] - m * projection_point[0]

    x_inputs, y_outputs = generate_segmentation_line(m, t, bbox, projection_point)
    if len(x_inputs) == 0:
        return None, (mask, None, None, projection_point, None, None)

    p1, p2 = get_mask_line_intersection_points(x_inputs, y_outputs, mask, x1, y1)
    if p1 is None or p2 is None:
        return None, (mask, x_inputs, y_outputs, projection_point, None, None)

    if switch_left_right:
        p1, p2 = p2, p1

    if percentage_thickness < 0 and use_central_projection:
        new_point = get_intermediate_point(projection_point, p1, np.abs(percentage_thickness))
    elif use_central_projection:
        new_point = get_intermediate_point(projection_point, p2, np.abs(percentage_thickness))
    else:
        new_point = get_intermediate_point(np.asarray(p1), p2, np.abs(percentage_thickness))

    mask_point = [int(round(new_point[1])) - y1, int(round(new_point[0])) - x1]
    if mask_point[0] < 0 or mask_point[0] >= mask.shape[0] or mask_point[1] < 0 or mask_point[1] >= mask.shape[1]:  # sometimes keypoints are located outside of the mask
        return None, None
    if mask[mask_point[0], mask_point[1]] == 0:
        return None, None
    return new_point, (mask, x_inputs, y_outputs, projection_point, p1, p2)


def calculate_anchor_point_and_orth_vecs(upper_vec, lower_vec, common_keypoint, angle_between_body_parts, mask, bbox):
    """
    Calculates the anchor point of an adjacent body part, meaning the inner point of a joint (e.g. the inner point of the elbow)
    :param upper_vec: Orientation vector of the upper body part
    :param lower_vec: Orientation vector of the lower body part
    :param common_keypoint: The coordinates of the common keypoint of the two body parts
    :param angle_between_body_parts: The oriented angle between the two body parts, these 4 parameters are returned by the function calculate_orientation_vecs
    :param mask: The segmentation mask
    :param bbox: The bounding box for which the segmentation mask is defined
    :return: the anchor point, the orthogonal vector of the upper body part and the orthogonal vector of the lower body part,
    meaning vectors that are orthogonal to the line between the enclosing keypoints of the respective body part
    """
    mid_point = (upper_vec * 0.5 + lower_vec * 0.5) * 10 + np.asarray(common_keypoint[:2])
    if (common_keypoint[0] - mid_point[0]) != 0:
        m = (common_keypoint[1] - mid_point[1]) / (common_keypoint[0] - mid_point[0])
    else:
        m = float('inf')
    t = common_keypoint[1] - m * common_keypoint[0]
    x_inputs, y_outputs = generate_segmentation_line(m, t, bbox, common_keypoint)
    x1, y1 = bbox[:2]
    p1, p2 = get_mask_line_intersection_points(x_inputs, y_outputs, mask, x1, y1)

    if p1 is None:
        return None, None, None

    if angle_between_body_parts > 180:
        orth_vec_up = np.asarray([-upper_vec[1], upper_vec[0]])
        orth_vec_low = np.asarray([lower_vec[1], -lower_vec[0]])
    else:
        orth_vec_up = np.asarray([upper_vec[1], -upper_vec[0]])
        orth_vec_low = np.asarray([-lower_vec[1], lower_vec[0]])

    s = np.dot(p1[:2] - p2[:2], orth_vec_up)
    if s >= 0:
        anchor_point = p2
    else:
        anchor_point = p1

    orth_vec_up /= np.linalg.norm(orth_vec_up)
    orth_vec_low /= np.linalg.norm(orth_vec_low)

    if orth_vec_low is None or orth_vec_low is None:
        return None, None, None

    return anchor_point, orth_vec_up, orth_vec_low


def calculate_anchor_point(keypoints, bodypart_upper, bodypart_lower, masks, bbox, crop_region=True):
    """
    Calculates the anchor point of an adjacent body part, meaning the inner point of a joint (e.g. the inner point of the elbow)
    :param keypoints: list of standard keypoint annotations
    :param bodypart_upper: the upper body part as tuple (kp id 1, kp id 2, bodypart id)
    :param bodypart_lower: the lower body part as tuple (kp id 1, kp id 2, bodypart id)
    :param masks: segmentation mask
    :param bbox: bounding box of the segmentation mask
    :param crop_region: If true, the segmentation mask is cropped to the bounding box of the body part for generation (speeds up generation)
    :return:
    """
    x1, y1 = bbox[:2]

    mask = np.zeros_like(masks)
    ys1, xs1 = np.where(masks == bodypart_upper[2])
    mask[ys1, xs1] = 1
    ys2, xs2 = np.where(masks == bodypart_lower[2])
    mask[ys2, xs2] = 1

    if crop_region:
        x1_ = max(0, min(np.min(xs1) - 2, np.min(xs2) - 2))
        x2_ = min(max(np.max(xs2) + 2, np.max(xs1) + 2), masks.shape[1])
        y1_ = max(0, min(np.min(ys1) - 2, np.min(ys2) - 2))
        y2_ = min(max(np.max(ys2) + 2, np.max(ys1) + 2), masks.shape[0])
        mask = mask[y1_:y2_, x1_:x2_]
        x2 = x1 + x2_
        y2 = y1 + y2_
        x1 = x1 + x1_
        y1 = y1 + y1_
        bbox = x1, y1, x2, y2

    upper_vec, lower_vec, angle_between_body_parts, keypoint = calculate_orientation_vecs(keypoints, bodypart_upper, bodypart_lower)

    anchor_point, orth_vec_up, orth_vec_low = calculate_anchor_point_and_orth_vecs(upper_vec, lower_vec, keypoint, angle_between_body_parts, mask, bbox)
    if anchor_point is None:
        return None

    intersec1 = get_intersection_point_of_vector_defined_lines(upper_vec, keypoint, orth_vec_up, anchor_point)
    intersec2 = get_intersection_point_of_vector_defined_lines(lower_vec, keypoint, orth_vec_low, anchor_point)

    _, keypoint1, keypoint2 = get_keypoints_from_adjacent_bodyparts(keypoints, bodypart_upper, bodypart_lower)
    dist1 = distance_matrix(np.asarray([intersec1[:2], keypoint1[:2]]), np.asarray([keypoint[:2], keypoint1[:2]]))
    dist2 = distance_matrix(np.asarray([intersec2[:2], keypoint2[:2]]), np.asarray([keypoint[:2], keypoint2[:2]]))

    angle_low = angle_between_points_oriented(orth_vec_low, np.asarray([0, 0]), np.asarray([0, 1]))
    angle_up = angle_between_points_oriented(orth_vec_up, np.asarray([0, 0]), np.asarray([0, 1]))

    left_right_changes = angle_low < 180 < angle_up or angle_low > 180 > angle_up

    if dist1[0, 0] + dist1[0, 1] - dist1[1, 0] < 1e-5 and dist2[0, 0] + dist2[0, 1] - dist2[1, 0] < 1e-5:
        percentage_upper = dist1[0, 0] / dist1[1, 0]
        percentage_lower = dist2[0, 0] / dist2[1, 0]

        return anchor_point, orth_vec_up, orth_vec_low, percentage_upper, percentage_lower, left_right_changes
    return None