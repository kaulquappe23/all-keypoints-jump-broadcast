# -*- coding: utf-8 -*-
"""
Created on 08.03.22

"""
from typing import Type

import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm

from datasets.general.bodypart_order import BodypartOrder
from datasets.general.bodypart_segmentation_utils import generate_segmentation_line, get_mask_line_intersection_points, calculate_orientation_vecs
from datasets.general.keypoint_generator import get_intermediate_point, angle_between_points, get_intersection_point_of_vector_and_slope_defined_line, get_correct_border_point


def thickness_metric(distances, threshold=0.2):
    """
    Calculate mean thickness error, standard deviation and the percentage of correct thickness at the given threshold
    :param distances: distances
    :param threshold: PCT threshold
    :return:
    """
    distances = np.asarray(distances)
    distances_valid = distances[distances >= 0]
    distances = distances_valid
    if len(distances) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(distances)
    std_dev = np.std(distances)
    if threshold is None:
        return mean, std_dev
    num_threshold = np.count_nonzero(np.asarray(distances) <= threshold)
    pct = num_threshold / len(distances)
    return mean, std_dev, pct


def thickness_percentage_differences(predictions_thick, annotations_thick, fixed_annos, bodypart_order: Type[BodypartOrder], bodypart_segmentation_func, bbox_dict, ids, thickness_vectors, anchors,
                                     return_all_distances=False, enlarge_area_for_bodypart_search=5):
    """
    Thickness accuracy
    @param predictions_thick: shape [num_examples, num_keypoints, 3], not including standard/fixed annotations
    @param annotations_thick: shape [num_examples, num_keypoints, 3], not including standard/fixed annotations
    @param fixed_annos: the fixed, standard annotations
    @param bodypart_order: BodypartOrder object
    @param bodypart_segmentation_func: function returning segmentation masks with id and bounding box as input
    @param bbox_dict: dictionary mapping from image ids to bounding boxes for segmentation mask in x1, y1, x2, y2. If None, mapped to full image
    @param ids: image ids in same order than predictions and annotations
    @return:    - the distance between the generated and the detected point in the image
                - the distance between the desired thickness of the thickness vector and the detected point
    @param thickness_vectors: desired thickness vectors, mapping from image ids to vectors
    @param anchors: anchors of adjacent body parts, also dict with ids
    @param return_all_distances: add nan values if distances are not computable. otherwise, they will be omitted
    @param enlarge_area_for_bodypart_search: enlarge the area around the keypoint on the segmentation mask to find the correct bodypart id. A bincount is executed on this area, if the exactly underlying keypoint is background
    """
    distances = []
    distances_vec = []
    for index in tqdm(range(predictions_thick.shape[0]), desc="Thickness Errors", position=0):  # sadly, this is not vectorizable
        pred, anno = predictions_thick[index], annotations_thick[index]
        fixed_anno = fixed_annos[index]

        p_round = np.asarray(np.round(pred[:, :2]), dtype=np.int32)
        a_round = np.asarray(np.round(anno[:, :2]), dtype=np.int32)

        if bbox_dict is not None:
            bbr = bbox_dict[ids[index]]
            bbr = np.array(bbr).astype(int)
            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
            bodypart_map = bodypart_segmentation_func(ids[index], (int(x2 - x1), int(y2 - y1)))

            a_round[:, 0] -= x1
            a_round[:, 1] -= y1

            # a_round = a_round[anno[num_std_points:, 2] != 0]
            # set all values to 0 if they are not visible
            a_round[anno[:, 2] == 0] = 0

            p_round[:, 0] -= x1
            p_round[:, 1] -= y1
        else:
            bodypart_map = bodypart_segmentation_func(ids[index])
            x1, y1 = 0, 0
            y2, x2 = bodypart_map.shape

        p_round = p_round[anno[:, 2] != 0]

        bodypart_ind = bodypart_map[a_round[:, 1], a_round[:, 0]]
        # set all invisible keypoints to bodypart id -1
        bodypart_ind[anno[:, 2] == 0] = -1

        for i in range(predictions_thick.shape[1]):
            if np.sum(annotations_thick[index, i]) <= 0:
                if return_all_distances:
                    distances.append(np.nan)
                    distances_vec.append(np.nan)
                continue
            bodypart_id = bodypart_ind[i]
            if bodypart_id == -1:
                if return_all_distances:
                    distances.append(np.nan)
                    distances_vec.append(np.nan)
                continue

            if bodypart_id not in bodypart_order.get_bodypart_to_keypoint_dict():
                bodypart_area = bodypart_map[a_round[i, 1] - enlarge_area_for_bodypart_search:a_round[i, 1] + enlarge_area_for_bodypart_search + 1,
                                a_round[i, 0] - enlarge_area_for_bodypart_search:a_round[i, 0] + enlarge_area_for_bodypart_search + 1]
                if len(bodypart_area.flatten()) == 0:
                    continue
                counts = np.bincount(bodypart_area.flatten())
                counts[0] = counts[0] // 2 # we count background only half. We can not leave out the background completely, but we want to make it less likely since the keypoint might be at the edge of a body part
                count_max = np.argmax(counts)
                if count_max != 0:
                    bodypart_ind[i] = count_max
                else:
                    raise RuntimeError("did not find bodypart from annotation, weird")

            bbox = (x1, y1, x2, y2)
            thickness_vector = None if thickness_vectors is None else thickness_vectors[index][i]
            dist, dist_vec, _ = thickness_percentage_difference(bodypart_order, bodypart_ind[i],
                                                             bodypart_map,
                                                             fixed_anno, anno[i], pred[i], bbox,
                                                             thickness_vector, anchors[ids[index]],
                                                             return_all_distances=return_all_distances)

            if dist is not None:
                distances.append(dist)
            if thickness_vectors is not None:
                distances_vec.append(dist_vec)
    return distances, distances_vec


def thickness_percentage_difference(bodypart_order: BodypartOrder, bodypart_id, bodypart_map, fixed_annos, anno, pred, bbox, thickness_vector, anchor,
                                    return_all_distances=False):


    vis_res = None
    if not bodypart_id in bodypart_order.get_endpoint_angle_bodyparts():
        # ------------------------------ standard body parts -------------------------------
        dist, dist_vec, thickness_a, thickness_p = thickness_keypoint(bodypart_order, bodypart_map, bodypart_id, fixed_annos, anno, pred, bbox, thickness_vector)
        bodypart = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_id]
        adjacent_bodypart, common_point, other_point = None, None, None
        if bodypart[0] in bodypart_order.keypoints_to_adjacent_bodyparts():
            adjacent_bodypart = bodypart_order.keypoints_to_adjacent_bodyparts()[bodypart[0]]
            common_point, other_point = 0, 1
        elif bodypart[1] in bodypart_order.keypoints_to_adjacent_bodyparts():
            adjacent_bodypart = bodypart_order.keypoints_to_adjacent_bodyparts()[bodypart[1]]
            common_point, other_point = 1, 0

        projection_point_a = thickness_a[0]
        projection_point_p = thickness_p[0]
        if adjacent_bodypart is not None:
            bodypart1, bodypart2 = bodypart_order.get_adjacent_bodyparts_from_bodypart(adjacent_bodypart)
            is_upper = bodypart_id == bodypart1[-1]
            if adjacent_bodypart in anchor:
                anchor_point, _, _, percentage_upper, percentage_lower, _ = anchor[adjacent_bodypart]
                percentage = percentage_upper if is_upper else percentage_lower
                distance_proj_a = distance_matrix(np.asarray([fixed_annos[bodypart[common_point]], fixed_annos[bodypart[other_point]]]), projection_point_a[None])[:, 0]
                distance_total = np.linalg.norm(fixed_annos[bodypart[0], :2] - fixed_annos[bodypart[1], :2])
                annotation_on_edge = distance_proj_a[1] > distance_total or distance_proj_a[0] / distance_total < percentage
                distance_proj_p = distance_matrix(np.asarray([fixed_annos[bodypart[common_point]], fixed_annos[bodypart[other_point]]]), projection_point_p[None])[:, 0]
                prediction_on_edge = distance_proj_p[1] > distance_total or distance_proj_p[0] / distance_total < percentage
                if annotation_on_edge or prediction_on_edge:  # the projection point of the annotation lies in the percentage boundary or on the other side of the common point
                    # ------------------------------ joints of adjacent body parts -------------------------------
                    thickness_a = None if annotation_on_edge else thickness_a
                    thickness_p = None if prediction_on_edge else thickness_p
                    dist, dist_vec, pred_upper, vis_res = thickness_adjacent_joint(bodypart_order, bodypart_map, adjacent_bodypart, fixed_annos, anno, pred, bbox, anchor_point, thickness_vector, thickness_a, thickness_p)

                    if thickness_a is not None:
                        vis_res = (None, projection_point_a, vis_res[2], vis_res[3])
                    if thickness_p is not None:
                        vis_res = (vis_res[0], vis_res[1], None, projection_point_p)

                    # the metric is wrong if one point is on the other side of the joint edge than the other point which is on the body part itself and not on the joint edge
                    # in this case, the detected points are too far and we set the error to 2
                    if pred_upper is not None and (is_upper and not pred_upper or not is_upper and pred_upper) and not (annotation_on_edge and prediction_on_edge):
                        return 2, 2, (projection_point_a, projection_point_p, vis_res)
        if dist is None:
            if return_all_distances:
                return np.nan, np.nan, vis_res
            return None, None, None
        vis_res = (projection_point_a, projection_point_p, vis_res)
    else:
        # ------------------------------ endpoint angle body parts -------------------------------
        dist, dist_vec = thickness_endpoint_angle(bodypart_order, bodypart_map, bodypart_id, fixed_annos, anno, pred, bbox, thickness_vector)
        if dist is None:
            if return_all_distances:
                return np.nan, np.nan, vis_res
            return None, None, None, vis_res
    return dist, dist_vec, vis_res


def thickness_keypoint(bodypart_order: BodypartOrder, bodypart_map, bodypart_id, fixed_annos, anno, pred, bbox, thickness_vector=None):
    """
    Calculate the thickness of "standard" body parts (i.e. not endpoint and not angle body parts)
    :param bodypart_order: BodypartOrder object
    :param bodypart_map: segmentation mask with body part ids
    :param bodypart_id: bodypart id of the body part to calculate the thickness for
    :param fixed_annos: standard/fixed keypoint annotations
    :param anno: annotation of point to calculate the thickness for
    :param pred: prediction of point to calculate the thickness for
    :param bbox: bounding box of the segmentation mask
    :param thickness_vector: vector indicating the thickness query, can be None
    :return: calculated thickness difference according to the points, calculated thickness difference according to the thickness vector (if it is not None)
    information for visualization for the annotation and the prediction thickness calculation (projection point and if computable, intersection points and thickness)
    """
    bodypart_mask = np.zeros_like(bodypart_map)
    bodypart_mask[np.where(bodypart_map == bodypart_id)] = 1
    bodypart = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_id]

    kp1, kp2 = bodypart[0], bodypart[1]
    if isinstance(kp1, int):
        kp1 = fixed_annos[kp1]
    else:
        kp1 = get_intermediate_point(fixed_annos[kp1[0]], fixed_annos[kp1[1]], 0.5)
    if isinstance(kp2, int):
        kp2 = fixed_annos[kp2]
    else:
        kp2 = get_intermediate_point(fixed_annos[kp2[0]], fixed_annos[kp2[1]], 0.5)

    keypoints = np.array([kp1, kp2])

    x1, y1, x2, y2 = bbox
    keypoints_box = keypoints.copy()
    keypoints_box[:, 0] -= x1
    keypoints_box[:, 1] -= y1
    if (keypoints[0][1] - keypoints[1][1]) != 0:
        m1 = - (keypoints[0][0] - keypoints[1][0]) / (keypoints[0][1] - keypoints[1][1])
        t1_a = anno[1] - m1 * anno[0]
        t1_p = pred[1] - m1 * pred[0]
        if np.abs(m1) > 1e-4:
            m2 = - 1 / m1
            t2 = keypoints[0][1] - m2 * keypoints[0][0]

            projection_a = [(t1_a - t2) / (m2 - m1)]
            projection_a.extend([m1 * projection_a[0] + t1_a, 1])
            projection_p = [(t1_p - t2) / (m2 - m1)]
            projection_p.extend([m1 * projection_p[0] + t1_p, 1])

        else:
            projection_a = [keypoints[0][0], anno[1], 1]
            projection_p = [keypoints[0][0], pred[1], 1]
    else:
        projection_a = [anno[0], keypoints[0][1], 1]
        projection_p = [pred[0], keypoints[0][1], 1]
        m1 = float('inf')
        t1_a, t1_p = 0, 0

    projection_a = np.asarray(projection_a)
    projection_p = np.asarray(projection_p)
    x_a, y_a = generate_segmentation_line(m1, t1_a, bbox, projection_a)
    x_p, y_p = generate_segmentation_line(m1, t1_p, bbox, projection_p)

    no_center = bodypart_id in bodypart_order.get_bps_without_central_projection()
    info_a = [projection_a]
    info_p = [projection_p]

    p1_a, p2_a = get_mask_line_intersection_points(x_a, y_a, bodypart_mask, x1, y1)
    if p1_a is None:  # might happen because of rounding issues, then we cannot calculate this metric
        return None, None, info_a, info_p

    test_points = np.asarray([p1_a[:2], projection_a[:2], p2_a[:2]])
    anno_point = anno[:2].copy()
    dist_a = distance_matrix(test_points, anno_point[None])[:, 0]
    if no_center:
        dist_total_a = dist_a[0] + dist_a[2]
        thickness_a = dist_a[0] / (dist_total_a + 1e-10)
        thickness_vec = thickness_vector[2] if thickness_vector is not None else None
    else:
        dist_total_a = min(dist_a[0], dist_a[2]) + dist_a[1]
        thickness_a = dist_a[1] / (dist_total_a + 1e-10)
        thickness_vec = 1 - thickness_vector[1] if thickness_vector is not None else None
        if dist_a[0] < dist_a[2]:
            info_a.append(p2_a)
        else:
            info_a.append(p1_a)
        info_a.append(thickness_a)

    p1_p, p2_p = get_mask_line_intersection_points(x_p, y_p, bodypart_mask, x1, y1)

    if p1_p is None:
        return 2, 2, info_a, info_p
    pred_point = pred[:2].copy()

    test_points = np.asarray([p1_p[:2], projection_p[:2], p2_p[:2]])
    dist_p = distance_matrix(test_points, pred_point[None, :])[:, 0]
    if no_center:
        dist_total_p = dist_p[0] + dist_p[2]
    else:
        dist_total_p = min(dist_p[0], dist_p[2]) + dist_p[1]

    if dist_p[0] < dist_p[2]:
        info_p.append(p2_p)
    else:
        info_p.append(p1_p)
    thickness_p = dist_p[1] / dist_total_p

    if no_center:
        thickness_p = dist_p[0] / dist_total_p
        dist = np.abs(thickness_a - thickness_p) * 2
        dist_vec = np.abs(thickness_p - thickness_vec) * 2 if thickness_vector is not None else None
    elif dist_a[0] < dist_a[2] and dist_p[0] > dist_p[2] or dist_a[0] > dist_a[2] and dist_p[0] < dist_p[2]:
        dist = thickness_a + thickness_p
        dist_vec = thickness_vec + thickness_p if thickness_vector is not None else None
    else:
        dist = np.abs(thickness_a - thickness_p)
        dist_vec = np.abs(thickness_p - thickness_vec) if thickness_vector is not None else None
    info_p.append(thickness_p)
    return dist, dist_vec, info_a, info_p


def thickness_endpoint_angle(bodypart_order: BodypartOrder, bodypart_map, bodypart_id, fixed_annos, anno, pred, bbox, thickness_vector=None):
    """
    Calculate the thickness of endpoint angle body parts
    :param bodypart_order: BodypartOrder object
    :param bodypart_map: segmentation mask with body part ids
    :param bodypart_id: bodypart id of the body part to calculate the thickness for
    :param fixed_annos: standard/fixed keypoint annotations
    :param anno: annotation of point to calculate the thickness for
    :param pred: prediction of point to calculate the thickness for
    :param bbox: bounding box of the segmentation mask
    :param thickness_vector: vector indicating the thickness query, can be None
    :return: calculated thickness difference according to the points, calculated thickness difference according to the thickness vector (if it is not None)
    """
    bodypart_mask = np.zeros_like(bodypart_map)
    bodypart_mask[np.where(bodypart_map == bodypart_id)] = 1
    bodypart = bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_id]

    kp = bodypart[1]
    if isinstance(kp, int):
        kp = fixed_annos[kp]
    else:
        kp = get_intermediate_point(fixed_annos[kp[0]], fixed_annos[kp[1]], 0.5)

    if (kp[0] - anno[0]) != 0:
        m1_a = (kp[1] - anno[1]) / (kp[0] - anno[0])
        t1_a = anno[1] - m1_a * anno[0]
    else:
        m1_a = float('inf')
        t1_a = 0

    if (kp[0] - pred[0]) != 0:
        m1_p = (kp[1] - pred[1]) / (kp[0] - pred[0])
        t1_p = pred[1] - m1_p * pred[0]
    else:
        m1_p = float('inf')
        t1_p = 0

    x1, y1, x2, y2 = bbox
    x_a, y_a = generate_segmentation_line(m1_a, t1_a, bbox, kp)
    x_p, y_p = generate_segmentation_line(m1_p, t1_p, bbox, kp)
    p1_a, p2_a = get_mask_line_intersection_points(x_a, y_a, bodypart_mask, x1, y1)
    if p1_a is None:  # might happen because of rounding issues, then we cannot calculate this metric
        return None, None

    kp_test = kp[:2].copy()
    test_points = np.asarray([p1_a[:2], kp_test, p2_a[:2]])
    anno_point = anno[:2].copy()

    dist_a = distance_matrix(test_points, anno_point[None])[:, 0]
    dist_total_a = min(dist_a[0], dist_a[2]) + dist_a[1]
    thickness_a = dist_a[1] / (dist_total_a + 1e-10)
    thickness_vec = 1 - thickness_vector[1] if thickness_vector is not None else None

    p1_p, p2_p = get_mask_line_intersection_points(x_p, y_p, bodypart_mask, x1, y1)
    if p1_p is None:
        return 2, 2

    pred_point = pred[:2].copy()

    test_points = np.asarray([p1_p[:2], kp_test, p2_p[:2]])
    dist_p = distance_matrix(test_points, pred_point[None, :])[:, 0]
    dist_total_p = min(dist_p[0], dist_p[2]) + dist_p[1]

    angle_anno_pred = angle_between_points(anno_point, kp_test, pred_point) if not np.all(anno_point == kp_test) else 0
    if angle_anno_pred > 90:
        dist = thickness_a + dist_p[1] / dist_total_p
        dist_vec = thickness_vec + dist_p[1] / dist_total_p if thickness_vector is not None else None
    else:
        thickness_p = dist_p[1] / dist_total_p
        dist = np.abs(thickness_a - thickness_p)
        dist_vec = np.abs(thickness_p - thickness_vec) if thickness_vector is not None else None
    return dist, dist_vec


def thickness_adjacent_joint(bodypart_order: BodypartOrder, bodypart_map, bodypart_id, fixed_annos, anno, pred, bbox, anchor, thickness_vector=None, thickness_info_a=None, thickness_info_p=None):
    """
    Calculate the thickness of endpoint angle body parts
    :param thickness_info_a: If thickness_info_a or thickness_info_p is set, these values are used instead of calculating them angle wise. This is used if one keypoint is not located on the elbow/knee part, but the other one is
    :param thickness_info_p: If thickness_info_a or thickness_info_p is set, these values are used instead of calculating them angle wise. This is used if one keypoint is not located on the elbow/knee part, but the other one is
    :param bodypart_order: BodypartOrder object
    :param bodypart_map: segmentation mask with body part ids
    :param bodypart_id: bodypart id of the body part to calculate the thickness for
    :param fixed_annos: standard/fixed keypoint annotations
    :param anno: annotation of point to calculate the thickness for
    :param pred: prediction of point to calculate the thickness for
    :param bbox: bounding box of the segmentation mask
    :param thickness_vector: vector indicating the thickness query, can be None
    :param anchor: anchor point for the joint
    :return: calculated thickness difference according to the points, calculated thickness difference according to the thickness vector (if it is not None)
    """
    bodypart_mask = np.zeros_like(bodypart_map)
    bodypart1, bodypart2 = bodypart_order.get_adjacent_bodyparts_from_bodypart(bodypart_id)
    bodypart_mask[np.where(bodypart_map == bodypart1[-1])] = 1
    bodypart_mask[np.where(bodypart_map == bodypart2[-1])] = 1
    upper_vec, lower_vec, angle_between_body_parts, common_keypoint = calculate_orientation_vecs(fixed_annos, bodypart1, bodypart2)

    annotation = anno.copy()
    prediction = pred.copy()

    if anchor[0] != annotation[0]:
        m_anno = (anchor[1] - annotation[1]) / (anchor[0] - annotation[0])
        t_anno = annotation[1] - m_anno * annotation[0]
    else:
        m_anno = float('inf')
        t_anno = 0
    if anchor[0] != prediction[0]:
        m_pred = (anchor[1] - prediction[1]) / (anchor[0] - prediction[0])
        t_pred = prediction[1] - m_pred * prediction[0]
    else:
        m_pred = float('inf')
        t_pred = 0
    x_a, y_a = generate_segmentation_line(m_anno, t_anno, bbox, anchor)
    x_p, y_p = generate_segmentation_line(m_pred, t_pred, bbox, anchor)

    if len(x_p) == 0:
        return 2, 2, None, (None, None, None, None)

    direction = annotation[:2] - anchor[:2]
    if np.all(direction == 0):
        direction = prediction[:2] - anchor[:2]
        if np.all(direction == 0):
            return 0, 0, None, (None, None, None, None)

    (x1, y1, x2, y2) = bbox
    p1_a, p2_a = get_mask_line_intersection_points(x_a, y_a, bodypart_mask, x1, y1, point=anchor, direction=direction)
    p1_p, p2_p = get_mask_line_intersection_points(x_p, y_p, bodypart_mask, x1, y1, point=anchor, direction=direction)

    if p1_a is None:
        return None, None, None, (None, None, None, None)
    if p1_p is None:
        return 2, 2, None, (None, None, None, None)

    # check if the keypoint is on the upper or lower body part
    # we check the scalar product of the vector between the anchor point and the common keypoint and the vector between the anchor point and the predicted/annotation point
    # for anchor right, upper body part means the scalar product between the orthogonal vector of the keypoint ... vector and the points is negative, otherwise positive
    # anchor right means angle between body parts is between 0 and 180
    anchor_right = angle_between_body_parts <= 180
    vec = common_keypoint[:2] - anchor[:2]
    orth_vec = np.asarray([-vec[1], vec[0]])
    s_pred = np.dot(orth_vec, prediction[:2] - anchor[:2])
    s_anno = np.dot(orth_vec, annotation[:2] - anchor[:2])

    pred_upper = not (anchor_right and s_pred < 0 or not anchor_right and s_pred > 0)
    anno_upper = not (anchor_right and s_anno < 0 or not anchor_right and s_anno > 0)

    thickness_a, thickness_p, dist_a, dist_p, border_point_a, border_point_p = None, None, None, None, None, None
    if np.linalg.norm(annotation[:2] - anchor[:2]) < 2:
        anno_upper = pred_upper
        thickness_a = 1.
        dist_a = np.asarray([0, 1, 2])
    if np.linalg.norm(prediction[:2] - anchor[:2]) < 2:
        pred_upper = anno_upper
        thickness_p = 1.
        dist_p = np.asarray([0, 1, 2])

    thickness_a = thickness_info_a if thickness_info_a is not None and dist_a is None else thickness_a
    thickness_p = thickness_info_p if thickness_info_p is not None and dist_p is None else thickness_p

    # we calculate the intermediate point, which is the intersection of the line between the keypoints of the body part and the line between the anchor point and the predicted/annotation point
    intermediate_point_a, intermediate_point_p = None, None
    if pred_upper and thickness_p is None:
        intermediate_point_p = get_intersection_point_of_vector_and_slope_defined_line(upper_vec, common_keypoint, m_pred, t_pred, anchor)
    elif thickness_p is None:
        intermediate_point_p = get_intersection_point_of_vector_and_slope_defined_line(lower_vec, common_keypoint, m_pred, t_pred, anchor)

    if anno_upper and thickness_a is None:
        intermediate_point_a = get_intersection_point_of_vector_and_slope_defined_line(upper_vec, common_keypoint, m_anno, t_anno, anchor)
    elif thickness_a is None:
        intermediate_point_a = get_intersection_point_of_vector_and_slope_defined_line(lower_vec, common_keypoint, m_anno, t_anno, anchor)

    if thickness_a is None:
        border_point_a = get_correct_border_point(anchor, intermediate_point_a, p1_a, p2_a)
        test_points = np.asarray([anchor, intermediate_point_a, border_point_a])
        dist_a = distance_matrix(test_points, annotation[None])[:, 0]
        dist_total_a = min(dist_a[0], dist_a[2]) + dist_a[1]
        thickness_a = dist_a[1] / (dist_total_a + 1e-10)
    elif dist_a is None:
        if len(thickness_a) < 3:
            return 2, 2, pred_upper, (None, None, None, None)
        projection_a, _, thickness_a = thickness_a
        if points_on_same_side(common_keypoint, projection_a, anchor, annotation):
            dist_a = np.asarray([0, 1, 2])
        else:
            dist_a = np.asarray([2, 1, 0])

    thickness_vec = 1 - thickness_vector[1] if thickness_vector is not None else None

    if thickness_p is None:
        border_point_p = get_correct_border_point(anchor, intermediate_point_p, p1_p, p2_p)
        test_points = np.asarray([anchor, intermediate_point_p, border_point_p])
        dist_p = distance_matrix(test_points, prediction[None])[:, 0]
        dist_total_p = min(dist_p[0], dist_p[2]) + dist_p[1]
        thickness_p = dist_p[1] / dist_total_p
    elif dist_p is None:
        if len(thickness_p) < 3:
            return 2, 2, pred_upper, (None, None, None, None)
        projection_p, _, thickness_p = thickness_p
        if points_on_same_side(common_keypoint, projection_p, anchor, prediction):
            dist_p = np.asarray([0, 1, 2])
        else:
            dist_p = np.asarray([2, 1, 0])

    if dist_a[0] < dist_a[2] and dist_p[0] > dist_p[2] or dist_a[0] > dist_a[2] and dist_p[0] < dist_p[2]:
        dist = thickness_a + thickness_p
        dist_vec = thickness_vec + thickness_p if thickness_vector is not None else None
    else:
        dist = np.abs(thickness_a - thickness_p)
        dist_vec = np.abs(thickness_p - thickness_vec) if thickness_vector is not None else None

    return dist, dist_vec, pred_upper, (border_point_a, intermediate_point_a, border_point_p, intermediate_point_p)


def format_thickness_result(mean, std, pct, prefix=""):
    """
    Format the thickness results, readable and for tex-tables
    :param prefix: prefix for the print
    :return:
    """
    print("{} - Thickness results: mean {:.2f}, std deviation {:.2f}, pct {:.2f}".format(prefix, mean, std, pct))
    print("{:.1f} & {:.1f} & {:.1f}".format(mean*100, pct*100, std))


def points_on_same_side(line_point_start, line_point_end, test_point1, test_point2):
    """
    Check if two points are on the same side of a line
    """
    line_vec = line_point_end - line_point_start
    test_vec1 = test_point1 - line_point_start
    test_vec2 = test_point2 - line_point_start
    return np.dot(np.cross(line_vec, test_vec1), np.cross(line_vec, test_vec2)) > 0


def format_thickness_bodyparts(distances, names, bodypart_order: BodypartOrder, combine_left_right=True):
    """
    Calculate the thickness results per body part and format them
    :param combine_left_right: Combine left and right body parts to one score
    :param distances:
    :param names:
    :param bodypart_order:
    :return:
    """
    text = ""
    header = ""
    finished = set()
    pairs = bodypart_order.left_right_pairs() if combine_left_right else []
    distance_order = [bodypart[-1] for bodypart in bodypart_order.get_used_bodypart_triples()] # [11, 13, 8, 10, 12, 14, 7, 9, 1, 6, 5, 2, 3, 4, 15, 16, 17, 18] for usage of the old code test points file
    for i in range(len(names)):
        name = names[i]
        idx = eval("bodypart_order." + name)
        distance_idx = distance_order.index(idx)
        if idx in finished:
            continue
        dist = distances[:, distance_idx]
        if pairs is not None:
            for jdx in range(len(pairs)):
                if pairs[jdx][0] == idx or pairs[jdx][1] == idx:
                    other = 0 if pairs[jdx][1] == idx else 1
                    name = name[2:]
                    distance_jdx = distance_order.index(pairs[jdx][other])
                    dist = np.concatenate((dist, distances[:, distance_jdx]), axis=1)
                    finished.add(pairs[jdx][other])
                    break
        mean, std, pct = thickness_metric(dist)
        finished.add(idx)
        text += " {:.1f}  &".format(pct * 100)
        header += "{:7}|".format(name)

    return header[:-1] + "\n" + text[:-1] + "\\\\"


def format_results(result_pck, result_thickness, thresholds=(0.1, 0.05)):
    pck_full, pck = result_pck
    text = ""
    header = ""

    for i in range(len(pck_full)):
        text += "{:.1f} & ".format(round(pck_full[i]*100, 1))
        header += "F{:.2f}| ".format(thresholds[i])

    mean, pct = result_thickness

    text += "{:.1f}  &".format(pct * 100)
    header += "PCT   |"

    return header + "\n" + text
