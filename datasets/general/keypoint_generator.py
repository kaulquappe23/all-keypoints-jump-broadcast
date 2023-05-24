# -*- coding: utf-8 -*-
"""
Created on 11.02.21

"""
import numpy as np


def get_intermediate_point(start, end, percentage=0.5):
    """
    Returns the point on the line between start and end that is percentage of the distance from start to end
    :param start:
    :param end:
    :param percentage:
    :return:
    """
    if start.shape == 3 and (start[2] == 0 or end[2] == 0):
        return np.zeros(3)
    length = np.sqrt(np.sum((start - end) ** 2))
    if length == 0:
        return start
    anno_len = percentage * length
    vec = (start - end) / length
    new_annotation = start - anno_len * vec

    # alternative = percentage * end + (1 - percentage) * start
    # assert np.sqrt(np.sum((new_annotation - alternative) ** 2)) < 1e-5
    return new_annotation


def get_mixed_point(points, percentages):
    """
    Returns the point that is the weighted average of the points in points with the weights in percentages
    :param points:
    :param percentages:
    :return:
    """
    assert np.sum(np.asarray(percentages)) - 1 < 1e-8
    new_annotation = np.zeros(3)
    for i, point in enumerate(points):
        if point[2] == 0:
            return np.zeros(3)
        new_annotation += percentages[i] * point

    return new_annotation


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1, v2):
    """
    Returns the angle in radians between vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(p1, anchor, p2):
    """
    Returns the angle enclosed by the lines between p1 - anchor - p2 in degrees, this angle is always smaller than 180 degrees
    """
    v1 = p1[:2] - anchor[:2]
    v2 = p2[:2] - anchor[:2]
    return angle_between_vectors(v1, v2) * 180 / np.pi


def angle_between_points_oriented(p1, anchor, p2):
    """
    Returns the angle enclosed by the lines between p1 - anchor - p2  in degrees with orientation, so this angle can be full
    360 degrees
    """
    v1 = p1[:2] - anchor[:2]
    v2 = p2[:2] - anchor[:2]
    small_angle = angle_between_vectors(v1, v2) * 180 / np.pi
    orthogonal = np.array([v1[1], -v1[0]])
    if np.dot(orthogonal, v2) < 0:
        return 360 - small_angle
    return small_angle


def get_intersection_point_of_vector_defined_lines(orientation1, point1, orientation2, point2):
    """
    Returns the intersection point of two lines defined by a point and an orientation vector
    """
    if orientation1[0] == 0 and orientation2[0] == 0:
        raise RuntimeError("Both lines are vertical and do not intersect")
    if orientation1[0] == 0:
        return get_intersection_point_of_vector_defined_lines(orientation2, point2, orientation1, point1)
    # We can now be sure that only orientation2[0] can be 0
    if orientation2[0] == 0:
        x = point2[0]
        a1 = orientation1[1] / orientation1[0]
        b1 = point1[1] - a1 * point1[0]
        y = a1 * x + b1
        return np.array([x, y, 1])
    a1 = orientation1[1] / orientation1[0]
    b1 = point1[1] - a1 * point1[0]
    a2 = orientation2[1] / orientation2[0]
    b2 = point2[1] - a2 * point2[0]
    if a1 == a2:
        raise RuntimeError("Both lines are parallel and do not intersect")
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return np.array([x, y, 1])


def get_intersection_point_of_vector_and_slope_defined_line(orientation, point, m, t, alt_point=None):
    """
    Returns the intersection point of a line defined by a point and an orientation vector and a line defined by y = mx + t.
    alt_point is used if m is inf
    """

    if m == float('inf'):
        assert orientation[0] != 0, "Both lines are vertical, hence parallel, and do not intersect"
        return get_intersection_point_of_vector_defined_lines(orientation, point, np.array([0, 1]), alt_point)
    if m * orientation[0] - orientation[1] == 0:
        raise RuntimeError("Both lines are parallel and do not intersect")
    x = (point[1] * orientation[0] - point[0] * orientation[1] - t * orientation[0]) / (m * orientation[0] - orientation[1])
    y = m * x + t
    return np.array([x, y, 1])


def get_correct_border_point(anchor_point, intermediate_point, p1, p2):
    """
    Check which point of the border points p1, p2 is not the anchor point or close to the anchor point, return the true border
    point
    """
    # first we check if the border points are located in the same direction as the intermediate point
    p1_right_dir = np.dot(p1[:2] - anchor_point[:2], intermediate_point[:2] - anchor_point[:2]) > 0
    p2_right_dir = np.dot(p2[:2] - anchor_point[:2], intermediate_point[:2] - anchor_point[:2]) > 0
    if p1_right_dir and not p2_right_dir:
        border_point = p1
    elif not p1_right_dir and p2_right_dir:
        border_point = p2
    else:
        # if both points are on the correct side, we use the point that is further away from the anchor point
        if np.linalg.norm(p1[:2] - anchor_point[:2]) < np.linalg.norm(p2[:2] - anchor_point[:2]):
            border_point = p2
        else:
            border_point = p1
    return border_point
