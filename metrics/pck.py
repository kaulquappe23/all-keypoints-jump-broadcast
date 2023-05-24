#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:57:39 2016

@author: einfalmo

Collection of helpers to calculate pose estimation metrics.
"""

import numpy as np


def eval_pck_array(prediction_array, annotation_array, joint_order, pck_thresholds, use_indices=None, separate_ref_lengths=None):
    """
    Evaluate predictions according to pck metric
    @param prediction_array: array containing predictions
    @param annotation_array: array containing annotations, in same order
    @param joint_order: class containing the following info: ref_length_indices: tuple of length 2, joint indices of reference distance and fallback_ref_length_indices if a distance is not computable
    @param pck_thresholds: PCK values for these thresholds are returned (typical values are 0.1, 0.2)
    @param use_indices: the PCK values are calculated only based on these joint indices. Set to None if you want to use all joints
    @return: list of total PCK scores for the given thresholds and list of lists of PCK scores for each joint
    @param separate_ref_lengths: if the reference lengths are not included in the predictions, they can be given separately here. This is used with priority
    """

    # Get normalized PCK distances
    normalized_distances = pck_normalized_distances_fast(prediction_array.copy(), annotation_array.copy(),
                                                         joint_order=joint_order,
                                                         separate_ref_lengths=separate_ref_lengths)
    # Calculate PCK scores
    pck_ts, pck_scores = pck_scores_from_normalized_distances(normalized_distances, ignore_negative_distances=True, relevant_indices=use_indices)

    # Collect PCK scores at fixed distance thresholds
    total_scores = list()
    for pck_threshold in pck_thresholds:
        if len(pck_ts) > 0 and len(pck_scores) > 0:
            total_scores.append(pck_score_at_threshold(pck_ts, pck_scores, applied_threshold=pck_threshold))
        else:
            total_scores.append(float("nan"))

    return total_scores


def pck_normalized_distances_fast(predictions, annotations, joint_order, separate_ref_lengths=None):
    """
    Calculates the normalized distances according to PCK-like metrics between joint predictions and annotations.
    The PCK-like reference distance is specified by the given pair of keypoint indices.
    Optionally, fallback reference distances can be specified for each individual pose.
    This is necessary when the reference distance using the keypoint-pair is ill defined for some poses.

    This function operates in the following way:
    If predictions and annotations are only 2D (x,y), the reference length will be blindly calculated for each pose,
    and every single keypoint prediction + annotation pair is converted into a normalized distances.
    For this to work, you have to ensure that only valid annotations are given to the function, i.e. all invalid cases
    have to be filtered out beforehand.

    If predictions and annotations are 3D (x,y,v), then the following will be applied:
    The v-flag is interpreted as "not available" if =0 and as "available" if >0.
    If the annotation of a reference keypoint is "not available", either the fallback reference length is used (if given),
    or all distances for that pose are set to -1. The second case can add a negative bias to any accumulated PCK result.
    You have to do a correct normalization of the PCK results yourself!
    If the annotation of any other joint is "not available",
    then the distance of the respective joint is again set to -1.

    :param predictions: numpy.ndarray [num_predictions x num_joints x 2(x,y)] or 3 with visibility flag
    :param annotations: numpy.ndarray [num_predictions x num_joints x 2(x,y)] or 3 with visibility flag
    :param joint_order: class containing the following info: ref_length_indices: tuple of length 2, joint indices of reference distance and fallback_ref_length_indices if a distance is not computable
    :param separate_ref_lengths: if the reference lengths are not included in the predictions, they can be given separately here. This is used with priority
    :return: numpy.ndarray [num_predictons x num_distances(joints)]
    """
    assert predictions.shape == annotations.shape, "Predictions and annotations need same shape, but are of shape {} and {}".format(predictions.shape, annotations.shape)

    ref_length_indices = joint_order.ref_length_indices
    fallback_ref_length_indices = joint_order.fallback_ref_length_indices

    assert len(ref_length_indices) == 2
    has_valid_flag = annotations.shape[2] == 3

    # Step 1: Calculate reference lengths
    if separate_ref_lengths is None:
        ref_lengths = np.sqrt(np.sum(np.power(
            (annotations[:, ref_length_indices[0], :2] - annotations[:, ref_length_indices[1], :2]), 2), axis=1))
    else:
        ref_lengths = np.sqrt(np.sum(np.power(
            (separate_ref_lengths[:, 0, :2] - separate_ref_lengths[:, 1, :2]), 2), axis=1))

    if has_valid_flag:
        if separate_ref_lengths is None:
            ref_lengths_invalid = np.logical_or(np.logical_or(ref_lengths == 0, annotations[:, ref_length_indices[0], 2] == 0),
                                                annotations[:, ref_length_indices[1], 2] == 0)
        else:
            ref_lengths_invalid = np.logical_or(np.logical_or(ref_lengths == 0, separate_ref_lengths[:, 0, 2] == 0),
                                                separate_ref_lengths[:, 1, 2] == 0)
        annotations_invalid = annotations[:, :, 2] == 0
        if fallback_ref_length_indices is not None:
            ref_lengths[ref_lengths_invalid] = np.sqrt(np.sum(np.power(
                (annotations[:, fallback_ref_length_indices[0], :2] - annotations[:, fallback_ref_length_indices[1], :2]), 2), axis=1))[ref_lengths_invalid]
            ref_lengths_invalid = np.logical_or(np.logical_or(ref_lengths == 0, annotations[:, fallback_ref_length_indices[0], 2] == 0),
                                                annotations[:, fallback_ref_length_indices[1], 2] == 0)
        # Set invalid ref lengths to some non-zero value for save division, will be filtered out later on
        ref_lengths[ref_lengths_invalid] = 1

    # Step 2: Calculate joint-wise distance between prediction and annotation
    distances = np.sqrt(np.sum(np.power(predictions[:, :, :2] - annotations[:, :, :2], 2), axis=2))

    # Step 3: Convert euclidean distances to normalized PCK distances
    norm_distances = distances / ref_lengths[:, np.newaxis]

    # Step 4: If possible, handle cases with invalid reference distances
    if has_valid_flag:
        norm_distances[annotations_invalid] = -1
        if fallback_ref_length_indices is None:
            norm_distances[ref_lengths_invalid, :] = -1

    return norm_distances


def pck_scores_from_normalized_distances(normalized_distances, ignore_negative_distances=True, relevant_indices=None):
    """
    Creates plottable PCK statistics from normalized all_pck distances.
    Returns a flattened list of all_pck thresholds and a list of respective all_pck scores.
    :param relevant_indices: include only these joints/angles/..., whatever the pck is computed on
    :param normalized_distances: numpy.ndarray of any dimension, with a total size of N.
    Normalized distances, will be flattened.
    :param ignore_negative_distances: If True, all negative distances are filtered out
    and excluded from the PCK score calculation.
    :return: numpy.ndarrays of sizes [N], [N]. First one are the PCK thresholds, second one are the PCK scores.
    """
    if relevant_indices is not None:
        normalized_distances = normalized_distances[:, relevant_indices]
    normalized_distances_flattened = normalized_distances.flatten()
    if ignore_negative_distances:
        normalized_distances_flattened = normalized_distances_flattened[np.where(normalized_distances_flattened >= 0)]
    n = normalized_distances_flattened.shape[0]
    pck_thresholds = np.sort(normalized_distances_flattened)
    pck_scores = np.arange(0, n, dtype=float) / n
    return pck_thresholds, pck_scores


def pck_score_at_threshold(pck_thresholds, pck_scores, applied_threshold):
    """
    Given a sorted (ascending) list of PCK thresholds and assigned PCK scores,
    return the PCK score at a specific threshold.
    :param pck_thresholds: numpy.ndarray, [N], sorted PCK thresholds (ascending).
    :param pck_scores: numpy.ndarray, [N], assigend PCK scores.
    :param applied_threshold: PCK threshold to apply.
    :return: The PCK score s for the score-threshold pair (s, t) with the largest t,
    such that t <= applied_threshold.
    """
    score_indices = np.where(pck_thresholds <= applied_threshold)[0]
    if score_indices.shape[0] > 0:
        if score_indices.shape[0] == pck_thresholds.shape[0]:
            if pck_thresholds[score_indices[-1]] <= applied_threshold:
                return 1.0
        return pck_scores[score_indices[-1]]
    else:
        return .0


