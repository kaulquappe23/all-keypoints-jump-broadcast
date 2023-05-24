# -*- coding: utf-8 -*-
"""
Created on 24.03.23

"""
import copy
import os.path
import pickle
from typing import Type

import cv2
import numpy as np
from tqdm import tqdm

from datasets.general.bodypart_order import BodypartOrder
from datasets.general.bodypart_segmentation_utils import get_endpoint, calculate_anchor_point, calculate_adjacent_kp, \
    calculate_endpoint_angle_kp, calculate_kp
from datasets.general.dataset_wrapper import DataWrapper
from datasets.general.keypoint_generator import get_intermediate_point


def get_boundaries(image, full_image_mask, bb=None):
    """
    Returns the boundaries for the segmentation mask
    :param image:
    :param full_image_mask: indicates if the mask covers the whole image or only the bounding box
    :param bb: the bounding box, can be None if full_image_mask is True
    :return:
    """
    if not full_image_mask:
        x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
        x2 = min([x2, image.shape[1]])
        y2 = min([y2, image.shape[0]])
    else:
        x1, y1 = 0, 0
        y2, x2 = image.shape[:2]
    return x1, y1, x2, y2


class IntermediateKeypointsDataWrapper(DataWrapper):
    def __init__(
            self, name, image_paths, image_ids, annotations, bodypart_indices, num_generation_kps: tuple, use_standard_kps=True
            ):
        """
        A wrapper for a dataset that provides an interface for generating intermediate keypoints.
        :param num_generation_kps: A tuple (min, max) indicating the minimum and maximum number of keypoints that should be
        generated
        :param use_standard_kps:
        """
        super().__init__(name, image_paths, image_ids, annotations)
        self.num_generation_points = num_generation_kps
        self.use_standard_points = use_standard_kps
        self.bodypart_indices = bodypart_indices

    def load_keypoints(self, image_id):
        annos = self.annotations[image_id]
        annos = self.generate_intermediate_kps(annos, self.num_generation_points)
        if not self.use_standard_points:
            anno, kp_vec = annos
            anno = anno[:, self.get_num_keypoints():]
            kp_vec = kp_vec[self.get_num_keypoints():]
            annos = anno, kp_vec
        return annos

    def generate_intermediate_kps(self, annotations, num, p=None):
        """
        Generates
        :param annotations:
        :param num: either an integer indicating the number of points to generate or a tuple indicating the minimum and maximum
        number of points to generate (minimum included, maximum excluded)
        :param p:
        :return:
        """
        if isinstance(num, tuple):
            if num[0] == num[1]:
                num = num[0]
            else:
                num = np.random.randint(num[0], num[1])
        new_annotations = np.zeros((1, self.get_num_keypoints() + num, 3))
        new_annotations[:, :self.get_num_keypoints()] = annotations

        percentages = np.zeros((self.get_num_keypoints() + num, self.get_num_keypoints()), dtype=np.float32)
        percentages[range(self.get_num_keypoints()), range(self.get_num_keypoints())] = 1

        idx = self.get_num_keypoints()
        p = p if p is not None else [1. / len(self.bodypart_indices) for _ in range(len(self.bodypart_indices))]
        r = np.random.choice(range(len(self.bodypart_indices)), num, replace=True, p=p)
        bodyparts = np.asarray(self.bodypart_indices)[r, :]
        for bodypart in bodyparts:
            percentage = np.random.rand()
            new_annotations[0, idx] = get_intermediate_point(annotations[0, bodypart[0]], annotations[0, bodypart[1]], percentage)
            percentages[idx, bodypart[0]] = 1 - percentage
            percentages[idx, bodypart[1]] = percentage
            idx += 1
        return new_annotations, percentages


class ArbitraryKeypointsDataWrapper(DataWrapper):
    """
    A wrapper for a dataset with segmentation masks that provides an interface for creating arbitrary keypoints
    """
    NORM_POSE = "norm_pose"
    KEYPOINT_VECTOR = "keypoint_vector"

    PREDEFINED_POINTS = 0
    GENERATE_POINTS = 1

    def __init__(
            self, base_data_wrapper: DataWrapper, bodypart_masks, bodypart_order: Type[BodypartOrder], representation_type: str,
            endpoints: str, anchors: str, full_image_masks=True,
            arbitrary_keypoint_mode=GENERATE_POINTS, test_points=None, num_points_to_generate=None
            ):
        """
        :param bodypart_masks: dictionary mapping image ids to file paths of the bodypart masks
        :param bodypart_order: class inheriting from BodypartOrder that defines the order of the bodyparts
        :param representation_type: either NORM_POSE or KEYPOINT_VECTOR
        :param full_image_masks: Indicates if the mask has the size of the whole image or the size of the bounding box
        :param endpoints: path to the endpoint file, if the file does not exist, it will be created
        :param anchors: path to the anchor file, if the file does not exist, it will be created
        :param test_points: a tuple (test_points_file, num_lines, points_per_line) indicating the path to the file containing
        the test points, the number of lines and the number of points per line
        the test points file should contain two "{}" placeholders, and is formatted with num_lines and points_per_line
        :param arbitrary_keypoint_mode: either PREDEFINED_POINTS (use a predefined number of equally spaced lines with equally
        spaced points, defined by the parameter test_points) or
        GENERATE_POINTS (generate a random number of random points, defined by the parameter num_points_to_generate)
        :param num_points_to_generate number of points to generate in GENERATE_POINTS mode. Can be a number or a tuple of
        numbers, indicating the minimum and maximum number of points to generate
        """
        super().__init__(
                base_data_wrapper.name, base_data_wrapper.image_paths, base_data_wrapper.image_ids, base_data_wrapper.joint_order,
                annotations=base_data_wrapper.annotations, bboxes=base_data_wrapper.bboxes
                )
        self.bodypart_masks = bodypart_masks
        self.bodypart_order = bodypart_order
        self.full_image_masks = full_image_masks

        self.representation_type = representation_type
        self.postfix = "seg"

        if representation_type == self.NORM_POSE:
            self.norm_pose_settings = self.setup_norm_pose()

        if not os.path.exists(endpoints):
            print("Endpoints file does not exist, calculating and saving to {}".format(endpoints))
            self.calculate_and_save_all_endpoints(endpoints)
        with open(endpoints, "rb") as f:
            self.endpoint_dict = pickle.load(f)

        if not os.path.exists(anchors):
            print("Anchors file does not exist, calculating and saving to {}".format(anchors))
            self.calculate_and_save_all_anchors(anchors)
        with open(anchors, "rb") as f:
            self.anchors = pickle.load(f)

        self.arbitrary_keypoint_mode = arbitrary_keypoint_mode
        if arbitrary_keypoint_mode == self.PREDEFINED_POINTS:
            assert test_points is not None
            test_file, num_lines, points_per_line = test_points
            self.num_test_points = (num_lines * 2 + 1) * points_per_line * len(
                    self.bodypart_order.get_used_bodypart_triples()
                    ) + self.joint_order.get_num_joints()
            if not os.path.exists(test_file):
                print("Test points file does not exist, generating and saving to {}".format(test_file))
                self.generate_and_save_predefined_test_points(test_file, num_lines, points_per_line)
            with open(test_file, "rb") as f:
                self.test_points = pickle.load(f)
        elif arbitrary_keypoint_mode == self.GENERATE_POINTS:
            self.num_points_to_generate = num_points_to_generate
            assert num_points_to_generate is not None
        else:
            raise ValueError("Unknown arbitrary_keypoint_mode")

    def setup_norm_pose(self):
        """
        Setup the norm pose information, endpoints, keypoints, and bodypart map
        :return: dictionary with keys "endpoints", "keypoints", "bodypart_map"
        """
        raise NotImplementedError

    def load_keypoints(self, image_id):
        if self.arbitrary_keypoint_mode == self.PREDEFINED_POINTS:
            annos = copy.deepcopy(self.test_points[image_id])
        elif self.arbitrary_keypoint_mode == self.GENERATE_POINTS:
            annos = self.generate_arbitrary_kps(image_id)
        else:
            raise ValueError("Unknown arbitrary_keypoint_mode")
        return annos

    def get_num_keypoints(self):
        """
        Returns the maximum number of keypoints returned by load_keypoints()
        :return:
        """
        if self.arbitrary_keypoint_mode == ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS:
            return self.num_test_points
        elif self.arbitrary_keypoint_mode == ArbitraryKeypointsDataWrapper.GENERATE_POINTS:
            if isinstance(self.num_points_to_generate, int):
                return self.joint_order.get_num_joints() + self.num_points_to_generate
            return self.joint_order.get_num_joints() + self.num_points_to_generate[-1]

    def get_annotations(
            self, ids, use_pbar=True, keypoint_vectors=None, use_standard=True, thickness_vectors=None, norm_poses=None
            ):
        """
        Returns annotations for the given image ids. Implementation for non-standard annotations is missing
        :param ids: list of image ids
        """
        if keypoint_vectors is not None or norm_poses is not None:
            raise NotImplementedError
            # This functionality is currently not implemented, and not used
        return super().get_annotations(ids, use_pbar=use_pbar)

    def load_bodypart_mask(self, image_id, size=None, scale_mask=1):
        """
        Loads the segmentation mask of the body parts
        :param image_id:
        :param size:
        :param scale_mask: Some masks are scaled that they can be analyzed with the human eye, this parameter scales them back
        :return:
        """
        segmentation_masks = self.bodypart_masks[image_id]
        if isinstance(segmentation_masks, str):
            if segmentation_masks.endswith(".pkl"):
                with open(segmentation_masks, "rb") as f:
                    segmentation_masks = pickle.load(f)
                    segmentation_masks = segmentation_masks.numpy()
            elif segmentation_masks.endswith(".png"):
                segmentation_masks = cv2.imread(segmentation_masks, cv2.IMREAD_UNCHANGED)
        else:
            if size is not None and (size[0] != segmentation_masks.shape[0] or size[1] != segmentation_masks.shape[1]):
                segmentation_masks = cv2.resize(segmentation_masks, size, interpolation=cv2.INTER_NEAREST)
        return segmentation_masks * scale_mask

    def check_bodypart_exists(self, bodypart, keypoints, masks, image_id):
        """
        Checks if a bodypart exists in the segmentation mask
        """
        if bodypart[-1] not in self.bodypart_order.get_adjacent_bodyparts():
            mask_ind = np.where(masks == bodypart[2])
            for i in range(2):
                if isinstance(bodypart[i], int):
                    if keypoints[bodypart[i]][2] == 0:
                        return False
                else:
                    bp1, bp2 = bodypart[i]
                    if keypoints[bp1][2] == 0 or keypoints[bp2][2] == 0:
                        return False
            if len(mask_ind[0]) == 0 or bodypart[-1] in self.bodypart_order.get_endpoint_bodyparts() and bodypart[-1] not in \
                    self.endpoint_dict[image_id]:
                return False
            return True
        bp1, bp2 = self.bodypart_order.get_bodypart_from_id(bodypart)
        return self.check_bodypart_exists(bp1, keypoints, masks, image_id) and \
            self.check_bodypart_exists(bp2, keypoints, masks, image_id)

    def generate_arbitrary_kps(self, image_id):
        """
        Generates arbitrary keypoints with the segmentation masks. Generates a certain number according to the values set
        during initialization of the dataset
        :param image_id:
        :return:
        """
        keypoints = self.annotations[image_id]
        image = self.load_image(image_id)
        bb = self.bboxes[image_id]
        bb = np.array(bb).astype(int)

        x1, y1, x2, y2 = get_boundaries(image, self.full_image_masks, bb)

        num = self.num_points_to_generate
        if isinstance(num, tuple):
            if num[0] == num[1]:
                num = num[0]
            else:
                num = np.random.randint(num[0], num[1])
        num_std_points = self.joint_order.get_num_joints()

        resized_masks = self.load_bodypart_mask(image_id, (int(x2 - x1), int(y2 - y1))) if num > 0 else None

        p = [1. / len(self.bodypart_order.get_used_bodypart_triples()) for _ in
             range(len(self.bodypart_order.get_used_bodypart_triples()))]
        count = 0
        tries = 0
        existent_bodyparts = self.bodypart_order.get_used_bodypart_triples().copy()

        new_annotations = np.zeros((num, 3), dtype=np.float32)
        new_annotations[:, 2] = 1

        kp_vector = np.zeros((num, num_std_points), dtype=np.float32)
        thickness_vector = np.zeros((num, 3))
        norm_pose_vector = np.zeros((num, 2), dtype=np.float32)
        angle_vector = np.zeros((num, 1), dtype=np.float32)

        x1_, y1_, x2_, y2_ = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
        bbox = (x1_, y1_, x2_, y2_)

        # We only try a few times, otherwise we might lose too much time
        # np.random.seed(3)
        while count < num and tries < num * 2:
            tries += 1
            r = np.random.choice(range(len(existent_bodyparts)), 1, replace=True, p=p)
            bodypart = existent_bodyparts[r[0]]
            # remove body parts that are not existent to save time
            used_bodyparts_exist = self.check_bodypart_exists(bodypart, keypoints, resized_masks, image_id)
            if not used_bodyparts_exist:
                existent_bodyparts.remove(bodypart)
                p = [1. / len(existent_bodyparts) for _ in range(len(existent_bodyparts))]
                if len(existent_bodyparts) == 0:
                    break
                continue

            percentage_projection = np.random.rand()
            percentage_thickness = np.random.rand()
            percentage_thickness = min(percentage_thickness * 2.05, 2) - 1  # assure that thickness 1 is also possible

            use_central_projection = not (bodypart[-1] in self.bodypart_order.get_bps_without_central_projection())
            if not use_central_projection:
                percentage_thickness += 1
                percentage_thickness /= 2
                if percentage_thickness < 0:
                    percentage_thickness = 0

            res = self.calculate_arbitrary_keypoint(
                    percentage_projection, percentage_thickness, bodypart, image_id, bbox, keypoints, resized_masks,
                    new_annotations,
                    kp_vector, thickness_vector, norm_pose_vector, angle_vector, count
                    )
            if res:
                count += 1

        if num > 0:
            keypoints = np.concatenate([keypoints, new_annotations[:count]])

        if self.representation_type == self.KEYPOINT_VECTOR:
            kp_vector = kp_vector[:count]
            original_kp_kp_vector = np.zeros((num_std_points, num_std_points), dtype=np.float32)
            original_kp_kp_vector[range(num_std_points), range(num_std_points)] = 1
            kp_vector = np.concatenate([original_kp_kp_vector, kp_vector], dtype=np.float32)

            thickness_vector = thickness_vector[:count]
            original_kp_thickness_vector = np.zeros((num_std_points, 3), dtype=np.float32)
            original_kp_thickness_vector[:, 1] = 1
            thickness_vector = np.concatenate([original_kp_thickness_vector, thickness_vector], dtype=np.float32)

            representations = {"keypoints": kp_vector, "thickness": thickness_vector}

            if len(self.bodypart_order.get_endpoint_angle_bodyparts()) > 0:
                angle_vector = angle_vector[:count]
                original_kp_angle_vector = np.zeros((num_std_points, 1), dtype=np.float32)
                angle_vector = np.concatenate([original_kp_angle_vector, angle_vector], dtype=np.float32)
                representations["angle"] = angle_vector

        elif self.representation_type == self.NORM_POSE:
            standard_points = self.norm_pose_settings["keypoints"][:, :2]
            norm_pose_vector = np.concatenate([standard_points, norm_pose_vector[:count]], dtype=np.float32)

            representations = {"norm_pose": norm_pose_vector}
        else:
            raise RuntimeError("Unknown representation type")

        if num > 0:
            keypoints = keypoints[None, :, :]
        return keypoints, representations

    def generate_equally_spaced_arbitrary_kps(self, image_id, num_lines, points_per_line):
        """
        Generates equally spaced lines of equally spaced keypoints along all body parts
        :param image_id:
        :param num_lines: number of lines per side
        :param points_per_line: number of points per line
        :return:
        """

        keypoints = self.annotations[image_id]
        image = self.load_image(image_id)
        bb = self.bboxes[image_id]
        bb = np.array(bb).astype(int)

        x1, y1, x2, y2 = get_boundaries(image, self.full_image_masks, bb)
        resized_masks = self.load_bodypart_mask(image_id, (int(x2 - x1), int(y2 - y1))) if points_per_line > 0 else None

        count = 0
        max_num_keypoints = (num_lines * 2 + 1) * points_per_line * len(self.bodypart_order.get_used_bodypart_triples())

        new_annotations = np.zeros((max_num_keypoints, 3), dtype=np.float32)
        new_annotations[:, 2] = 1

        num_std_points = self.joint_order.get_num_joints()

        kp_vector = np.zeros((max_num_keypoints, num_std_points), dtype=np.float32)
        thickness_vector = np.zeros((max_num_keypoints, 3), dtype=np.float32)
        norm_pose_vector = np.zeros((max_num_keypoints, 2), dtype=np.float32)
        angle_vector = np.zeros((max_num_keypoints, 1), dtype=np.float32)

        x1_, y1_, x2_, y2_ = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
        bbox = (x1_, y1_, x2_, y2_)

        for bodypart in self.bodypart_order.get_used_bodypart_triples():
            bodyparts_exists = self.check_bodypart_exists(bodypart, keypoints, resized_masks, image_id)
            if not bodyparts_exists:
                spare = (num_lines * 2 + 1) * points_per_line
                new_annotations[count: count + spare, 2] = 0
                count += spare
                continue

            for idx_line in range(- num_lines, num_lines + 1):

                percentage_thickness = idx_line / num_lines
                use_central_projection = not (bodypart[-1] in self.bodypart_order.get_bps_without_central_projection())
                endpoint_angle = (bodypart[-1] in self.bodypart_order.get_endpoint_angle_bodyparts())
                if not use_central_projection or endpoint_angle:
                    percentage_thickness += 1
                    percentage_thickness /= 2
                    if percentage_thickness < 0:
                        percentage_thickness = 0

                for idx_point in range(points_per_line):

                    percentage_projection = idx_point / (points_per_line - 1)

                    res = self.calculate_arbitrary_keypoint(
                            percentage_projection, percentage_thickness, bodypart, image_id, bbox, keypoints, resized_masks,
                            new_annotations, kp_vector, thickness_vector, norm_pose_vector,
                            angle_vector, count
                            )
                    if not res:
                        new_annotations[count, 2] = 0
                    # we always increase count so that we know which keypoint corresponds to which bodypart
                    count += 1

        keypoints = np.concatenate([keypoints, new_annotations[:count]])

        if self.representation_type == self.KEYPOINT_VECTOR:
            kp_vector = kp_vector[:count]
            original_kp_kp_vector = np.zeros((num_std_points, num_std_points), dtype=np.float32)
            original_kp_kp_vector[range(num_std_points), range(num_std_points)] = 1
            kp_vector = np.concatenate([original_kp_kp_vector, kp_vector], dtype=np.float32)

            thickness_vector = thickness_vector[:count]
            original_kp_thickness_vector = np.zeros(
                    (num_std_points, 3), dtype=np.float32
                    )  # if len(endpoint_bodyparts) == 0 else np.zeros((self.get_num_keypoints(), 4), dtype=np.float32)
            original_kp_thickness_vector[:, 1] = 1
            thickness_vector = np.concatenate([original_kp_thickness_vector, thickness_vector], dtype=np.float32)

            representations = {"kp_vector": kp_vector, "thickness": thickness_vector}

            if len(self.bodypart_order.get_endpoint_angle_bodyparts()) > 0:
                angle_vector = angle_vector[:count]
                original_kp_angle_vector = np.zeros((num_std_points, 1), dtype=np.float32)
                angle_vector = np.concatenate([original_kp_angle_vector, angle_vector], dtype=np.float32)
                representations["angle"] = angle_vector
        elif self.representation_type == self.NORM_POSE:
            standard_points = self.norm_pose_settings["keypoints"][:, :2]
            norm_pose_vector = np.concatenate([standard_points, norm_pose_vector[:count]], dtype=np.float32)

            representations = {"norm_pose": norm_pose_vector}
        else:
            raise ValueError("Unknown representation type: {}".format(self.representation_type))

        keypoints = keypoints[None, :, :]
        return keypoints, representations

    def calculate_arbitrary_keypoint(
            self, percentage_projection, percentage_thickness, bodypart, image_id, bbox, keypoints, masks,
            new_annotations, kp_vector, thickness_vector, norm_pose_vector, angle_vector, count,
            switch_left_right_if_necessary=True
            ):
        """
        Calculates an arbitrary keypoint based on the percentage of the projection and the percentage of the thickness. The
        coordinates of the new point as well as the information for the keypoint/thickness/angle vector or norm pose
        are inserted in the given arrays at position count.
        In case that the body part belongs to a body part that is connected to an adjacent body part, the percentages are
        adjusted such that the generated keypoint lies not in the adjacent are
        :param percentage_projection: If the bodypart is created with the endpoint angle strategy, this is the percentage of
        the angle.
        :param percentage_thickness:
        :param bodypart: tuple (enclosing keypoint id 1, enclosing keypoint id 2, bodypart id)
        :param bbox: the bounding box of the segmentation mask
        :param keypoints: list of coordinates of standard keypoints
        :param masks: segmentation masks
        :param new_annotations:
        :param kp_vector:
        :param thickness_vector:
        :param norm_pose_vector:
        :param angle_vector:
        :param count:
        :param switch_left_right_if_necessary: If true, the left and right bodyparts are switched for body parts connected with
        adjacent body parts (if false, collapse at the left-right change position will happen)
        :return:
        """
        percentage_angle = 0
        is_upper = False
        adjacent_bodypart = None
        use_central_projection = not bodypart[-1] in self.bodypart_order.get_bps_without_central_projection()
        # Endpoint body part
        if bodypart[-1] in self.bodypart_order.get_endpoint_bodyparts():
            endpoints = self.endpoint_dict[image_id][bodypart[-1]]
            bodypart_gen = (keypoints.shape[0], keypoints.shape[0] + 1, bodypart[-1])
            keypoints_gen = np.concatenate((keypoints, np.array(endpoints)), axis=0)
            new_point, vis_res = calculate_kp(
                    keypoints_gen, bodypart_gen, masks, percentage_thickness, percentage_projection, bbox,
                    use_central_projection=use_central_projection,
                    crop_region=True
                    )
        # Adjacent body part
        elif bodypart[-1] in self.bodypart_order.get_adjacent_bodyparts():
            bodypart1, bodypart2 = self.bodypart_order.get_bodypart_from_id(bodypart)
            if bodypart[-1] in self.anchors[image_id]:
                anchor = self.anchors[image_id][bodypart[-1]]
            else:
                return False
            new_point, percentage_projection, is_upper, _ = calculate_adjacent_kp(
                    keypoints, bodypart1, bodypart2, masks, percentage_thickness, percentage_angle=percentage_projection,
                    bbox=bbox, crop_region=True, anchor=anchor, switch_left_right=switch_left_right_if_necessary
                    )
        # Endpoint angle body part
        elif bodypart[-1] in self.bodypart_order.get_endpoint_angle_bodyparts():
            percentage_angle = percentage_projection
            if percentage_angle == 0:
                percentage_angle = 1
            percentage_projection = 1
            percentage_thickness = np.abs(percentage_thickness)
            new_point, vis_res = calculate_endpoint_angle_kp(
                    keypoints, bodypart, masks, percentage_thickness, percentage_angle, bbox, crop_region=True
                    )
        # Other body part
        else:
            # Check if body part is connected with adjacent body part, we might need to switch left and right
            kps_of_adjacent_bodyparts = self.bodypart_order.keypoints_to_adjacent_bodyparts()
            adjacent_bodypart, common_point = None, None
            if bodypart[0] in kps_of_adjacent_bodyparts:
                adjacent_bodypart = kps_of_adjacent_bodyparts[bodypart[0]]
                common_point = 0
            elif bodypart[1] in kps_of_adjacent_bodyparts:
                adjacent_bodypart = kps_of_adjacent_bodyparts[bodypart[1]]
                common_point = 1

            switch_left_right_necessary = False
            if adjacent_bodypart is not None:
                bodypart1, bodypart2 = self.bodypart_order.get_bodypart_from_id(adjacent_bodypart)
                if bodypart[-1] == bodypart1[-1]:
                    is_upper = True
                if adjacent_bodypart in self.anchors[image_id]:
                    # We adjust the percentages, such that the resulting point lies not in the adjacent area (would belong to
                    # the adjacent body part)
                    _, _, _, percentage_upper, percentage_lower, switch_left_right_lower = self.anchors[image_id][
                        adjacent_bodypart]
                    if is_upper:
                        percentage_projection = (1 - percentage_upper) * percentage_projection
                    else:
                        percentage_projection = (1 - percentage_lower) * percentage_projection
                        switch_left_right_necessary = switch_left_right_lower
                    if common_point == 0:
                        percentage_projection = 1 - percentage_projection

            projection_point_outside_mask = bodypart[-1] in self.bodypart_order.get_bodyparts_with_projection_point_outside_mask()
            bodypart_min_max = self.bodypart_order.get_bodyparts_with_min_max()
            if bodypart[-1] in bodypart_min_max:
                mini, maxi = bodypart_min_max[bodypart[-1]]
                percentage_projection = percentage_projection * (maxi - mini) + mini
            do_switch_left_right = switch_left_right_necessary and switch_left_right_if_necessary
            new_point, vis_res = calculate_kp(
                    keypoints, bodypart, masks, percentage_thickness, percentage_projection, bbox,
                    use_central_projection=use_central_projection,
                    crop_region=True, projection_point_outside_mask=projection_point_outside_mask,
                    switch_left_right=do_switch_left_right
                    )

        if new_point is None:
            return False

        new_annotations[count, :2] = new_point[:2]

        # Add representation
        if self.representation_type == self.KEYPOINT_VECTOR:
            if bodypart[-1] in self.bodypart_order.get_adjacent_bodyparts():
                bp1, bp2 = self.bodypart_order.get_bodypart_from_id(bodypart)
                common_keypoint = np.intersect1d(np.asarray(bp1), np.asarray(bp2))
                kp_vector[count, common_keypoint] = percentage_projection
                if is_upper:
                    other_kp = bp1[0] if common_keypoint == bp1[1] else bp1[1]
                else:
                    other_kp = bp2[0] if common_keypoint == bp2[1] else bp2[1]
                kp_vector[count, other_kp] = 1 - percentage_projection
            else:
                if isinstance(bodypart[0], int):
                    kp_vector[count, bodypart[0]] = 1 - percentage_projection
                else:
                    kp_vector[count, bodypart[0][0]] = 0.5 * (1 - percentage_projection)
                    kp_vector[count, bodypart[0][1]] = 0.5 * (1 - percentage_projection)
                if isinstance(bodypart[1], int):
                    kp_vector[count, bodypart[1]] = percentage_projection
                else:
                    kp_vector[count, bodypart[1][0]] = 0.5 * percentage_projection
                    kp_vector[count, bodypart[1][1]] = 0.5 * percentage_projection
            if percentage_angle != 0:
                angle_vector[count] = percentage_angle
            if percentage_thickness < 0:  # not use central projection: percentage thickness can not be < 0
                percentage_thickness = np.abs(percentage_thickness)
                thickness_vector[count, 0] = percentage_thickness
                thickness_vector[count, 1] = 1 - percentage_thickness
            elif use_central_projection:
                thickness_vector[count, 1] = 1 - percentage_thickness
                thickness_vector[count, 2] = percentage_thickness
            else:
                thickness_vector[count, 0] = 1 - percentage_thickness
                thickness_vector[count, 2] = percentage_thickness
        elif self.representation_type == self.NORM_POSE:
            if adjacent_bodypart is not None:
                bp1, bp2 = self.bodypart_order.get_bodypart_from_id(bodypart)
                if bodypart[-1] == bp1[-1]:
                    adjacent_bodypart = bp2[-1]
                else:
                    adjacent_bodypart = bp1[-1]
            elif bodypart[-1] in self.bodypart_order.get_adjacent_bodyparts():
                bp1, bp2 = self.bodypart_order.get_bodypart_from_id(bodypart)
                common_keypoint = np.intersect1d(np.asarray(bp1), np.asarray(bp2))
                if is_upper:
                    bodypart = bp1
                    adjacent_bodypart = bp2[-1]
                else:
                    bodypart = bp2
                    adjacent_bodypart = bp1[-1]
                if common_keypoint == bodypart[0]:
                    percentage_projection = 1 - percentage_projection
            norm_pose_point = self.get_norm_pose_point(
                    percentage_thickness, percentage_projection, percentage_angle, bodypart, adjacent_bodypart
                    )
            if norm_pose_point is None:
                return False
            norm_pose_vector[count] = norm_pose_point
        else:
            raise ValueError("Unknown representation type: {}".format(self.representation_type))

        return True

    def calculate_and_save_all_endpoints(self, destination_file):
        """
        Calculates all endpoints for all images and all endpoint bodyparts and saves them to a file
        :param destination_file:
        :return:
        """
        endpoint_dict = {}
        for image_id in tqdm(self.image_ids):
            endpoint_dict[image_id] = {}
            keypoints = self.annotations[image_id]
            image = self.load_image(image_id)
            bb = self.bboxes[image_id]
            bb = np.array(bb).astype(int)

            x1, y1, x2, y2 = get_boundaries(image, self.full_image_masks, bb)

            resized_masks = self.load_bodypart_mask(image_id, (int(x2 - x1), int(y2 - y1)))
            x1_, y1_, x2_, y2_ = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            bbox = (x1_, y1_, x2_, y2_)

            for bodypart_id_endpoint in self.bodypart_order.get_endpoint_bodyparts():
                bodypart = list(self.bodypart_order.get_bodypart_to_keypoint_dict()[bodypart_id_endpoint]) + [
                        bodypart_id_endpoint]
                result = get_endpoint(keypoints, bodypart, resized_masks, bbox)
                if result is not None:
                    endpoint_dict[image_id][bodypart_id_endpoint] = result

        with open(destination_file, "wb") as f:
            pickle.dump(endpoint_dict, f)

    def get_norm_pose_point(
            self, percentage_thickness, percentage_projection, percentage_angle, bodypart, adjacent_bodypart=None
            ):
        """
        Calculates the norm pose point coordinates (normalized) for a given projection, thickness and angle percentage for a
        certain body part
        :param percentage_thickness:
        :param percentage_projection:
        :param percentage_angle:
        :param bodypart:
        :param adjacent_bodypart: If the body part is an adjacent body part or is connected to an adjacent body part,
        the area considered for the norm pose point
        covers the masks for the original
        :return:
        """

        return get_norm_pose_point(
                self.norm_pose_settings, self.bodypart_order, percentage_thickness, percentage_projection, percentage_angle,
                bodypart,
                adjacent_bodypart
                )

    def calculate_anchor_points(self, image_id, crop_region=True):
        """
        Calculates the anchor points for a given image and the defined adjacent body parts
        :param image_id:
        :param crop_region:
        :return:
        """
        keypoints = self.annotations[image_id]
        image = self.load_image(image_id)
        bb = self.bboxes[image_id]
        bb = np.array(bb).astype(int)

        x1, y1, x2, y2 = get_boundaries(image, self.full_image_masks, bb)

        anchor_points = {}

        for bodypart in self.bodypart_order.get_adjacent_bodyparts():

            resized_masks = self.load_bodypart_mask(image_id, (int(x2 - x1), int(y2 - y1)))

            x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            bbox = [x1, y1, x2, y2]

            bodypart1, bodypart2 = self.bodypart_order.get_bodypart_from_id(bodypart)
            if not self.check_bodypart_exists(bodypart1, keypoints, resized_masks, image_id) or not self.check_bodypart_exists(
                    bodypart2, keypoints, resized_masks, image_id
                    ):
                continue

            res = calculate_anchor_point(keypoints, bodypart1, bodypart2, resized_masks, bbox, crop_region=crop_region)
            if res is not None:
                anchor_point, orth_vec_up, orth_vec_low, percentage_upper, percentage_lower, left_right_changes = res
                anchor_points[
                    bodypart] = anchor_point, orth_vec_up, orth_vec_low, percentage_upper, percentage_lower, left_right_changes

        return anchor_points

    def calculate_and_save_all_anchors(self, destination_file):
        """
        Calculates and saves all anchor points for all images
        :param destination_file:
        :return:
        """
        anchor_dict = {}
        for image_id in tqdm(self.image_ids):
            anchor_dict[image_id] = self.calculate_anchor_points(image_id)

        with open(destination_file, "wb") as f:
            pickle.dump(anchor_dict, f)
        return anchor_dict

    def generate_and_save_predefined_test_points(self, destination_file, num_lines, points_per_line):
        """
        Generates and saves predefined test points for all images
        :param destination_file:
        :return:
        """
        if not os.path.exists(os.path.dirname(destination_file)):
            os.makedirs(os.path.dirname(destination_file))
        test_points_dict = {}
        for image_id in tqdm(self.image_ids):
            test_points_dict[image_id] = self.generate_equally_spaced_arbitrary_kps(image_id, num_lines, points_per_line)

        with open(destination_file, "wb") as f:
            pickle.dump(test_points_dict, f)
        return test_points_dict


def get_norm_pose_point(
        norm_pose_settings, bodypart_order, percentage_thickness, percentage_projection, percentage_angle, bodypart,
        adjacent_bodypart=None
        ):
    """
    Calculates the norm pose point coordinates (normalized) for a given projection, thickness and angle percentage for a
    certain body part
    :param bodypart_order:
    :param norm_pose_settings:
    :param percentage_thickness:
    :param percentage_projection:
    :param percentage_angle:
    :param bodypart:
    :param adjacent_bodypart: If the body part is an adjacent body part or is connected to an adjacent body part,
    the area considered for the norm pose point
    covers the masks for the original
    :return:
    """

    keypoints = norm_pose_settings["keypoints"]
    bodypart_map = norm_pose_settings["bodypart_map"]
    endpoints = norm_pose_settings["endpoints"]
    bbox = (0, 0, bodypart_map.shape[1], bodypart_map.shape[0])

    if bodypart[-1] in bodypart_order.get_endpoint_bodyparts():
        endpoints = endpoints[bodypart[-1]]
        bodypart_gen = (keypoints.shape[0], keypoints.shape[0] + 1, bodypart[-1])
        keypoints_gen = np.concatenate((keypoints, np.array(endpoints)), axis=0)
        orig_size_point, _ = calculate_kp(
                keypoints_gen, bodypart_gen, bodypart_map, percentage_thickness, percentage_projection, bbox, crop_region=True
                )

    elif bodypart[-1] in bodypart_order.get_endpoint_angle_bodyparts():
        percentage_thickness = np.abs(percentage_thickness)
        orig_size_point, _ = calculate_endpoint_angle_kp(
                keypoints, bodypart, bodypart_map, percentage_thickness, percentage_angle, bbox, crop_region=True
                )

    else:
        if adjacent_bodypart is not None:
            bodypart_map = bodypart_map.copy()
            bodypart_map[bodypart_map == adjacent_bodypart] = bodypart[-1]
        orig_size_point, _ = calculate_kp(
                keypoints, bodypart, bodypart_map, percentage_thickness, percentage_projection, bbox, crop_region=True
                )

    if orig_size_point is None:
        return None
    return np.asarray([orig_size_point[0] / bodypart_map.shape[1], orig_size_point[1] / bodypart_map.shape[0]])
