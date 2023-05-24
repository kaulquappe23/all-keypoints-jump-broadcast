# -*- coding: utf-8 -*-
"""
Created on 30.03.23

"""
import os.path
from pathlib import Path
from typing import Type

import cv2
import numpy as np

from datasets.general.arbitrary_keypoints_data_wrapper import ArbitraryKeypointsDataWrapper
from datasets.general.bodypart_order import BodypartOrder
from datasets.general.bodypart_segmentation_utils import get_endpoint
from datasets.general.csv_annotation_utils import read_csv_annotations, read_csv_bboxes, write_csv_boxes, CSVInformation
from datasets.general.dataset_wrapper import DataWrapper, save_single_person_keypoints
from datasets.jump.jump_joint_order import JumpJointOrder
from paths import YTJumpLoc


class YTJumpCSVInformation(CSVInformation):
    yt_jump_annotation_header = ["event", "frame_num", "athlete", "slowmotion"] + [joint_name + suffix for joint_name in
                                                                                   JumpJointOrder.names() for suffix in
                                                                                   ["_x", "_y", "_s"]]

    @classmethod
    def get_header(cls):
        return cls.yt_jump_annotation_header

    @classmethod
    def image_id_from_info(cls, info):
        event, frame_num, athlete, slowmotion = info
        return "{}_({:05d})".format(event, int(frame_num))

    @classmethod
    def columns_from_image_id(cls, image_id):
        return Jump.id_convert(image_id)

    @classmethod
    def image_path(cls, info, subset=None):
        event, frame_num, athlete, slowmotion = info
        image_id = cls.image_id_from_info(info)
        return os.path.join(YTJumpLoc.frames_path, event, image_id + ".jpg")

    @classmethod
    def csv_path(cls, subset, num_annotations_to_use=None):
        csv_file = os.path.join(YTJumpLoc.annotation_path, "{}.csv".format(subset))
        if num_annotations_to_use is None:
            return csv_file
        return csv_file.replace(".csv", "") + "_" + str(num_annotations_to_use) + ".csv"


class Jump(DataWrapper):

    @staticmethod
    def id_convert(image_id):
        """
        Get csv column info from image id
        :param image_id:
        :return: athlete name, event, frame number
        """

        space1 = image_id.find(" ")
        space2 = -1 if space1 == -1 else image_id.find(" ", space1 + 1)
        space3 = -1 if space2 == -1 else image_id.find(" ", space2 + 1)
        if space3 == -1:
            athlete = "UNKNOWN"
        else:
            athlete = image_id[space2 + 1: space3]
        event = image_id[: image_id.rfind("_(")]
        frame_num = str(int(image_id[image_id.rfind("_(") + 2: image_id.rfind(").jpg")]))
        return event, frame_num, athlete, "UNKNOWN"

    @classmethod
    def from_csv_file(cls, subset, num_annotations_to_use=None, crop=True):
        """
        Create an object from this class from a csv file
        :param subset: subset of dataset
        :param num_annotations_to_use: if set, a specified number of annotations is used and not all of them
        :param is_youtube: youtube dataset or not
        :param crop: crop humans (True, default) or use whole images (False)
        :return:
        """
        csv_info = YTJumpCSVInformation
        header = csv_info.get_header()

        csv_file = csv_info.csv_path(subset, num_annotations_to_use)
        if not os.path.exists(csv_file):
            if num_annotations_to_use is not None:
                csv_file = csv_info.csv_path(subset)
                if not os.path.exists(csv_file):
                    raise RuntimeError("File " + csv_file + " does not exist")
            else:
                raise RuntimeError("File " + csv_file + " does not exist")
        additional_info, keypoints = read_csv_annotations(csv_file, header, 20)
        csv_file = csv_info.csv_path(subset, num_annotations_to_use)

        img_dir = YTJumpLoc.frames_path

        image_paths = {}
        image_ids = []
        annotations = {}
        for i, info in enumerate(additional_info):
            image_name = "{}_({:05d})".format(info[0], int(info[1]))

            image_ids.append(image_name)
            annotations[image_name] = keypoints[i]

            image_paths[image_name] = os.path.join(img_dir, info[0], image_name + ".jpg")

        jump = Jump("jump", image_paths, image_ids, JumpJointOrder, annotations=annotations)

        if num_annotations_to_use is not None and len(jump.image_ids) > num_annotations_to_use:
            image_ids = sorted(np.random.choice(image_ids, num_annotations_to_use, replace=False).tolist())
            annotations = dict((k, annotations[k]) for k in image_ids)
            image_paths = dict((k, image_paths[k]) for k in image_ids)
            assert len(image_ids) == len(annotations) == len(image_paths)
            print("USING ONLY {} ANNOTATIONS FROM DATASET".format(len(image_ids)))
            save_single_person_keypoints(csv_file, annotations, header, image_ids, cls.id_convert)
            jump.image_ids = image_ids
            jump.annotations = annotations
            jump.image_paths = image_paths

        if crop and not os.path.exists(csv_file.replace(".csv", "_bbox.csv")):
            print("Creating bbox file...")
            bboxes = {}
            for image_name in image_ids:
                x_coords = keypoints[np.where(keypoints[:, 2] > 0), 0]
                y_coords = keypoints[np.where(keypoints[:, 2] > 0), 1]
                min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
                min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
                w = max_x - min_x
                h = max_y - min_y
                offset_w = int(w * 0.25)
                offset_h = int(h * 0.25)
                min_x = max(0, min_x - offset_w)
                min_y = max(0, min_y - offset_h)
                image = jump.load_image(image_name)
                max_x = min(image.shape[1], max_x + offset_w)
                w = max_x - min_x
                max_y = min(image.shape[0], max_y + offset_h)
                h = max_y - min_y
                bboxes[image_name] = [min_x, min_y, w, h]
            jump.bboxes = bboxes
            write_csv_boxes(csv_file.replace(".csv", "_bbox.csv"), bboxes)
        elif crop:
            jump.bboxes = read_csv_bboxes(csv_file.replace(".csv", "_bbox.csv"))

        return jump


class JumpArbitraryKeypoints(ArbitraryKeypointsDataWrapper, Jump):

    def __init__(
            self, subset, params, bodypart_order: Type[BodypartOrder], representation_type: str,
            arbitrary_keypoint_mode=ArbitraryKeypointsDataWrapper.GENERATE_POINTS, test_points=None, num_points_to_generate=None,
            num_annotations_to_use=None, verbose=True
            ):
        """
        Jump dataset for arbitrary keypoint detection
        :param subset: test, train or val
        :param dictionary containing "is_youtube": youtube or bisp dataset, dictionary used for general applicability
        :param bodypart_order: BodyPartOrder class for representation of head
        :param representation_type: ArbitraryKeypointsDataWrapper.NORM_POSE or ArbitraryKeypointsDataWrapper.KEYPOINT_VECTOR
        :param arbitrary_keypoint_mode: either ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS (use a predefined number of
        equally spaced lines with equally spaced points, defined by the parameter test_points) or
        ArbitraryKeypointsDataWrapper.GENERATE_POINTS (generate a random number of random points, defined by the parameter
        num_points_to_generate)
        :param test_points: (num_lines, points_per_line) if arbitrary_keypoint_mode ==
        ArbitraryKeypointsDataWrapper.PREDEFINED_POINTS
        :param num_points_to_generate number of points to generate in GENERATE_POINTS mode. Can be a number or a tuple of
        numbers, indicating the minimum and maximum number of points to generate
        :param num_annotations_to_use: if set, a specified number of annotations is used and not all of them
        """

        repr = "_norm_pose" if representation_type == ArbitraryKeypointsDataWrapper.NORM_POSE else ""
        is_youtube = params["is_youtube"]
        if is_youtube:
            endpoints = YTJumpLoc.segmentation_endpoint_path.format(subset, bodypart_order.head_strategy())
            anchors = YTJumpLoc.segmentation_anchor_path.format(subset)
            if test_points is not None:
                num_lines, points_per_line = test_points
                test_file = Path(YTJumpLoc.test_points) / "test_points{}_{}_{}_{}{}.pkl".format(bodypart_order.head_strategy(),
                                                                                                subset, num_lines,
                                                                                                points_per_line, repr)
                test_points = test_file, num_lines, points_per_line
            seg_boxes = YTJumpLoc.segmentation_bbox_path
            seg_im_path = YTJumpLoc.segmentation_images
        else:
            endpoints = BispJumpLoc.segmentation_endpoint_path.format(subset)
            anchors = BispJumpLoc.segmentation_anchor_path.format(subset)
            if test_points is not None:
                num_lines, points_per_line = test_points
                test_file = Path(BispJumpLoc.test_points) / "test_points{}_{}_{}_{}{}.pkl".format(bodypart_order.head_strategy(),
                                                                                                  subset, num_lines,
                                                                                                  points_per_line, repr)
                test_points = test_file, num_lines, points_per_line
            seg_im_path = BispJumpLoc.segmentation_images.format(subset)
            seg_boxes = BispJumpLoc.segmentation_bbox_path.format(subset)

        jump_base_data = Jump.from_csv_file(subset, num_annotations_to_use)

        bboxes = read_csv_bboxes(seg_boxes, header=True)
        bodypart_masks = {}
        for image_name, bbox in bboxes.items():
            image_name = image_name.replace(".jpg", "")
            if image_name in jump_base_data.image_ids:
                mask_path = os.path.join(seg_im_path, image_name + ".png")
                if not os.path.exists(mask_path):
                    continue
                jump_base_data.bboxes[image_name] = bbox
                bodypart_masks[image_name] = mask_path
        # For some images, we do not have segmentation masks, so we remove them from the dataset
        jump_base_data.image_ids = sorted(list(bodypart_masks.keys()))
        if verbose:
            print("Number of images with segmentation masks (that are used in subset {}): {}".format(subset,
                                                                                                     len(jump_base_data.image_ids)))

        super().__init__(jump_base_data, bodypart_masks, bodypart_order, representation_type, endpoints, anchors,
                         full_image_masks=False, arbitrary_keypoint_mode=arbitrary_keypoint_mode,
                         test_points=test_points, num_points_to_generate=num_points_to_generate)

    def setup_norm_pose(self):
        return setup_jump_norm_pose(self.bodypart_order)

    def load_bodypart_mask(self, image_id, size=None, scale_mask=0.1):
        return super(JumpArbitraryKeypoints, self).load_bodypart_mask(image_id, size, scale_mask).astype(int)


def setup_jump_norm_pose(bodypart_order):
    """
    Setup the norm pose for the jump dataset
    :param bodypart_order: JumpBodyPartOrder, either HeadAngle or HeadEndpoint one
    :return: dict containing the keypoint coordinates, the bodypart map and the endpoint dict
    """
    pose_shape = (382, 388)
    points = [[40, 80], [98, 80], [146, 80],
              [172, 370], [172, 282], [172, 210],
              [20, 80], [144, 384], [178, 385]
              ]
    for point in points.copy():
        points.append([382 - point[0], point[1]])
    points.extend([[191, 65], [191, 30]])
    indices = [JumpJointOrder.l_wrist, JumpJointOrder.l_elbow, JumpJointOrder.l_shoulder,
               JumpJointOrder.l_ankle, JumpJointOrder.l_knee, JumpJointOrder.l_hip,
               JumpJointOrder.l_hand, JumpJointOrder.l_heel, JumpJointOrder.l_toetip,
               JumpJointOrder.r_wrist, JumpJointOrder.r_elbow, JumpJointOrder.r_shoulder,
               JumpJointOrder.r_ankle, JumpJointOrder.r_knee, JumpJointOrder.r_hip,
               JumpJointOrder.r_hand, JumpJointOrder.r_heel, JumpJointOrder.r_toetip,
               JumpJointOrder.neck, JumpJointOrder.head]
    keypoints = np.zeros((20, 3))
    keypoints[indices, :2] = np.asarray(points)
    keypoints[JumpJointOrder.r_toetip][1] = 383
    keypoints[:, 2] = 1

    bodypart_file = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "general" / "norm_pose_bodyparts_v3.png"
    bodypart_map = cv2.imread(str(bodypart_file), cv2.IMREAD_GRAYSCALE) // 10

    endpoint_dict = {}
    bodypart_dict = bodypart_order.get_bodypart_to_keypoint_dict()
    for bodypart_id_endpoint in bodypart_order.get_endpoint_bodyparts():
        bodypart = list(bodypart_dict[bodypart_id_endpoint]) + [bodypart_id_endpoint]
        result = get_endpoint(keypoints, bodypart, bodypart_map, (0, 0, pose_shape[0], pose_shape[1]))
        if result is not None:
            endpoint_dict[bodypart_id_endpoint] = result

    return {
            "keypoints":    keypoints,
            "bodypart_map": bodypart_map,
            "endpoints":    endpoint_dict
            }
