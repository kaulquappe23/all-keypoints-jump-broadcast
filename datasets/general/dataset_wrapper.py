# -*- coding: utf-8 -*-
"""
Created on 02.02.21

"""
import os
from typing import Type

import cv2
import numpy as np
from tqdm import tqdm

from datasets.general.joint_order import JointOrder


class DataWrapper:
    """
    A wrapper for a dataset that provides a unified interface for loading images and annotations.
    """

    def __init__(self, name, image_paths, image_ids, joint_order: Type[JointOrder], annotations=None, bboxes=None):
        self.name = name
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.annotations = annotations
        self.bboxes = bboxes
        self.joint_order: Type[JointOrder] = joint_order
        self.postfix = ""

    def __len__(self):
        return len(self.image_ids)

    @classmethod
    def from_csv_file(cls, csv_file, num_annotations_to_use=None):
        """
        Load the dataset from a csv file. If less annotations than used in the file (e.g. for tests) should be used,
        set the parameter. The filename is then altered to <csv_file>_<num_annotations_to_use>.csv
        :param csv_file: filename without extension
        :param num_annotations_to_use: None if all should be used, otherwise int
        :return: A DataWrapper object
        """
        raise NotImplementedError

    def load_keypoints(self, image_id):
        return self.annotations[image_id]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        path = self.image_paths[image_id]
        image = cv2.imread(path)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_bbox(self, image_id):
        if self.bboxes is not None:
            return self.bboxes[image_id]
        else:
            raise RuntimeError("Bboxes are not set for this dataset")

    def get_num_keypoints(self):
        return self.joint_order.get_num_joints()

    def get_annotations(self, ids, use_pbar=False):
        """
        Returns the annotations for the given ids
        :param ids:
        :param use_pbar: uses tqdm if True
        :return:
        """
        annotations = []
        iterator = tqdm(enumerate(ids), position=0, total=len(ids), desc="Loading annotations") if use_pbar else enumerate(ids)
        if use_pbar:
            print("Loading annotations...")
        for i, im_id in iterator:
            annos = self.annotations[im_id]
            if len(annos.shape) == 3:
                annos = annos[0]
            annotations.append(annos)
        return np.asarray(annotations)


def save_single_person_keypoints(filename, keypoints, header, ordered_ids, id_convert_func, save_as_float=True):
    """
    Saves keypoints in a csv file for single person datasets
    :param filename: filename to save
    :param keypoints: dictionary mapping from image ids to array with shape (num_joints, 3) or array with shape (num_images,
    num_joints, 3) and the annotations are in the same order as the ordered_ids
    :param header: list of header strings
    :param ordered_ids: image ids
    :param id_convert_func: function mapping from image_id to a list of strings that is used before the keypoints in the file
    :param save_as_float: coordinates as float or int
    :return:
    """
    os.makedirs(filename[:filename.rfind(os.sep)], exist_ok=True)
    with open(filename, "w") as f:
        num_detections = len(ordered_ids)
        f.write("#{}\n".format(num_detections))

        write_header = ""
        for header_item in header:
            write_header += header_item + ";"
        f.write(write_header[:-1] + "\n")

        num_joints = keypoints[list(keypoints.keys())[0]].shape[0]

        for image_id in ordered_ids:
            ident = id_convert_func(image_id)
            kpt_string = ""
            for i in range(len(ident)):
                kpt_string += ident[i] + ";"

            if image_id in keypoints:
                coords = keypoints[image_id]
                for i in range(coords.shape[0]):
                    if save_as_float:
                        kpt_string += "{:.2f};{:.2f};{:.2f};".format(coords[i, 0], coords[i, 1], coords[i, 2])
                    else:
                        kpt_string += "{:.0f};{:.0f};{:.0f};".format(coords[i, 0], coords[i, 1], coords[i, 2])
            else:
                for i in range(num_joints):
                    if save_as_float:
                        kpt_string += "{:.2f};{:.2f};{:.2f};".format(-1, -1, 0)
                    else:
                        kpt_string += "{:.0f};{:.0f};{:.0f};".format(-1, -1, 0)
            f.write("{}\n".format(kpt_string[:-1]))
