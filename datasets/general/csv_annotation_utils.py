# -*- coding: utf-8 -*-
"""
Created on 30.03.23

"""

import numpy as np
import csv

class CSVInformation:

    @classmethod
    def get_header(cls):
        raise NotImplementedError

    @classmethod
    def image_id_from_info(cls, info):
        raise NotImplementedError

    @classmethod
    def columns_from_image_id(cls, image_id):
        raise NotImplementedError

    @classmethod
    def image_path(cls, info, subset=None):
        raise NotImplementedError

    @classmethod
    def csv_path(cls, subset, num_annotations_to_use=None):
        raise NotImplementedError


class SkipCSVCommentsIterator:
    """
    Simple file-iterator wrapper to skip empty and '#'-prefixed lines.
    Taken from https://bytes.com/topic/python/answers/513222-csv-comments
    (User: skip)
    """

    def __init__(self, fp):
        self.fp = fp

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.fp)
        if not line.strip() or line[0] == "#":
            return next(self)
        return line


def read_csv_annotations(csv_path, header, num_joints, convert_visible_to_binary=True):
    """
    Reads annotations from the given csv file. Format is expected to be the header and then per row one annotation, starting with some offset columns followed by x, y, visibility for each joint.
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file
        - num_joints: Number of joints
        - header: List of column names (see top of this file)
        - convert_visible_to_binary: If True, convert the visibility flag to 0/1 (values greater than 1 are set to 1, values smaller than 0 are set to 0)

    returns_
        - List of offset column entries
        - Numpy.ndarray (num_images x num_joints x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=float)
    num_offsets = len(header) - 3 * num_joints
    offset_values = []

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                for header_csv, header_val in zip(row, header):
                    assert header_val == header_csv
                continue
            assert (len(row) == num_offsets + 3 * num_joints)
            offset_values.append(row[:num_offsets])
            joint_count = 0
            for i in range(num_offsets, len(row), 3):
                joints[row_count, joint_count] = [float(row[i]), float(row[i + 1]), float(row[i + 2])]
                joint_count += 1
            row_count += 1

    if convert_visible_to_binary:
        joints[:, :, 2] = np.clip(joints[:, :, 2], 0, 1)
    return offset_values, joints


def read_csv_bboxes(csv_path, header=True):
    """
    Read bounding boxes from a csv file. Format is expected to be image id, min x, min y, width, height
    :param csv_path: path to file
    :param header: If true, header is skipped
    :return:
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=',')
        if header:
            next(reader)
        bboxes = {}
        for row in reader:
            bboxes[row[0]] = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
    return bboxes


def write_csv_boxes(csv_path, boxes):
    """
    Write bounding boxes to a csv file. Format is image id, min x, min y, width, height
    :param csv_path: path to file
    :param boxes: dict of boxes, mapping from image id to box (min_x, min_y, width, height)
    """
    with open(csv_path, "w") as f:
        lines = ["image_id, min_x, min_y, width, height\n"]
        for image_id, box in boxes.items():
            min_x, min_y, w, h = box
            lines.append("{},{},{},{},{}\n".format(image_id, min_x, min_y, w, h))
        f.writelines(lines)