# -*- coding: utf-8 -*-
"""
Created on 08.10.21

"""
import numpy as np

from datasets.general.joint_order import JointOrder


class JumpJointOrder(JointOrder):
    """
    Capsules the order of the keypoints in the jump dataset.
    """

    head = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    r_hand = 5
    l_shoulder = 6
    l_elbow = 7
    l_wrist = 8
    l_hand = 9
    r_hip = 10
    r_knee = 11
    r_ankle = 12
    r_heel = 13
    r_toetip = 14
    l_hip = 15
    l_knee = 16
    l_ankle = 17
    l_heel = 18
    l_toetip = 19

    num_joints = 20
    num_bodyparts = 16

    ref_length_indices = [l_shoulder, r_hip]
    fallback_ref_length_indices = [r_shoulder, l_hip]

    @classmethod
    def indices(cls):
        return [cls.head, cls.neck,
                cls.r_shoulder, cls.r_elbow, cls.r_wrist, cls.r_hand,
                cls.l_shoulder, cls.l_elbow, cls.l_wrist, cls.l_hand,
                cls.r_hip, cls.r_knee, cls.r_ankle, cls.r_heel, cls.r_toetip,
                cls.l_hip, cls.l_knee, cls.l_ankle, cls.l_heel, cls.l_toetip]

    @classmethod
    def bodypart_indices(cls):
        return [[cls.head, cls.neck], [cls.r_shoulder, cls.neck], [cls.l_shoulder, cls.neck], [cls.l_hip, cls.r_hip],
                [cls.r_shoulder, cls.r_elbow], [cls.r_elbow, cls.r_wrist], [cls.r_wrist, cls.r_hand],
                [cls.l_shoulder, cls.l_elbow], [cls.l_elbow, cls.l_wrist], [cls.l_wrist, cls.l_hand],
                [cls.r_hip, cls.r_knee], [cls.r_knee, cls.r_ankle], [cls.r_ankle, cls.r_heel], [cls.r_heel, cls.r_toetip],
                [cls.l_hip, cls.l_knee], [cls.l_knee, cls.l_ankle], [cls.l_ankle, cls.l_heel], [cls.l_heel, cls.l_toetip]]

    @classmethod
    def names(cls):
        return ["head", "neck",
                "rsho", "relb", "rwri", "rhan",
                "lsho", "lelb", "lwri", "lhan",
                "rhip", "rkne", "rank", "rhee", "rtoe",
                "lhip", "lkne", "lank", "lhee", "ltoe"]

    @classmethod
    def pretty_names(cls):
        return ["Head", "Neck",
                "R. shoulder", "R. elbow", "R. wrist", "R. hand",
                "L. shoulder", "L. elbow", "L. wrist", "L. hand",
                "R. hip", "R. knee", "R. ankle", "R. heel", "R. toetip",
                "L. hip", "L. knee", "L. ankle", "L. heel", "L. toetip"]

    @classmethod
    def flip_pairs(cls):
        return [[2, 6], [3, 7], [4, 8], [5, 9], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19]]

    @classmethod
    def get_num_joints(cls):
        return cls.num_joints
