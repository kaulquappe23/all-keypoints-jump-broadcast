# -*- coding: utf-8 -*-
"""
Created on 23.03.23

"""
import numpy as np

from datasets.general.bodypart_order import BodypartOrder
from datasets.jump.jump_joint_order import JumpJointOrder


class JumpBodypartOrder(BodypartOrder):
    """
    Capsules the order of the body parts in segmentation files and the usage of them
    This class does not include the head in the used body parts, see subclasses. Do not use this class, only the subclasses
    """
    """ Order of segmentation masks:
        1: Torso, 2: Head 3: Right Hand, 4: Left Hand, 5: Left Foot, 6: Right Foot, 7: Upper Leg Right, 8: Upper Leg Left, 9: Lower Leg Right,
        10: Lower Leg Left, 11: Upper Arm Left, 12: Upper Arm Right, 13: Lower Arm Left, 14: Lower Arm Right
    """
    torso = 1
    head = 2
    r_hand = 3
    l_hand = 4
    l_foot = 5
    r_foot = 6
    r_thigh = 7
    l_thigh = 8
    r_lleg = 9
    l_lleg = 10
    l_uarm = 11
    r_uarm = 12
    l_farm = 13
    r_farm = 14

    l_elbow = 15
    r_elbow = 16
    l_knee = 17
    r_knee = 18

    @classmethod
    def get_max_bodypart_num(cls):
        return 14


    @classmethod
    def get_used_bodypart_triples(cls):
        return [(JumpJointOrder.l_shoulder, JumpJointOrder.l_elbow, cls.l_uarm), (JumpJointOrder.l_elbow, JumpJointOrder.l_wrist, cls.l_farm),
                (JumpJointOrder.l_hip, JumpJointOrder.l_knee, cls.l_thigh), (JumpJointOrder.l_knee, JumpJointOrder.l_ankle, cls.l_lleg),
                (JumpJointOrder.r_shoulder, JumpJointOrder.r_elbow, cls.r_uarm), (JumpJointOrder.r_elbow, JumpJointOrder.r_wrist, cls.r_farm),
                (JumpJointOrder.r_hip, JumpJointOrder.r_knee, cls.r_thigh), (JumpJointOrder.r_knee, JumpJointOrder.r_ankle, cls.r_lleg),
                (JumpJointOrder.neck, (JumpJointOrder.r_hip, JumpJointOrder.l_hip), cls.torso),
                (JumpJointOrder.r_toetip, JumpJointOrder.r_heel, cls.r_foot), (JumpJointOrder.l_toetip, JumpJointOrder.l_heel, cls.l_foot),
                (JumpJointOrder.r_wrist, JumpJointOrder.r_hand, cls.r_hand), (JumpJointOrder.l_wrist, JumpJointOrder.l_hand, cls.l_hand),
                (cls.l_uarm, cls.l_farm, cls.l_elbow), (cls.r_uarm, cls.r_farm, cls.r_elbow), (cls.l_thigh, cls.l_lleg, cls.l_knee), (cls.r_thigh, cls.r_lleg, cls.r_knee)
                ]

    @classmethod
    def names(cls):
        return ["background", "torso", "head", "r_hand", "l_hand", "l_foot", "r_foot", "r_thigh", "l_thigh", "r_lleg", "l_lleg", "l_uarm", "r_uarm", "l_farm", "r_farm", "l_elbow", "r_elbow", "l_knee", "r_knee"]

    @classmethod
    def pretty_name_bodypart_order(cls):
        names = cls.names()
        names = [names[bodypart[-1]] for bodypart in cls.get_used_bodypart_triples()]
        idx = [17, 8, 0, 13, 1, 12, 2, 15, 3, 10, 4, 14, 5, 11, 6, 16, 7, 9]
        names = np.asarray(names)[idx].tolist()
        return names

    @classmethod
    def get_colors(cls):
        colors = ["black",  # background
                  "dimgray", # torso
                  "silver",  # head
                  "yellow", # r_hand
                  "green", # l_hand
                  "firebrick", # l_foot
                  "lightskyblue", # r_foot
                  "blue", # r_thigh
                  "orchid", # l_thigh
                  "dodgerblue", # r_lleg
                  "red", # l_lleg
                  "greenyellow", # l_uarm
                  "orange", # r_uarm
                  "limegreen", # l_farm
                  "gold", # r_farm
                  ]
        return colors

    @classmethod
    def get_endpoint_bodyparts(cls):
        return {cls.r_hand, cls.l_hand}

    @classmethod
    def get_endpoint_angle_bodyparts(cls):
        return {}

    @classmethod
    def get_adjacent_bodyparts(cls):
        return {cls.l_elbow, cls.r_elbow, cls.l_knee, cls.r_knee}

    @classmethod
    def keypoints_to_adjacent_bodyparts(cls):
        return  {
                    JumpJointOrder.l_elbow: cls.l_elbow,
                    JumpJointOrder.r_elbow: cls.r_elbow,
                    JumpJointOrder.l_knee: cls.l_knee,
                    JumpJointOrder.r_knee: cls.r_knee
                }

    @classmethod
    def get_bodyparts_with_projection_point_outside_mask(cls):
        return {cls.torso, cls.l_foot, cls.r_foot}

    @classmethod
    def get_bodyparts_with_min_max(cls):
        return {cls.torso: (0.1, 0.9)}

    @classmethod
    def get_left_mapping(cls):
        """ Returns a dictionary mapping left bodyparts to right bodyparts"""
        return {cls.l_hand: cls.r_hand, cls.l_foot: cls.r_foot, cls.l_thigh: cls.r_thigh, cls.l_lleg: cls.r_lleg, cls.l_uarm: cls.r_uarm, cls.l_farm: cls.r_farm,
                cls.l_elbow: cls.r_elbow, cls.l_knee: cls.r_knee}

    @classmethod
    def left_right_pairs(cls):
        return [(cls.l_hand, cls.r_hand), (cls.l_foot, cls.r_foot), (cls.l_thigh, cls.r_thigh), (cls.l_lleg, cls.r_lleg), (cls.l_uarm, cls.r_uarm), (cls.l_farm, cls.r_farm),
                (cls.r_knee, cls.l_knee), (cls.r_elbow, cls.l_elbow)]

    @classmethod
    def head_strategy(cls):
        return BodypartOrder.NO_HEAD


class JumpHeadAngleBodypartOrder(JumpBodypartOrder):

    @classmethod
    def get_used_bodypart_triples(cls):
        return super().get_used_bodypart_triples() + [(JumpJointOrder.neck, JumpJointOrder.head, cls.head)]

    @classmethod
    def get_endpoint_bodyparts(cls):
        return {cls.r_hand, cls.l_hand}

    @classmethod
    def get_endpoint_angle_bodyparts(cls):
        return {cls.head}

    @classmethod
    def head_strategy(cls):
        return BodypartOrder.HEAD_ANGLE


class JumpHeadEndpointBodypartOrder(JumpBodypartOrder):

    @classmethod
    def get_used_bodypart_triples(cls):
        return super().get_used_bodypart_triples() + [(JumpJointOrder.neck, JumpJointOrder.head, cls.head)]

    @classmethod
    def get_endpoint_bodyparts(cls):
        return {cls.r_hand, cls.l_hand, cls.head}

    @classmethod
    def head_strategy(cls):
        return BodypartOrder.HEAD_ENDPOINT