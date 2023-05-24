# -*- coding: utf-8 -*-
"""
Created on 14.07.22

"""
import numpy as np
from matplotlib import colors as pltcolors


class BodypartOrder:

    NO_HEAD = "_no_head"
    HEAD_ENDPOINT = "_head_endpoint"
    HEAD_ANGLE = "_head_angle"

    @classmethod
    def names(cls):
        """
        Get the names of the body parts in the order defined here. 0 is usually the background
        :return:
        """
        raise NotImplementedError

    @classmethod
    def pretty_name_bodypart_order(cls):
        """
        Get the names of the body parts without the background in the order they should be printed in tables etc.
        :return:
        """
        return cls.names()[1:]

    @classmethod
    def get_rgb_colors(cls):
        colors = [pltcolors.to_rgb(c) for c in cls.get_colors()]
        return colors

    @classmethod
    def get_max_bodypart_num(cls):
        """
        Get the maximum id for a bodypart.
        :return:
        """
        raise NotImplementedError

    @classmethod
    def get_colors(cls):
        raise NotImplementedError

    @classmethod
    def get_used_bodypart_triples(cls):
        """
        Get a list of tuples (keypoint1, keypoint2, bodypart) where keypoint1 and keypoint2 are the ids of the keypoints that enclose the body part
        and body part is the id of the body part as defined in this class.
        :return:
        """
        raise NotImplementedError

    @classmethod
    def get_bodypart_to_keypoint_dict(cls):
        """
        Get a dictionary that maps bodypart ids to the ids of the keypoints that enclose the body part.
        :return:
        """
        result_dict = {}
        for keypoint1, keypoint2, bodypart in cls.get_used_bodypart_triples():
            result_dict[bodypart] = (keypoint1, keypoint2)
        return result_dict

    @classmethod
    def get_adjacent_bodyparts(cls):
        """
        Get a set of body part ids that correspond to body areas of adjacent body parts where it is unclear where one bodypart begins and the other ends.
        Examples are knee and elbow. These body parts are in some cases considered as own body parts due to the different creation scheme. The list of ids of these body parts is returned here.
        :return:
        """
        return set()

    @classmethod
    def get_bodyparts_with_adjacent_bodypart(cls):
        """
        Get a dictionary that maps bodypart ids of so-called adjacent body parts to the ids of the bodyparts that are adjacent.
        E.g. the adjacent body part is the knee, then it is mapped to the body part ids of the thigh and the lower leg.
        :return:
        """
        bp_to_adjacent_bp = {}
        for bodypart, (bp1, bp2) in cls.get_bodypart_to_keypoint_dict().items():
            if bodypart in cls.get_adjacent_bodyparts():
                bp_to_adjacent_bp[bp1] = bp2
                bp_to_adjacent_bp[bp2] = bp1
        return bp_to_adjacent_bp

    @classmethod
    def get_adjacent_bodyparts_from_bodypart(cls, bodypart):
        """
        Get the full bodyparts (keypoints and id) of the two body parts that are adjacent to the given bodypart.
        :param bodypart: full bodypart or bodypart id of an adjacent bodypart
        :return:
        """
        bodypart_dict = cls.get_bodypart_to_keypoint_dict()
        if isinstance(bodypart, int):
            bodypart = bodypart_dict[bodypart]
        bp_id1, bp_id2 = bodypart[:2]
        bp10, bp11 = bodypart_dict[bp_id1]
        bodypart1 = (bp10, bp11, bp_id1)
        bp20, bp21 = bodypart_dict[bp_id2]
        bodypart2 = (bp20, bp21, bp_id2)
        return bodypart1, bodypart2

    @classmethod
    def get_bodypart_from_id(cls, bodypart):
        """
        If bodypart is and int, the corresponding keypoint ids are returned. If the bodypart is a tuple or a list, the first two entries of the tuple are considered as bodypart ids of
        an adjacent body part and the corresponding bodypart triples for both bodyparts are returned.
        :param bodypart:
        :return:
        """
        bodypart_dict = cls.get_bodypart_to_keypoint_dict()
        if isinstance(bodypart, int):
            bodypart = bodypart_dict[bodypart]
        bp_id1, bp_id2 = bodypart[:2]
        bp10, bp11 = bodypart_dict[bp_id1]
        bodypart1 = (bp10, bp11, bp_id1)
        bp20, bp21 = bodypart_dict[bp_id2]
        bodypart2 = (bp20, bp21, bp_id2)
        return bodypart1, bodypart2

    def get_bodypart(self, keypoint_indices):
        for bodypart in self.get_used_bodypart_triples():
            bp = np.asarray(bodypart)[:2]
            if keypoint_indices[0] in bp and keypoint_indices[1] in bp:
                return bodypart

    @classmethod
    def get_endpoint_bodyparts(cls):
        """
        Get a set of body parts that are endpoints of the human body without a keypoint located at the end of the body part -> these body parts have only one enclosing keypoint
        :return:
        """
        return set()

    @classmethod
    def get_endpoint_angle_bodyparts(cls):
        """
        Similar to get_endpoint_bodyparts, but these endpoints are not included via a generated enclosing keypoint, but via an angle.
        :return:
        """
        return set()

    @classmethod
    def get_bps_without_central_projection(cls):
        """
        Get a set of body part ids for which no central projection is used, meaning that there is not a difference between left, central point and right, but only between left and right.
        :return:
        """
        return set()

    @classmethod
    def keypoints_to_adjacent_bodyparts(cls):
        """
        Adjacent body parts are located around keypoints. This method returns a dictionary that maps the ids of the keypoints to the ids of the adjacent body parts.
        :return:
        """
        return {}

    @classmethod
    def get_bodyparts_with_projection_point_outside_mask(cls):
        """
        Get the set of body parts for which it is allowed that projection points are outside the segmentation mask.
        :return:
        """
        return {}

    @classmethod
    def get_bodyparts_with_min_max(cls):
        """
        Get a dictionary that maps bodypart ids to a tuple (min, max) where min and max are the minimum and maximum
        allowed percentage for the projection points
        """
        return {}

    @classmethod
    def left_right_pairs(cls):
        """
        Get tuples with the ids of corresponding left and right body parts.
        :return:
        """
        raise NotImplementedError

    @classmethod
    def head_strategy(cls):
        """
        Returns the strategy to encode keypoints on the head.
        :return:
        """
        raise NotImplementedError


class DenseposeOrder(BodypartOrder):
    torso = [1, 2]
    r_hand = 3
    l_hand = 4
    l_foot = 5
    r_foot = 6
    r_thigh = [7, 9]
    l_thigh = [8, 10]
    r_lleg = [11, 13]
    l_lleg = [12, 14]
    l_uarm = [15, 17]
    r_uarm = [16, 18]
    l_farm = [19, 21]
    r_farm = [20, 22]
    head = [23, 24]

    @classmethod
    def get_max_bodypart_num(cls):
        return 24

    @classmethod
    def get_colors(cls):
        colors = ["black",
                  "silver", "silver",                                       # torso
                  "yellow",                                                 # right hand
                  "blue",                                                   # left hand
                  "dodgerblue",                                             # left foot
                  "orchid",                                                 # right foot
                  "red", "green", "firebrick", "mediumseagreen",            # thigh, r, l, r, l
                  "lightsalmon", "limegreen", "orange", "greenyellow",      # lower leg, r, l, r, l
                  "aquamarine", "darkviolet", "darkturquoise", "blueviolet",# upper arm, l, r, l, r
                  "lightskyblue", "gold", "deepskyblue", "khaki",           # forearm, l, r, l, r
                  "dimgrey", "darkblue"                                     # head
                  ]
        return colors


