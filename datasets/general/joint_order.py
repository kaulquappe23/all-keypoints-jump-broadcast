# -*- coding: utf-8 -*-
"""
Created on 23.03.23

"""


class JointOrder:
    ref_length_indices = None
    fallback_ref_length_indices = None

    @classmethod
    def get_num_joints(cls):
        raise NotImplementedError

    @classmethod
    def indices(cls):
        raise NotImplementedError

    @classmethod
    def bodypart_indices(cls):
        raise NotImplementedError

    @classmethod
    def names(cls):
        raise NotImplementedError

    @classmethod
    def pretty_names(cls):
        raise NotImplementedError

    @classmethod
    def flip_pairs(cls):
        raise NotImplementedError

    def __init__(self):
        pass
