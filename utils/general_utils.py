# -*- coding: utf-8 -*-
"""
Created on 13.04.23

"""
import os

import cv2
import torch


def get_dict(dic_obj, key, default=None):
    if dic_obj is not None and key in dic_obj and dic_obj[key] is not None:
        return dic_obj[key]
    return default