# -*- coding: utf-8 -*-
"""
Created on 31.03.23

"""
import numpy as np


def adjust_color(color, ratio):
    if ratio == 0:
        return color
    else:
        color = np.asarray(color)
        white = np.asarray([255, 255, 255])
        diff = white - color
        adjust = diff * ratio
        return color + adjust