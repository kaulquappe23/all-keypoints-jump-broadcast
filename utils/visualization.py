# -*- coding: utf-8 -*-
"""
Created on 13.04.23

"""
import numpy as np


def cutout_image(annos, image, cutout_ratio, scale=1):
    """
    Cut out the image according to the annotations, adding 20% margin to all sides
    :param annos: num_keypoints x 2 or 3
    :param image:
    :param cutout_ratio: width to height ratio that is desired
    :param scale: scale coordinates
    :return:
    """
    x_max = int(round(np.max(annos[:, 0])))
    x_min = int(round(np.min(annos[:, 0])))
    y_max = int(round(np.max(annos[:, 1])))
    y_min = int(round(np.min(annos[:, 1])))
    w = x_max - x_min
    h = y_max - y_min
    x_max = min(x_max + int(0.2 * w), image.shape[1] * 1 / scale)
    x_min = max(x_min - int(0.2 * w), 0)
    y_max = min(y_max + int(0.2 * h), image.shape[0] * 1 / scale)
    y_min = max(y_min - int(0.2 * h), 0)
    w = x_max - x_min
    h = y_max - y_min

    ratio_w, ratio_h = cutout_ratio
    if w / ratio_w > h / ratio_h:
        h_final = int(w / ratio_w * ratio_h)
        offset = (h_final - h) // 2
        if y_min - offset < 0:
            offset = 2 * offset - y_min
            y_min = 0
        else:
            y_min = y_min - offset
        y_max = min(y_max + offset, image.shape[0] * 1 / scale)
    elif w / ratio_w < h / ratio_h:
        w_final = int(h / ratio_h * ratio_w)
        offset = (w_final - w) // 2
        if x_min - offset < 0:
            offset = 2 * offset - x_min
            x_min = 0
        else:
            x_min = x_min - offset
        x_max = min(x_max + offset, image.shape[1] * 1 / scale)
    y_min = int(round(y_min * scale))
    y_max = int(round(y_max * scale)) + 1
    x_min = int(round(x_min * scale))
    x_max = int(round(x_max * scale)) + 1
    image = image[y_min: y_max, x_min: x_max]
    return image