# -*- coding: utf-8 -*-
"""
Created on 06.11.20

"""


def convert_hrnet_state_dict(state_dict):
    """
    This is necessary as I modified the HRNet Code for me to be more readable
    :param state_dict:
    :return:
    """
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("conv") or key.startswith("bn") or key.startswith("layer1"):
            state_dict["1.stem." + key] = state_dict.pop(key)
        elif key.startswith("stage"):
            state_dict["1." + key[:7] + "stage." + key[7:]] = state_dict.pop(key)
        elif key.startswith("transition"):
            state_dict["1." + key] = state_dict.pop(key)
        elif key.startswith("final_layers.0"):
            state_dict["1.final_layer_non_deconv" + key[14:]] = state_dict.pop(key)
        elif key.startswith("final_layers.1"):
            state_dict["1.final_layer_after_deconv" + key[14:]] = state_dict.pop(key)
        elif key.startswith("deconv_layers.0"):
            state_dict["1.deconv_layer.deconv_layer" + key[15:]] = state_dict.pop(key)
    return state_dict


def get_hrnet_key(key):
    if key.startswith("conv") or key.startswith("bn") or key.startswith("layer1"):
        return "1.stem." + key
    elif key.startswith("stage"):
        return "1." + key[:7] + "stage." + key[7:]
    elif key.startswith("transition"):
        return "1." + key
    elif key.startswith("final_layers.0"):
        return "1.final_layer_non_deconv" + key[14:]
    elif key.startswith("final_layers.1"):
        return "1.final_layer_after_deconv" + key[14:]
    elif key.startswith("deconv_layers.0"):
        return "1.deconv_layer.deconv_layer" + key[15:]
    raise RuntimeError("key not known {}".format(key))
