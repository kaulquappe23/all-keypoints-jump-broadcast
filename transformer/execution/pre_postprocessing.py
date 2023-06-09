# -*- coding: utf-8 -*-
"""
Created on 13.04.23

"""
import cv2
import numpy as np
import torch


def additional_vectors_to_list(additional_vectors, device=None):
    """
    Convert additional vectors to list
    :param device: move vectors to specific device if desired. This method works also for numpy arrays if device is None
    :param additional_vectors: dictionary containing additional vectors, could be 'keypoint_vector', 'thickness_vector', 'angle_vector', 'norm_pose_vector'
    :return:
    """
    additional_vectors_list = []
    if 'keypoint_vector' in additional_vectors:
        additional_vectors_list.append(additional_vectors['keypoint_vector'])
        if 'thickness_vector' in additional_vectors:
            additional_vectors_list.append(additional_vectors['thickness_vector'])
        if 'angle_vector' in additional_vectors:
            additional_vectors_list.append(additional_vectors['angle_vector'])
    elif 'norm_pose_vector' in additional_vectors:
        additional_vectors_list.append(additional_vectors['norm_pose_vector'])
    if device is not None:
        additional_vectors_list = [x.to(device) for x in additional_vectors_list]
    return additional_vectors_list


def flip_additional_vectors(additional_vectors, flip_pairs):
    """
    Prepare additional vectors for flipped image. Works with batched and non-batched vectors.
    :param additional_vectors: dictionary containing additional vectors, could be 'keypoint_vector', 'thickness_vector', 'angle_vector', 'norm_pose_vector'
    :param flip_pairs:
    :return:
    """
    if 'keypoint_vector' in additional_vectors:
        tensor = isinstance(additional_vectors['keypoint_vector'], torch.Tensor)
        if tensor:
            device = additional_vectors['keypoint_vector'].device
            additional_vectors['keypoint_vector'] = additional_vectors['keypoint_vector'].cpu().numpy()
        temp_vec = fliplr_keypoint_vectors(additional_vectors['keypoint_vector'], flip_pairs)
        if tensor:
            temp_vec = torch.from_numpy(temp_vec).to(device)
        additional_vectors['keypoint_vector'] = temp_vec
    if 'thickness_vector' in additional_vectors:
        if 'angle_vector' not in additional_vectors:
            if len(additional_vectors['thickness_vector'].shape) == 2:
                additional_vectors['thickness_vector'][:, [0, 2]] = additional_vectors['thickness_vector'][:, [2, 0]]
            else:
                additional_vectors['thickness_vector'][:, :, [0, 2]] = additional_vectors['thickness_vector'][:, :, [2, 0]]
        else:
            if len(additional_vectors['thickness_vector'].shape) == 2:
                to_swap = additional_vectors['thickness_vector'][(additional_vectors['angle_vector'] == 0)[:, 0]]
                to_swap[:, [0, 2]] = to_swap[:, [2, 0]]
                additional_vectors['thickness_vector'][(additional_vectors['angle_vector'] == 0)[:, 0]] = to_swap
            else:
                to_swap = additional_vectors['thickness_vector'][(additional_vectors['angle_vector'] == 0)[:, :, 0]]
                to_swap[:, [0, 2]] = to_swap[:, [2, 0]]
                additional_vectors['thickness_vector'][(additional_vectors['angle_vector'] == 0)[:, :, 0]] = to_swap
    if 'angle_vector' in additional_vectors:
        angle_vector = additional_vectors['angle_vector']
        if isinstance(angle_vector, torch.Tensor):
            angle_vector[torch.logical_and(angle_vector != 0, angle_vector != 1)] = 1 - angle_vector[torch.logical_and(angle_vector != 0, angle_vector != 1)]
        else:
            angle_vector[np.logical_and(angle_vector != 0, angle_vector != 1)] = 1 - angle_vector[np.logical_and(angle_vector != 0, angle_vector != 1)]
        additional_vectors['angle_vector'] = angle_vector
    if 'norm_pose_vector' in additional_vectors:
        if len(additional_vectors['norm_pose_vector'].shape) == 2:
            additional_vectors['norm_pose_vector'][:, 0] = 1 - additional_vectors['norm_pose_vector'][:, 0]
        else:
            additional_vectors['norm_pose_vector'][:, :, 0] = 1 - additional_vectors['norm_pose_vector'][:, :, 0]
    return additional_vectors

def fliplr_keypoint_vectors(keypoint_vector, matched_parts):
    """
    flip left right for keypoint vectors
    """

    if keypoint_vector.ndim == 2:
        for pair in matched_parts:
            keypoint_vector[:, pair[0]], keypoint_vector[:, pair[1]] = keypoint_vector[:, pair[1]], keypoint_vector[:, pair[0]].copy()
    else:  # batched version
        for pair in matched_parts:
            keypoint_vector[:, :, pair[0]], keypoint_vector[:, :, pair[1]] = keypoint_vector[:, :, pair[1]], keypoint_vector[:, :, pair[0]].copy()

    return keypoint_vector

def fliplr_joints(joints, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def flip_back(output_flipped, matched_parts):
    """
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    """
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i,j] = dr[border: -border, border: -border].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
    return hm


def get_final_preds(hm, center, scale, rot=0, flip_width=None):
    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    if True:  # config.TEST.POST_PROCESS:
        hm = gaussian_blur(hm, 11)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n,p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height], 0 if isinstance(rot, int) and rot == 0 else rot[i]
        )
        if flip_width is not None:
            preds[i, :, 0] = flip_width[i] - preds[i, :, 0] - 1

    return preds, maxvals


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (src_w-1) * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
