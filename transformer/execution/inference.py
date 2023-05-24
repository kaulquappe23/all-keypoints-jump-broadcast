# -*- coding: utf-8 -*-
"""
Created on 27.08.21

"""
import copy

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from transformer.data.transformer_dataset import TransformerDatasetWrapper
from transformer.execution.pre_postprocessing import get_final_preds, get_affine_transform, flip_back, flip_additional_vectors, \
    additional_vectors_to_list


def inference_data_loader(data_loader, dataset: TransformerDatasetWrapper, model, use_flip=True):
    """
    Calculate predictions for all images in the data loader
    :param data_loader: iterator over the dataset
    :param dataset: the dataset
    :param model: model to use for inference
    :param use_flip: use flip test or not
    :return: predictions, bounding boxes, image ids and additional output dict, can contain representations,
    original annotations, transformed annotations
    """
    # switch to evaluate mode
    model.eval()

    num_samples = len(dataset)
    num_joints = dataset.dataset.get_num_keypoints()
    all_preds = np.zeros(
            (num_samples, num_joints, 3),
            dtype=np.float32
            )
    all_boxes = np.zeros((num_samples, 6))
    image_ids = []
    idx = 0
    joints = []
    representations = {}
    original_annotations = []

    with torch.no_grad():
        for i, (images, target, target_weight, meta) in enumerate(tqdm(data_loader)):
            # compute output

            additional_vectors = meta["representations"] if "representations" in meta else None
            orig_annos = meta['non_transformed_annotations'].numpy() if 'non_transformed_annotations' in meta else None
            if additional_vectors is not None:
                if len(representations) == 0:
                    for key in additional_vectors:
                        representations[key] = []
                for key, value in additional_vectors.items():
                    representations[key].append(copy.deepcopy(value.cpu().numpy()))
            if orig_annos is not None:
                original_annotations.append(copy.deepcopy(orig_annos))

            output = get_model_output(model, images, additional_vectors, use_flip, dataset.dataset.joint_order.flip_pairs())

            num_images = images.size(0)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            preds, maxvals = copy.deepcopy(get_final_preds(output.clone().cpu().numpy(), c, s))

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :num_joints, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals[:, :num_joints]
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = 1

            image_ids.extend(copy.deepcopy(meta['image_id']))
            if "transformed_annotations" in meta:
                joints.append(copy.deepcopy(meta["transformed_annotations"].numpy()))

            idx += num_images

        additional_output = {
                "transformed_annotations": np.concatenate(joints).reshape(-1, num_joints, 3),
                "original_annotations":    np.concatenate(original_annotations).reshape(-1, num_joints, 3)
                }
        for key, value in representations.items():
            additional_output[key] = np.concatenate(value).reshape(-1, num_joints, value[0].shape[-1])

        return all_preds, all_boxes, image_ids, additional_output


def inference_single_image(
        image, center, scale, model, additional_vectors=None, use_flip=False, flip_pairs=None, with_confidence=False
        ):
    """
    Calculate model output for a single image
    :param image: loaded image, will be normalized and converted to tensor, so a numpy array is expected
    :param center: center of image, as returned from prepare_single_image
    :param scale: scale of image, as returned from prepare_single_image
    :param model: model for inference_data_loader
    :param additional_vectors: norm_pose, keypoint, thickness, angle vectors as needed for the model
    :param use_flip: use flip test
    :param flip_pairs: flip pairs for flip test
    :param with_confidence: return confidences for keypoints or not
    :return:
    """
    model.eval()

    with torch.no_grad():
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
        image = transforms.ToTensor()(image)
        image = normalize(image)
        image = image[None, :, :, :]

        output = get_model_output(model, image, additional_vectors, use_flip=use_flip, flip_pairs=flip_pairs)

        preds, maxvals = get_final_preds(output.clone().cpu().numpy(), center, scale)

        if with_confidence:
            preds = np.concatenate([preds, maxvals], axis=2)

    return output[0], preds


def get_model_output(model, image_input, additional_vectors, use_flip=False, flip_pairs=None):
    """
    Get output from the model, with or without flip test
    :param model: pose model
    :param image_input: image batch
    :param additional_vectors: keypoint, norm pose, thickness, angle vector as matching the model
    :param use_flip: use flip test or not
    :param flip_pairs: flip pairs if flip test is used
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)

    additional_vectors_list = additional_vectors_to_list(additional_vectors, device=device)
    output = model(image_input, additional_vectors=additional_vectors_list)

    if use_flip:
        input_flipped = np.flip(image_input.cpu().numpy(), 3).copy()
        input_flipped = torch.from_numpy(input_flipped)
        if torch.cuda.is_available():
            input_flipped = input_flipped.cuda()

        flipped_vectors = flip_additional_vectors(additional_vectors, flip_pairs)
        additional_vectors_list = additional_vectors_to_list(flipped_vectors, device=device)
        outputs_flipped = model(input_flipped, additional_vectors=additional_vectors_list)

        if isinstance(outputs_flipped, list):
            output_flipped = outputs_flipped[-1]
        else:
            output_flipped = outputs_flipped

        output_flipped = output_flipped.cpu().numpy()
        if 'keypoint_vector' not in additional_vectors and 'norm_pose' not in additional_vectors:  # otherwise, the "query" is
            # already updated
            output_flipped = flip_back(output_flipped, flip_pairs)
        else:
            output_flipped = flip_back(output_flipped, [])

        output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

        output = (output + output_flipped) * 0.5

    return output


def prepare_single_image(image_path, bbox):
    """
    Prepare a single image for inference_data_loader
    :param image_path: path to load image
    :param bbox: x_min, y_min, width, height of image crop
    :return:
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if not x2 >= x1 and y2 >= y1:
        raise RuntimeError
    clean_box = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    x, y, w, h = clean_box

    center = np.zeros((2), dtype=np.float32)
    center[0] = x + (w - 1) * 0.5
    center[1] = y + (h - 1) * 0.5

    aspect_ratio = 192.0 / 256
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
            [w * 1.0 / 200, h * 1.0 / 200],
            dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    r = 0
    trans = get_affine_transform(center, scale, r, (192, 256))

    image = cv2.warpAffine(
            image,
            trans,
            (192, 256),
            flags=cv2.INTER_LINEAR)
    return image, center, scale
