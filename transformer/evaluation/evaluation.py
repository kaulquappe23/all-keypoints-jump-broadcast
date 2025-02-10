# -*- coding: utf-8 -*-
"""
Created on 04.05.23

"""
import gc
import os
import pickle

import numpy as np
import torch

from datasets.general.dataset_utils import get_dataset_from_config, get_dataloader
from metrics.pck import eval_pck_array
from metrics.thickness_metric import thickness_percentage_differences, thickness_metric, format_results, \
    format_thickness_bodyparts
from paths import GeneralLoc
from transformer.data.transformer_dataset import TransformerDataset
from transformer.execution.inference import inference_data_loader
from transformer.model import token_pose
from transformer.model.token_pose import convert_state_dict


def general_eval_pipeline(
        inference_again, distances_again, weights_file, title, config=None, model_name=None, subset="test", batch_size=None
        ):
    """
    Evaluate a model with PCK and thickness metrics
    :param batch_size: Overwrite the default batch size for the evaluation
    :param inference_again: calculate the inference again or use saved file
    :param distances_again: calculate the distances again or use saved file
    :param weights_file: weights file to load for the model
    :param title: title of the run
    :param config: configuration file, if None, loaded from the checkpoint
    :param model_name: model name for the checkpoint, None if only weights are in the checkpoint
    :param subset: subset to test on
    :return:
    """
    if config is None:
        settings = torch.load(weights_file, weights_only=False)
        config = settings["config"]

    if "20" in weights_file:
        run = weights_file.split(os.sep)[-2]
        step = weights_file.split(os.sep)[-1][5:-8]
    else:
        run = weights_file.split(os.sep)[-1]
        step = ""

    print("\n------------------------------ {} ---------------------------------\n".format(title))
    print("Run: {}, Step: {}".format(run, step))

    save_pred_path = os.path.join(GeneralLoc.log_path, config.NAME, "eval",
                                  "preds_{}_{}_{}_{}.pkl".format(config.NAME, run, step, subset))
    save_dist_path = os.path.join(GeneralLoc.log_path, config.NAME, "eval",
                                  "dist_{}_{}_{}_{}.pkl".format(config.NAME, run, step, subset))
    os.makedirs(os.path.join(GeneralLoc.log_path, config.NAME, "eval"), exist_ok=True)

    if inference_again:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_file, map_location=device, weights_only=False)
        if config is None:
            config = checkpoint["config"]
        if model_name is not None:
            checkpoint = checkpoint[model_name]

        if batch_size is not None:
            config.SINGLE_GPU_BATCH_SIZE = batch_size

        model = token_pose.get_token_pose_net(config, load_weights_from_config=False)
        model.load_state_dict(convert_state_dict(checkpoint, config.TOKEN_EMBEDDING))
        model = model.to(device)

        dataset = TransformerDataset(config, params={"subset": subset, "arbitrary_points": True})

        data_loader = get_dataloader(dataset, is_train=False, batch_size=config.SINGLE_GPU_BATCH_SIZE, num_workers=config.WORKERS)
        res = inference_data_loader(data_loader, dataset, model, use_flip=True)
        with open(save_pred_path, "wb") as f:
            pickle.dump(res, f)
        model = model.to("cpu")  # free up GPU memory
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    with open(save_pred_path, "rb") as f:
        res = pickle.load(f)
        preds, boxes, image_ids, additional_output = res
        annos = additional_output["original_annotations"]
        thick_vec = None if "thickness_vector" not in additional_output else additional_output["thickness_vector"]

        pck_metric_res = [eval_pck_array(preds, annos, config.JOINT_ORDER, pck_thresholds=[0.1, 0.05]),
                          eval_pck_array(preds, annos, config.JOINT_ORDER, pck_thresholds=[0.1, 0.05],
                                         use_indices=[i for i in range(config.JOINT_ORDER.get_num_joints())])]

        if distances_again:
            bodypart_order = config.BODYPART_ORDER
            dataset = get_dataset_from_config(config, subset, params={"arbitrary_points": True})
            fixed_annos = annos[:, :dataset.joint_order.get_num_joints()]
            annos = annos[:, dataset.joint_order.get_num_joints():]
            preds = preds[:, dataset.joint_order.get_num_joints():]
            thickness_res = thickness_percentage_differences(preds, annos, fixed_annos,
                                                             bodypart_order, dataset.load_bodypart_mask, dataset.bboxes,
                                                             image_ids, thick_vec,
                                                             anchors=getattr(dataset, "anchors", {}),
                                                             return_all_distances=True
                                                             )
            with open(save_dist_path, "wb") as f:
                pickle.dump(thickness_res, f)
        with open(save_dist_path, "rb") as f:
            thickness_res = pickle.load(f)

        mean, std, pct = thickness_metric(thickness_res[0])
        totals = format_results(pck_metric_res, (mean, pct)).split("\n")

        num_bodyparts = len(config.BODYPART_ORDER.get_used_bodypart_triples())
        distances = thickness_res[0]
        assert len(distances) % num_bodyparts == 0
        distances = np.asarray(distances)
        distances = distances.reshape(len(annos), num_bodyparts, -1)

        names = config.BODYPART_ORDER.pretty_name_bodypart_order()
        per_kp = (format_thickness_bodyparts(distances, names, config.BODYPART_ORDER)).split("\n")
        header = totals[0] + per_kp[0]
        text = totals[1] + per_kp[1]
        print(header + "\n" + text)

    print("\n---------------------------------------------------------------\n\n")
