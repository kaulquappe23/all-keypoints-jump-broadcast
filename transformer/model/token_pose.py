# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao
# Modified by Yanjie Li
# Modified by Katja Ludwig
# ------------------------------------------------------------------------------

import os
import re
from functools import partial

import einops as einops
import torch
from torch import nn

from hrnet.model.model_converter import get_hrnet_key
from hrnet.model.pose_higher_hrnet import HRNetW32Backbone
from transformer.model.token_embedding import TokenEmbedding
from transformer.model.vision_transformer import VisionTransformer


class TokenPoseNet(nn.Module):

    def __init__(
            self, keypoint_transform_config, input_size=(256 // 4, 192 // 4, 32), patch_size=(4, 3), embed_dim=192, depth=12,
            num_heads=8, mlp_ratio=3, heatmap_size=(64, 48), pos_encoding="sine", cnn=None,
            correlate_everything=False
            ):
        """

        :param keypoint_transform_config: Configuration dict for the keypoint embedding. Needs to contain 'num_joints',
        'representation_type' and if necessary 'num_layers' and 'concat_after_layers'. See class TokenEmbedding for more details.
        :param input_size: size of the input for the transformer - important, the default value is the output size of the
        HRNetW32 backbone
        :param patch_size: feature or image patch size
        :param embed_dim: token length
        :param heatmap_size: output heatmap size
        :param pos_encoding: sine or learnable
        :param cnn: CNN backbone for feature extraction before transformer
        :param correlate_everything: correlate the tokens with the image patches or not
        """
        super().__init__()

        self.heatmap_size = heatmap_size if isinstance(heatmap_size, tuple) else (heatmap_size, heatmap_size)
        self.cnn = nn.Identity() if cnn is None else cnn
        self.keypoint_transform_config = keypoint_transform_config
        token_embedding = TokenEmbedding(keypoint_transform_config['num_joints'], embed_dim, keypoint_transform_config)
        self.description = token_embedding.description

        self.vision_transformer = VisionTransformer(
                in_feature_size=input_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                mlp_ratio=mlp_ratio, scale_head=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), pos_encoding=pos_encoding,
                num_tokens=keypoint_transform_config['num_joints'], token_embedding=token_embedding,
                correlate_everything=correlate_everything
                )

        hidden_heatmap_dim = heatmap_size[0] * heatmap_size[1] // 8
        heatmap_dim = heatmap_size[0] * heatmap_size[1]

        self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_heatmap_dim),
                nn.LayerNorm(hidden_heatmap_dim),
                nn.Linear(hidden_heatmap_dim, heatmap_dim)
                ) if embed_dim <= hidden_heatmap_dim * 0.5 else nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, heatmap_dim)
                )

    def forward(self, x, additional_vectors=None):
        x = self.cnn(x)
        x = self.vision_transformer(x, additional_vectors=additional_vectors)
        x = self.mlp_head(x)
        x = einops.rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x

    def init_weights(self, pretrained_transformer=None, model_name=None, load_head=True, pretrained_hrnet=None, verbose=True):
        """
        Load weights for the model
        :param pretrained_transformer: file with weights containing the whole transformer (if necessary, including the HRNet
        backbone), the head does not need to match (if not, set load_head=False)
        :param model_name: if multiple models are saved in the file, specify which one to load (if not set, "state_dict" is
        used, if the given file contains a dictionary)
        :param load_head: load the head weights or not
        :param pretrained_hrnet: file with hrnet weights to initialize the feature extractor ONLY. This param is ignored if
        pretrained_transformer is set.
        :param verbose: Print information about the loaded weights
        :return:
        """
        if pretrained_transformer is not None and os.path.isfile(pretrained_transformer):
            pretrained_state_dict = torch.load(pretrained_transformer)
            if verbose:
                print("Loaded {}".format(pretrained_transformer))
            if "state_dict" in pretrained_state_dict and model_name is None:
                model_name = "state_dict"
            if model_name is not None:
                pretrained_state_dict = pretrained_state_dict[model_name]
            # Convert keys, as I modified the names slightly
            pretrained_state_dict = convert_state_dict(pretrained_state_dict, self.keypoint_transform_config)
            if load_head:
                # load weights strictly
                res = self.load_state_dict(pretrained_state_dict)
            else:
                # remove all head weights
                for key in list(pretrained_state_dict.keys()):
                    if key.startswith("vision_transformer.token"):
                        pretrained_state_dict.pop(key)
                    if key.startswith("vision_transformer.token_embed"):
                        pretrained_state_dict.pop(key)
                    if key.startswith("mlp_head"):
                        pretrained_state_dict.pop(key)
                # load backbone weights
                res = self.load_state_dict(pretrained_state_dict, strict=False)
                if verbose:
                    print("IGNORED HEAD")
            if verbose:
                print("LOADED WEIGHTS FROM {}, MISSING KEYS: {}".format(pretrained_transformer, len(res[0])))
        elif pretrained_hrnet is not None and os.path.isfile(pretrained_hrnet):
            self.cnn.init_weights(pretrained_hrnet)
        elif pretrained_transformer is not None and not os.path.isfile(
                pretrained_transformer
                ) or pretrained_hrnet is not None and not os.path.isfile(pretrained_hrnet):
            raise RuntimeError("Initialization failed. Check if weights file really exists!")

    def freeze_backbone(self):
        """
        Freeze backbone weights of HRNet and not of the last layers (the head)
        @return:
        """
        if isinstance(self.cnn, nn.Identity):
            raise RuntimeError("Backbone should be frozen, but TokenPoseNet has no CNN backbone")
        for name, param in self.cnn.named_parameters():
            param.requires_grad = False
        for name, param in self.cnn.named_buffers():
            param.requires_grad = False


def get_token_pose_net(cfg, load_weights_from_config, verbose=True):
    """
    Get the TokenPoseNet model from the config file
    :param cfg: Configuration file for the network, containing information about the representation of the keypoint tokens,
    model input/output size, pretraining, etc.
    :param load_weights_from_config: If set, the weights are loaded from the pretrained file defined in the config file.
    Otherwise, the weights are not initialized
    :param verbose:
    :return:
    """
    num_joints = cfg.JOINT_ORDER.get_num_joints()
    if getattr(cfg, "GENERATE_KEYPOINTS", None) is not None:
        keypoint_transform_config = cfg.TOKEN_EMBEDDING
        keypoint_transform_config["num_joints"] = num_joints
    else:
        keypoint_transform_config = {
                'representation_type': TokenEmbedding.LEARNABLE_TOKEN,
                'num_joints':          num_joints
                }
    correlate_tokens = getattr(cfg, "KEYPOINT_TOKEN_ATTENTION", False)
    if hasattr(cfg, "CNN") and cfg.CNN == "hrnet_stage3":
        patch_size = getattr(cfg, "PATCH_SIZE", (4, 3))
        input_size = (cfg.INPUT_SIZE[1] // 4, cfg.INPUT_SIZE[0] // 4, 32)
        heatmap_size = (cfg.OUTPUT_SIZE[0][1], cfg.OUTPUT_SIZE[0][0])

        model = TokenPoseNet(
                input_size=input_size, pos_encoding=cfg.POS_ENCODING, embed_dim=cfg.EMBED_SIZE, patch_size=patch_size,
                heatmap_size=heatmap_size,
                keypoint_transform_config=keypoint_transform_config, cnn=HRNetW32Backbone(num_stages=3),
                correlate_everything=correlate_tokens
                )
        if verbose:
            print("Using HRNet stage 3 backbone")
    else:
        model = TokenPoseNet(
                patch_size=(16, 12), embed_dim=cfg.EMBED_SIZE, depth=12, num_heads=16,
                keypoint_transform_config=keypoint_transform_config, correlate_everything=correlate_tokens
                )
        if verbose:
            print("Using pure transformer net")
    if load_weights_from_config:
        model_name = None if not hasattr(cfg, "PRETRAINED_MODEL_NAME") else cfg.PRETRAINED_MODEL_NAME
        load_head = False if not hasattr(cfg, "PRETRAINED_HEAD") else cfg.PRETRAINED_HEAD
        pretrained_hrnet = None if not hasattr(cfg, "PRETRAINED_HRNET") else cfg.PRETRAINED_HRNET
        model.init_weights(cfg.PRETRAINED_FILE, model_name=model_name, load_head=load_head, pretrained_hrnet=pretrained_hrnet)
    return model


def convert_state_dict(state_dict, keypoint_transform_config=None):
    """
    As I modified the code of the transformer and the HRNet backbone, the names of the weights need to be adjusted.
    :param keypoint_transform_config: Containing information about how the keypoint tokens are created. Dict with keys
    "representation_type" (referring to the options in the class TokenEmbedding), and, if
    applicable, "num_layers" (for the number of layers used in the embedding MLP) and "concat_after_layers" (for the number of
    layers after which the different parts of the keypoint tokens are concatenated)
    :param state_dict: loaded state dict
    :return: state dict with converted names
    """
    keys = list(state_dict.keys())
    num_concat_after = 1
    for key in keys:
        if key.startswith("module.pre_feature"):
            hrnet_key = get_hrnet_key(key[len("module.pre_feature") + 1:])[2:]
            state_dict['cnn.' + hrnet_key] = state_dict.pop(key)
        elif key == 'module.transformer.keypoint_token':
            state_dict['vision_transformer.token'] = state_dict.pop(key)
        elif key == 'module.transformer.pos_embedding':
            state_dict['vision_transformer.pos_encoding'] = state_dict.pop(key)
        elif key.startswith('module.transformer.patch_to_embedding'):
            state_dict[
                'vision_transformer.patch_embed.proj' + key[len('module.transformer.patch_to_embedding'):]] = state_dict.pop(key)
        elif key.startswith('module.transformer.mlp_head'):
            state_dict['mlp_head' + key[len('module.transformer.mlp_head'):]] = state_dict.pop(key)
        elif key.startswith('vision_transformer.token_embed.'):
            state_dict['vision_transformer.token.0' + key[len('vision_transformer.token_embed'):]] = state_dict.pop(key)
        elif key.startswith('vision_transformer.thickness_embed'):
            state_dict['vision_transformer.token.1' + key[len('vision_transformer.thickness_embed'):]] = state_dict.pop(key)
        elif key.startswith('vision_transformer.token.'):  # conversion from weights before refactoring
            representation_type = keypoint_transform_config["representation_type"]
            assert representation_type != TokenEmbedding.LEARNABLE_TOKEN
            if representation_type == TokenEmbedding.NORM_POSE:
                if keypoint_transform_config["num_layers"] == 1:
                    state_dict[
                        'vision_transformer.token_embedding.embedding' + key[len('vision_transformer.token'):]] = state_dict.pop(
                            key
                            )
                else:
                    state_dict['vision_transformer.token_embedding.embedding' + key[len(
                            'vision_transformer.token'
                            ) + 2:]] = state_dict.pop(key)
            else:
                if keypoint_transform_config["num_layers"] == 1:
                    new_key = 'vision_transformer.token_embedding.embeddings.before_concat' + key[
                                                                                              len('vision_transformer.token'):len(
                                                                                                      'vision_transformer.token'
                                                                                                      ) + 2] + '.0' + key[len(
                            'vision_transformer.token'
                            ) + 2:]
                    state_dict[new_key] = state_dict.pop(key)
                elif "concat_after_layers" in keypoint_transform_config and keypoint_transform_config["num_layers"] == \
                        keypoint_transform_config["concat_after_layers"]:
                    new_key = 'vision_transformer.token_embedding.embeddings.before_concat' + key[
                                                                                              len('vision_transformer.token'):]
                    state_dict[new_key] = state_dict.pop(key)
                else:
                    num_streams = representation_type - 1
                    num_stream = int(key[len('vision_transformer.token.'):len('vision_transformer.token.') + 1])
                    if "concat_after_layers" in keypoint_transform_config and keypoint_transform_config[
                        "concat_after_layers"] == 0:
                        new_key = 'vision_transformer.token_embedding.embeddings.after_concat.' + str(
                                (num_concat_after - 1) - (num_concat_after - 1) % 2
                                ) + key[len('vision_transformer.token.x'):]
                        num_concat_after += 1
                    elif num_stream < num_streams:
                        new_key = 'vision_transformer.token_embedding.embeddings.before_concat.' + key[len(
                                'vision_transformer.token.'
                                ):]
                        digits = list(re.finditer(r"\d", new_key))
                        if len(digits) == 1:
                            new_key = new_key[:digits[0].end() + 1] + '0.' + new_key[digits[0].end() + 1:]
                    else:
                        new_key = 'vision_transformer.token_embedding.embeddings.after_concat.' + str(
                                num_concat_after + num_concat_after % 2 - 1
                                ) + key[len('vision_transformer.token.x'):]
                        num_concat_after += 1
                    state_dict[new_key] = state_dict.pop(key)
        else:
            for layer_idx in range(12):
                if key.startswith('module.transformer.transformer.layers.{}.0.fn.norm'.format(layer_idx)):
                    state_dict['vision_transformer.blocks.{}.norm1'.format(layer_idx) + key[len(
                            'module.transformer.transformer.layers.{}.0.fn.norm'.format(layer_idx)
                            ):]] = state_dict.pop(key)
                elif key.startswith('module.transformer.transformer.layers.{}.0.fn.fn.to_out.0'.format(layer_idx)):
                    state_dict['vision_transformer.blocks.{}.attn.proj'.format(layer_idx) + key[len(
                            'module.transformer.transformer.layers.{}.0.fn.fn.to_out.0'.format(layer_idx)
                            ):]] = state_dict.pop(key)
                elif key == 'module.transformer.transformer.layers.{}.0.fn.fn.to_qkv.weight'.format(layer_idx):
                    state_dict['vision_transformer.blocks.{}.attn.qkv.weight'.format(layer_idx)] = state_dict.pop(key)
                elif key.startswith('module.transformer.transformer.layers.{}.1.fn.norm'.format(layer_idx)):
                    state_dict['vision_transformer.blocks.{}.norm2'.format(layer_idx) + key[len(
                            'module.transformer.transformer.layers.{}.1.fn.norm'.format(layer_idx)
                            ):]] = state_dict.pop(key)
                elif key.startswith('module.transformer.transformer.layers.{}.1.fn.fn.net.0'.format(layer_idx)):
                    state_dict['vision_transformer.blocks.{}.mlp.fc1'.format(layer_idx) + key[len(
                            'module.transformer.transformer.layers.{}.1.fn.fn.net.0'.format(layer_idx)
                            ):]] = state_dict.pop(key)
                elif key.startswith('module.transformer.transformer.layers.{}.1.fn.fn.net.3'.format(layer_idx)):
                    state_dict['vision_transformer.blocks.{}.mlp.fc2'.format(layer_idx) + key[len(
                            'module.transformer.transformer.layers.{}.1.fn.fn.net.3'.format(layer_idx)
                            ):]] = state_dict.pop(key)

    return state_dict
