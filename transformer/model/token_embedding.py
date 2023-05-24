# -*- coding: utf-8 -*-
"""
Created on 05.05.23

"""
import torch
from torch import nn
from torch.nn.init import trunc_normal_

from transformer.model.vision_transformer import Concat
from utils.general_utils import get_dict


class TokenEmbedding(nn.Module):
    """
    This class capsules the embedding of the non-visual tokens. The options are
    - learnable token: a learnable token is used as the embedding for each keypoint (only usable with fixed keypoints, original
    TokenPose variant)
    - Keypoint vectors: a vector for the projection point, the thickness and the angle is used
    - Normalized pose: the coordinates on a pose template are used
    """

    LEARNABLE_TOKEN = 0
    NORM_POSE = 1
    KEYPOINT_VECTOR_WITHOUT_THICKNESS = 2
    KEYPOINT_VECTOR_WITH_THICKNESS = 3
    KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE = 4

    def __init__(self, num_keypoints, embed_dim, config):
        super().__init__()

        representation_type = config["representation_type"]
        num_layers = get_dict(config, "num_layers", 1)
        concat_after_layers = get_dict(config, "concat_after_layers", 1)
        use_reduction = get_dict(config, "use_reduction", True)
        not_reduce = get_dict(config, "not_reduce", 0)

        self.description = "{}L".format(num_layers)
        if concat_after_layers != 1:
            self.description += "_{}C".format(concat_after_layers)
        if representation_type == TokenEmbedding.NORM_POSE:
            self.description += "_NP"

        self.representation_type = representation_type
        if representation_type == TokenEmbedding.LEARNABLE_TOKEN:
            self.token = nn.Parameter(torch.zeros(1, num_keypoints, embed_dim))
            trunc_normal_(self.token, std=.02)
        elif representation_type == TokenEmbedding.NORM_POSE:
            reduction_factor = 2 ** (num_layers - 1) if use_reduction else 1
            self.embedding = [nn.Linear(2, embed_dim // reduction_factor)]
            for i in range(1, num_layers):
                self.embedding.append(nn.ReLU())
                reduction_factor = 2 ** (num_layers - i - 1) if use_reduction else 1
                reduction_last = 2 ** (num_layers - i) if use_reduction else 1
                self.embedding.append(nn.Linear(embed_dim // reduction_last, embed_dim // reduction_factor))
            self.embedding = nn.Sequential(*self.embedding)
        else:
            if representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITHOUT_THICKNESS:
                reduction_factor = 1
                num_vectors = 1
            elif representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS:
                reduction_factor = 2
                num_vectors = 2
            elif representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE:
                reduction_factor = 3
                num_vectors = 3
            else:
                raise ValueError("Unknown representation type")
            self.embeddings = {
                "before_concat": [],
                "after_concat": []
            }
            if concat_after_layers > 0:
                embed_size = embed_dim // (2 ** max(0, num_layers - 1 - not_reduce)) if concat_after_layers == 1 and use_reduction else embed_dim
                # We need to make sure the embed size fits if we concatenate the embeddings
                self.embeddings["before_concat"].append([nn.Linear(num_keypoints, embed_size - (num_vectors - 1) * embed_size // reduction_factor)])
                if num_vectors > 1:
                    self.embeddings["before_concat"].append([nn.Linear(3, embed_size // reduction_factor)])
                if num_vectors > 2:
                    self.embeddings["before_concat"].append([nn.Linear(1, embed_size // reduction_factor)])
            else:
                embed_size = embed_dim // (2 ** (num_layers - 1))
                if representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITHOUT_THICKNESS:
                    start_size = num_keypoints
                elif representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS:
                    start_size = num_keypoints + 3
                elif representation_type == TokenEmbedding.KEYPOINT_VECTOR_WITH_THICKNESS_AND_ANGLE:
                    start_size = num_keypoints + 4
                else:
                    raise ValueError("Unknown representation type")
                self.embeddings["after_concat"].append(nn.Linear(start_size, embed_size))

            for i in range(1, num_layers):
                embed_size = embed_dim // (2 ** max(0, num_layers - i - 1 - not_reduce)) if concat_after_layers == 1 and use_reduction or concat_after_layers == 0 else embed_dim
                embed_size_last = embed_dim // (2 ** max(0, num_layers - i - not_reduce)) if concat_after_layers == 1 and use_reduction or concat_after_layers == 0 else embed_dim
                if concat_after_layers > i:
                    # We keep multiple 'branches' since the features are not concatenated yet
                    for j in range(num_vectors):
                        self.embeddings["before_concat"][j].append(nn.ReLU())
                        if j == 0:
                            # We need to make sure the embed size fits if we concatenate the embeddings
                            self.embeddings["before_concat"][j].append(nn.Linear(embed_size_last - (num_vectors - 1) * embed_size_last // reduction_factor, embed_size - (num_vectors - 1) * embed_size // reduction_factor))
                        else:
                            self.embeddings["before_concat"][j].append(nn.Linear(embed_size_last // reduction_factor, embed_size // reduction_factor))
                else:
                    # Everything is already concatenated
                    self.embeddings["after_concat"].append(nn.ReLU())
                    self.embeddings["after_concat"].append(nn.Linear(embed_size_last, embed_size))

            self.embeddings["before_concat"] = nn.ModuleList([nn.Sequential(*l) for l in self.embeddings["before_concat"]])
            self.embeddings["after_concat"] = nn.Sequential(*self.embeddings["after_concat"])
            self.embeddings = nn.ModuleDict(self.embeddings)


    def forward(self, x=None, B=None):
        if self.representation_type == TokenEmbedding.LEARNABLE_TOKEN:
            return self.token.expand(B, -1, -1)
        elif self.representation_type == TokenEmbedding.NORM_POSE:
            return self.embedding(x[0])
        else:
            if len(self.embeddings["before_concat"]) > 0:
                x = [self.embeddings["before_concat"][i](x[i]) for i in range(len(x))]
            x = Concat(dim=2)(x)
            x = self.embeddings["after_concat"](x)
            return x
