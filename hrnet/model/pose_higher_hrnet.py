# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao
# Modified by Bowen Cheng 
# Modified by Katja Ludwig
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from hrnet.model.blocks import BasicBlock
from hrnet.model.modules import Stem, DeconvLayer, Stage, NoneModule


class HRNet(nn.Module):

    def __init__(self, stage_config):
        super(HRNet, self).__init__()

        self.num_stages = stage_config["num_stages"]

        self.stem = Stem()

        num_blocks = 4
        block = BasicBlock

        if self.num_stages >= 2:
            stage2_channels = stage_config["stage2_channels"]
            stage2_modules = stage_config["stage2_modules"]
            num_channels = [
                    stage2_channels[i] * block.expansion for i in range(len(stage2_channels))
                    ]
            self.transition1 = self._make_transition_layer([256], num_channels)
            self.stage2 = Stage(num_in_channels=num_channels, num_out_channels=stage2_channels, num_modules=stage2_modules,
                                num_blocks=num_blocks, block=block, multi_scale_output=self.num_stages != 2)

        if self.num_stages >= 3:
            stage3_channels = stage_config["stage3_channels"]
            stage3_modules = stage_config["stage3_modules"]
            pre_stage_channels = self.stage2.get_num_channels()
            num_channels = [
                    stage3_channels[i] * block.expansion for i in range(len(stage3_channels))
                    ]
            self.transition2 = self._make_transition_layer(
                    pre_stage_channels, num_channels)
            self.stage3 = Stage(num_in_channels=num_channels, num_out_channels=stage3_channels, num_modules=stage3_modules,
                                num_blocks=num_blocks, block=block, multi_scale_output=self.num_stages != 3)

        if self.num_stages >= 4:
            stage4_channels = stage_config["stage4_channels"]
            stage4_modules = stage_config["stage4_modules"]
            pre_stage_channels = self.stage3.get_num_channels()
            num_channels = [
                    stage4_channels[i] * block.expansion for i in range(len(stage4_channels))
                    ]
            self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
            self.stage4 = Stage(num_in_channels=num_channels, num_out_channels=stage4_channels, num_modules=stage4_modules,
                                num_blocks=num_blocks, block=block, multi_scale_output=False)

        if self.num_stages >= 5 and stage_config["stage5"] == "classification":
            self.stage5 = "classification"
            self.last_layer = nn.Sequential(
                    nn.Conv2d(
                            in_channels=32,
                            out_channels=2048,
                            kernel_size=1,
                            stride=1,
                            padding=0
                            ),
                    nn.BatchNorm2d(2048),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                    )
            self.pretrained_layers = ["*"]

        elif self.num_stages >= 5 and stage_config["stage5"] == "hpe":
            self.stage5 = "hpe"
            self.output_list = stage_config["output_as_list"]
            cfg = stage_config["cfg"]
            pre_stage_channels = self.stage4.get_num_channels()

            self._make_final_hpe_layers(cfg, pre_stage_channels[0])
            self.deconv_layer = DeconvLayer(cfg, pre_stage_channels[0])

            self.pretrained_layers = cfg.PRETRAINED_LAYERS
        else:
            self.pretrained_layers = ["*"]

    def set_dropout(self, dropout):
        """
        Set dropout after fusion layers
        @param dropout: None if dropout should not be used, otherwise dropout factor is set
        """
        self.stage2.set_dropout(dropout)
        self.stage3.set_dropout(dropout)
        self.stage3.set_dropout(dropout)

    def _make_final_hpe_layers(self, cfg, input_channels):
        """
        Build the final layers, one with deconvolution and one without
        Tag per joint defines it associative embedding is applied joint-wise or once for all joints
        """
        dim_tag = cfg.NUM_JOINTS if cfg.TAG_PER_JOINT else 1

        output_channels = cfg.NUM_JOINTS + dim_tag \
            if cfg.USE_ASSOCIATIVE_EMBEDDING_LOSS[0] else cfg.NUM_JOINTS
        self.final_layer_non_deconv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0
                )

        input_channels = 32
        output_channels = cfg.NUM_JOINTS + dim_tag if cfg.USE_ASSOCIATIVE_EMBEDDING_LOSS[1] else cfg.NUM_JOINTS
        self.final_layer_after_deconv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
                padding=0
                )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        Create a transition layer, analogous to the fusion layer, but with creation of a new branch
        @param num_channels_pre_layer: the number of channels in the module from the layer before the transition, in a list: [
        num_channels_branch1, num_channels_branch2, ...]
        @param num_channels_cur_layer: the number of channels in the module from the layer after the transition, in a list: [
        num_channels_branch1, num_channels_branch2, ...]
        @return:
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i],
                                      num_channels_cur_layer[i],
                                      3,
                                      1,
                                      1,
                                      bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(NoneModule())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                            nn.Conv2d(
                                    inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def forward(self, x):
        x = self.stem(x)

        if self.num_stages >= 2:
            x_list = []
            for i in range(2):
                if not isinstance(self.transition1[i], NoneModule):
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(x)
            y_list = self.stage2(x_list)
            x = y_list[0]

        if self.num_stages >= 3:
            x_list = []
            for i in range(3):
                if not isinstance(self.transition2[i], NoneModule):
                    x_list.append(self.transition2[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage3(x_list)
            x = y_list[0]

        if self.num_stages >= 4:
            x_list = []
            for i in range(4):
                if not isinstance(self.transition3[i], NoneModule):
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)
            x = y_list[0]

        if self.num_stages >= 5 and self.stage5 == "classification":
            x = self.last_layer(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.num_stages >= 5 and self.stage5 == "hpe":
            y0 = self.final_layer_non_deconv(x)

            x = torch.cat((x, y0), 1)
            x = self.deconv_layer(x)
            y1 = self.final_layer_after_deconv(x)

            if self.output_list:
                return [y0, y1]
            else:
                return (y0, y1)
        return x


class HRNetW32Backbone(HRNet):
    """
    Capsules the backbone HRNet. Contains 4 stages with 4 branches. Number of channels and modules per stage are set according
    to the paper HigherHRNet.
    """

    def __init__(self, num_stages=5):
        stage2_channels = [32, 64]
        stage2_modules = 1
        stage3_channels = [32, 64, 128]
        stage3_modules = 4
        stage4_channels = [32, 64, 128, 256]
        stage4_modules = 3

        stage_config = {
                "num_stages":      num_stages,
                "stage2_channels": stage2_channels,
                "stage2_modules":  stage2_modules,
                "stage3_channels": stage3_channels,
                "stage3_modules":  stage3_modules,
                "stage4_channels": stage4_channels,
                "stage4_modules":  stage4_modules,
                }

        super(HRNetW32Backbone, self).__init__(stage_config)


def get_hrnet(cfg, is_train, num_stages=5):
    """
    Build the HRNet Backbone. Load weights if a pretrained model is defined.
    @param cfg: Config file with base class BaseConfig
    @param is_train: get model in train or inference_data_loader mode
    @return:
    """
    model = HRNetW32Backbone(num_stages=num_stages)

    if is_train:
        is_strict = cfg.PRETRAINED_FILE is not None and cfg.PRETRAINED_FILE.endswith(".tar")
        model.init_weights(cfg.PRETRAINED_FILE, is_strict=is_strict, verbose=False)

    return model
