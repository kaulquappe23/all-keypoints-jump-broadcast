# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao
# Modified by Bowen Cheng
# Modified by Katja Ludwig
# ------------------------------------------------------------------------------

import logging

from torch import nn as nn

from hrnet.model.blocks import Bottleneck, BasicBlock, BN_MOMENTUM

logger = logging.getLogger(__name__)


class Stem(nn.Module):
    """
    Stem of HRNet, containing two convolutions reducing the size and the fist layer with only one branch
    """

    def __init__(self):
        super(Stem, self).__init__()
        # Stem
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        assert 1 != Bottleneck.expansion
        downsample = nn.Sequential(
                nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64 * Bottleneck.expansion, momentum=BN_MOMENTUM),
                )

        # First Stage
        layers = [Bottleneck(in_channels=64, out_channels=64, stride=1, downsample=downsample)]
        in_channels = 64 * Bottleneck.expansion
        for i in range(1, 4):
            layers.append(Bottleneck(in_channels, out_channels=64))
        self.layer1 = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x


class DeconvLayer(nn.Module):
    """
    This is the deconvolution layer for the HPE task. It contains a ConvTranspose layer and 4 basic blocks
    """

    def __init__(self, cfg, input_channels):
        super(DeconvLayer, self).__init__()
        dim_tag = cfg.NUM_JOINTS if cfg.TAG_PER_JOINT else 1

        final_output_channels = cfg.NUM_JOINTS + dim_tag if cfg.USE_ASSOCIATIVE_EMBEDDING_LOSS[0] else cfg.NUM_JOINTS
        input_channels += final_output_channels
        output_channels = 32
        padding = 1
        output_padding = 0
        deconv_kernel = 4

        layers = [nn.Sequential(
                nn.ConvTranspose2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
                )]
        for _ in range(4):
            layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                    ))
        self.deconv_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layer(x)


class NoneModule(nn.Module):
    """
    Does nothing, but tracing models does not support None "Objects", needed in fusion layers
    """

    def __init__(self):
        super(NoneModule, self).__init__()

    def forward(self, x):
        return x


class HighResolutionModule(nn.Module):
    """
    A HighResolutionModule contains several branches and consists of a certain number of residual blocks before it is finished
    with a fusion layer.
    Fusion layers share information between the branches with different resolutions. A stage is made of multiple modules.
    """

    def __init__(self, num_branches, blocks, num_blocks, num_in_channels, num_out_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_out_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

        self.dropout = None

    def set_dropout(self, dropout):
        """
        Set dropout factor if wanted. Dropout is applied at the end of the fusion layers
        """
        if dropout is not None:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        """
        Build the branch number branch_index. It contains num_blocks blocks of type block (basic or bottleneck blocks are
        possible).
        """
        downsample = None
        if stride != 1 or self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.num_in_channels[branch_index],
                              num_channels[branch_index] * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                                   momentum=BN_MOMENTUM),
                    )

        layers = []
        layers.append(block(self.num_in_channels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.num_in_channels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """
        Build all branches, the number of branches is dependent on the stage.
        """
        branches = []

        for i in range(num_branches):
            branches.append(
                    self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        Build the fusion layer at the end of the module that shares information between the branches.
        Branch output with lower resolution than the target branch needs to be upsampled, branch output with higher resolution
        is downsampled with strided convolutions.
        There is an information exchange between all branches.
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_in_channels
        fuse_layers = []
        for out_branch_idx in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for in_branch_idx in range(num_branches):
                if in_branch_idx > out_branch_idx:
                    fuse_layer.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[in_branch_idx],
                                      num_inchannels[out_branch_idx],
                                      1,
                                      1,
                                      0,
                                      bias=False),
                            nn.BatchNorm2d(num_inchannels[out_branch_idx]),
                            nn.Upsample(scale_factor=2 ** (in_branch_idx - out_branch_idx), mode='nearest')))
                elif in_branch_idx == out_branch_idx:
                    fuse_layer.append(NoneModule())
                else:
                    conv3x3s = []
                    for k in range(out_branch_idx - in_branch_idx):
                        if k == out_branch_idx - in_branch_idx - 1:
                            num_outchannels_conv3x3 = num_inchannels[out_branch_idx]
                            conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[in_branch_idx],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[in_branch_idx]
                            conv3x3s.append(nn.Sequential(
                                    nn.Conv2d(num_inchannels[in_branch_idx],
                                              num_outchannels_conv3x3,
                                              3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            y_final = self.relu(y)
            if self.dropout is not None:
                y_final = self.dropout(y_final)
            x_fuse.append(y_final)

        return x_fuse


class Stage(nn.Module):
    """
    Capsules a complete HRNet stage. Each stage has a certain number of HRNet modules. Stage 1 has 1 branch, stage 2 has 2
    branches, ...
    """

    def __init__(self, num_in_channels, num_out_channels, block, num_blocks, num_modules, multi_scale_output=True):
        super(Stage, self).__init__()

        num_branches = len(num_out_channels)
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                    HighResolutionModule(
                            num_branches,
                            block,
                            num_blocks,
                            num_in_channels,
                            num_out_channels,
                            reset_multi_scale_output)
                    )
            num_in_channels = modules[-1].get_num_inchannels()

        self.stage = nn.Sequential(*modules)
        self.num_inchannels = num_in_channels

    def set_dropout(self, dropout):
        for module in self.stage.modules():
            if isinstance(module, HighResolutionModule):
                module.set_dropout(dropout)

    def get_num_channels(self):
        return self.num_inchannels

    def forward(self, x):
        return self.stage(x)
