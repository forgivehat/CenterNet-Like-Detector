# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from mmcv.ops.carafe import CARAFEPack

from mmdet.models.builder import  NECKS

@NECKS.register_module()
class SPPFCSPBlock(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer with CSP for
     YOLOv7

     Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         expand_ratio (float): Expand ratio of SPPCSPBlock.
            Defaults to 0.5.
         kernel_sizes (int, tuple[int]): Sequential or number of kernel
             sizes of pooling layers. Defaults to 5.
         is_tiny_version (bool): Is tiny version of SPPFCSPBlock. If True,
            it means it is a yolov7 tiny model. Defaults to False.
         conv_cfg (dict): Config dict for convolution layer. Defaults to None.
             which means using conv2d. Defaults to None.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.03, eps=0.001).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         init_cfg (dict or list[dict], optional): Initialization config dict.
             Defaults to None.
     """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 kernel_sizes=5,
                 is_tiny_version: bool = False,
                 conv_cfg= None,
                 norm_cfg= dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg= dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.is_tiny_version = is_tiny_version

        mid_channels = int(2 * out_channels * expand_ratio)

        if is_tiny_version:
            self.main_layers = ConvModule(
                in_channels,
                mid_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.main_layers = nn.Sequential(
                ConvModule(
                    in_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])

        if is_tiny_version:
            self.fuse_layers = ConvModule(
                4 * mid_channels,
                mid_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.fuse_layers = nn.Sequential(
                ConvModule(
                    4 * mid_channels,
                    mid_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule(
                    mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.short_layer = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x1 = self.main_layers(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x1)
            y2 = self.poolings(y1)
            concat_list = [x1] + [y1, y2, self.poolings(y2)]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))
        else:
            concat_list = [x1] + [m(x1) for m in self.poolings]
            if self.is_tiny_version:
                x1 = self.fuse_layers(torch.cat(concat_list[::-1], 1))
            else:
                x1 = self.fuse_layers(torch.cat(concat_list, 1))

        x2 = self.short_layer(x)
        return self.final_conv(torch.cat((x1, x2), dim=1))


class SPPFBottleneck(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5,9,13),
                 use_conv_first=True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg= dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@NECKS.register_module()
class CTResNetUlikeCARAFENeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 use_dcn=True,
                 init_cfg=None,
                 upsample_cfg=dict(
                     type='carafe',
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1)):
        super(CTResNetUlikeCARAFENeck, self).__init__(init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels

        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')

        self.layers = nn.ModuleList()
        self.layers.append(SPPFCSPBlock(
            self.in_channel,
            self.in_channel,
            kernel_sizes=(5, 9, 13),
        ))

        """use deconv layers to upsample backbone's output."""
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            self.layers.append(conv_module)
            upsample_cfg_ = self.upsample_cfg.copy()
            if self.upsample == 'deconv':
                upsample_module = ConvModule(
                    feat_channel,
                    feat_channel,
                    num_deconv_kernels[i],
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='deconv'),
                    norm_cfg=dict(type='BN'))
            elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(
                        in_channels=feat_channel,
                        out_channels=feat_channel,
                        scale_factor=2,
                        upsample_kernel=self.upsample_kernel)
            elif self.upsample == 'carafe':
                upsample_cfg_.update(channels=feat_channel, scale_factor=2)
            upsample_module = build_upsample_layer(upsample_cfg_)
            self.layers.append(upsample_module)
            self.in_channel = feat_channel
    

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
    
        self.start_level = 0
        self.backbone_end_level = 2
        # for i, feature in enumerate(inputs) :
        #     print(f"inputs[{i}] shape:", feature.shape)

        # 初始化特征图为骨干网络最后一层特征图
        outs = inputs[-1]
        # SPPFCSP
        outs = self.layers[0](outs)

       
        for i, (conv_module, upsample_module) in enumerate(zip(self.layers[1::2], self.layers[2::2])):
            outs = conv_module(outs)  
            outs = upsample_module(outs) 
            outs = torch.add(outs, inputs[self.backbone_end_level - i]) 

        # print(f"outs shape",outs.shape)
        return outs,

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()
