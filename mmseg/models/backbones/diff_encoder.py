# DIFF Module
# Implementation of DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# By Yuxiang Ji


import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner import BaseModule

from ..builder import BACKBONES

from .diff.src.models.diff import DIFFEncoder


@BACKBONES.register_module()
class DIFF(BaseModule):
    def __init__(self,
                 in_channels=3,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 style=None,
                 style_hallucination_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 batch_size=1,):
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        config_related_path = '../../configs/stridehyperfeature.yaml'
        self.diff_model = DIFFEncoder(batch_size=batch_size, config_related_path=config_related_path)

    def forward(self, x, gt_semantic_seg=None):
        return self.forward_features(x, gt_semantic_seg=gt_semantic_seg)

    def forward_features(self, x, return_style=False, gt_semantic_seg=None):
        x = x.to(dtype=torch.float16)
        x = self.diff_model(x, gt_semantic_seg)
        return x
        
    def init_weights(self):
        return


if __name__ == '__main__':
    pass