#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb

from network.cylinder_fea_generator import cylinder_fea
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.BEV_Unet import BEV_Unet


class PFC(nn.Module):

    def __init__(self, cfgs, nclasses):
        super(PFC, self).__init__()
        self.nclasses = nclasses
        self.cylinder_3d_generator = cylinder_fea(
            cfgs,
            grid_size=cfgs['dataset']['grid_size'],
            fea_dim=9,
            out_pt_fea_dim=256,
            fea_compre=16,
            nclasses=nclasses,
            use_sara=cfgs['model']['use_sara'],
            use_att=cfgs['model']['use_att'] if 'use_att' in cfgs['model'] else False)
        self.cylinder_3d_spconv_seg = Asymm_3d_spconv(
            cfgs,
            output_shape=cfgs['dataset']['grid_size'],
            use_norm=True,
            num_input_features=16,
            init_size=32,
            nclasses=nclasses)
        self.UNet = BEV_Unet(128)

    def forward(self, train_dict):
        train_pt_fea_ten, train_vox_ten = train_dict['return_fea'], train_dict['pol_voxel_ind'] # [34656, 9] [34656, 3]
        # 对每个网格的点进行池化
        # [13391, 4] [13391, 16] [20810, 17] [17, 6, 180, 320]
        coords, features_3d, softmax_pix_logits, cam = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten, train_dict)
        # 主干网络
        sem_prediction, center, offset, instmap = self.cylinder_3d_spconv_seg(features_3d, coords, len(train_pt_fea_ten), train_dict)

        center = self.UNet(center) # [1, 128, 480, 360] ->  [1, 1, 480, 360]

        return sem_prediction, center, offset, instmap, softmax_pix_logits, cam
