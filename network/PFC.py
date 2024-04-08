#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
from mmcv.ops import DynamicScatter
from network.cylinder_fea_generator import cylinder_fea
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.BEV_Unet import BEV_Unet


class PFC(nn.Module):

    def __init__(self, cfgs, nclasses):
        super(PFC, self).__init__()
        self.nclasses = nclasses
        self.fcclip_vision_dim = cfgs['model']['fcclip_vision_dim']
        self.fcclip_text_dim = cfgs['model']['fcclip_text_dim']
        self.grid_size = np.array(cfgs['dataset']['grid_size'])
        self.img_size = cfgs['model']['IMAGE_SIZE']
        self.point_cloud_range = np.array(cfgs['dataset']['min_volume_space'] + cfgs['dataset']['max_volume_space'])
        self.voxel_size = ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / (self.grid_size - 1)).tolist()
        self.deconv = nn.Sequential(
            nn.Conv2d(self.fcclip_vision_dim,64 , kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2), # x2
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2), # x2
            # nn.ConvTranspose2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingNearest2d(scale_factor=2), # x2
            nn.Conv2d(64, self.fcclip_vision_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
            )
        self.pre_norm = nn.BatchNorm1d(self.fcclip_vision_dim)
        
        self.vfe_scatter = DynamicScatter(self.voxel_size,
                                    self.point_cloud_range,
                                    ( cfgs['model']['scatter_points_mode']!= 'max'))
            
    def extract_clip_features(self,train_dict):
        clip_vision_channel = train_dict['ori_clip_vision_channel']
        pixel_coordinates = train_dict['pixel_coordinates']
        pol_voxel_ind_channel = train_dict['pol_voxel_ind']
        # point2img_index_channel = train_dict['point2img_index_channel']
        # img_indices_channel = train_dict['img_indices_channel'] 
        B,N,D,H,W = clip_vision_channel.shape
        clip_vision_channel = clip_vision_channel.view(B*N,D,H,W)
        fcclip_deconv_fea = self.deconv(clip_vision_channel)
        _,_,new_w,new_h = fcclip_deconv_fea.shape
        fcclip_deconv_fea = fcclip_deconv_fea.view(B,N,self.fcclip_vision_dim,new_h,new_w)

        voxels = []
        coors = []
        for batch in range(B):
            num_point = pixel_coordinates[batch][0].shape[0]
            pol_voxel_ind = pol_voxel_ind_channel[batch]
            coors.append(F.pad(pol_voxel_ind, (1, 0), mode='constant', value=batch))
            point_fcclip_features = fcclip_deconv_fea.new_zeros([N, num_point, self.fcclip_vision_dim])
            for view in range(N):
                batch_view_fcclip_vision = fcclip_deconv_fea[batch,view,...].unsqueeze(0)
                # point2img_index = point2img_index_channel[batch][view]
                # img_indices = img_indices_channel[batch][view]
                pixel_coord = pixel_coordinates[batch][view]
                normalized_coordinates = pixel_coord.clone()
                normalized_coordinates[..., 0] = normalized_coordinates[..., 0] / (self.img_size[0]*0.5) - 1  # 对宽度进行归一化
                normalized_coordinates[..., 1] = normalized_coordinates[..., 0] / (self.img_size[1]*0.5) - 1
                normalized_coordinates = normalized_coordinates[None, ..., None, :]
                sampled_features = F.grid_sample(batch_view_fcclip_vision.float() , normalized_coordinates.float())
                sampled_features = sampled_features.view(-1,self.fcclip_vision_dim)
                point_fcclip_features[view] = sampled_features
            # 转换到voxel fcclip features
            point_fcclip_features = point_fcclip_features.mean(dim=0)
            voxels.append(point_fcclip_features)

        voxels = torch.cat(voxels,dim=0)
        coors = torch.cat(coors,dim=0)
        voxels = self.pre_norm(voxels)
        # mmdet版本
        voxel_feats, voxel_coors = self.vfe_scatter(voxels, coors)
        # pytorch版本
            
        del fcclip_deconv_fea,clip_vision_channel,  pixel_coordinates,pol_voxel_ind_channel,point_fcclip_features,voxels,coors,
        batch_view_fcclip_vision, normalized_coordinates, pixel_coord        
        return voxel_feats,voxel_coors

    def forward(self, train_dict):
        fcclip_voxel_features = self.extract_clip_features(train_dict)
        # [34656, 9] [34656, 3]
        # 对每个网格的点进行池化
        # [13391, 4] [13391, 16] [20810, 17] [17, 6, 180, 320]
        # coords, features_3d, softmax_pix_logits, cam = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten, train_dict)
        # # 主干网络
        # sem_prediction, center, offset, instmap = self.cylinder_3d_spconv_seg(features_3d, coords, len(train_pt_fea_ten), train_dict)

        # center = self.UNet(center) # [1, 128, 480, 360] ->  [1, 1, 480, 360]

        return fcclip_voxel_features
