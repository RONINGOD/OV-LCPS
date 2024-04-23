#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
from mmcv.ops import DynamicScatter
from network.transformer_decoder import _Transformer_Decoder, get_classification_logits, MaskPooling
from mmengine.structures import InstanceData
from network.util.mask_pseduo_sample import _MaskPseudoSampler
from mmengine.model import kaiming_init, xavier_init
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import SubMConv3d
from mmdet3d.registry import MODELS,TASK_UTILS            

def freeze_everything(model):
    for param in model.parameters():
        param.requires_grad = False

class PFC(nn.Module):

    def __init__(self, cfgs, nclasses,pos_dim=3,use_pa_seg=True,use_sem_loss=True,
                 mask_channels=(256, 256, 256, 256, 256),
                 pa_seg_weight = 0.2,
                 num_decoder_layers=6,
                 score_thr = 0.4,
                 iou_thr = 0.8,
                 geometric_ensemble_alpha = 0.0, # 0
                 geometric_ensemble_beta = 1.0, # 1
                 ignore_index = 16,
                 init_logit_scale = 4.6052,
                 clip_model_name = 'convnext_large_d_320',
                 clip_model_pretrain = 'laion2b_s29b_b131k_ft_soup',
                 timm_kwargs = {'drop_path_rate': 0.1},
                 assigner_zero_layer_cfg=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                            dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                            dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
                 assigner_cfg=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                            dict(type='mmdet.FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                            dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                            dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
                 transformer_decoder_cfg=dict(type='_Transformer_Decoder'),):
        super(PFC, self).__init__()
        self.nclasses = nclasses
        self.ignore_index = nclasses-1
        self.score_thr = score_thr
        self.num_decoder_layers = num_decoder_layers
        self.clip_vision_dim = cfgs['model']['clip_vision_dim']
        self.fcclip_text_dim = cfgs['model']['clip_text_dim']
        cls_channels=(256, 256, self.fcclip_text_dim)
        self.deconv_layers = cfgs['model']['deconv_layers']
        self.deconv_hidden_dim = cfgs['model']['deconv_hidden_dim']
        self.grid_size = np.array(cfgs['dataset']['grid_size'])
        self.img_size = cfgs['model']['IMAGE_SIZE']
        self.point_cloud_range = np.array(cfgs['dataset']['min_volume_space'] + cfgs['dataset']['max_volume_space'])
        self.voxel_size = ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / (self.grid_size - 1)).tolist()
        self.deconv = nn.Sequential()
        self.pe_type = cfgs['model']['pe_type']
        self.query_embed_dims = cfgs['model']['query_embed_dims']
        self.num_queries = cfgs['model']['num_queries']
        self.use_pa_seg = use_pa_seg
        self.use_sem_loss = use_sem_loss
        self.label_map = None
        self.label_inverse_map = None
        self.thing_class = None
        self.stuff_class = None
        self.thing_map = None
        self.thing_inverse_map = None
        self.novel_class = None
        self.base_class = None
        self.total_map = None
        self.total_inverse_map = None
        self.total_class = None
        self.categroy_overlapping_mask = None
        self.map_stuff_class = None
        self.map_thing_class = None
        self.iou_thr = iou_thr
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.loss_cls = MODELS.build(cfgs['model']['loss_cls'])
        self.loss_mask = MODELS.build(cfgs['model']['loss_mask'])
        self.loss_dice = MODELS.build(cfgs['model']['loss_dice'])
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        # for i in range(self.deconv_layers):
        #     if i==0 and i==self.deconv_layers-1:
        #         self.deconv.add_module(f'conv_{i}',nn.Conv2d(self.fcclip_vision_dim,self.fcclip_vision_dim,kernel_size=7,stride=1, padding=3, bias=False))
        #     elif i==0:
        #         self.deconv.add_module(f'conv_{i}',nn.Conv2d(self.fcclip_vision_dim,self.deconv_hidden_dim,kernel_size=7,stride=1, padding=3, bias=False))
        #     elif i==self.deconv_layers-1:
        #         self.deconv.add_module(f'conv_{i}',nn.Conv2d(self.deconv_hidden_dim,self.fcclip_vision_dim, kernel_size=3, stride=1, padding=1, bias=False)) 
        #     else:
        #         self.deconv.add_module(f'conv_{i}',nn.Conv2d(self.deconv_hidden_dim,self.deconv_hidden_dim,kernel_size=7,stride=1, padding=3, bias=False))
        #     xavier_init(self.deconv[-1])
        #     self.deconv.add_module(f'act_{i}',nn.ReLU(inplace=True))
        #     self.deconv.add_module(f'up_{i}', nn.UpsamplingNearest2d(scale_factor=2) if i != self.deconv_layers - 1 else nn.UpsamplingBilinear2d(scale_factor=2))
        self.queries = SubMConv3d(self.query_embed_dims, self.num_queries, indice_key="logit", 
                                    kernel_size=1, stride=1, padding=0, bias=False)
        xavier_init(self.queries)
        if self.use_sem_loss:
            self.loss_ce = MODELS.build(dict(
                            type='mmdet.CrossEntropyLoss',
                            use_sigmoid=False,
                            class_weight=None,
                            loss_weight=1.0))
            self.loss_lovasz = MODELS.build(dict(type='LovaszLoss',
                                                reduction='none',))
        self.sem_queries = nn.Conv3d(self.query_embed_dims, self.nclasses, kernel_size=1, stride=1, padding=0, bias=False)
        xavier_init(self.sem_queries)
        self.pre_norm = nn.BatchNorm1d(self.clip_vision_dim)
        self.void_embedding = nn.Embedding(1,self.fcclip_text_dim)
        xavier_init(self.void_embedding)
        self.query_embed = nn.Embedding(self.num_queries,self.num_queries)
        self.vfe_scatter = DynamicScatter(self.voxel_size,
                                    self.point_cloud_range,
                                    ( cfgs['model']['scatter_points_mode']!= 'max'))

        # self.pe_vision_proj = nn.Linear(self.fcclip_vision_dim,self.query_embed_dims)
        self.decoder_norm = nn.LayerNorm(self.query_embed_dims)
        self.mask_embed = MLP((self.query_embed_dims,self.query_embed_dims,self.query_embed_dims,self.query_embed_dims),act_type='ReLU',bias=True)
        self.mask_proj = MLP((self.nclasses,self.query_embed_dims,self.query_embed_dims,self.query_embed_dims,self.query_embed_dims),act_type='ReLU',bias=True)
        self.pe_vision_proj = MLP((self.clip_vision_dim,self.query_embed_dims,self.query_embed_dims,self.query_embed_dims),act_type='ReLU',bias=True)
        self.mask_features = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.mask_pooling = MaskPooling()
        self._mask_pooling_proj = MLP((self.query_embed_dims,self.query_embed_dims,self.query_embed_dims,self.fcclip_text_dim))
        if self.pe_type == 'polar' or self.pe_type == 'cart':
            self.position_proj = nn.Linear(pos_dim, self.query_embed_dims)
            self.position_norm = build_norm_layer(dict(type='LN'),
                                                  self.query_embed_dims)[1]
            self.feat_conv = nn.Sequential(
                nn.Linear(self.query_embed_dims, self.query_embed_dims, bias=False),
                build_norm_layer(dict(type='LN'), self.query_embed_dims)[1],
                build_activation_layer(dict(type='GELU')))
        elif self.pe_type == 'mpe':
            self.polar_proj = nn.Linear(pos_dim, self.query_embed_dims)
            self.polar_norm = build_norm_layer(dict(type='LN'), self.query_embed_dims)[1]
            self.cart_proj = nn.Linear(pos_dim, self.query_embed_dims)
            self.cart_norm = build_norm_layer(dict(type='LN'), self.query_embed_dims)[1]
            
            self.pe_conv = nn.ModuleList()
            self.pe_conv.append(
                nn.Linear(self.query_embed_dims, self.query_embed_dims, bias=False))
            self.pe_conv.append(
                build_norm_layer(dict(type='LN'), self.query_embed_dims)[1])
            self.pe_conv.append(build_activation_layer(dict(type='ReLU', inplace=True),))    
            
        else:
            self.feat_conv = nn.Sequential(
                nn.Linear(self.query_embed_dims, self.query_embed_dims, bias=False),
                build_norm_layer(dict(type='LN'), self.query_embed_dims)[1],
                build_activation_layer(dict(type='GELU')))
        
        # build transformer decoder            
        transformer_decoder_cfg.update(embed_dims=self.query_embed_dims)
        self.transformer_decoder = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.transformer_decoder.append(_Transformer_Decoder(transformer_decoder_cfg))
                
        # build pa_seg
        self.fc_cls = nn.ModuleList()
        self.fc_cls.append(None)
        self.fc_mask = nn.ModuleList()
        self.fc_mask.append(MLP(mask_channels))
        if self.use_pa_seg:
            self.fc_coor_mask = nn.ModuleList()
            self.fc_coor_mask.append(MLP(mask_channels))
            self.pa_seg_weight = pa_seg_weight    
        for _ in range(num_decoder_layers):
            self.fc_cls.append(MLP(cls_channels))
            self.fc_mask.append(MLP(mask_channels))
            if use_pa_seg:
                self.fc_coor_mask.append(MLP(mask_channels))
        
        # build assigner      
        if assigner_zero_layer_cfg is not None:
            self.zero_assigner = TASK_UTILS.build(assigner_zero_layer_cfg)
        if assigner_cfg is not None:
            self.assigner = TASK_UTILS.build(assigner_cfg)
        self.sampler = _MaskPseudoSampler()
        
            
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
        fcclip_deconv_fea = fcclip_deconv_fea.view(B,N,self.clip_vision_dim,new_h,new_w)

        voxels = []
        coors = []
        for batch in range(B):
            num_point = pixel_coordinates[batch][0].shape[0]
            pol_voxel_ind = pol_voxel_ind_channel[batch]
            coors.append(F.pad(pol_voxel_ind, (1, 0), mode='constant', value=batch))
            point_fcclip_features = fcclip_deconv_fea.new_zeros([N, num_point, self.clip_vision_dim])
            for view in range(N):
                batch_view_fcclip_vision = fcclip_deconv_fea[batch,view,...].unsqueeze(0)
                pixel_coord = pixel_coordinates[batch][view]
                normalized_coordinates = pixel_coord.clone()
                normalized_coordinates[..., 0] = normalized_coordinates[..., 0] / (self.img_size[0]*0.5) - 1  # 对宽度进行归一化
                normalized_coordinates[..., 1] = normalized_coordinates[..., 0] / (self.img_size[1]*0.5) - 1
                normalized_coordinates = normalized_coordinates[None, ..., None, :]
                sampled_features = F.grid_sample(batch_view_fcclip_vision.float() , normalized_coordinates.float())
                sampled_features = sampled_features.squeeze(0).squeeze(-1)
                sampled_features = sampled_features.permute(1,0)
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

    def voxelize_clip_features(self,train_dict):
        clip_features = train_dict['clip_features'] # [34752, 768]
        pol_voxel_ind_channel = train_dict['pol_voxel_ind'] # [34752, 3]
        B = len(clip_features)
        voxels = []
        coors = []
        for batch in range(B):
            pol_voxel_ind = pol_voxel_ind_channel[batch]
            coors.append(F.pad(pol_voxel_ind, (1, 0), mode='constant', value=batch))
            voxels.append(clip_features[batch])

        voxels = torch.cat(voxels,dim=0)
        coors = torch.cat(coors,dim=0)
        voxels = self.pre_norm(voxels)
        # mmdet版本
        voxel_feats, voxel_coors = self.vfe_scatter(voxels, coors)
        return voxel_feats,voxel_coors

    def mpe(self, features, voxel_coors, batch_size):
        """Encode features with sparse indices."""

        if self.pe_type is not None:
            normed_polar_coors = [
                voxel_coor[:, 1:] / voxel_coor.new_tensor(self.grid_size)[None, :].float()
                for voxel_coor in voxel_coors
            ]

        if self.pe_type == 'cart' or self.pe_type == 'mpe':
            normed_cat_coors = []
            for idx in range(len(normed_polar_coors)):
                normed_polar_coor = normed_polar_coors[idx].clone()
                polar_coor = normed_polar_coor.new_zeros(normed_polar_coor.shape)
                for i in range(3):
                    polar_coor[:, i] = normed_polar_coor[:, i]*(
                                        self.point_cloud_range[i+3] -
                                        self.point_cloud_range[i]) + \
                                        self.point_cloud_range[i]
                x = polar_coor[:, 0] * torch.cos(polar_coor[:, 1])
                y = polar_coor[:, 0] * torch.sin(polar_coor[:, 1])
                cat_coor = torch.stack([x, y, polar_coor[:, 2]], 1)
                normed_cat_coor = cat_coor / (
                    self.point_cloud_range[3] - self.point_cloud_range[0])
                normed_cat_coors.append(normed_cat_coor)

        if self.pe_type == 'polar':
            mpe = []
            for i in range(batch_size):
                pe = self.position_norm(
                    self.position_proj(normed_polar_coors[i].float()))
                features[i] = features[i] + pe
                features[i] = self.feat_conv(features[i])
                if self.use_pa_seg:
                    mpe.append(pe)

        elif self.pe_type == 'cart':
            mpe = []
            for i in range(batch_size):
                pe = self.position_norm(
                    self.position_proj(normed_cat_coors[i].float()))
                features[i] = features[i] + pe
                features[i] = self.feat_conv(features[i])
                if self.use_pa_seg:
                    mpe.append(pe)

        elif self.pe_type == 'mpe':
            mpe = []
            for i in range(batch_size):
                cart_pe = self.cart_norm(
                    self.cart_proj(normed_cat_coors[i].float()))
                polar_pe = self.polar_norm(
                    self.polar_proj(normed_polar_coors[i].float()))
                for pc in self.pe_conv:
                    polar_pe = pc(polar_pe)
                    cart_pe = pc(cart_pe)
                    features[i] = pc(features[i])
                pe = cart_pe + polar_pe
                features[i] = features[i] + pe # 
                if self.pa_seg:
                    mpe.append(pe)

        else:
            for i in range(batch_size):
                features[i] = self.feat_conv(features[i])
            mpe = None

        return features, mpe

    def init_inputs(self, features,voxel_coors,batch_size,text_features):
        pe_features, mpe = self.mpe(features, voxel_coors, batch_size)
        queries = self.queries.weight.clone().squeeze(0).squeeze(0).repeat(batch_size,1,1).permute(0,2,1) # [1,1,1,256,128] -> [1,128,256]
        queries = [queries[i] for i in range(queries.shape[0])]

        sem_preds = []
        if self.use_sem_loss:
            sem_queries = self.sem_queries.weight.clone().squeeze(-1).squeeze(-1).repeat(1,1,batch_size).permute(2,0,1) # [20,256,1,1,1] -> [1,20,256]
            sem_queries = self.decoder_norm(sem_queries)
            sem_queries = self.mask_embed(sem_queries) # [1, 17, 256]
            outputs_mask = self.mask_proj(sem_queries.transpose(1,2)).transpose(1,2).unsqueeze(3) # [1,256,768]  

            for b in range(len(pe_features)):
                maskpool_embeddings = self.mask_pooling(x=pe_features[b].unsqueeze(0).unsqueeze(3),mask=outputs_mask)
                maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings.transpose(1,2)).squeeze(0)
                sem_pred = get_classification_logits(maskpool_embeddings,text_features,self.logit_scale)
                # sem_pred = F.normalize(sem_pred, p=2, dim=-1)
                sem_preds.append(sem_pred)
                stuff_queries = sem_queries[b][self.stuff_class] # [5,256]
                queries[b] = torch.cat([queries[b], stuff_queries], dim=0) # [133,256]

        return queries, pe_features, mpe, sem_preds

    def pa_seg(self, queries, features, mpe, layer):
        if mpe is None:
            mpe = [None] * len(features)
        class_preds, mask_preds, pos_mask_preds = multi_apply(
            self.pa_seg_single, queries, features, mpe, [layer] * len(features))
        return class_preds, mask_preds, pos_mask_preds

    def pa_seg_single(self, queries, features, mpe, layer):
        """Get Predictions of a single sample level."""
        mask_queries = queries # [133,256]
        mask_queries = self.fc_mask[layer](mask_queries) # [133,256]
        mask_pred = torch.einsum('nc,vc->nv', mask_queries, features)  # [133, 37660]

        if self.use_pa_seg:
            pos_mask_queries = queries
            pos_mask_queries = self.fc_coor_mask[layer](pos_mask_queries)
            pos_mask_pred = torch.einsum('nc,vc->nv', pos_mask_queries, mpe) # [139,37660]
            mask_pred = mask_pred + pos_mask_pred
            
            # 考虑添加sem_queries
            # sem_queries = queries
        else:
            pos_mask_pred = None

        if layer != 0:
            cls_queries = queries
            cls_pred = self.fc_cls[layer](cls_queries) # [139,20]
        else:
            cls_pred = None

        return cls_pred, mask_pred, pos_mask_pred
    
    def forward_vision_features(self,features,voxel_coors,text_features):
        class_preds_buffer = []
        mask_preds_buffer = []
        pos_mask_preds_buffer = []
        sem_preds_buffer = []
        batch_size = voxel_coors[:, 0].max().item() + 1
        feature_split = []
        voxel_coor_split = []
        for i in range(batch_size):
            feature_split.append(features[voxel_coors[:, 0] == i])
            voxel_coor_split.append(voxel_coors[voxel_coors[:, 0] == i])
        queries, features, mpe, sem_preds = self.init_inputs(
            feature_split, voxel_coor_split, batch_size,text_features)
        _, mask_preds, pos_mask_preds = self.pa_seg(queries, features, mpe, layer=0)
        class_preds_buffer.append(None)
        mask_preds_buffer.append(mask_preds)
        pos_mask_preds_buffer.append(pos_mask_preds)
        for i in range(self.num_decoder_layers):
            queries = self.transformer_decoder[i](queries, features, mask_preds) # queries [133,256] features [37510, 256] mask_preds [139, 37510]
            class_preds, mask_preds, pos_mask_preds = self.pa_seg(queries, features, mpe, layer=i+1)
            class_preds_buffer.append(class_preds)
            mask_preds_buffer.append(mask_preds)
            pos_mask_preds_buffer.append(pos_mask_preds)
        
        
        return class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds

    def get_targets(
        self,
        sampling_results,
        gt_sem_masks=None,
        gt_sem_classes=None,
        positive_weight=1.0,
    ):
        if gt_sem_masks is None:
            gt_sem_masks = [None] * len(sampling_results)
            gt_sem_classes = [None] * len(sampling_results)

        pos_inds = [sr.pos_inds for sr in sampling_results]
        neg_inds = [sr.neg_inds for sr in sampling_results]
        pos_gt_masks = [sr.pos_gt_masks for sr in sampling_results]
        if hasattr(sampling_results[0], 'pos_gt_labels'):
            pos_gt_labels = [sr.pos_gt_labels for sr in sampling_results]
        else:
            pos_gt_labels = [None] * len(sampling_results)

        (labels, mask_targets, label_weights, mask_weights) = multi_apply(
            self._get_target_single,
            pos_inds,
            neg_inds,
            pos_gt_masks,
            pos_gt_labels,
            gt_sem_masks,
            gt_sem_classes,
            positive_weight=positive_weight)

        return (labels, mask_targets, label_weights, mask_weights)

    def _get_target_single(
        self,
        positive_indices,
        negative_indices,
        positive_gt_masks,
        positive_gt_labels,
        gt_sem_masks,
        gt_sem_classes,
        positive_weight,
    ):
        num_pos = positive_indices.shape[0]
        num_neg = negative_indices.shape[0]
        num_samples = num_pos + num_neg
        num_points = positive_gt_masks.shape[-1]
        labels = positive_gt_masks.new_full((num_samples, ),
                                            len(self.total_class),
                                            dtype=torch.long)
        label_weights = positive_gt_masks.new_zeros(num_samples,
                                                    len(self.total_class))
        mask_targets = positive_gt_masks.new_zeros(num_samples, num_points)
        mask_weights = positive_gt_masks.new_zeros(num_samples, num_points)

        if num_pos > 0:
            positive_weight = 1.0 if positive_weight <= 0 else positive_weight

            if positive_gt_labels is not None:
                labels[positive_indices] = positive_gt_labels
            label_weights[positive_indices] = positive_weight
            mask_targets[positive_indices, ...] = positive_gt_masks
            mask_weights[positive_indices, ...] = positive_weight

        if num_neg > 0:
            label_weights[negative_indices] = 1.0

        if gt_sem_masks is not None and gt_sem_classes is not None:
            sem_labels = positive_gt_masks.new_full((len(self.stuff_class), ),
                                                    len(self.total_class),
                                                    dtype=torch.long)
            sem_targets = positive_gt_masks.new_zeros(len(self.stuff_class),
                                                      num_points)
            sem_weights = positive_gt_masks.new_zeros(len(self.stuff_class),
                                                      num_points)
            sem_stuff_weights = torch.eye(
                len(self.stuff_class), device=positive_gt_masks.device)
            sem_label_weights = label_weights.new_zeros(len(self.stuff_class), len(self.total_class)).float()
            sem_label_weights[:, self.stuff_class] = sem_stuff_weights

            if len(gt_sem_classes > 0):
                sem_inds = gt_sem_classes - self.stuff_class[0]
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_classes.long()
                sem_targets[sem_inds] = gt_sem_masks
                sem_weights[sem_inds] = 1

            label_weights[:, self.stuff_class] = 0
            label_weights[:, len(self.total_class)-1] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

        target_dict_assign = dict()
        target_dict_assign['labels'] = labels
        target_dict_assign['masks'] = mask_targets

        weight_dict_assign = dict()
        weight_dict_assign['labels'] = label_weights
        weight_dict_assign['masks'] = mask_weights

        return labels, mask_targets, label_weights, mask_weights
 
    def generate_mask_class_target(self, batch_data_samples):
        labels = []
        masks = []

        for idx in range(len(batch_data_samples['voxel_semantic_labels'])):
            semantic_label = batch_data_samples['voxel_semantic_labels'][idx] # [11982]
            instance_label = batch_data_samples['voxel_instance_labels'][idx]# [11982]

            gt_panoptici_label = (instance_label << 16) + semantic_label
            unique_semantic_label = torch.unique(semantic_label)
            unique_panoptic_label = torch.unique(gt_panoptici_label)

            mask = []
            label = []

            for unq_pan in unique_panoptic_label:
                unq_sem = unq_pan & 0xFFFF
                if unq_sem in self.thing_class:
                    label.append(unq_sem)
                    mask.append(gt_panoptici_label == unq_pan)

            for unq_sem in unique_semantic_label:
                if (unq_sem in self.thing_class) or (unq_sem
                                                     == self.ignore_index):
                    continue
                label.append(unq_sem)
                mask.append(semantic_label == unq_sem)

            if len(label) > 0:
                label = torch.stack(label, dim=0)
                mask = torch.stack(mask, dim=0)
            else:
                label = semantic_label.new_zeros(size=[0])
                mask = semantic_label.new_zeros(
                    size=[0, semantic_label.shape[-1]])

            label, mask = label.long(), mask.long()
            labels.append(label)
            masks.append(mask)

        return (labels, masks)
    
    def bipartite_matching(self, class_preds, mask_preds, pos_mask_preds, batch_data_samples,text_features):
        gt_classes, gt_masks = self.generate_mask_class_target(batch_data_samples) # [7] [19,41589]

        gt_thing_classes = []
        gt_thing_masks = []
        gt_stuff_classes = []
        gt_stuff_masks = []

        cls_targets_buffer = []
        mask_targets_buffer = []
        label_weights_buffer = []

        for b in range(len(gt_classes)):
            # update 
            is_thing_class = (torch.isin(gt_classes[b],torch.tensor(self.thing_class,device=gt_classes[b].device))) & (gt_classes[b]!=self.ignore_index)
            is_stuff_class = (torch.isin(gt_classes[b],torch.tensor(self.stuff_class,device=gt_classes[b].device))) & (gt_classes[b]!=self.ignore_index)
            gt_thing_classes.append(gt_classes[b][is_thing_class])
            gt_thing_masks.append(gt_masks[b][is_thing_class])
            gt_stuff_classes.append(gt_classes[b][is_stuff_class])
            gt_stuff_masks.append(gt_masks[b][is_stuff_class])

        sampling_results = []
        for b in range(len(mask_preds[0])):
            thing_masks_pred_detach = mask_preds[0][b][:self.num_queries,:].detach()
            # 取可见的部分
            voxel2point_map = batch_data_samples['voxel2point_map'][b]
            seenmask = batch_data_samples['seenmask'][b]
            seen_unique_indices = batch_data_samples['seen_unique_indices'][b]
            thing_masks_pred_detach = thing_masks_pred_detach.permute(1,0)[voxel2point_map][seenmask[:,0]][seen_unique_indices].permute(1,0)
            
            sampled_gt_instances = InstanceData(
                labels=gt_thing_classes[b], masks=gt_thing_masks[b])
            sampled_pred_instances = InstanceData(masks=thing_masks_pred_detach)

            assign_result = self.zero_assigner.assign(
                sampled_pred_instances,
                sampled_gt_instances,
                img_meta=None,)
            sampling_result = self.sampler.sample(assign_result,
                                                    sampled_pred_instances,
                                                    sampled_gt_instances)
            sampling_results.append(sampling_result)

        cls_targets, mask_targets, label_weights, _ = self.get_targets(sampling_results, gt_stuff_masks, gt_stuff_classes)
        cls_targets_buffer.append(cls_targets)
        mask_targets_buffer.append(mask_targets)
        label_weights_buffer.append(label_weights)

        for layer in range(self.num_decoder_layers):
            sampling_results = []
            for b in range(len(mask_preds[0])):
                if class_preds[layer] is not None:
                    thing_class_pred_detach = class_preds[layer][b][:self.num_queries,:].detach()
                else:
                    # for layer 1, we don't have class_preds from layer 0, so we use class_preds from layer 1 for matching
                    thing_class_pred_detach = class_preds[layer+1][b][:self.num_queries,:].detach()
                # cos
                thing_class_pred_detach = get_classification_logits(thing_class_pred_detach,text_features,self.logit_scale)
                # thing_class_pred_detach = F.normalize(thing_class_pred_detach, p=2, dim=-1)
                # thing_class_pred_detach = thing_class_pred_detach[...,:-1]
                thing_masks_pred_detach = thing_masks_pred_detach = mask_preds[layer][b][:self.num_queries,:].detach()
                # 取可见的部分
                voxel2point_map = batch_data_samples['voxel2point_map'][b]
                seenmask = batch_data_samples['seenmask'][b]
                seen_unique_indices = batch_data_samples['seen_unique_indices'][b]
                thing_masks_pred_detach = thing_masks_pred_detach.permute(1,0)[voxel2point_map][seenmask[:,0]][seen_unique_indices].permute(1,0)
                
                sampled_gt_instances = InstanceData(
                    labels=gt_thing_classes[b], masks=gt_thing_masks[b])
                sampled_pred_instances = InstanceData(
                    scores=thing_class_pred_detach, masks=thing_masks_pred_detach)
                assign_result = self.assigner.assign(
                    sampled_pred_instances,
                    sampled_gt_instances,
                    img_meta=None)
                sampling_result = self.sampler.sample(assign_result,
                                                      sampled_pred_instances,
                                                      sampled_gt_instances)
                sampling_results.append(sampling_result)

            cls_targets, mask_targets, label_weights, _ = self.get_targets(sampling_results, gt_stuff_masks, gt_stuff_classes)
            cls_targets_buffer.append(cls_targets)
            mask_targets_buffer.append(mask_targets)
            label_weights_buffer.append(label_weights)

        return cls_targets_buffer, mask_targets_buffer, label_weights_buffer
    
    def generate_panoptic_results(self, class_preds, mask_preds):
        """Get panoptic results from mask predictions and corresponding class
        predictions.

        Args:
            class_preds (list[torch.Tensor]): Class predictions.
            mask_preds (list[torch.Tensor]): Mask predictions.

        Returns:
            tuple[list[torch.Tensor]]: Semantic predictions and
                instance predictions.
        """
        semantic_preds = []
        instance_ids = []
        for i in range(len(class_preds)):
            class_pred = class_preds[i] 
            mask_pred = mask_preds[i]

            scores = class_pred[:self.num_queries][:, self.thing_class] # 139 = 128+num_class
            thing_scores, thing_labels = scores.sigmoid().max(dim=1)
            thing_scores *= 2
            thing_labels += self.thing_class[0]
            # 修改stuff_class对应到base里的和novel里的
            stuff_scores = class_pred[self.num_queries:][:, self.stuff_class].diag().sigmoid()
            stuff_labels = torch.tensor(self.stuff_class)
            stuff_labels = stuff_labels.to(thing_labels.device)


            scores = torch.cat([thing_scores, stuff_scores], dim=0)
            labels = torch.cat([thing_labels, stuff_labels], dim=0)
            print(torch.unique(scores))
            print(self.score_thr)

            keep = ((scores > self.score_thr) & (labels != self.ignore_index))
            cur_scores = scores[keep]  # [pos_proposal_num]

            cur_classes = labels[keep]  # [pos_proposal_num]
            cur_masks = mask_pred[keep]  # [pos_proposal_num, pt_num]
            cur_masks = cur_masks.sigmoid()

            semantic_pred = cur_classes.new_full((cur_masks.shape[-1], ),
                                                 self.ignore_index)
            instance_id = cur_classes.new_full((cur_masks.shape[-1], ),
                                               0)

            if cur_masks.shape[0] == 0:
                # semantic_pred = torch.stack(list(map(lambda x: torch.tensor(self.total_inverse_map[x.item()] if x!=self.ignore_index else x,device=semantic_pred.device), semantic_pred)))
                semantic_preds.append(semantic_pred)
                instance_ids.append(instance_id)
                continue

            cur_prob_masks = cur_masks * cur_scores.reshape(-1, 1)
            cur_mask_ids = cur_prob_masks.argmax(0) # cur_mask_ids全0学不到东西
            id = 1

            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class in self.thing_class
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                if mask_area > 0 and original_area > 0: 
                    if mask_area / original_area < self.iou_thr:
                        continue
                    semantic_pred[mask] = pred_class
                    if isthing:
                        instance_id[mask] = id
                        id += 1
            # map到CB+CN
            # semantic_pred = torch.stack(list(map(lambda x: torch.tensor(self.total_inverse_map[x.item()] if x!=self.ignore_index else x,device=semantic_pred.device), semantic_pred)))
            semantic_preds.append(semantic_pred)
            instance_ids.append(instance_id)
        return (semantic_preds, instance_ids)

    def loss_single_layer(self, class_preds, mask_preds, pos_mask_preds, class_targets, mask_targets, label_weights, layer, train_dict,text_features,sem_preds,reduction_override=None):
        batch_size = len(mask_preds)
        losses = dict()

        class_targets = torch.cat(class_targets, 0)
        pos_inds = (class_targets != len(self.total_class)) & (
            class_targets < len(self.total_class))
        bool_pos_inds = pos_inds.type(torch.bool)
        bool_pos_inds_split = bool_pos_inds.reshape(batch_size, -1)

        if class_preds is not None:
            # alpha = self.geometric_ensemble_alpha
            # beta = self.geometric_ensemble_beta
            # category_overlapping_mask = self.categroy_overlapping_mask.to(class_preds[0].device)
            class_preds = [get_classification_logits(preds,text_features,self.logit_scale).softmax(-1) for preds in class_preds]
            class_preds = torch.cat(class_preds, 0)  # [B*N] [133, 12]
            label_weights = torch.cat(label_weights, 0)  # [B*N]
            num_pos = pos_inds.sum().float()
            avg_factor = reduce_mean(num_pos)

            losses[f'loss_cls_{layer}'] = self.loss_cls(
                class_preds, # [133, 12]
                class_targets, # [133]
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

        # mask loss
        loss_mask = 0
        valid_bs = 0
        for mask_idx, (mpred, mtarget) in enumerate(
                zip(mask_preds, mask_targets)):
            mp = mpred[bool_pos_inds_split[mask_idx]]
            # 取seen部分
            voxel2point_map = train_dict['voxel2point_map'][mask_idx]
            seenmask = train_dict['seenmask'][mask_idx]
            seen_unique_indices = train_dict['seen_unique_indices'][mask_idx]
            mp = mp.permute(1,0)[voxel2point_map][seenmask[:,0]][seen_unique_indices].permute(1,0)
            mt = mtarget[bool_pos_inds_split[mask_idx]]
            if len(mp) > 0:
                valid_bs += 1
                loss_mask += self.loss_mask(
                    mp.reshape(-1, 1), (1 - mt).long().reshape(
                        -1))  # (1 - mt) for binary focal loss  [2529252, 1] [2529252]
        if valid_bs > 0:
            losses[f'loss_mask_{layer}'] = loss_mask / valid_bs
        else:
            losses[f'loss_mask_{layer}'] = class_preds.sum() * 0.0

        loss_dice = 0
        valid_bs = 0
        for mask_idx, (mpred, mtarget) in enumerate(
                zip(mask_preds, mask_targets)):
            mp = mpred[bool_pos_inds_split[mask_idx]]
            # 取seen部分
            voxel2point_map = train_dict['voxel2point_map'][mask_idx]
            seenmask = train_dict['seenmask'][mask_idx]
            seen_unique_indices = train_dict['seen_unique_indices'][mask_idx]
            mp = mp.permute(1,0)[voxel2point_map][seenmask[:,0]][seen_unique_indices].permute(1,0)
            mt = mtarget[bool_pos_inds_split[mask_idx]]
            if len(mp) > 0:
                valid_bs += 1
                loss_dice += self.loss_dice(mp, mt) # [54, 46838] [54, 46838]

        if valid_bs > 0:
            losses[f'loss_dice_{layer}'] = loss_dice / valid_bs
        else:
            losses[f'loss_dice_{layer}'] = class_preds.sum() * 0.0

        if self.use_pa_seg:
            loss_dice_pos = 0
            valid_bs = 0
            for mask_idx, (mpred, mtarget) in enumerate(
                    zip(pos_mask_preds, mask_targets)):
                mp = mpred[bool_pos_inds_split[mask_idx]]
                # 取seen部分
                voxel2point_map = train_dict['voxel2point_map'][mask_idx]
                seenmask = train_dict['seenmask'][mask_idx]
                seen_unique_indices = train_dict['seen_unique_indices'][mask_idx]
                mp = mp.permute(1,0)[voxel2point_map][seenmask[:,0]][seen_unique_indices].permute(1,0)
                mt = mtarget[bool_pos_inds_split[mask_idx]]
                if len(mp) > 0:
                    valid_bs += 1
                    loss_dice_pos += self.loss_dice(mp, mt) * self.pa_seg_weight

            if valid_bs > 0:
                losses[f'loss_dice_pos_{layer}'] = loss_dice_pos / valid_bs
            else:
                losses[f'loss_dice_pos_{layer}'] = class_preds.sum() * 0.0

        return losses

    def forward(self, train_dict):
        clip_voxel_features, voxel_coors = self.voxelize_clip_features(train_dict)
        text_features = train_dict['text_features'][0].float() # [12, 768]
        voxel_features = self.pe_vision_proj(clip_voxel_features) # [V，256]
        masked_voxel_features = self.mask_features(voxel_features.unsqueeze(1)).squeeze(1)
        # add void class weight
        # text_features = torch.cat([text_features,F.normalize(self.void_embedding.weight,dim=-1)],dim=0)
        class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds = self.forward_vision_features(masked_voxel_features,voxel_coors,text_features)
        if self.training:
            cls_targets_buffer, mask_targets_buffer, label_weights_buffer = self.bipartite_matching(class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, train_dict,text_features)
            losses = dict()
            for i in range(self.num_decoder_layers+1):
                losses.update(self.loss_single_layer(class_preds_buffer[i], mask_preds_buffer[i], pos_mask_preds_buffer[i],
                                                    cls_targets_buffer[i], mask_targets_buffer[i], label_weights_buffer[i], i,train_dict,text_features,sem_preds))
            # if self.use_sem_loss:
            #     seg_label = train_dict['voxel_semantic_labels']# [46838]
            #     for b in range(len(sem_preds)):
            #         voxel2point_map = train_dict['voxel2point_map'][b]
            #         seenmask = train_dict['seenmask'][b]
            #         seen_unique_indices = train_dict['seen_unique_indices'][b]
            #         sem_preds[b] = sem_preds[b][voxel2point_map][seenmask[:,0]][seen_unique_indices]
            #     seg_label = torch.cat(seg_label, dim=0)
            #     sem_preds = torch.cat(sem_preds, dim=0) # [46838, 20]
            #     losses['loss_ce'] = self.loss_ce(
            #         sem_preds, seg_label, ignore_index=self.ignore_index)
            #     losses['loss_lovasz'] = self.loss_lovasz(
            #         sem_preds, seg_label, ignore_index=self.ignore_index)
            return losses
        else:
            mask_cls_results = class_preds_buffer[-1] # [134, 768]
            mask_pred_results = mask_preds_buffer[-1] # [134, 9813]
            batch_size = voxel_coors[:,0].max().item()+1
            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta
            category_overlapping_mask = self.categroy_overlapping_mask.to(mask_cls_results[0].device)
            class_results_buffer = []
            clip_feature_split = []
            voxel_coor_split = []
            for i in range(batch_size):
                clip_feature_split.append(clip_voxel_features[voxel_coors[:, 0] == i])
                voxel_coor_split.append(voxel_coors[voxel_coors[:, 0] == i])
            for b in range(batch_size):
                clip_feature = clip_feature_split[b].transpose(0,1).unsqueeze(0).unsqueeze(3) # [V, 1536]
                mask_cls = mask_cls_results[b] #  [134,768] [embed_dim+len(self.stuff_class),text_features]
                mask_for_pooling = mask_pred_results[b].unsqueeze(0).unsqueeze(3)  # [134,12826]
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling).squeeze(0)
                in_vocabulary_class_preds = get_classification_logits(mask_cls, text_features, self.logit_scale) # [134, 17]
                out_vocabulary_class_preds = get_classification_logits(pooled_clip_feature,text_features,self.logit_scale)
                # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
                in_vocabulary_class_preds = in_vocabulary_class_preds.softmax(-1)
                out_vocabulary_class_preds = out_vocabulary_class_preds.softmax(-1)
                cls_logits_seen = (
                    (in_vocabulary_class_preds ** (1 - alpha) * out_vocabulary_class_preds**alpha)
                    * category_overlapping_mask
                )
                cls_logits_unseen = (
                    (in_vocabulary_class_preds ** (1 - beta) * out_vocabulary_class_preds**beta)
                    * (~ category_overlapping_mask)
                ) 
                cls_results = cls_logits_seen + cls_logits_unseen
                class_results_buffer.append(cls_results)
                
            semantic_preds, instance_ids = self.generate_panoptic_results(class_results_buffer, mask_pred_results)
            semantic_preds = torch.cat(semantic_preds)
            instance_ids = torch.cat(instance_ids)
            print(torch.unique(semantic_preds))
            print(torch.unique(instance_ids))
            pts_semantic_preds = []
            pts_instance_preds = []
            for batch_idx in range(batch_size):
                semantic_sample = semantic_preds[voxel_coors[:, 0] == batch_idx]
                instance_sample = instance_ids[voxel_coors[:, 0] == batch_idx]
                voxel2point_map = train_dict['voxel2point_map'][batch_idx]
                point_semantic_sample = semantic_sample[voxel2point_map]
                point_instance_sample = instance_sample[voxel2point_map]
                pts_semantic_preds.append(point_semantic_sample.cpu().numpy())
                pts_instance_preds.append(point_instance_sample.cpu().numpy())
            
            return pts_semantic_preds, pts_instance_preds

            
        


class MLP(nn.Module):
    def __init__(self, channels,act_type='GELU',bias=False):
        super().__init__()
        self.mlp = nn.ModuleList()
        for cc in range(len(channels) - 2):
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(
                        channels[cc],
                        channels[cc + 1],
                        bias=bias),
                    build_norm_layer(
                        dict(type='LN'), channels[cc + 1])[1],
                    build_activation_layer(
                        dict(type=act_type))))
        self.mlp.append(
            nn.Linear(channels[-2], channels[-1]))
        
    def forward(self, input):
        for layer in self.mlp:
            input = layer(input)
        return input
