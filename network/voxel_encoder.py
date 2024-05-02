import torch
from mmcv.cnn import build_norm_layer
from torch import nn
import torch_scatter
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule,constant_init

@MODELS.register_module()
class _CylinderVFE(BaseModule):#TODO transfer into dynamic scatter
    """Simple voxel feature encoder used in SECOND.
    It simply averages the values of points in a voxel.

    Args:
        num_features (int): Number of features to use. Default: 4.
    """


    def __init__(self, grid_size, fea_dim=3,
                 out_fea_dim=64, mid_size=64, max_pt_per_encode=64, fea_compre=None, norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__()

        self.PPmodel = nn.Sequential(
            build_norm_layer(norm_cfg, fea_dim)[1],

            nn.Linear(fea_dim, mid_size),
            build_norm_layer(norm_cfg, mid_size)[1],
            nn.ReLU(),

            nn.Linear(mid_size, mid_size*2),
            build_norm_layer(norm_cfg, mid_size*2)[1],
            nn.ReLU(),

            nn.Linear(mid_size*2, mid_size*4),
            build_norm_layer(norm_cfg, mid_size*4)[1],
            nn.ReLU(),

            nn.Linear(mid_size*4, out_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    #@force_fp32(out_fp16=True)
    def forward(self, cat_pt_fea, cat_pt_ind): 
        cur_dev = cat_pt_fea.get_device()
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        processed_cat_pt_fea = self.PPmodel(cat_pt_fea) # MLP 9->256    
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # [34752,256] [34752]
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return processed_pooled_data,unq