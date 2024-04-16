import torch
import torch.nn as nn

# # Create a random tensor to represent your input
# input = torch.rand([1, 1536, 24, 42])

# # Define the layer
# layer = nn.ModuleList([
#     # nn.ConvTranspose2d(in_channels=1536, out_channels=1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.Conv2d(1536,256 , kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2), # x2
            
#             nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2), # x2
            
#             # nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
#             # nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingNearest2d(scale_factor=2), # x2
            
#             # nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            
#             nn.Conv2d(64, 1536, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.UpsamplingBilinear2d(scale_factor=2)
#             # nn.ConvTranspose2d(64, 1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(64, 1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.UpsamplingNearest2d(scale_factor=4),
#             ])

# # Pass the input through the layer
# output = input
# for fc in layer:
#     output = fc(output)

# # Print the output size
# print(output.shape)

# # data = np.load('/home/coisini/data/nuscenes_fcclip_features/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.npy')
# # print(data)
# # print(data.shape)

# import numpy as np

# res_voxel_coors = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]])  # 假设这是你的原始数据
# i = 7  # 要填充的值

# # 在最后一维（列）后面添加一列，填充值为i
# res_voxel_coors = np.pad(res_voxel_coors, ((0, 0), (1, 0)), constant_values=i)

# print(res_voxel_coors)

# point_fcclip_features = torch.rand(6, 34720, 1536) 
# pol_voxel_ind = torch.randint(0, 10000, size=(34720, 3))

# # 步骤1: 获取唯一的 voxel 索引和对应的计数
# unique_voxel_idx, inverse_idx, voxel_counts = torch.unique(pol_voxel_ind[:, 0], return_inverse=True, return_counts=True)
# num_voxels = len(unique_voxel_idx)
# print(pol_voxel_ind[:, 0])

# # 步骤2: 计算六个摄像头 point_fcclip_features 的平均值
# avg_point_fcclip_features = point_fcclip_features.mean(dim=0)
# print(avg_point_fcclip_features.shape)

# # 步骤3: 创建 voxel_fcclip_features 张量并填充数据
# voxel_fcclip_features = torch.zeros(num_voxels, 1536, device=point_fcclip_features.device)
# voxel_fcclip_features.index_add_(0, inverse_idx, avg_point_fcclip_features)
# voxel_fcclip_features /= voxel_counts.unsqueeze(1)

# print(voxel_fcclip_features.shape)

# gt_classes = [torch.zeros(size=[0])]

# gt_masks = [torch.zeros(size=[0,7938])]


# checkpoint = torch.load('/home/coisini/project/fc-clip/checkpoints/fcclip_cocopan.pth')
# print(checkpoint['model'].keys())
import open_clip
print(open_clip.list_pretrained())

