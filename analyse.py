import torch
import torch.nn as nn
import numpy as np

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
# import open_clip
# print(open_clip.list_pretrained())

# import clip
# print(clip.available_models())

# import os
# print(os.path.basename("/home/coisini/data/nuscenes_openseg_features/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951"))
# print(os.path.dirname("/home/coisini/data/nuscenes_openseg_features/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.npy"))

# preds = torch.rand([133,13])
# targes = torch.randint(low=0,high=12,size=[133])
# weights = torch.rand([13,13])
# from mmdet.models.losses import cross_entropy
# print(cross_entropy(preds,targes))

NUSCENES_LABELS_16 = ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person', 'traffic cone',
                    'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
NUSCENES_LABELS_DETAILS = ['barrier', 'barricade', 'bicycle', 'bus', 'car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
                        'motorcycle', 'person', 'pedestrian','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                        'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                        'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods']
import yaml
with open("nuscenes.yaml", 'r') as stream:
    nuscenesyaml = yaml.safe_load(stream)
base_thing_list = [cl for cl, is_thing in nuscenesyaml['base_thing_class'].items() if is_thing]
base_stuff_list = [cl for cl, is_stuff in nuscenesyaml['base_stuff_class'].items() if is_stuff]
novel_thing_list = [cl for cl, is_thing in nuscenesyaml['novel_thing_class'].items() if is_thing]
novel_stuff_list = [cl for cl, is_stuff in nuscenesyaml['novel_stuff_class'].items() if is_stuff]

MAPPING_NUSCENES_DETAILS = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4,
                            5, 6, 6, 7, 8, 8, 8, 8, 8,
                            9, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13,
                            14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15])+1
base_thing = []
base_stuff = []
novel_thing = []
novel_stuff = []
mapping_dict = dict()
for i in range(len(NUSCENES_LABELS_DETAILS)):
    label = NUSCENES_LABELS_DETAILS[i]
    id = MAPPING_NUSCENES_DETAILS[i]
    mapping_dict[label] = id
for k in mapping_dict.keys():
    if mapping_dict[k] in base_thing_list:
        base_thing.append(k)
    if mapping_dict[k] in base_stuff_list:
        base_stuff.append(k)
    if mapping_dict[k] in novel_thing_list:
        novel_thing.append(k)
    if mapping_dict[k] in novel_stuff_list:
        novel_stuff.append(k)
base_total = base_thing+base_stuff
novel_total = novel_thing+novel_stuff
total = base_total+novel_total
base_total_add_noise = ['noise'] + base_total
total_add_noise = ['noise'] + total
# print(len(NUSCENES_LABELS_DETAILS))
# print(MAPPING_NUSCENES_DETAILS+1)
print(base_thing)
print(base_stuff)
print(novel_thing)
print(novel_stuff)
print(base_total,len(base_total))
print(total,len(total))
transfer_dict = list(np.vectorize(mapping_dict.__getitem__)(total))
transfer_dict  = transfer_dict
transfer_list = []
count = {}

for num in transfer_dict:
    if num not in count:
        count[num] = len(count) + 1
    transfer_list.append(count[num])
transfer_list  = [0]+transfer_list
print(transfer_list)
print(transfer_dict)
print(mapping_dict)

import torch

# 假设你的class_scores是一个133x32的Tensor
class_scores = torch.randn(133, 32)

# 这是你的多标签索引
labels = [0, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 9, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12]
num_classes = max(labels) + 1  # 计算总类别数

# 创建一个新的Tensor，用于存储映射后的得分
class_preds = torch.zeros(133, num_classes)

# 对于每个类别，找到对应的标签，然后获取最大得分
for i, class_idx in enumerate(labels):
    class_preds[:, class_idx], _ = torch.max(torch.stack([class_scores[:, i], class_preds[:, class_idx]], dim=-1), dim=-1)

print(class_preds.size())  # 输出新的class_score