from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP

def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if not tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {('module.' + k): v
                for k, v in state_dict.items()})
    return state_dict

def revise_ckpt_2(state_dict):
    param_names = list(state_dict.keys())
    for param_name in param_names:
        if 'img_neck.lateral_convs' in param_name or 'img_neck.fpn_convs' in param_name:
            del state_dict[param_name]
    return state_dict

# 将0-16的语义label转化为0-15和255,为了让语义模型输出16类，不输出noise
def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def transform_map(class_list):
    return {i:class_list[i] for i in range(len(class_list))}

def inverse_transform(learning_map):
    return {v: k for k, v in learning_map.items()} 

def SemKITTI2train_single(label):
    return label - 1  # uint8 trick, transform null area 0->255

def get_model(model):
    if isinstance(model, DDP):
        return model.module
    else:
        return model