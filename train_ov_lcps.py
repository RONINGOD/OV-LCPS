#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
import yaml
import torch
import torch.optim as optim
import datetime
import warnings
import pickle
import shutil

from tqdm import tqdm
from nuscenes import NuScenes
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.AppLogger import AppLogger
from network.PFC import PFC
from dataloader.dataset import collate_fn_OV, Nuscenes_pt, spherical_dataset, OV_Nuscenes_pt,collate_dataset_info, SemKITTI_pt,ov_spherical_dataset
from dataloader.eval_sampler import SequentialDistributedSampler
from network.util.instance_post_processing import get_panoptic_segmentation
from network.util.loss import PanopticLoss, PixelLoss
from utils.eval_pq import PanopticEval,OV_PanopticEval
from utils.metric_util import per_class_iu, fast_hist_crop

warnings.filterwarnings("ignore")


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

def load_pretrained_model(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model)
    model.load_state_dict(model_dict)
    return model


def main():
    torch.set_num_threads(6)
    # torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--configs', default='configs/pa_po_nuscenes.yaml')
    parser.add_argument('-l', '--logdir', default='train.log')
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('-r', "--resume", action="store_true", default=False)
    args = parser.parse_args()
    print(os.environ)
    print()
    if "WORLD_SIZE" in os.environ.keys() and int(os.environ["WORLD_SIZE"]) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        if args.local_rank == 0:
            print("|| MASTER_ADDR:",os.environ["MASTER_ADDR"],
                  "|| MASTER_PORT:",os.environ["MASTER_PORT"],
                  "|| LOCAL_RANK:",os.environ["LOCAL_RANK"],
                  "|| RANK:",os.environ["RANK"], 
                  "|| WORLD_SIZE:",os.environ["WORLD_SIZE"])
    

    # 分布式初始化
    if args.local_rank != -1:
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
        torch.cuda.set_device(args.local_rank)

    # log文件初始化
    logger = AppLogger("logA", args.logdir)

    torch.backends.cudnn.benchmark = True  # 是否自动加速，自动选择合适算法，false选择固定算法
    torch.backends.cudnn.deterministic = True  # 为了消除该算法本身的不确定性
    
    # 加载cfg文件
    with open(args.configs, 'r') as s:
        cfgs = yaml.safe_load(s)
    logger.info(cfgs)

    datasetname = cfgs['dataset']['name']
    version = cfgs['dataset']['version']
    data_path = cfgs['dataset']['path']
    num_worker = cfgs['dataset']['num_worker']
    train_batch_size = cfgs['model']['train_batch_size']
    val_batch_size = cfgs['model']['val_batch_size']
    model_load_path = cfgs['model']['model_load_path']
    model_save_path = cfgs['model']['model_save_path']
    lr = cfgs['model']['learning_rate']
    lr_step = cfgs['model']['LR_MILESTONES']
    lr_gamma = cfgs['model']['LR_GAMMA']
    grid_size = cfgs['dataset']['grid_size']
    pix_fusion = cfgs['model']['pix_fusion']
    min_points = cfgs['dataset']['min_points']
    grad_accumu = 1
    # 初始化类别名称和数量，不包括noise类。
    unique_label, unique_label_str = collate_dataset_info(cfgs)
    
    # 加noise类 
    nclasses = len(unique_label) + 1

    my_model = PFC(cfgs, nclasses)

    # 加载模型
    if args.resume:
        pretrained_model = torch.load(model_load_path, map_location=torch.device('cpu'))
        # 消除分布式训练时在保存参数的时候多出来的module.
        weights_dict = {}
        for k, v in pretrained_model.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        # # debug的时候查看参数量
        # model_dict = my_model.state_dict()
        my_model.load_state_dict(weights_dict, strict=False)
        print(f'load checkpoint file {model_load_path}')

    # DDP的sync_bn，让多卡训练的bn范围正常
    if args.local_rank != -1:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
    my_model.cuda()
    if args.local_rank != -1:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

    # NuScenes: MultiStepLR; SemanticKitti: CosineAnnealingLR, CosineAnnealingWarmRestarts
    optimizer = optim.Adam(my_model.parameters(), lr=lr)
    scheduler_steplr = MultiStepLR(optimizer, milestones=lr_step, gamma=lr_gamma)
    # scheduler_steplr = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8, verbose=True)
    # scheduler_steplr = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-8, verbose=True)

    loss_fn = PanopticLoss(center_loss_weight=cfgs['model']['center_loss_weight'],
                            offset_loss_weight=cfgs['model']['offset_loss_weight'],
                            center_loss=cfgs['model']['center_loss'], offset_loss=cfgs['model']['offset_loss'])
    pix_loss_fn = PixelLoss()


    if datasetname == 'SemanticKitti':
        train_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfgs, split='train', return_ref=True)
        val_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfgs, split='val', return_ref=True)
    elif datasetname == 'nuscenes':
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        assert version == "v1.0-trainval" or version == "v1.0-mini"
        train_pt_dataset = OV_Nuscenes_pt(data_path, split='train', cfgs=cfgs, nusc=nusc, version=version)
        val_pt_dataset = OV_Nuscenes_pt(data_path, split='val', cfgs=cfgs, nusc=nusc, version=version)
    else:
        raise NotImplementedError

    train_dataset = ov_spherical_dataset(train_pt_dataset, cfgs, ignore_label=0)
    val_dataset = ov_spherical_dataset(val_pt_dataset, cfgs, ignore_label=0, use_aug=False)

    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = SequentialDistributedSampler(val_dataset, val_batch_size)
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=train_batch_size,
                                                           collate_fn=collate_fn_OV,
                                                           sampler=train_sampler,
                                                           pin_memory=True,
                                                           num_workers=num_worker)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_batch_size,
                                                         collate_fn=collate_fn_OV,
                                                         sampler=val_sampler,
                                                         pin_memory=True,
                                                         num_workers=num_worker)
    else:
        train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=train_batch_size,
                                                           collate_fn=collate_fn_OV,
                                                           shuffle=True,
                                                           pin_memory=True,
                                                           num_workers=num_worker)
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=val_batch_size,
                                                         collate_fn=collate_fn_OV,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=num_worker)

    if datasetname == 'nuscenes':
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        learning_map = nuscenesyaml['learning_map']

    # training
    epoch = 0
    best_val_PQ = 0
    start_training = False
    # my_model.train()
    global_iter = 0
    evaluator = OV_PanopticEval(len(unique_label)+1+1, None, [0,len(unique_label)+1], min_points=min_points,offset=65536)
    loss_fn_dict ={
        'sem_loss':[],
        'class_loss':[],
        'mask_loss':[],
        'dice_loss':[],
        'dice_pos_loss':[]
    }
    last_avg_loss = 0.0
    while epoch < cfgs['model']['max_epoch']:
        avg_loss = 0.0
        if args.local_rank < 1:
            logger.info("epoch: %d   lr: %.5f\n" % (epoch, optimizer.param_groups[0]['lr']))

        # validation
        if (epoch > 0):
            if args.local_rank <= 0:
                print(f"Epoch {epoch} => Start Evaluation...")
            my_model.eval()
            evaluator.reset()
            sem_hist_list = []
            pbar_val = tqdm(total=len(val_dataset_loader))
            if args.local_rank > 0:
                save_dict = {
                    'item1': [],          
                    'item2': [],
                    'item3': [],
                    'item4': [],
                    'item5': [],
                }

            if args.local_rank != -1:
                torch.distributed.barrier()
            get_model(my_model).label_map = transform_map(SemKITTI2train(np.hstack([val_pt_dataset.base_thing_list,val_pt_dataset.base_stuff_list,val_pt_dataset.novel_thing_list,val_pt_dataset.novel_stuff_list,17])))
            get_model(my_model).label_inverse_map = inverse_transform(get_model(my_model).label_map)
            get_model(my_model).thing_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(val_pt_dataset.thing_list)))
            get_model(my_model).stuff_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(val_pt_dataset.stuff_list)))
            get_model(my_model).total_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(np.hstack([val_pt_dataset.base_thing_list+val_pt_dataset.base_stuff_list,17]))))
            get_model(my_model).categroy_overlapping_mask = torch.from_numpy(np.hstack((np.full(len(val_pt_dataset.base_thing_list+val_pt_dataset.base_stuff_list), True, dtype=bool),np.full(len(val_pt_dataset.novel_thing_list+val_pt_dataset.novel_stuff_list),False,dtype=bool),np.full(1,True,dtype=bool))))
            with torch.no_grad():
                for i_iter_val, val_dict in enumerate(val_dataset_loader):
                    val_dict['voxel2point_map'] = [torch.from_numpy(i) for i in val_dict['voxel2point_map']]
                    val_dict['point2voxel_map'] = [torch.from_numpy(i) for i in val_dict['point2voxel_map']]
                    val_dict['seenmask'] = [torch.from_numpy(i).cuda() for i in val_dict['seenmask']]
                    val_dict['seen_unique_indices'] = [torch.from_numpy(i).cuda() for i in val_dict['seen_unique_indices']]
                    val_dict['pol_voxel_ind'] = [torch.from_numpy(i).cuda() for i in val_dict['pol_voxel_ind']]
                    val_dict['voxel_semantic_labels'] = [SemKITTI2train(torch.from_numpy(i)).type(torch.LongTensor).cuda() for i in val_dict['voxel_semantic_labels']]
                    val_dict['voxel_instance_labels'] = [torch.from_numpy(i).type(torch.LongTensor).cuda() for i in val_dict['voxel_instance_labels']]
                    if pix_fusion:
                        val_dict['pixel_coordinates'] = [torch.from_numpy(i).cuda() for i in val_dict['pixel_coordinates']]
                        val_dict['ori_clip_vision_channel'] = torch.from_numpy(val_dict['ori_clip_vision_channel']).cuda()
                        val_dict['text_features'] = torch.from_numpy(val_dict['text_features']).cuda()

                    predict_labels_sem, pts_instance_preds = my_model(val_dict)
                    predict_labels_sem = np.vectorize(get_model(my_model).label_map.__getitem__)(predict_labels_sem)
                    predict_labels_sem = [sem + 1 for sem in predict_labels_sem]
                    val_grid = val_dict['pol_voxel_ind']
                    val_pt_labels = val_dict['pt_sem_label']
                    val_pt_inst = val_dict['pt_ins_label']
                    for count, i_val_grid in enumerate(val_grid):
                        panoptic = pts_instance_preds[count]
                        # 语义分割预测出是前景类别，但是全景分割预测它是背景(全景ID预测为0)，当作noise处理
                        # if datasetname == 'SemanticKitti':
                        #     panoptic_mask1 = (panoptic <= 8) & (panoptic > 0)
                        #     panoptic[panoptic_mask1] = 0
                        # elif datasetname == 'nuscenes':
                        #     panoptic_mask1 = (panoptic <= 10) & (panoptic > 0)
                        #     panoptic[panoptic_mask1] = 0
                        # else:
                        #     raise NotImplementedError

                        if args.local_rank < 1:
                            # 用实例标签的语义
                            if datasetname == 'SemanticKitti':
                                sem_gt = np.squeeze(val_pt_labels[count])
                                inst_gt = np.squeeze(val_pt_inst[count])      
                            elif datasetname == 'nuscenes':
                                # sem_gt = np.squeeze(val_pt_inst[count]) // 1000
                                # sem_gt = np.vectorize(learning_map.__getitem__)(sem_gt)
                                sem_gt = np.squeeze(val_pt_labels[count])
                                inst_gt = np.squeeze(val_pt_inst[count])
                            else:
                                raise NotImplementedError

                            evaluator.addBatch(predict_labels_sem[count], panoptic, 
                                                sem_gt, inst_gt)
                            # PQ, SQ, RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ() # for debug
                            sem_hist_list.append(fast_hist_crop(
                                predict_labels_sem[count],
                                val_pt_labels[count],
                                unique_label))
                        else:
                            save_dict['item1'].append(predict_labels_sem[count])
                            save_dict['item2'].append(panoptic)
                            save_dict['item3'].append(val_pt_labels[count])
                            save_dict['item4'].append(val_pt_inst[count])
                            save_dict['item5'].append(fast_hist_crop(
                                predict_labels_sem[count],
                                val_pt_labels[count],
                                unique_label))
                        
                        #######################################################################################################
                        # debugging area

                        # debugging area
                        #######################################################################################################

                    pbar_val.update(1)
                    # if i_iter_val==10:
                    #     break
                    del val_dict
                # end for
                pbar_val.close()

                if args.local_rank != -1:
                    torch.distributed.barrier()
                    if args.local_rank > 0:
                        os.makedirs('./tmpdir', exist_ok=True)
                        pickle.dump(save_dict,
                                    open(os.path.join('./tmpdir', 'result_part_{}.pkl'.format(args.local_rank)), 'wb'))
                    torch.distributed.barrier()
                if args.local_rank < 1:
                    if args.local_rank == 0:
                        world_size = torch.distributed.get_world_size()
                        for i in range(world_size - 1):
                            part_file = os.path.join('./tmpdir', 'result_part_{}.pkl'.format(i + 1))
                            cur_dict = pickle.load(open(part_file, 'rb'))
                            for j in range(len(cur_dict['item1'])):
                                
                                # 用实例标签的语义
                                if datasetname == 'SemanticKitti':
                                    sem_gt = np.squeeze(cur_dict['item3'][j])
                                    inst_gt = np.squeeze(cur_dict['item4'][j])
                                elif datasetname == 'nuscenes':
                                    # sem_gt = np.squeeze(cur_dict['item4'][j] // 1000)
                                    # sem_gt = np.vectorize(learning_map.__getitem__)(sem_gt)
                                    sem_gt = np.squeeze(cur_dict['item3'][j])
                                    inst_gt = np.squeeze(cur_dict['item4'][j])
                                else:
                                    raise NotImplementedError

                                evaluator.addBatch(cur_dict['item1'][j], cur_dict['item2'][j], sem_gt,
                                                inst_gt)
                                sem_hist_list.append(cur_dict['item5'][j])
                        if os.path.isdir('./tmpdir'):
                            shutil.rmtree('./tmpdir')
                    # end args.local_rank == 0
                    ######################################################################################################
                    # get PQ results, only for rank 0(Distributed GPU) or rank -1(single GPU)
                    ######################################################################################################
                    PQ, SQ, RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ()
                    miou, ious = evaluator.getSemIoU()
                    logger.info('Validation per class PQ, SQ, RQ and IoU: ')
                    for class_name, class_pq, class_sq, class_rq, class_iou in zip(unique_label_str, class_all_PQ[1:-1],
                                                                                class_all_SQ[1:-1], class_all_RQ[1:-1],
                                                                                ious[1:-1]):
                        logger.info('%20s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%' % (
                            class_name, class_pq * 100, class_sq * 100, class_rq * 100, class_iou * 100))
                    
                    thing_upper_idx_dict = {"nuscenes": 10, "SemanticKitti":8} # thing label index: nusc 1-9, kitti 1-8
                    upper_idx = thing_upper_idx_dict[datasetname]
                    from utils.metric_util import cal_PQ_dagger
                    PQ_dagger = cal_PQ_dagger(class_all_PQ, class_all_SQ, upper_idx + 1)
                    PQ_th = np.nanmean(class_all_PQ[1: upper_idx + 1]) # exclude 0
                    SQ_th = np.nanmean(class_all_SQ[1: upper_idx + 1])
                    RQ_th = np.nanmean(class_all_RQ[1: upper_idx + 1])
                    PQ_st = np.nanmean(class_all_PQ[upper_idx+1: -1]) # exlucde 17 or 20
                    SQ_st = np.nanmean(class_all_SQ[upper_idx+1: -1])
                    RQ_st = np.nanmean(class_all_RQ[upper_idx+1: -1])
                
                    logger_msg1 = 'PQ %.1f  PQ_dagger  %.1f  SQ %.1f  RQ %.1f  |  PQ_th %.1f  SQ_th %.1f  RQ_th %.1f  |  PQ_st %.1f  SQ_st %.1f  RQ_st %.1f  |  mIoU %.1f' %(
                        PQ * 100, PQ_dagger * 100, SQ * 100, RQ * 100,
                        PQ_th * 100, SQ_th * 100, RQ_th * 100,
                        PQ_st * 100, SQ_st * 100, RQ_st * 100,
                        miou * 100)
                    logger.info(logger_msg1)
                    ######################################################################################################
                    
                    # save model if performance is improved
                    if best_val_PQ < PQ:
                        best_val_PQ = PQ
                        torch.save(my_model.state_dict(), model_save_path)
                    
                    logger_msg2 = 'Current val PQ is %.1f while the best val PQ is %.1f' %(
                            PQ * 100, best_val_PQ * 100)
                    logger_msg3 = 'Current val miou is %.1f' % (miou * 100)
                    logger.info(logger_msg2)
                    logger.info(logger_msg3)

                    ######################################################################################################
                    # get loss and mIoU result
                    ######################################################################################################
                    if start_training:
                        sem_l, class_l, dice_l, dice_pos_l = np.nanmean(loss_fn_dict['sem_loss']),np.nanmean(loss_fn_dict['class_loss']),\
                                np.nanmean(loss_fn_dict['dice_loss']), np.nanmean(loss_fn_dict['dice_pos_loss'])
                        logger.info(
                            'epoch %d iter %5d, avg_loss: %.1f, semantic loss: %.1f, class loss: %.1f, dice loss: %.1f, dice position loss: %.1f\n' %
                            (epoch, i_iter, last_avg_loss, sem_l, class_l, dice_l, dice_pos_l))

                    iou = per_class_iu(sum(sem_hist_list))
                    logger.info('Validation per class iou: ')
                    for class_name, class_iou in zip(unique_label_str, iou):
                        logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
                    val_miou = np.nanmean(iou) * 100
                    logger.info('Current val miou is %.1f' %
                                val_miou)
                    print('*' * 40)
                # end if args.local_rank < 1
                
                if args.local_rank != -1:
                    torch.distributed.barrier()
                loss_fn.reset_loss_dict()
            # end with torch.no_grad():
        
        if args.local_rank <= 0:
            print(f"Epoch {epoch} => Start Training...")
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        get_model(my_model).label_map = transform_map(SemKITTI2train(np.hstack([train_pt_dataset.base_thing_list,train_pt_dataset.base_stuff_list,train_pt_dataset.novel_thing_list,train_pt_dataset.novel_stuff_list,17])))
        get_model(my_model).label_inverse_map = inverse_transform(get_model(my_model).label_map)
        get_model(my_model).thing_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(train_pt_dataset.thing_list))
        get_model(my_model).stuff_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(train_pt_dataset.stuff_list))
        get_model(my_model).total_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(np.hstack([train_pt_dataset.base_thing_list+train_pt_dataset.base_stuff_list,17])))
        get_model(my_model).categroy_overlapping_mask = torch.from_numpy(np.full(len(get_model(my_model).total_class),True,dtype=bool))
        # for i,c in enumerate(my_model.novel_class):
        #     my_model.total_map[i+len()]
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter, train_dict in enumerate(train_dataset_loader):
            # training data process
            my_model.train()
            train_dict['voxel2point_map'] = [torch.from_numpy(i) for i in train_dict['voxel2point_map']]
            train_dict['point2voxel_map'] = [torch.from_numpy(i) for i in train_dict['point2voxel_map']]
            # train_dict['unique_grid_ind'] = [torch.from_numpy(i) for i in train_dict['unique_grid_ind']]
            train_dict['seenmask'] = [torch.from_numpy(i).cuda() for i in train_dict['seenmask']]
            train_dict['seen_unique_indices'] = [torch.from_numpy(i).cuda() for i in train_dict['seen_unique_indices']]
            train_dict['pol_voxel_ind'] = [torch.from_numpy(i).cuda() for i in train_dict['pol_voxel_ind']]
            # train_dict['return_fea'] = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
            #                             train_dict['return_fea']]
            train_dict['voxel_semantic_labels'] = [torch.from_numpy(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(SemKITTI2train(i))).type(torch.LongTensor).cuda() for i in train_dict['voxel_semantic_labels']]
            train_dict['voxel_instance_labels'] = [torch.from_numpy(i).type(torch.LongTensor).cuda() for i in train_dict['voxel_instance_labels']]
            if pix_fusion:
                # train_dict['camera_channel'] = torch.from_numpy(train_dict['camera_channel']).float().cuda()
                train_dict['pixel_coordinates'] = [torch.from_numpy(i).cuda() for i in train_dict['pixel_coordinates']]
                train_dict['ori_clip_vision_channel'] = torch.from_numpy(train_dict['ori_clip_vision_channel']).cuda()
                train_dict['text_features'] = torch.from_numpy(train_dict['text_features']).cuda()
                # train_dict['thing_list'] = torch.from_numpy(train_dict['thing_list']).cuda()
                # train_dict['img_indices_channel'] = [[torch.from_numpy(i).cuda() for i in split ] for split in train_dict['img_indices_channel']]
                # train_dict['point2img_index_channel'] = [[torch.from_numpy(i) for i in split ] for split in train_dict['point2img_index_channel']]
                # train_dict['masks'] = [torch.from_numpy(i).cuda() for i in train_dict['masks']]
                # train_dict['valid_mask'] = [torch.from_numpy(i).cuda() for i in train_dict['valid_mask']]
                # train_dict['im_label'] = SemKITTI2train(torch.cat([torch.from_numpy(i) for i in train_dict['im_label']], dim=0)).cuda()
            loss_dict = my_model(train_dict)
            loss = torch.sum(torch.stack(list(loss_dict.values())),dim=0)
            sem_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k=='loss_ce'or k=='loss_lovasz'])          
            cls_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_cls')])
            mask_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_mask')])
            dice_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_dice') and not k.startswith('loss_dice_pos')])            
            dice_pos_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_dice_pos')])
          
            loss_cpu = torch.tensor(loss, device="cpu").item()
            avg_loss = i_iter / (i_iter + 1) * avg_loss + 1 / (i_iter + 1) * loss_cpu

            # backward + optimize
            loss.backward()
            # grad accumulation
            if grad_accumu > 1:
                if (i_iter + 1) % grad_accumu == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
            loss_fn_dict['sem_loss'].append(sem_loss)
            loss_fn_dict['class_loss'].append(cls_loss)
            loss_fn_dict['dice_loss'].append(dice_loss)
            loss_fn_dict['dice_pos_loss'].append(dice_pos_loss)
            pbar.set_postfix({"sem_loss": sem_loss, 
                            "class_loss": cls_loss,
                            "mask_loss": mask_loss,
                            "dice_loss": dice_loss,
                            "dice_pos_loss": dice_pos_loss,
                            "avg_loss": avg_loss})
            pbar.update(1)
            start_training = True
            global_iter += 1
            del train_dict
        sem_l, class_l, dice_l, dice_pos_l = sem_loss,cls_loss,mask_loss,dice_loss,
        pbar.close()
        scheduler_steplr.step()
        epoch += 1
        last_avg_loss = avg_loss

    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
