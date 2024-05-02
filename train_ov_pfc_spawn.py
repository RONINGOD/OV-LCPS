
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.parallel import DistributedDataParallel as DDP
from mmengine import Config
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingParamScheduler
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler # 0.4.12
# from torch.optim.lr_scheduler import 
from mmengine.optim import CosineAnnealingParamScheduler
from utils.AppLogger import AppLogger
import torch.optim as optim
from network.PFC import PFC
from nuscenes import NuScenes
from dataloader.dataset import collate_fn_OV, Nuscenes_pt, spherical_dataset, OV_Nuscenes_pt,collate_dataset_info, SemKITTI_pt,ov_spherical_dataset
import warnings
warnings.filterwarnings("ignore")
from mmengine import ProgressBar
import yaml
from tqdm import tqdm
from utils.load_save_util import revise_ckpt,revise_ckpt_2,SemKITTI2train,transform_map,inverse_transform, SemKITTI2train_single, get_model
from utils.eval_pq import PanopticEval,OV_PanopticEval
from utils.metric_util import per_class_iu, fast_hist_crop
from utils.metric_util import cal_PQ_dagger
from mmengine import Config
import pickle
import shutil
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True  # 是否自动加速，自动选择合适算法，false选择固定算法
    torch.backends.cudnn.deterministic = True  # 为了消除该算法本身的不确定性

    # load config
    with open(args.configs, 'r') as s:
        cfg = Config(yaml.safe_load(s))
    cfg.work_dir = args.work_dir

    # init DDP
    if args.launcher == 'none':
        distributed = False
        rank = 0
        cfg.gpu_ids = [0]         # debug
    else:
        distributed = True
        seed = 3407
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if dist.get_rank() != 0:
            import builtins
            builtins.print = pass_print

    # configure logger
    if local_rank == 0 and rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.configs)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='train_log', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')
    datasetname = cfg['dataset']['name']
    version = cfg['dataset']['version']
    data_path = cfg['dataset']['path']
    num_worker = cfg['dataset']['num_worker']
    train_batch_size = cfg['model']['train_batch_size']
    val_batch_size = cfg['model']['val_batch_size']
    model_load_path = cfg['model']['model_load_path']
    model_save_path = cfg['model']['model_save_path']
    lr = cfg['model']['learning_rate']
    lr_step = cfg['model']['LR_MILESTONES']
    lr_gamma = cfg['model']['LR_GAMMA']
    grid_size = cfg['dataset']['grid_size']
    pix_fusion = cfg['model']['pix_fusion']
    min_points = cfg['dataset']['min_points']
    cumulative_iters = 1
    # 初始化类别名称和数量，不包括noise类。
    unique_label, unique_label_str = collate_dataset_info(cfg)
    # 加noise类 
    nclasses = len(unique_label) + 1
    my_model = PFC(cfg, nclasses)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')
    # NuScenes: MultiStepLR; SemanticKitti: CosineAnnealingLR, CosineAnnealingWarmRestarts
    optimizer = optim.Adam(my_model.parameters(), lr=lr,weight_decay=0.01)
    scheduler_steplr = MultiStepLR(optimizer, milestones=lr_step, gamma=lr_gamma,)
    # scheduler_steplr = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8, verbose=True)
    # scheduler_steplr = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-8, verbose=True)
    
    if datasetname == 'SemanticKitti':
        train_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfg, split='train', return_ref=True)
        val_pt_dataset = SemKITTI_pt(os.path.join(data_path, 'dataset', 'sequences'), cfg, split='val', return_ref=True)
    elif datasetname == 'nuscenes':
        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        assert version == "v1.0-trainval" or version == "v1.0-mini"
        train_pt_dataset = OV_Nuscenes_pt(data_path, split='train', cfgs=cfg, nusc=nusc, version=version)
        val_pt_dataset = OV_Nuscenes_pt(data_path, split='val', cfgs=cfg, nusc=nusc, version=version)
    else:
        raise NotImplementedError

    train_dataset = ov_spherical_dataset(train_pt_dataset, cfg, ignore_label=0)
    val_dataset = ov_spherical_dataset(val_pt_dataset, cfg, ignore_label=0, use_aug=False)
    collate_fn = collate_fn_OV
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,drop_last=False)
    else:
        sampler = None
        val_sampler = None
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       collate_fn=collate_fn,
                                                       pin_memory=True,
                                                       sampler=sampler,
                                                       num_workers=num_worker)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_batch_size,
                                                     collate_fn=collate_fn,
                                                     pin_memory=True,
                                                     sampler=val_sampler,
                                                     num_workers=num_worker)
    if datasetname == 'nuscenes':
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
    learning_map = nuscenesyaml['learning_map']
    # resume and load
    epoch = 0
    best_miou, best_pq = 0.0, 0.0
    global_iter = 0
    print_freq = cfg.model.print_freq
    cfg.resume = ''
    if osp.exists(osp.join(osp.abspath(args.work_dir), 'latest.pth')):
        cfg.resume = osp.join(osp.abspath(args.work_dir), 'latest.pth')
    if args.resume!='':
        cfg.resume = args.resume
    
    print('resume from: ', cfg.resume)
    print('work dir: ', args.work_dir)

    if cfg.resume and osp.exists(cfg.resume):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler_steplr.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_miou' in ckpt:
            best_miou = ckpt['best_miou']
        if 'best_pq' in ckpt:
            best_pq = ckpt['best_pq']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.model.model_load_path:
        ckpt = torch.load(cfg.model.model_load_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    evaluator = OV_PanopticEval(nclasses, None, [0], min_points=min_points,offset=2**32)
    loss_fn_dict ={
        'sem_loss':[],
        'class_loss':[],
        'mask_loss':[],
        'dice_loss':[],
        'dice_pos_loss':[]
    }
    avg_loss = 0.0

    while epoch < cfg['model']['max_epoch']:
        if local_rank < 1:
            print(f"Epoch {epoch} => Start Training...")
        my_model.train()
        
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        # for cumulative_iters > 1
        if cumulative_iters > 1:
            total_iters = len(train_dataset_loader)
            divisible_iters = total_iters // cumulative_iters * cumulative_iters
            remainder_iters = total_iters - divisible_iters
            logger.info(f'cumulative_iters: {cumulative_iters}, total_iters: {total_iters}, \
                        divisible_iters: {divisible_iters}, remainder_iters: {remainder_iters}')
        loss_list = []
        # time.sleep(1)
        data_time_s = time.time()
        time_s = time.time()
        bar = tqdm(total=len(train_dataset_loader))
        to_cuda_list = ['voxel2point_map','point2voxel_map','pol_voxel_ind','grid_mask',
                        'voxel_instance_labels','point_mask','clip_features','text_features']
        get_model(my_model).label_map = transform_map(np.hstack([0,train_pt_dataset.base_thing_list,train_pt_dataset.base_stuff_list,train_pt_dataset.novel_thing_list,train_pt_dataset.novel_stuff_list]))
        get_model(my_model).label_inverse_map = inverse_transform(get_model(my_model).label_map)
        get_model(my_model).thing_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(train_pt_dataset.thing_list)
        get_model(my_model).stuff_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(train_pt_dataset.stuff_list)
        get_model(my_model).total_class = np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(np.hstack([0,train_pt_dataset.base_thing_list+train_pt_dataset.base_stuff_list]))
        get_model(my_model).categroy_overlapping_mask = torch.from_numpy(np.full(len(get_model(my_model).total_class),True,dtype=bool))
        if distributed:
            torch.distributed.barrier()
        for i_iter, data in enumerate(train_dataset_loader):
            for k in to_cuda_list:
                if isinstance(data[k],list):
                    for i in range(len(data[k])):
                        data[k][i] = torch.from_numpy(data[k][i]).cuda()
                else:
                    data[k] = torch.from_numpy(data[k]).cuda()
            data['voxel_semantic_labels'] = [torch.from_numpy(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(i)).type(torch.LongTensor).cuda() for i in data['voxel_semantic_labels']]
        
            data_time_e = time.time()
            # with torch.cuda.amp.autocast():
            loss_dict = my_model(data)
            loss = torch.sum(torch.stack(list(loss_dict.values())),dim=0)
            sem_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k=='loss_ce'or k=='loss_lovasz'])          
            cls_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_cls')])
            mask_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_mask')])
            dice_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_dice') and not k.startswith('loss_dice_pos')])            
            dice_pos_loss = np.nanmean([loss_dict[k].detach().cpu().numpy()   for k in loss_dict.keys() if k.startswith('loss_dice_pos')])
            loss_cpu = torch.tensor(loss, device="cpu").item()
            avg_loss = i_iter / (i_iter + 1) * avg_loss + 1 / (i_iter + 1) * loss_cpu
            if cumulative_iters > 1:
                loss_factor = cumulative_iters if i_iter < divisible_iters else remainder_iters
                loss_list.append(loss.item())
                loss = loss / loss_factor
                loss.backward()
                # scaler.scale(loss).backward()
                if (i_iter+1) % cumulative_iters == 0 or i_iter + 1 == len(train_dataset_loader):
                    # scaler.unscale_(optimizer)
                    # grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    # scaler.step(optimizer)
                    # scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.zero_grad()
                loss_list.append(loss.item())

            # scheduler.step()
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('\n[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), lr: %.7f, time: %.3f (%.3f)'%(
                    epoch+1, i_iter, len(train_dataset_loader), 
                    loss_list[-1], np.mean(loss_list), lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
                loss_list = []
            loss_fn_dict['sem_loss'].append(sem_loss)
            loss_fn_dict['class_loss'].append(cls_loss)
            loss_fn_dict['dice_loss'].append(dice_loss)
            loss_fn_dict['mask_loss'].append(mask_loss)
            loss_fn_dict['dice_pos_loss'].append(dice_pos_loss)
            data_time_s = time.time()
            time_s = time.time()
            bar.set_postfix({"sem_loss": sem_loss, 
                "class_loss": cls_loss,
                "mask_loss": mask_loss,
                "dice_loss": dice_loss,
                "dice_pos_loss": dice_pos_loss,
                "avg_loss": avg_loss})
            bar.update(1)
        bar.close()
        if distributed:
            torch.distributed.barrier()
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_steplr.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_miou':best_miou,
                'best_pq':best_pq,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), 'latest.pth')
            torch.save(dict_to_save, save_file_name)
            # dst_file = osp.join(args.work_dir, 'latest.pth')
            # symlink(save_file_name, dst_file)
        sem_l, class_l, mask_l,dice_l, dice_pos_l = sem_loss,cls_loss,mask_loss,dice_loss,dice_pos_loss
        scheduler_steplr.step()
        epoch += 1
        # eval
        my_model.eval()
        evaluator.reset()
        sem_hist_list = []
        get_model(my_model).label_map = transform_map(np.hstack([0,val_pt_dataset.base_thing_list,val_pt_dataset.base_stuff_list,val_pt_dataset.novel_thing_list,val_pt_dataset.novel_stuff_list]))
        get_model(my_model).label_inverse_map = inverse_transform(get_model(my_model).label_map)
        get_model(my_model).thing_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(val_pt_dataset.thing_list))
        get_model(my_model).stuff_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(val_pt_dataset.stuff_list))
        get_model(my_model).total_class = np.sort(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(np.hstack([0,val_pt_dataset.base_thing_list+val_pt_dataset.base_stuff_list])))
        get_model(my_model).categroy_overlapping_mask = torch.from_numpy(np.hstack((np.full(1,True,dtype=bool),np.full(len(val_pt_dataset.base_thing_list+val_pt_dataset.base_stuff_list), True, dtype=bool),np.full(len(val_pt_dataset.novel_thing_list+val_pt_dataset.novel_stuff_list),False,dtype=bool))))

        with torch.no_grad():
            logger.info("epoch: %d   lr: %.5f\n" % (epoch, optimizer.param_groups[0]['lr']))

            if local_rank < 1:
                print(f"Epoch {epoch} => Start Evaluation...")
            if local_rank > 0:
                save_dict = {
                    'item1': [],          
                    'item2': [],
                    'item3': [],
                    'item4': [],
                    'item5': [],
                }
            val_bar = tqdm(total=len(val_dataset_loader))
            for i_iter_val, data in enumerate(val_dataset_loader):                  
                for k in to_cuda_list:
                    if isinstance(data[k],list):
                        for i in range(len(data[k])):
                            data[k][i] = torch.from_numpy(data[k][i]).cuda()
                    else:
                        data[k] = torch.from_numpy(data[k]).cuda()
                data['voxel_semantic_labels'] = [torch.from_numpy(np.vectorize(get_model(my_model).label_inverse_map.__getitem__)(i)).type(torch.LongTensor).cuda() for i in data['voxel_semantic_labels']]
                predict_labels_sem, pts_instance_preds = my_model(data)
                predict_labels_sem = [np.vectorize(get_model(my_model).label_map.__getitem__)(sem) for sem in predict_labels_sem]
                val_grid = data['pol_voxel_ind']
                val_pt_labels = data['pt_sem_label']
                val_pt_inst = data['pt_ins_label']
                for count, i_val_grid in enumerate(val_grid):
                    panoptic = pts_instance_preds[count]
                    if local_rank<1:
                        if datasetname == 'SemanticKitti':
                            sem_gt = np.squeeze(val_pt_labels[count])
                            inst_gt = np.squeeze(val_pt_inst[count])      
                        elif datasetname == 'nuscenes':
                            sem_gt = np.squeeze(val_pt_labels[count])
                            inst_gt = np.squeeze(val_pt_inst[count])
                        else:
                            raise NotImplementedError
                        evaluator.addBatch(predict_labels_sem[count], panoptic,sem_gt, inst_gt)
                        sem_hist_list.append(fast_hist_crop(predict_labels_sem[count],val_pt_labels[count],unique_label))
                    else:
                        save_dict['item1'].append(predict_labels_sem[count])
                        save_dict['item2'].append(panoptic)
                        save_dict['item3'].append(val_pt_labels[count])
                        save_dict['item4'].append(val_pt_inst[count])
                        save_dict['item5'].append(fast_hist_crop(
                            predict_labels_sem[count],
                            val_pt_labels[count],
                            unique_label))
                    val_bar.set_postfix({"semantic": np.unique(predict_labels_sem[count]), 
                            "instance_id": np.unique(panoptic)})
                val_bar.update(1)
        val_bar.close()
        if distributed:
            torch.distributed.barrier()
            if local_rank > 0:
                os.makedirs(osp.join(osp.abspath(args.work_dir),'tmpdir'), exist_ok=True)
                pickle.dump(save_dict,
                            open(os.path.join(osp.abspath(args.work_dir),'tmpdir', 'result_part_{}.pkl'.format(local_rank)), 'wb'))
            torch.distributed.barrier()
        if local_rank < 1:
            if local_rank == 0:
                world_size = torch.distributed.get_world_size()
                for i in range(world_size - 1):
                    part_file = os.path.join(osp.abspath(args.work_dir),'tmpdir', 'result_part_{}.pkl'.format(i + 1))
                    cur_dict = pickle.load(open(part_file, 'rb'))
                    for j in range(len(cur_dict['item1'])):
                        
                        # 用实例标签的语义
                        if datasetname == 'SemanticKitti':
                            sem_gt = np.squeeze(cur_dict['item3'][j])
                            inst_gt = np.squeeze(cur_dict['item4'][j])
                        elif datasetname == 'nuscenes':
                            sem_gt = np.squeeze(cur_dict['item3'][j])
                            inst_gt = np.squeeze(cur_dict['item4'][j])
                        else:
                            raise NotImplementedError

                        evaluator.addBatch(cur_dict['item1'][j], cur_dict['item2'][j], sem_gt,
                                        inst_gt)
                        sem_hist_list.append(cur_dict['item5'][j])
                if os.path.isdir(osp.join(osp.abspath(args.work_dir),'tmpdir')):
                    shutil.rmtree(osp.join(osp.abspath(args.work_dir),'tmpdir'))
            PQ, SQ, RQ, class_all_PQ, class_all_SQ, class_all_RQ = evaluator.getPQ()
            miou, ious = evaluator.getSemIoU()
            logger.info('Validation per class PQ, SQ, RQ and IoU: ')
            for class_name, class_pq, class_sq, class_rq, class_iou in zip(unique_label_str, class_all_PQ[1:],
                                                                                    class_all_SQ[1:], class_all_RQ[1:],
                                                                                    ious[1:]):
                logger.info('%20s : %6.8f%%  %6.8f%%  %6.8f%%  %6.8f%%' % (class_name, class_pq * 100, class_sq * 100, class_rq * 100, class_iou * 100))
            thing_upper_idx_dict = {"nuscenes": 10, "SemanticKitti":8}
            upper_idx = thing_upper_idx_dict[datasetname]
            PQ_dagger = cal_PQ_dagger(class_all_PQ, class_all_SQ, upper_idx + 1)
            PQ_th = np.nanmean(class_all_PQ[1: upper_idx + 1]) # exclude 0
            SQ_th = np.nanmean(class_all_SQ[1: upper_idx + 1])
            RQ_th = np.nanmean(class_all_RQ[1: upper_idx + 1])
            PQ_st = np.nanmean(class_all_PQ[upper_idx+1:])
            SQ_st = np.nanmean(class_all_SQ[upper_idx+1:])
            RQ_st = np.nanmean(class_all_RQ[upper_idx+1:])
            PQ_N_th = np.nanmean(class_all_PQ[val_pt_dataset.novel_thing_list])
            PQ_N_st = np.nanmean(class_all_PQ[val_pt_dataset.novel_stuff_list])
            RQ_N_th = np.nanmean(class_all_RQ[val_pt_dataset.novel_thing_list])
            RQ_N_st = np.nanmean(class_all_RQ[val_pt_dataset.novel_stuff_list])
            SQ_N_th = np.nanmean(class_all_SQ[val_pt_dataset.novel_thing_list])
            SQ_N_st = np.nanmean(class_all_SQ[val_pt_dataset.novel_stuff_list])
            
            logger_msg1 = 'PQ %.8f  PQ_dagger  %.8f  SQ %.8f  RQ %.8f  |  PQ_th %.8f  SQ_th %.8f  RQ_th %.8f  |  PQ_st %.8f  SQ_st %.8f  RQ_st %.8f  |  PQ_N_th %.8f  PQ_N_st %.8f  RQ_N_th %.8f  RQ_N_st %.8f  SQ_N_th %.8f  SQ_N_st %.8f  |  mIoU %.8f' %(
                    PQ * 100, PQ_dagger * 100, SQ * 100, RQ * 100,
                    PQ_th * 100, SQ_th * 100, RQ_th * 100,
                    PQ_st * 100, SQ_st * 100, RQ_st * 100,
                    PQ_N_th * 100,PQ_N_st*100, RQ_N_th*100, RQ_N_st*100, SQ_N_th*100, SQ_N_st*100,
                    miou * 100)
            logger.info(logger_msg1)
            if PQ>best_pq:
                best_pq = PQ
                dict_to_save = {
                    'state_dict': my_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler_steplr.state_dict(),
                    'epoch': epoch + 1,
                    'global_iter': global_iter,
                    'best_miou':best_miou,
                    'best_pq':best_pq,
                }
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'best_pq_{PQ}.pth')
                torch.save(dict_to_save, save_file_name)
            best_miou = max(best_miou,miou)
            logger.info('Current val miou is %.8f while the best val miou is %.8f' %
                    (miou*100, best_miou*100))
            logger.info('Current val PQ is %.8f while the best val PQ is %.8f' %
                    (PQ*100, best_pq*100))
            iou = per_class_iu(sum(sem_hist_list))
            logger.info('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                logger.info('%s : %.8f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100
            logger.info('Current val miou is %.1f' %
                        val_miou)
            logger.info('*' * 40)
            # print('*' * 40)
            sem_l, class_l, dice_l, dice_pos_l,mask_l = np.nanmean(loss_fn_dict['sem_loss']),np.nanmean(loss_fn_dict['class_loss']),\
                                    np.nanmean(loss_fn_dict['dice_loss']), np.nanmean(loss_fn_dict['dice_pos_loss']),np.nanmean(loss_fn_dict['mask_loss'])
            
            logger.info(
                'epoch %d iter %5d, avg_loss: %.4f, semantic loss: %.4f, class loss: %.4f, mask_loss: %.4f ,dice loss: %.4f, dice position loss: %.4f\n' %
                (epoch, i_iter, avg_loss, sem_l, class_l, mask_l, dice_l, dice_pos_l))
        if distributed:
            torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()

            

        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('-c', '--configs', default='configs/pa_po_nuscenes.yaml')
    parser.add_argument('-w', '--work_dir', default='work_dir/nusc_pfc/')
    # parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('-r', "--resume", type=str, default='')
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
# python train_ov_pfc_spawn.py --launcher pytorch -c configs/open_pa_po_nuscenes_mini.yaml -w work_dir/nusc_pfc/mini
# python train_openseg_pfc.py --launcher pytorch -c configs/open_pa_po_nuscenes.yaml -w work_dir/nusc_pfc/
# python train_ov_pfc_spawn.py --launcher pytorch -c configs/open_pa_po_nuscenes_mini.yaml -w work_dir/nusc_pfc/mini