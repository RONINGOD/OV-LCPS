import os
import torch
import argparse
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper,adjust_intrinsic
from nuscenes.nuscenes import NuScenes
import pickle
import time
from dataloader.utils import PCDTransformTool
from pyquaternion import Quaternion
import clip
import copy

def get_parser():
    parser = argparse.ArgumentParser(description="openseg demo for builtin configs")
    parser.add_argument(
        "--input",
        default='/home/coisini/data/nuscenes',
        type=str,
        help="nuscenes data root",
    )
    parser.add_argument("--version",default='v1.0-mini',type=str,help='nuscenes data version')
    parser.add_argument("--split",default='train',type=str,help='nuscenes data split')
    parser.add_argument("--start",default=0,type=int,help='nuscenes data start id')
    parser.add_argument(
        "--output",
        default='/home/coisini/data/nuscenes_openseg_features',
        help="A file or directory to save output features."
    )
    parser.add_argument(
        '--openseg_model', 
        type=str, 
        default='/home/coisini/project/lcps/checkpoints/openseg_exported_clip', 
        help='Where is the exported OpenSeg model'
    )
    return parser

def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_text_embedding(categories):
    model, preprocess = clip.load("ViT-L/14@336px")
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
            texts = clip.tokenize(category)  #tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = model.encode_text(texts)  #embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
            
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    del model
    return all_text_embeddings.cpu().numpy().T

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #### Dataset specific parameters #####
    img_dim = (800, 450)
    ######################################

    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.feat_dim = 768 # CLIP feature dimension
    args.img_dim = img_dim
    data_root = args.input
    version = args.version
    split = args.split
    output = args.output
    assert data_root, "The input path(s) was not found"
    make_file(output) 

    cut_bound = 5 # do not use the features on the image boundary
    img_size = (800, 450) # resize image
    nusc = NuScenes(version=version,dataroot=data_root,verbose=True)

    with open(f'{data_root}/nuscenes_pkl/nuscenes_infos_{split}_mini.pkl', 'rb') as f:
        nusc_data = pickle.load(f)['infos']
    CAM_NAME_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
                     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
    NUSCENES_LABELS_16 = ['barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person', 'traffic cone',
                      'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
    base_things = ['barrier','bicycle','car','construction_vehicle','traffic_cone',
                    'trailer','truck']
    base_stuff = ['driveable_surface','other_flat','sidewalk','terrain','manmade']
    base_total = base_things+base_stuff
    novel_things = ['bus','motorcycle','pedestrian']
    novel_stuff = ['vegetation']
    novel_total = novel_things+novel_stuff
    total = base_total+novel_total
    base_total = ['noise'] + base_total
    total = ['noise'] + total
    text_features = build_text_embedding(base_total)
    np.save(os.path.join(output,'base_text_features.npy'),text_features)
    text_features = build_text_embedding(total)
    np.save(os.path.join(output,'total_text_features.npy'),text_features)
    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None
    
    start_id=args.start
    pbar = tqdm(total=len(nusc_data))
    for index,sample_data in enumerate(nusc_data):
        start_time = time.time()
        info = nusc_data[index]
        token = sample_data['token']
        lidar_path = info['lidar_path']
        lidar_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidar_channel = nusc.get("sample_data", lidar_token)
        if version == "v1.0-trainval":
            lidar_path = lidar_path[16:]
        elif version == "v1.0-mini":
            lidar_path = lidar_path[44:]
        elif version == "v1.0-test":
            lidar_path = lidar_path[16:]
        pcd_data_name = lidar_path.split('.')[0]
        img_features_path = os.path.join(output,pcd_data_name+'.npy')
        # if os.path.exists(img_features_path) or index<start_id:
        #     pbar.set_postfix({
        #         "token":sample_data['token'],
        #         "finished in ":"{:.2f}s".format(time.time()-start_time)
        #     })
        #     pbar.update(1)
        #     continue

        points = np.fromfile(os.path.join(data_root, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        pcd = PCDTransformTool(points[:, :3])
        n_points_cur = points.shape[0]
        rec = nusc.get('sample', token)
        num_img = len(CAM_NAME_LIST)
        img_list = []
        counter = torch.zeros((n_points_cur, 1))
        sum_features = torch.zeros((n_points_cur, args.feat_dim))
        vis_id = torch.zeros((n_points_cur, num_img), dtype=int)
        for img_id,cam_name in enumerate(CAM_NAME_LIST):
            cam_token = info['cams'][cam_name]['sample_data_token']
            cam_channel = nusc.get('sample_data', cam_token)
            camera_sample = nusc.get('sample_data', rec['data'][cam_name])
            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = nusc.get('calibrated_sensor', cam_channel['calibrated_sensor_token'])
            pcd_trans_tool = copy.deepcopy(pcd)
            pcd_trans_tool.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(cs_record['translation']))
            # Second step: transform from ego to the global frame at timestamp of the first frame in the sequence pack.
            poserecord = nusc.get('ego_pose', lidar_channel['ego_pose_token'])
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pcd_trans_tool.translate(np.array(poserecord['translation']))
            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.get('ego_pose', cam_channel['ego_pose_token'])
            pcd_trans_tool.translate(-np.array(poserecord['translation']))
            pcd_trans_tool.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
            # Fifth step: project from 3d coordinate to 2d coordinate
            K = np.array(cs_record['camera_intrinsic'])
            K = adjust_intrinsic(K, intrinsic_image_dim=(1600, 900), image_dim=img_size)
            pcd_trans_tool.pcd2image(K)
            pixel_coord = pcd_trans_tool.pcd[:3, :]
            pixel_coord = np.round(pixel_coord).astype(int)
            inside_mask = (pixel_coord[0] >= cut_bound) * (pixel_coord[1] >= cut_bound) \
            * (pixel_coord[0] < img_size[0]-cut_bound) \
            * (pixel_coord[1] < img_size[1]-cut_bound)
            
            front_mask = pixel_coord[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask
            mapping = np.zeros((3, n_points_cur), dtype=int)
            mapping[0][inside_mask] = pixel_coord[1][inside_mask]
            mapping[1][inside_mask] = pixel_coord[0][inside_mask]
            mapping[2][inside_mask] = 1
            mapping_3d = np.ones([n_points_cur, 4], dtype=int)
            mapping_3d[:, 1:4] = mapping.T
            if mapping_3d[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue
            mapping_3d = torch.from_numpy(mapping_3d)
            mask = mapping_3d[:, 3]
            vis_id[:, img_id] = mask
            img_path = os.path.join(data_root, camera_sample['filename'])
            # openseg
            feat_2d = extract_openseg_img_feature(
                img_path, args.openseg_model, args.text_emb, img_size=[img_size[1], img_size[0]])
            feat_2d_3d = feat_2d[:, mapping_3d[:, 1], mapping_3d[:, 2]].permute(1, 0)
            counter[mask!=0]+= 1
            sum_features[mask!=0] += feat_2d_3d[mask!=0]
            
        counter[counter==0] = 1e-5
        feat_bank = sum_features/counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

        mask = torch.zeros(n_points_cur, dtype=torch.bool)
        mask[point_ids] = True
        feat_save = feat_bank[mask].numpy()
        mask = mask.numpy()
        dir_name = os.path.dirname(img_features_path)
        make_file(dir_name)
        save_dict = {"point_feat": feat_save,
                     "point_mask": mask}
        np.save(img_features_path,save_dict)   
        pbar.set_postfix({
            "token":sample_data['token'],
            "finished in ":"{:.2f}s".format(time.time()-start_time)
        })
        pbar.update(1)
    pbar.close()
    return

if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Arguments:")
    print(args)
    main(args)