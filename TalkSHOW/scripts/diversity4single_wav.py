import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["EGL_DEVICE_ID"] = '1'
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(os.getcwd())
os.environ['DISPLAY'] = ':0'
from transformers import Wav2Vec2Processor
from glob import glob

import numpy as np
import json
import smplx as smpl
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool

import time
from data_utils.consts import speaker_id
import pickle
from data_utils.lower_body import c_index_3d
import trimesh
from trimesh.exchange.export import export_mesh
from trimesh.exchange.load import load_mesh

def init_model(model_name, model_path, args, config):
    if model_name == 's2g_face':
        generator = s2g_face(
            args,
            config,
        )
    elif model_name == 's2g_body_vq':
        generator = s2g_body_vq(
            args,
            config,
        )
    elif model_name == 's2g_body_pixel':
        generator = s2g_body_pixel(
            args,
            config,
        )
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if model_name == 'smplx_S2G':
        generator.generator.load_state_dict(model_ckpt['generator']['generator'])

    elif 'generator' in list(model_ckpt.keys()):
        generator.load_state_dict(model_ckpt['generator'])
    else:
        model_ckpt = {'generator': model_ckpt}
        generator.load_state_dict(model_ckpt)

    return generator

with open(' /Code/TalkSHOW/data_utils/hand_component.json') as file_obj:
    comp = json.load(file_obj)
    left_hand_c = np.asarray(comp['left'])
    right_hand_c = np.asarray(comp['right'])
def to3d(data):
    left_hand_pose = np.einsum('bi,ij->bj', data[:, 75:87], left_hand_c[:12, :])
    right_hand_pose = np.einsum('bi,ij->bj', data[:, 87:99], right_hand_c[:12, :])
    data = np.concatenate((data[:, :75], left_hand_pose, right_hand_pose), axis=-1) # 99 - 12*2 + 45*2
    return data


def get_vertices(smplx_model, betas, result_list, config=None, require_pose=False, isGT=True):
    
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    exp = config.Data.pose.expression

    for i in result_list:
        vertices = []
        poses = []
        joints = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas, # torch.Size([1, 300])
                                 expression=i[j][165:265].unsqueeze_(dim=0) if exp else expression, # torch.Size([1, 100])
                                 jaw_pose=i[j][0:3].unsqueeze_(dim=0), # torch.Size([1, 3])
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            # output = smplx_model(betas=betas,
            #                      expression=i[j][165:265].unsqueeze_(dim=0) if exp else expression,
            #                      return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)

            if config.Infer.save_mesh.save_obj and not isGT:
                vertices_save_obj = output.vertices[0].detach().cpu().numpy()
                out_mesh = trimesh.Trimesh(
                vertices_save_obj, smplx_model.faces,
                process=False)
                out_mesh.apply_transform(trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0]))

                save_mesh_dir = os.path.join(config.Infer.save_root_path, config.Infer.file_name, 'mesh')
                # save_mesh_dir = os.path.join(config.Infer.save_root_path, 'mesh')

                os.makedirs(save_mesh_dir, exist_ok=True)
                output_name=os.path.join(save_mesh_dir, str(j) + '.obj')
                export_mesh(out_mesh, output_name, file_type='obj')

            pose = output.body_pose
            poses.append(pose.detach().cpu())

            joints.append(output.joints)

        if config.Infer.save_joints:
            assert len(result_list) == 1
            joints_tensor = torch.cat(joints, dim=0)
            save_joints_dir = os.path.join(config.Infer.save_root_path, config.Infer.file_name, 'joints')
            os.makedirs(save_joints_dir, exist_ok=True)
            output_name = os.path.join(save_joints_dir, 'all_joints.pt')
            torch.save(joints_tensor, output_name)

        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(data_root, g_body, g_face, g_body2, exp_name, infer_loader, infer_set, device, norm_stats, smplx,
          smplx_model, rendertool, args=None, config=None, wav_files=None, speaker_name=None, speaker_beta_path=None):
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
    num_sample = 1
    face = False
    if face:
        body_static = torch.zeros([1, 162], device='cuda')
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)
    stand = config.Infer.stand
    j = 0
    gt_0 = None

    if type(wav_files) is not list:
        wav_files = [wav_files]
    wav_file = wav_files[0]
    if config.Infer.show_vis.show_visualise: # just for show's output visualize
        
        pkl_path = config.Infer.show_vis.show_pkl_path
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file)[0]
        
        # jaw_pose = np.array(data['jaw_pose']) # 3
        # leye_pose = np.array(data['leye_pose']) # 3
        # reye_pose = np.array(data['reye_pose']) # 3
        # global_orient = np.array(data['global_orient']).squeeze() # 3
        # body_pose = np.array(data['body_pose_axis']) # 63
        # left_hand_pose = np.array(data['left_hand_pose']) # 12
        # right_hand_pose = np.array(data['right_hand_pose']) # 12
        # expression = np.array(data['expression'])

        # full_body = np.concatenate(
        #     (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)
        # assert full_body.shape[1] == 99

        # full_body = to3d(full_body) # (300, 99) -> (300, 165)
        # poses = full_body[:, c_index_3d] # torch.Size([300, 129])

        # jaw_pose = torch.tensor(jaw_pose, dtype=torch.float64).to('cuda') 
        # poses = torch.tensor(poses, dtype=torch.float64).to('cuda') 
        # expression = torch.tensor(expression, dtype=torch.float64).to('cuda') 
        
        # smplx_param = torch.cat([jaw_pose, poses, expression], dim=-1) # (300, 232)
        # smplx_param = part2full(smplx_param, stand) # torch.Size([300, 265])
        # betas = torch.tensor(np.array(data['betas']), dtype=torch.float64).to('cuda') # (1, 300)

        # # smplx_param=smplx_param[1000:2000, ...]
        # result_list = [smplx_param]
        # # full_body = np.concatenate((full_body, expression), axis=1) # (300, 265)
        # vertices_list, _ = get_vertices(smplx_model, betas, result_list, config) # (300, 10475, 3) gt&pred
        # if data['transl'].shape[0] < data['batch_size']:
        #     data['transl'] = np.tile(data['transl'], (data['batch_size'] // data['transl'].shape[0] + 1, 1))[:data['batch_size'], :]
        
        with open(config.Infer.show_vis.hand_component_path) as file_obj:
            comp = json.load(file_obj)
            left_hand_c = np.asarray(comp['left'])
            right_hand_c = np.asarray(comp['right'])
        smplx_param = data
        vertices_list = []
        vertices = []

        if config.Infer.save_mesh.save_obj:
            for j in range(smplx_param['batch_size']):
                dtype = torch.float64
                left_hand_pose = np.einsum('bi,ij->bj', smplx_param['left_hand_pose'][j][None, ...], left_hand_c[:12, :])
                right_hand_pose = np.einsum('bi,ij->bj', smplx_param['right_hand_pose'][j][None, ...], right_hand_c[:12, :])
                output = smplx_model(betas=torch.tensor(smplx_param['betas'], dtype=dtype).to('cuda'), # torch.Size([1, 300]) 
                                expression=torch.tensor(smplx_param['expression'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 100]) 
                                jaw_pose=torch.tensor(smplx_param['jaw_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 3]) 
                                leye_pose=torch.tensor(smplx_param['leye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                reye_pose=torch.tensor(smplx_param['reye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                global_orient=torch.tensor(smplx_param['global_orient'][j], dtype=dtype).to('cuda'), 
                                body_pose=torch.tensor(smplx_param['body_pose_axis'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                left_hand_pose=torch.tensor(left_hand_pose, dtype=dtype).to('cuda'),  
                                right_hand_pose=torch.tensor(right_hand_pose, dtype=dtype).to('cuda'), 
                                transl=torch.tensor(smplx_param['transl'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                return_verts=True)
                vertices = output.vertices[0].detach().cpu().numpy()
                # for idx in range(vertices_.shape[0]):  # idx=0
                #     vertices = vertices_[idx]
                out_mesh = trimesh.Trimesh(
                vertices, smplx_model.faces,
                process=False)
                out_mesh.apply_transform(trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0]))

                save_mesh_dir = os.path.join(config.Infer.save_root_path, config.Infer.file_name, 'mesh')
                # save_mesh_dir = os.path.join(config.Infer.save_root_path, 'mesh')

                os.makedirs(save_mesh_dir, exist_ok=True)
                output_name=os.path.join(save_mesh_dir, f'{j:05d}' + '.obj')
                export_mesh(out_mesh, output_name, file_type='obj')
            exit()


        for j in range(smplx_param['batch_size']):
            dtype = torch.float64
            left_hand_pose = np.einsum('bi,ij->bj', smplx_param['left_hand_pose'][j][None, ...], left_hand_c[:12, :])
            right_hand_pose = np.einsum('bi,ij->bj', smplx_param['right_hand_pose'][j][None, ...], right_hand_c[:12, :])
            output = smplx_model(betas=torch.tensor(smplx_param['betas'], dtype=dtype).to('cuda'), # torch.Size([1, 300]) 
                            expression=torch.tensor(smplx_param['expression'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 100]) 
                            jaw_pose=torch.tensor(smplx_param['jaw_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 3]) 
                            leye_pose=torch.tensor(smplx_param['leye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                            reye_pose=torch.tensor(smplx_param['reye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                            global_orient=torch.tensor(smplx_param['global_orient'][j], dtype=dtype).to('cuda'), 
                            body_pose=torch.tensor(smplx_param['body_pose_axis'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                            left_hand_pose=torch.tensor(left_hand_pose, dtype=dtype).to('cuda'),  
                            right_hand_pose=torch.tensor(right_hand_pose, dtype=dtype).to('cuda'), 
                            return_verts=True)

            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        vertices = np.asarray(vertices)
        # vertices = vertices[:1000, ...]
        vertices_list.append(vertices)

        rendertool._render_sequences(wav_file, vertices_list[:], stand=stand, face=face, whole_body=config.Infer.whole_body)


    else:
        for wav_file in wav_files:
            if config.Infer.test_with_gt and not config.Infer.test_gt_from_show: 
                pkl_path = config.Infer.test_gt_pkl_path
                with open(pkl_path, 'rb') as file:
                    data = pickle.load(file)
                with open(config.Infer.show_vis.hand_component_path) as file_obj:
                    comp = json.load(file_obj)
                    left_hand_c = np.asarray(comp['left'])
                    right_hand_c = np.asarray(comp['right'])
                smplx_param = data
                # vertices_list = []
                gt_vertices = []
                for j in range(smplx_param['batch_size']):
                    dtype = torch.float64
                    left_hand_pose = np.einsum('bi,ij->bj', smplx_param['left_hand_pose'][j][None, ...], left_hand_c[:12, :])
                    right_hand_pose = np.einsum('bi,ij->bj', smplx_param['right_hand_pose'][j][None, ...], right_hand_c[:12, :])
                    output = smplx_model(betas=torch.tensor(smplx_param['betas'], dtype=dtype).to('cuda'), # torch.Size([1, 300]) 
                                    expression=torch.tensor(smplx_param['expression'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 100]) 
                                    jaw_pose=torch.tensor(smplx_param['jaw_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), # torch.Size([1, 3]) 
                                    leye_pose=torch.tensor(smplx_param['leye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                    reye_pose=torch.tensor(smplx_param['reye_pose'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                    global_orient=torch.tensor(smplx_param['global_orient'][j], dtype=dtype).to('cuda'), 
                                    body_pose=torch.tensor(smplx_param['body_pose_axis'][j], dtype=dtype).unsqueeze_(dim=0).to('cuda'), 
                                    left_hand_pose=torch.tensor(left_hand_pose, dtype=dtype).to('cuda'),  
                                    right_hand_pose=torch.tensor(right_hand_pose, dtype=dtype).to('cuda'), 
                                    return_verts=True)

                    gt_vertices.append(output.vertices.detach().cpu().numpy().squeeze())

                    

                gt_vertices = np.asarray(gt_vertices)

                # vertices = vertices[:1000, ...]
                # vertices_list.append(vertices)
            
            elif config.Infer.test_with_gt and config.Infer.test_gt_from_show:
                pkl_path = config.Infer.test_gt_pkl_path
                with open(pkl_path, 'rb') as file:
                    # data = pickle.load(file)[0]
                    data = pickle.load(file)

                # gt_vertices_list = []

                jaw_pose = np.array(data['jaw_pose']) # 3
                leye_pose = np.array(data['leye_pose']) # 3
                reye_pose = np.array(data['reye_pose']) # 3
                global_orient = np.array(data['global_orient']).squeeze() # 3
                body_pose = np.array(data['body_pose_axis']) # 63
                left_hand_pose = np.array(data['left_hand_pose']) # 12
                right_hand_pose = np.array(data['right_hand_pose']) # 12
                expression = np.array(data['expression'])

                full_body = np.concatenate(
                    (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)
                assert full_body.shape[1] == 99

                full_body = to3d(full_body) # (300, 99) -> (300, 165)
                poses = full_body[:, c_index_3d] # torch.Size([300, 129])

                jaw_pose = torch.tensor(jaw_pose, dtype=torch.float64).to('cuda') 
                poses = torch.tensor(poses, dtype=torch.float64).to('cuda') 
                expression = torch.tensor(expression, dtype=torch.float64).to('cuda') 
                
                smplx_param = torch.cat([jaw_pose, poses, expression], dim=-1) # (300, 232)
                smplx_param = part2full(smplx_param, stand) # torch.Size([300, 265])
                betas = torch.tensor(np.array(data['betas']), dtype=torch.float64).to('cuda') # (1, 300)

                # smplx_param=smplx_param[1000:2000, ...]
                result_list = [smplx_param]
                # full_body = np.concatenate((full_body, expression), axis=1) # (300, 265)
                gt_vertices_list, _ = get_vertices(smplx_model, betas, result_list, config, isGT=True) # (300, 10475, 3) gt&pred


            id = torch.tensor(speaker_id[speaker_name], dtype=torch.int64).to('cuda') - 20
            # id = torch.tensor(1, dtype=torch.int64).to('cuda')

            # if config.Data.pose.expression:
            #     expression = bat['expression'].to(device).to(torch.float32)
            #     poses = torch.cat([poses_, expression], dim=1)
            # else:
            #     poses = poses_
            # cur_wav_file = bat['aud_file'][0]
            f = open(speaker_beta_path, 'rb+')
            # data = pickle.load(f)
            data = pickle.load(f)[0]

            betas = torch.tensor(np.array(data['betas']), dtype=torch.float64).to('cuda') # (1, 300)
            assert betas.shape == (1,300)
            # betas = torch.zeros([1, 300], dtype=torch.float64).to('cuda')
            # gt = poses.to('cuda').squeeze().transpose(1, 0) # (300, 265)
            # if config.Data.pose.normalization:
            #     gt = denormalize(gt, norm_stats[0], norm_stats[1]).squeeze(dim=0)
            # if config.Data.pose.convert_to_6d:
            #     if config.Data.pose.expression:
            #         gt_exp = gt[:, -100:]
            #         gt = gt[:, :-100]

            #     gt = gt.reshape(gt.shape[0], -1, 6)

            #     gt = matrix_to_axis_angle(rotation_6d_to_matrix(gt)).reshape(gt.shape[0], -1)
            #     gt = torch.cat([gt, gt_exp], -1)
            # if face:
            #     gt = torch.cat([gt[:, :3], body_static.repeat(gt.shape[0], 1), gt[:, -100:]], dim=-1)

            # result_list = [gt]
            result_list = []


            pred_face = g_face.infer_on_audio(wav_file,
                                                initial_pose=None, # 传进去没用
                                                norm_stats=None,
                                                w_pre=False,
                                                id=id, # 源代码注释了
                                                frame=None,
                                                am=am,
                                                am_sr=am_sr,
                                                num_classes=config.Model.face_num_classes,
                                                fps=config.Infer.fps,
                                                # num_classes=4s
                                                )

            pred_face = torch.tensor(pred_face).squeeze().to('cuda') # （300， 103）
            # pred_face = torch.zeros([gt.shape[0], 105])

            # if config.Data.pose.convert_to_6d:
            #     pred_jaw = pred_face[:, :6].reshape(pred_face.shape[0], -1, 6)
            #     pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred_face.shape[0], -1)
            #     pred_face = pred_face[:, 6:]
            
            pred_jaw = pred_face[:, :3] # (300, 3)
            pred_face = pred_face[:, 3:] # (300, 100)

            # id = torch.tensor([0], device='cuda')

            for i in range(num_sample):
                # id = torch.tensor(0, dtype=torch.int64).to('cuda')

                pred_res = g_body.infer_on_audio(   wav_file,
                                                    initial_pose=None, # 传进去没用
                                                    norm_stats=norm_stats,
                                                    txgfile=None, # 没用
                                                    id=id,
                                                    # var=var,
                                                    fps=config.Infer.fps,
                                                    w_pre=False
                                                    )
                pred = torch.tensor(pred_res).squeeze().to('cuda')

                if pred.shape[0] < pred_face.shape[0]:
                    repeat_frame = pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
                    pred = torch.cat([pred, repeat_frame], dim=0)
                else:
                    pred = pred[:pred_face.shape[0], :]

                body_or_face = False
                if pred.shape[1] < 275:
                    body_or_face = True
                if config.Data.pose.convert_to_6d:
                    pred = pred.reshape(pred.shape[0], -1, 6)
                    pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
                    pred = pred.reshape(pred.shape[0], -1)

                pred = torch.cat([pred_jaw, pred, pred_face], dim=-1) # (300, 232)
                # pred[:, 9:12] = global_orient
                pred = part2full(pred, stand) # torch.Size([300, 265])
                if face:
                    pred = torch.cat([pred[:, :3], body_static.repeat(pred.shape[0], 1), pred[:, -100:]], dim=-1)
                
                # result_list[0] = poses2pred(result_list[0], stand) # GT
                # if gt_0 is None:
                #     gt_0 = gt
                # pred = pred2poses(pred, gt_0)
                # result_list[0] = poses2poses(result_list[0], gt_0)

                result_list.append(pred)
                # result_list.append(pred[0:300])

            if config.Infer.just_smplx_param:
                torch.save(result_list[0], os.path.join(config.Infer.smplx_param_save_base, 'pose.pt'))
                torch.save(betas, os.path.join(config.Infer.smplx_param_save_base, 'body_shape_betas.pt'))
                exit()


            vertices_list, _ = get_vertices(smplx_model, betas, result_list, config, isGT=False) # (300, 10475, 3) gt&pred
            if config.Infer.test_with_gt and not config.Infer.test_gt_from_show: 
                vertices_list.append(vertices)
            elif config.Infer.test_with_gt and config.Infer.test_gt_from_show:
                vertices_list.append(gt_vertices_list[0])

            # result_list = [res.to('cpu') for res in result_list]
            # dict = np.concatenate(result_list[1:], axis=0)
            # file_name = '/data/lirenda621/code/TalkSHOW/visualise/video/' + config.Log.name + '/' + \
            #             wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
            # np.save(file_name, dict)

            # rendertool._render_sequences(cur_wav_file, vertices_list[1:], stand=stand, face=face)
            # rendertool._render_sequences(cur_wav_file, vertices_list[:1], stand=stand, face=face)
            rendertool._render_sequences(wav_file, vertices_list[:], stand=stand, face=face, whole_body=config.Infer.whole_body)


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    body_model_name = args.body_model_name
    body_model_path = args.body_model_path
    smplx_path = ' /Code/TalkSHOW/visualise'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = init_model(body_model_name, body_model_path, args, config)
    generator2 = None
    generator_face = init_model(face_model_name, face_model_path, args, config)

    print('init smlpx model...')
    dtype = torch.float64
    model_params = dict(model_path=smplx_path,
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True if config.Infer.save_mesh.save_obj_for_mesh_texture else False,
                        use_face_contour = True if config.Infer.save_mesh.save_obj_for_mesh_texture else False,
                        # create_transl=False,
                        # use_face_contour =False,
                        # gender='ne',
                        dtype=dtype,)
    smplx_model = smpl.create(**model_params).to('cuda')
    print('init rendertool...')
    rendertool = RenderTool(infer_cfg=config.Infer)

    infer(config.Data.data_root, generator, generator_face, generator2, args.exp_name, None, None, device,
          None, True, smplx_model, rendertool, args, config, config.Infer.test_wav_files, speaker_name=config.Infer.speaker_name,
          speaker_beta_path=config.Infer.speaker_beta_path)


if __name__ == '__main__':
    main()
