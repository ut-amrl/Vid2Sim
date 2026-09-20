import os
import sys
import torch
import numpy as np
import argparse
import math
import imageio
from tqdm import tqdm
from pathlib import Path

from PIL import Image
from transformers import pipeline
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
import torch.nn.functional as F

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import asyncio
import aiofiles
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from colmap_utils.loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, read_points3D_binary, storePly
from colmap_utils.utils import visualize_depth, visualize_normal

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return positions

def gen_ply(bin_path, ply_path):
    if os.path.exists(ply_path):
        return
    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def readColmapSceneInfo(path):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(path, 'images', os.path.basename(extr.name))
        K = np.array([[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
        
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = T
        # c2w = np.linalg.inv(w2c)

        cam = {"uid": uid, "image_path": image_path, "K": K, "c2w": c2w, "FovX": FovX, "FovY": FovY, "H": height, "W": width}
        cam_infos.append(cam)
    return cam_infos


@torch.no_grad()
def process_image(args, cam, pts, pipe, device):
    K = torch.tensor(cam["K"], dtype=torch.float32, device=device)
    c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device=device)
    H, W = cam["H"], cam["W"]

    pts = torch.tensor(pts, device=device)
    pts_h = torch.cat([pts, torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)], dim=1)
    pts_cam = torch.mm(c2w, pts_h.T).T
    pts_cam = pts_cam[:, :3]
    pts_cam = torch.mm(K, pts_cam.T).T
    depth = pts_cam[:, 2]
    pts_cam = pts_cam / depth.unsqueeze(1)
    
    depth_mask = depth > 0
    vis_mask = (pts_cam[:, 0] >= 0) & (pts_cam[:, 0] < W) & (pts_cam[:, 1] >= 0) & (pts_cam[:, 1] < H)
    mask = torch.logical_and(depth_mask, vis_mask)
    
    pts_cam = pts_cam[mask]
    depth = depth[mask]
    sfm_depth_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    sfm_depth_mask[pts_cam[:, 1].long(), pts_cam[:, 0].long()] = 1
    
    sfm_depth = {
        'depth': depth.cpu().numpy(),
        'pts_cam': pts_cam[:,:2].cpu().numpy()
    }

    image = Image.open(cam["image_path"])
    result = pipe(image, device=device)
    pred_dis = result['predicted_depth'].to(device).squeeze(0)
    pred_H, pred_W = pred_dis.shape
    if pred_H != H or pred_W != W:
        pred_dis = F.interpolate(pred_dis[None,None,...], (H, W), mode='bilinear', align_corners=False)[0][0]
    pred_dis = pred_dis / pred_dis.max()
    
    pred_dis_sparse = pred_dis[pts_cam[:, 1].long(), pts_cam[:, 0].long()]
    zero_mask = pred_dis_sparse != 0
    sfm_dis = 1 / depth
    
    pred_dis_sparse = pred_dis_sparse[zero_mask]
    sfm_dis = sfm_dis[zero_mask]

    t_colmap = torch.median(sfm_dis)
    s_colmap = torch.mean(torch.abs(sfm_dis - t_colmap))

    t_mono = torch.median(pred_dis_sparse)
    s_mono = torch.mean(torch.abs(pred_dis_sparse - t_mono))
    scale = s_colmap / s_mono
    offset = t_colmap - t_mono * scale

    full_zero_mask = pred_dis != 0
    pred_dis[full_zero_mask] = pred_dis[full_zero_mask] * scale + offset

    return pred_dis.cpu().numpy(), sfm_depth

async def save_results(
        depth_out_path, 
        sfm_depth_out_path,
        vis_depth_out_path, 
        vis_normal_out_path, 
        image_name, pred_depth, sfm_depth, vis_depth, vis_normal):
    depth_file = os.path.join(depth_out_path, image_name[:-4] + ".npy")
    sfm_depth_file = os.path.join(sfm_depth_out_path, image_name[:-4] + ".npz")
    depth_vis_file = os.path.join(vis_depth_out_path, image_name)
    normal_vis_file = os.path.join(vis_normal_out_path, image_name)
    
    np.save(depth_file, pred_depth)
    np.savez(sfm_depth_file, **sfm_depth)
    imageio.imsave(depth_vis_file, vis_depth)
    imageio.imsave(normal_vis_file, vis_normal)

def process_batch(args, cam_batch, all_cams, pts, model_path, device):
    # Initialize the pipeline inside the process
    pipe = pipeline(task="depth-estimation", model=model_path, device=device)
    results = []
    for cam in cam_batch:
        pred_dis, sfm_depth = process_image(args, cam, pts, pipe, device)
        vis_depth = visualize_depth(pred_dis).permute(1, 2, 0).cpu().numpy()
        vis_normal = visualize_normal(pred_dis, cam).cpu().numpy()
        vis_depth = (vis_depth[:, :, :3] * 255).astype(np.uint8)
        vis_normal = (vis_normal[:, :, :3] * 255).astype(np.uint8)
        results.append((os.path.basename(cam["image_path"]), pred_dis, sfm_depth, vis_depth, vis_normal))
    return results

@torch.no_grad()
def process(args):
    path = args.path
    cam_infos = readColmapSceneInfo(path)
    # Ordering only needs to be deterministic. Stereo clips name frames "00007_left", so
    # sort on the leading integer where there is one and fall back to the plain name.
    def _order(cam):
        stem = os.path.splitext(os.path.basename(cam['image_path']))[0]
        head = stem.split('_')[0]
        return (int(head), stem) if head.isdigit() else (0, stem)
    cam_infos = sorted(cam_infos, key=_order)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    gen_ply(bin_path, ply_path)
    
    pts = fetchPly(os.path.join(path, "sparse/0/points3D.ply"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "depth-anything/Depth-Anything-V2-Large-hf"

    depth_out_path = os.path.join(path, "depths")
    sfm_depth_out_path = os.path.join(path, "sfm_gt_depths")
    vis_depth_out_path = os.path.join(path, "vis_depths")
    vis_normal_out_path = os.path.join(path, "vis_normal")
    os.system(f'rm -rf {depth_out_path}/*')
    os.system(f'rm -rf {sfm_depth_out_path}/*')
    
    os.makedirs(depth_out_path, exist_ok=True)
    os.makedirs(sfm_depth_out_path, exist_ok=True)
    os.makedirs(vis_depth_out_path, exist_ok=True)
    os.makedirs(vis_normal_out_path, exist_ok=True)

    # Debug
    batch_size = 4  # Adjust based on your GPU memory
    num_processes = min(multiprocessing.cpu_count(), 4)  # Limit to 4 or CPU count, whichever is smaller

    with multiprocessing.Pool(num_processes) as pool:
        for i in tqdm(range(0, len(cam_infos), batch_size)):
            cam_batch = cam_infos[i:i+batch_size]
            results = pool.apply(process_batch, args=(args, cam_batch, cam_infos, pts, model_path, device))
            
            loop = asyncio.get_event_loop()
            tasks = [save_results(depth_out_path, sfm_depth_out_path, vis_depth_out_path, vis_normal_out_path, image_name, pred_depth, sfm_depth, vis_depth, vis_normal) 
                     for image_name, pred_depth, sfm_depth, vis_depth, vis_normal in results]
            loop.run_until_complete(asyncio.gather(*tasks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the scene")
    args = parser.parse_args()

    process(args)