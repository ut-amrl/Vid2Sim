import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
import copy

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

@torch.no_grad()
def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               max_depth=5.0, angle_threshold=15.0, volume=None, use_ground_mask=False, use_sky_mask=False):
    if use_ground_mask or use_sky_mask:
        from sam_hq_prompt import SAM_HQ_PROMPT
        seg_model = SAM_HQ_PROMPT()

    depths_tsdf_fusion, rgbs_tsdf_fusion = [], []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        _, H, W = gt.shape

        mask = torch.zeros(H, W, dtype=torch.bool, device="cuda")
        rgb = gt*255

        if use_ground_mask:
            ground_mask = seg_model.infer(rgb, 'ground')
            mask = mask | ground_mask
            
        if use_sky_mask:
            mask = torch.zeros(H, W, dtype=torch.bool, device="cuda")
            sky_mask = seg_model.infer(rgb, 'sky')
            mask = mask | sky_mask

        out = render(view, gaussians, pipeline, background)
        rendering = out["render"].clamp(0.0, 1.0)

        depth = out["depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()

        normal = out["normals"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)

        # Compute angle between other normal and ground normal
        ground_normal = torch.tensor([0,-1,0], dtype=torch.float32, device="cuda")
        dot = torch.sum(normal*ground_normal, dim=-1)
        angle = torch.acos(dot)
        mask = mask | (angle < (angle_threshold / 180 * 3.14159))

        if volume is not None:
            depth_tsdf[mask] = 0
            depths_tsdf_fusion.append(depth_tsdf.cpu())
            rgb = (rendering.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            rgbs_tsdf_fusion.append(rgb.copy())
        
    if volume is not None:
        # depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx]
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            # color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            color = o3d.geometry.Image(rgbs_tsdf_fusion[idx])
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            H, W = ref_depth.shape
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose
            )

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                 max_depth : float, voxel_size : float, use_ground_mask : bool, use_sky_mask : bool,
                 angle_threshold : float = 15.0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras()+scene.getTestCameras(), scene, gaussians, pipeline, background, 
                    max_depth=max_depth, volume=volume, use_ground_mask=use_ground_mask, use_sky_mask=use_sky_mask,
                    angle_threshold=angle_threshold)
        
        print(f"extract_triangle_mesh")
        mesh = volume.extract_triangle_mesh()

        path = os.path.join(dataset.model_path, "mesh")
        os.makedirs(path, exist_ok=True)
        
        o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        mesh = clean_mesh(mesh)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=999.0, type=float)
    parser.add_argument("--voxel_size", default=0.1, type=float)
    parser.add_argument("--use_ground_mask", action="store_true")
    parser.add_argument("--use_sky_mask", action="store_true")
    # Surfaces whose normal is within this angle of straight up are dropped from the TSDF,
    # which removes the ground. Right for Unity, where the ground plane is added back in
    # the editor; set 0 to keep the ground for a collision mesh.
    parser.add_argument("--angle_threshold", default=15.0, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        args.max_depth, 
        args.voxel_size, 
        args.use_ground_mask,
        args.use_sky_mask,
        args.angle_threshold
    )