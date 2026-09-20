"""Re-walk the recorded route through the trained splat, off the rails, in metres.

The point of a flythrough is to leave the training trajectory: rendering the exact
recorded poses only shows how well 3DGS memorised its inputs. A monocular reconstruction
of a forward-walking camera is weakest to the *side* of the path, where no camera ever
looked, so that is what this probes -- a constant lateral/vertical offset held for the
whole route.

Offsets are given in metres and converted through ``sfm_metric_alignment.json``. SfM
units are arbitrary, so "0.5" alone is meaningless; the Umeyama fit against the LiDAR
trajectory is what makes a metre a metre here.

Shifting the camera is one subtraction. With ``W2C = [R_w2c | t]`` and Vid2Sim storing
``cam.R = R_w2c^T``, ``cam.T = t``, a displacement ``d`` expressed in camera axes
(x right, y down, z forward) moves the centre to ``C + cam.R @ d``, which is exactly
``T_new = T - d``. Rotation is untouched, so the view direction stays parallel to the
original -- a dolly, not an orbit.

    micromamba run -n vid2sim-recon python tools/render_flythrough.py \
        -m output/urban_walk_20m --right 0.5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import getWorld2View2  # noqa: E402


def shifted(cam, d_cam: np.ndarray):
  """Copy of ``cam`` dollied by ``d_cam`` (SfM units, camera axes)."""
  out = copy.copy(cam)
  out.T = cam.T - d_cam
  wvt = torch.tensor(getWorld2View2(out.R, out.T)).transpose(0, 1).cuda()
  out.world_view_transform = wvt
  out.full_proj_transform = wvt.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0)).squeeze(0)
  out.camera_center = wvt.inverse()[3, :3]
  return out


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--right', type=float, default=0.0, help='Metres to dolly right.')
  parser.add_argument('--up', type=float, default=0.0, help='Metres to dolly up.')
  parser.add_argument('--forward', type=float, default=0.0, help='Metres to dolly forward.')
  parser.add_argument('--fps', type=float, default=10.0)
  # get_combined_args drops any argument left as None, so these default to '' not None.
  parser.add_argument('--tag', default='', help='Output subdirectory name.')
  # A stereo scene holds both cameras as ordinary views. Walking all of them would hop
  # between eyes every frame; restricting to one eye reproduces the physical path, and
  # matches the monocular run view for view.
  parser.add_argument('--only', default='', help='Keep cameras whose name contains this.')
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                    dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  align = json.loads((pathlib.Path(dataset.source_path) / 'sfm_metric_alignment.json').read_text())
  m_per_unit = align['scale']
  d_cam = np.array([args.right, -args.up, args.forward]) / m_per_unit
  print(f'{m_per_unit:.4f} m per SfM unit | offset {args.right:+.2f}R {args.up:+.2f}U '
        f'{args.forward:+.2f}F m = {np.round(d_cam, 4)} units')

  cams = scene.getTrainCameras()
  if args.only:
    cams = [c for c in cams if args.only in c.image_name]
  cams = sorted(cams, key=lambda c: c.image_name)
  print(f'{len(cams)} cameras')
  tag = args.tag or f'r{args.right:+.2f}_u{args.up:+.2f}_f{args.forward:+.2f}'
  out_dir = pathlib.Path(dataset.model_path) / 'flythrough' / tag
  out_dir.mkdir(parents=True, exist_ok=True)

  psnrs = []
  on_rails = args.right == args.up == args.forward == 0.0
  for cam in tqdm(cams, desc=f'render {tag}'):
    view = cam if on_rails else shifted(cam, d_cam)
    rgb = render(view, gaussians, pipe, bg)['render'].clamp(0, 1)
    if on_rails:
      mse = ((rgb - cam.original_image[:3].to(rgb.device)) ** 2).mean()
      psnrs.append(float(-10 * torch.log10(mse)))
    torchvision.utils.save_image(rgb, out_dir / f'{cam.image_name}.png')

  if psnrs:
    print(f'on-rails PSNR vs GT: mean {np.mean(psnrs):.2f} dB  min {np.min(psnrs):.2f} dB')

  video = out_dir.parent / f'{tag}.mp4'
  os.system(f"ffmpeg -y -loglevel error -framerate {args.fps} -pattern_type glob "
            f"-i '{out_dir}/*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p "
            f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video}")
  print(f'-> {video}  ({len(cams)} frames)')


if __name__ == '__main__':
  main()
