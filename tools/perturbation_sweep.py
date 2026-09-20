"""Probe the splat at posed offsets from the recorded route, translating *and* turning.

``render_flythrough.py`` dollies the whole route sideways while keeping the view direction
parallel to the original. That is the right probe for "how far can the camera stray", but
it is not the situation a policy in Isaac Lab actually creates. A policy that drifts also
*turns*, and turning is the harsher test: a lateral dolly keeps looking at surfaces the
capture saw head-on, whereas a yaw swings the frame toward regions the camera only ever
caught at the edge of its field of view, or never caught at all.

So this samples waypoints along the recorded path and, at each one, renders a grid of
combined position and heading perturbations. Rendering the grid at one pose would say
nothing about the route as a whole -- an open stretch of sidewalk tolerates far more
deviation than a spot hemmed in by a wall -- so the sweep reports where along the route
the envelope is tight.

Composing the two perturbations needs care. With ``W2C = [R_w2c | t]`` and Vid2Sim storing
``cam.R = R_w2c^T``, ``cam.T = t``, displacing the centre by ``d`` in camera axes and then
rotating the body by ``R_d`` about that *new* centre gives::

    cam.R' = cam.R @ R_d
    cam.T' = R_d^T @ (cam.T - d)

Rotating about the new centre rather than the old is what makes the two controls
independent: yaw changes heading only, and never smuggles in extra translation.

There is no ground truth off the path, so quality is scored by the rasteriser's alpha: the
share of the frame where accumulated opacity is too low to call the pixel explained. That
is the failure a policy actually trips over -- the splat having nothing to show -- and,
unlike a sharpness statistic, it does not move merely because the frame now contains
different stuff. Sharpness is reported alongside but cannot be read across headings:
turning right here swings high-frequency foliage into view and so scores *better* than the
on-path frame, which is a fact about the scene rather than about the reconstruction.

    micromamba run -n vid2sim-recon python tools/perturbation_sweep.py \
        -m output/stereo_lidar_w0.5 -s data/urban_walk_20m_stereo --only left
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import getWorld2View2  # noqa: E402


def yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
  """Rotation in camera axes: x right, y down, z forward.

  Yaw turns about the down axis, so positive is a turn to the right; pitch turns about the
  right axis, so positive tips the view down.
  """
  y, p = np.radians(yaw_deg), np.radians(pitch_deg)
  Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
  Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
  return Ry @ Rx


def perturbed(cam, d_cam: np.ndarray, R_d: np.ndarray):
  """Copy of ``cam`` displaced by ``d_cam`` then rotated by ``R_d`` about the new centre."""
  out = copy.copy(cam)
  out.R = cam.R @ R_d
  out.T = R_d.T @ (cam.T - d_cam)
  wvt = torch.tensor(getWorld2View2(out.R, out.T)).transpose(0, 1).cuda()
  out.world_view_transform = wvt
  out.full_proj_transform = wvt.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0)).squeeze(0)
  out.camera_center = wvt.inverse()[3, :3]
  return out


def detail(img: np.ndarray) -> float:
  """Variance of the Laplacian: high-frequency energy, which smearing destroys."""
  return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


def label(img: np.ndarray, text: str, colour=(255, 255, 255)) -> np.ndarray:
  cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
  cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
  return img


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--only', default='', help='Keep cameras whose name contains this.')
  parser.add_argument('--waypoints', type=int, default=24,
                      help='Poses sampled evenly along the recorded route.')
  parser.add_argument('--lateral', type=float, nargs='+', default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                      help='Metres right, negative is left.')
  parser.add_argument('--yaw', type=float, nargs='+', default=[-30.0, -15.0, 0.0, 15.0, 30.0],
                      help='Degrees, positive turns right.')
  parser.add_argument('--forward', type=float, default=0.0, help='Metres.')
  parser.add_argument('--up', type=float, default=0.0, help='Metres.')
  parser.add_argument('--pitch', type=float, default=0.0, help='Degrees, positive tips down.')
  parser.add_argument('--alpha-floor', type=float, default=0.5,
                      help='Accumulated opacity below which a pixel counts as empty.')
  parser.add_argument('--fps', type=float, default=4.0)
  parser.add_argument('--tag', default='perturb')
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                    dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  src = pathlib.Path(dataset.source_path)
  fitted = src / 'lidar_depth_scale.json'
  m_per_unit = (json.loads(fitted.read_text())['m_per_unit'] if fitted.exists()
                else json.loads((src / 'sfm_metric_alignment.json').read_text())['scale'])

  cams = [c for c in scene.getTrainCameras() if args.only in c.image_name]
  cams = sorted(cams, key=lambda c: c.image_name)
  step = max(len(cams) // args.waypoints, 1)
  waypoints = cams[::step][:args.waypoints]
  print(f'{len(cams)} cameras, {len(waypoints)} waypoints, '
        f'{len(args.lateral)}x{len(args.yaw)} perturbations, {m_per_unit:.4f} m/unit')

  out_dir = pathlib.Path(dataset.model_path) / 'flythrough' / args.tag
  out_dir.mkdir(parents=True, exist_ok=True)

  # Rows are headings, columns are lateral offsets, so reading down a column shows the
  # cost of turning and reading across a row the cost of stepping sideways.
  shape = (len(args.yaw), len(args.lateral))
  holes, sharp, worst = np.zeros(shape), np.zeros(shape), []
  for wi, cam in enumerate(tqdm(waypoints, desc='waypoints')):
    rows, wp_holes = [], 0.0
    for yi, yaw in enumerate(args.yaw):
      tiles = []
      for li, lat in enumerate(args.lateral):
        d = np.array([lat, -args.up, args.forward]) / m_per_unit
        view = perturbed(cam, d, yaw_pitch(yaw, args.pitch))
        out = render(view, gaussians, pipe, bg)
        rgb = out['render'].clamp(0, 1)
        hole = float((out['alpha'].squeeze() < args.alpha_floor).float().mean())
        holes[yi, li] += hole
        wp_holes = max(wp_holes, hole)
        img = (rgb.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8).copy()
        sharp[yi, li] += detail(img)
        tiles.append(label(img, f'{lat:+.1f} m {yaw:+.0f} deg  {hole:.0%} empty'))
      rows.append(np.hstack(tiles))
    cv2.imwrite(str(out_dir / f'wp{wi:03d}_{cam.image_name}.png'), np.vstack(rows))
    worst.append((wp_holes, wi, cam.image_name))

  n = len(waypoints)
  cy, cx = len(args.yaw) // 2, len(args.lateral) // 2
  print('\nshare of frame with no geometry behind it (mean over waypoints)')
  print('              ' + ''.join(f'{l:+8.1f} m' for l in args.lateral))
  for yi, yaw in enumerate(args.yaw):
    print(f'  yaw {yaw:+5.0f} deg ' + ''.join(f'{holes[yi, li] / n:8.1%} '
                                              for li in range(len(args.lateral))))
  print(f'\nsharpness relative to on-path, for reference only -- rotation changes what is '
        f'in frame,\nso these are not comparable across rows:')
  print('              ' + ''.join(f'{l:+8.1f} m' for l in args.lateral))
  for yi, yaw in enumerate(args.yaw):
    print(f'  yaw {yaw:+5.0f} deg ' + ''.join(f'{sharp[yi, li] / sharp[cy, cx]:9.2f}'
                                              for li in range(len(args.lateral))))
  print('\ntightest waypoints (largest empty share at any perturbation):')
  for h, wi, name in sorted(worst, reverse=True)[:5]:
    print(f'  wp{wi:03d} {name}: {h:.1%}')

  video = out_dir.parent / f'{args.tag}.mp4'
  os.system(f"ffmpeg -y -loglevel error -framerate {args.fps} -pattern_type glob "
            f"-i '{out_dir}/wp*.png' -c:v libx264 -crf 20 -pix_fmt yuv420p "
            f"-vf 'scale=trunc(iw/4)*2:trunc(ih/4)*2' {video}")
  print(f'\n-> {video}')
  print(f'-> {out_dir} ({len(waypoints)} grids)')


if __name__ == '__main__':
  main()
