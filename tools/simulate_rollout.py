"""Fly the recorded route and make repeated random excursions off it, as a policy would.

``perturbation_sweep.py`` answers "how bad is a 1 m, 15 degree deviation here" by holding
one pose and rendering a grid. This answers a different question: what a *rollout* looks
like when the camera keeps leaving the demonstrated path and coming back. The distinction
matters because the frames a policy actually sees are the ones during a departure, where
position and heading are both part-way to their extremes and changing frame to frame --
not the static corner cases of a grid.

Each excursion picks a side at random, eases out to a sampled fraction of the configured
limits, holds, and eases back. Position and heading share that one fraction rather than
being drawn separately, so an excursion is a single coherent "how far off did it get"
rather than an unphysical mixture like a full metre sideways while still facing straight
ahead. The heading turns *away* from the path by default, which is both what steering off
looks like and the harsher case: it swings the frame toward regions the capture only
caught at the edge of its field of view. ``--yaw-toward-path`` inverts it to model a robot
that has drifted but is already correcting.

Motion is eased rather than linear because a linear ramp corners instantly at the start
and end of an excursion, which no robot does and which puts a visible jerk in the output.

Coverage is scored per frame from the rasteriser's alpha -- the share of the frame with no
geometry behind it. Over a rollout that reads as the fraction of frames a policy would
spend looking at holes, which is the number that decides whether this asset is usable.

    micromamba run -n vid2sim-recon python tools/simulate_rollout.py \
        -m output/stereo_lidar_w0.5 -s data/urban_walk_20m_stereo --only left
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from perturbation_sweep import perturbed, yaw_pitch  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def excursion_profile(n: int, rise: float) -> np.ndarray:
  """Eased 0 -> 1 -> 0 over ``n`` frames, holding at 1 between the two ramps."""
  t = np.linspace(0.0, 1.0, n, endpoint=False)
  edge = max(rise, 1e-6)
  s = np.minimum(np.clip(t / edge, 0, 1), np.clip((1.0 - t) / edge, 0, 1))
  return s * s * (3.0 - 2.0 * s)  # smoothstep


def build_schedule(n_frames: int, rng: np.random.Generator, args) -> list[dict]:
  """Per-frame lateral offset (m) and yaw offset (deg) for the whole route."""
  plan = [{'lateral': 0.0, 'yaw': 0.0, 'episode': -1} for _ in range(n_frames)]
  start, ep = args.settle, 0
  while start + args.duration <= n_frames - args.settle:
    side = 1.0 if rng.random() < 0.5 else -1.0
    # One fraction drives both channels, so position and heading stay consistent with
    # each other instead of combining into poses no robot would hold.
    frac = rng.uniform(args.min_fraction, 1.0)
    shape = excursion_profile(args.duration, args.rise)
    yaw_sign = -side if args.yaw_toward_path else side
    for k in range(args.duration):
      plan[start + k] = {
        'lateral': side * frac * args.max_lateral * shape[k],
        'yaw': yaw_sign * frac * args.max_yaw * shape[k],
        'episode': ep,
      }
    start += args.duration + args.gap
    ep += 1
  return plan


def stamp(img: np.ndarray, lines: list[str]) -> np.ndarray:
  for i, text in enumerate(lines):
    y = 24 + 24 * i
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
  parser.add_argument('--max-lateral', type=float, default=1.0, help='Metres.')
  parser.add_argument('--max-yaw', type=float, default=15.0, help='Degrees.')
  parser.add_argument('--min-fraction', type=float, default=0.4,
                      help='Smallest excursion as a fraction of the limits. Excursions '
                           'drawn near zero are indistinguishable from staying on the '
                           'path and waste frames.')
  parser.add_argument('--duration', type=int, default=24, help='Frames per excursion.')
  parser.add_argument('--gap', type=int, default=12, help='Frames back on the path.')
  parser.add_argument('--settle', type=int, default=8,
                      help='Frames left alone at each end of the route, where little has '
                           'been observed to the side and any deviation looks broken.')
  parser.add_argument('--rise', type=float, default=0.35,
                      help='Fraction of an excursion spent easing out, and again easing '
                           'back in.')
  parser.add_argument('--yaw-toward-path', action='store_true',
                      help='Turn back toward the path instead of away from it.')
  parser.add_argument('--alpha-floor', type=float, default=0.5)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--side-by-side', action='store_true', default=True,
                      help='Show the expert view beside the deviated one.')
  parser.add_argument('--no-side-by-side', dest='side_by_side', action='store_false')
  parser.add_argument('--fps', type=float, default=10.0)
  parser.add_argument('--tag', default='rollout')
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

  cams = sorted([c for c in scene.getTrainCameras() if args.only in c.image_name],
                key=lambda c: c.image_name)
  plan = build_schedule(len(cams), np.random.default_rng(args.seed), args)
  n_ep = max(p['episode'] for p in plan) + 1
  print(f'{len(cams)} frames, {n_ep} excursions, up to {args.max_lateral:.1f} m and '
        f'{args.max_yaw:.0f} deg, seed {args.seed}, {m_per_unit:.4f} m/unit')

  out_dir = pathlib.Path(dataset.model_path) / 'flythrough' / args.tag
  out_dir.mkdir(parents=True, exist_ok=True)

  record, holes_off, holes_on = [], [], []
  for i, (cam, p) in enumerate(tqdm(list(zip(cams, plan)), desc=args.tag)):
    d = np.array([p['lateral'], 0.0, 0.0]) / m_per_unit
    view = perturbed(cam, d, yaw_pitch(p['yaw'], 0.0))
    out = render(view, gaussians, pipe, bg)
    hole = float((out['alpha'].squeeze() < args.alpha_floor).float().mean())
    img = (out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
           * 255).astype(np.uint8).copy()
    stamp(img, [f'{p["lateral"]:+.2f} m  {p["yaw"]:+.1f} deg',
                f'{hole:.1%} of frame empty'])

    (holes_on if p['episode'] < 0 else holes_off).append(hole)
    if args.side_by_side:
      ref = render(cam, gaussians, pipe, bg)['render'].clamp(0, 1)
      ref = (ref.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8).copy()
      img = np.hstack([stamp(ref, ['expert']), img])
    cv2.imwrite(str(out_dir / f'{i:05d}.png'), img)

    C = -view.R @ view.T  # camera centre in SfM units
    record.append({'frame': i, 'image': cam.image_name, 'episode': p['episode'],
                   'lateral_m': p['lateral'], 'yaw_deg': p['yaw'],
                   'empty_fraction': hole, 'centre_sfm': [float(x) for x in C],
                   'R_cam_to_world': [[float(x) for x in r] for r in view.R]})

  poses = out_dir.parent / f'{args.tag}_poses.json'
  poses.write_text(json.dumps(
    {'m_per_unit': m_per_unit, 'max_lateral_m': args.max_lateral,
     'max_yaw_deg': args.max_yaw, 'seed': args.seed, 'frames': record}, indent=2))

  off, on = np.array(holes_off), np.array(holes_on)
  print(f'\non-path frames  ({len(on):3d}): empty median {np.median(on):.1%}  '
        f'p90 {np.percentile(on, 90):.1%}')
  print(f'off-path frames ({len(off):3d}): empty median {np.median(off):.1%}  '
        f'p90 {np.percentile(off, 90):.1%}  max {off.max():.1%}')
  for thr in (0.02, 0.05, 0.10):
    print(f'  frames worse than {thr:.0%} empty: {(off > thr).mean():.1%} of the rollout')
  peak = max(record, key=lambda r: r['empty_fraction'])
  print(f'worst frame: {peak["image"]} at {peak["lateral_m"]:+.2f} m '
        f'{peak["yaw_deg"]:+.1f} deg -> {peak["empty_fraction"]:.1%} empty')

  video = out_dir.parent / f'{args.tag}.mp4'
  os.system(f"ffmpeg -y -loglevel error -framerate {args.fps} -pattern_type glob "
            f"-i '{out_dir}/*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p "
            f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {video}")
  print(f'\n-> {video}\n-> {poses}')


if __name__ == '__main__':
  main()
