"""Is the splat's geometry accurate enough to survive a sideways step?

A lateral move converts depth error into misregistration. Shift the camera by ``b`` and a
surface at depth ``z`` carrying depth error ``dz`` lands about ``f*b*dz/z^2`` pixels away
from where it belongs. Texture then appears in the wrong place -- error spread over every
textured pixel regardless of how well that pixel was sampled, which is exactly the
signature the error attribution found.

So this measures the depth directly, against something the splat never saw: block-matched
disparity from the raw stereo pair. That is an independent metric measurement, good in the
near field where a 24.5 cm baseline still yields tens of pixels of disparity, and it is
the near field that dominates what a novel view looks like.

The final column is the point. It restates depth error as the pixel displacement it would
cause at the stereo baseline, which is directly comparable to the misregistration visible
in the held-out renders.

    micromamba run -n vid2sim-recon python tools/check_depth_accuracy.py \
        -m output/urban_walk_20m_leftonly -s data/urban_walk_20m_stereo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def sgbm_depth(left: np.ndarray, right: np.ndarray, fx: float, baseline: float):
  """Metric depth from block matching, plus a validity mask."""
  m = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  disp = m.compute(left, right).astype(np.float32) / 16.0
  # Below ~2 px the depth is dominated by quantisation; above 128 is out of range.
  valid = disp > 2.0
  depth = np.zeros_like(disp)
  depth[valid] = fx * baseline / disp[valid]
  return depth, valid


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--samples', type=int, default=25)
  parser.add_argument('--max-depth', type=float, default=15.0, help='Metres.')
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  src = pathlib.Path(dataset.source_path)
  meta = json.loads((src / 'meta.json').read_text())
  fx, baseline = meta['fx'], meta['baseline_m']
  # Prefer the factor fit against this reconstruction's own triangulated points. The
  # trajectory-derived one in sfm_metric_alignment.json gets copied between sequences,
  # and a scale borrowed from a different reconstruction would show up here as depth
  # error that the model does not actually have.
  fitted = src / 'lidar_depth_scale.json'
  if fitted.exists():
    m_per_unit = json.loads(fitted.read_text())['m_per_unit']
  else:
    m_per_unit = json.loads((src / 'sfm_metric_alignment.json').read_text())['scale']

  cams = sorted([c for c in scene.getTrainCameras() if 'left' in c.image_name],
                key=lambda c: c.image_name)
  probe = cams[::max(len(cams) // args.samples, 1)][:args.samples]

  rel, shift, zs = [], [], []
  for cam in probe:
    stem = cam.image_name
    L = cv2.imread(str(src / 'images' / f'{stem}.jpg'))
    R = cv2.imread(str(src / 'images' / f'{stem.replace("_left", "_right")}.jpg'))
    ref, valid = sgbm_depth(L, R, fx, baseline)

    pred = render(cam, gaussians, pipe, bg)['depth'].squeeze().cpu().numpy() * m_per_unit
    ok = valid & (ref > 0.5) & (ref < args.max_depth) & (pred > 0.1)
    if ok.sum() < 500:
      continue
    z, p = ref[ok], pred[ok]
    rel.append(np.abs(p - z) / z)
    # f * b * dz / z^2, the misregistration a lateral step of one baseline would produce.
    shift.append(fx * baseline * np.abs(p - z) / z ** 2)
    zs.append(z)

  rel, shift, zs = np.concatenate(rel), np.concatenate(shift), np.concatenate(zs)
  print(f'{len(probe)} frames, {len(rel)} pixels with block-matched depth '
        f'(median {np.median(zs):.1f} m)')
  print(f'absolute relative depth error: median {np.median(rel):.1%}  '
        f'p90 {np.percentile(rel, 90):.1%}')
  print(f'implied misregistration at the {baseline * 100:.1f} cm baseline: '
        f'median {np.median(shift):.2f} px  p90 {np.percentile(shift, 90):.2f} px')
  for d in (0.5, 1.0):
    s = shift * d / baseline
    print(f'  extrapolated to a {d:.1f} m lateral step: median {np.median(s):5.1f} px  '
          f'p90 {np.percentile(s, 90):6.1f} px')


if __name__ == '__main__':
  main()
