"""Attribute novel-view error to the capture or to the reconstruction, against real GT.

``analyze_coverage.py`` says what fraction of an off-path view the capture could not
support. It cannot say whether that is where the render actually goes wrong -- for that
you need ground truth off the path, which normally does not exist.

The stereo rig provides it. Train on the left camera only and the right camera becomes a
24.5 cm lateral novel view with a recorded image behind it. Classifying every right-view
pixel by what the *left* cameras sampled, then measuring error separately inside each
class, splits the error budget:

* error concentrated in the unobserved and undersampled pixels means the capture geometry
  is the binding constraint, and better method work cannot recover it;
* error spread evenly, including across well-sampled pixels, means the data was there and
  the reconstruction failed to use it.

Pixels where the ground truth is masked (pedestrians) are excluded, since the model was
trained never to reproduce them.

    micromamba run -n vid2sim-recon python tools/attribute_error.py \
        -m output/urban_walk_20m_leftonly -s data/urban_walk_20m_stereo
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from analyze_coverage import MIN_DEPTH, OCCL_TOL, backproject, cam_pose  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def psnr(se: torch.Tensor) -> float:
  """PSNR from a set of per-pixel squared errors on [0,1] RGB."""
  return float(-10 * torch.log10(se.mean().clamp(min=1e-12)))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--train-tag', default='left', help='Cameras the model trained on.')
  parser.add_argument('--eval-tag', default='right', help='Held-out cameras to score.')
  parser.add_argument('--samples', type=int, default=40)
  # The right camera runs about 4.6% darker than the left. Left uncorrected that DC gap
  # lands on every pixel equally and masquerades as reconstruction error, so remove a
  # per-image, per-channel offset and measure structure instead of exposure.
  parser.add_argument('--fix-exposure', action='store_true')
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  allcams = scene.getTrainCameras()
  train = sorted([c for c in allcams if args.train_tag in c.image_name],
                 key=lambda c: c.image_name)
  evalc = sorted([c for c in allcams if args.eval_tag in c.image_name],
                 key=lambda c: c.image_name)
  print(f'{len(train)} training cameras, {len(evalc)} held-out cameras')

  with torch.no_grad():
    depths, Rs, ts, Ks = [], [], [], []
    for c in train:
      depths.append(render(c, gaussians, pipe, bg)['depth'].squeeze().half())
      R, t = cam_pose(c)
      Rs.append(R)
      ts.append(t)
      Ks.append(torch.tensor([c.Fx, c.Fy, c.Cx, c.Cy], device='cuda'))
    depths = torch.stack(depths)
    Rs, ts, Ks = torch.stack(Rs), torch.stack(ts), torch.stack(Ks)
  N, H, W = depths.shape

  probe = evalc[::max(len(evalc) // args.samples, 1)][:args.samples]
  buckets = {'unobserved': [], 'undersampled': [], 'well sampled': []}
  with torch.no_grad():
    for view in probe:
      out = render(view, gaussians, pipe, bg)
      rgb = out['render'].clamp(0, 1)
      gt = view.original_image[:3].to(rgb.device)
      if args.fix_exposure:
        gt = (gt - gt.mean((1, 2), keepdim=True) + rgb.mean((1, 2), keepdim=True)).clamp(0, 1)
      se = ((rgb - gt) ** 2).mean(0)                       # (H, W)

      pts, nrm, z = backproject(out['depth'].squeeze(), view)
      valid = z > MIN_DEPTH
      se = se[1:-1, 1:-1]
      if view.gt_alpha_mask is not None:
        valid &= view.gt_alpha_mask.squeeze().to(valid.device)[1:-1, 1:-1] > 0.5

      Rv, tv = cam_pose(view)
      ray = pts.reshape(-1, 3) - (-Rv.T @ tv)
      ray = ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-12)
      n = nrm.reshape(-1, 3)
      cos_novel = (ray * n).sum(-1).abs().clamp(min=1e-3)
      dens_novel = view.Fx * view.Fy * cos_novel / z.reshape(-1) ** 2

      pc = torch.einsum('nij,pj->npi', Rs, pts.reshape(-1, 3)) + ts[:, None, :]
      zc = pc[..., 2]
      u = pc[..., 0] / zc.clamp(min=1e-6) * Ks[:, None, 0] + Ks[:, None, 2]
      v = pc[..., 1] / zc.clamp(min=1e-6) * Ks[:, None, 1] + Ks[:, None, 3]
      inside = (zc > MIN_DEPTH) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
      seen_z = torch.gather(depths.reshape(N, -1).float(), 1,
                            v.clamp(0, H - 1).long() * W + u.clamp(0, W - 1).long())
      visible = inside & (zc <= seen_z * (1 + OCCL_TOL)) & (seen_z > MIN_DEPTH)

      rayt = pts.reshape(-1, 3)[None] - (-torch.einsum('nij,ni->nj', Rs, ts))[:, None, :]
      rayt = rayt / rayt.norm(dim=-1, keepdim=True).clamp(min=1e-12)
      cos_t = (rayt * n[None]).sum(-1).abs().clamp(min=1e-3)
      dens_t = torch.where(visible,
                           Ks[:, None, 0] * Ks[:, None, 1] * cos_t / zc.clamp(min=1e-6) ** 2,
                           torch.zeros_like(zc))

      any_seen = visible.any(0)
      enough = any_seen & (dens_t.max(0).values >= dens_novel)
      flat_se, flat_ok = se.reshape(-1), valid.reshape(-1)
      buckets['unobserved'].append(flat_se[flat_ok & ~any_seen])
      buckets['undersampled'].append(flat_se[flat_ok & any_seen & ~enough])
      buckets['well sampled'].append(flat_se[flat_ok & enough])

  total = sum(b.numel() for v in buckets.values() for b in v)
  all_se = torch.cat([b for v in buckets.values() for b in v])
  print(f'\nheld-out {args.eval_tag} views, {len(probe)} sampled, '
        f'overall PSNR {psnr(all_se):.2f} dB\n')
  print(f'{"region":>14} {"share of pixels":>16} {"PSNR":>9} {"share of squared error":>24}')
  tot_se = all_se.sum()
  for k, v in buckets.items():
    x = torch.cat(v)
    print(f'{k:>14} {x.numel() / total:15.1%} {psnr(x):8.2f} dB {x.sum() / tot_se:23.1%}')


if __name__ == '__main__':
  main()
