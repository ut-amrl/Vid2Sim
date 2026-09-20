"""Two standard 3DGS off-path failure modes, tested against held-out stereo ground truth.

Error attribution puts the off-path loss in pixels the capture sampled perfectly well, so
the reconstruction is what is failing. The two usual culprits are both measurable here.

View-dependent overfitting
    Spherical harmonics give every Gaussian a free per-direction colour. On a capture
    whose viewing directions barely vary, that freedom is nearly unconstrained: the model
    can absorb geometry and exposure error into apparent view-dependence and still fit
    every training image. Rendering at reduced SH degree asks what the fit costs. If
    novel-view PSNR *rises* when the extra bands are discarded, they were memorising.

Anisotropic sliver Gaussians
    A Gaussian stretched along the viewing direction is nearly free to the training loss
    -- it projects to almost the same footprint from every camera on the path -- but
    swings across the image the moment the camera steps aside. That is what "melted"
    off-path renders are made of. The axis-ratio distribution shows how much of the model
    is built from them.

    micromamba run -n vid2sim-recon python tools/diagnose_offpath.py \
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

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


@torch.no_grad()
def score(cams, gaussians, pipe, bg) -> float:
  se = []
  for c in cams:
    rgb = render(c, gaussians, pipe, bg)['render'].clamp(0, 1)
    gt = c.original_image[:3].to(rgb.device)
    m = ((rgb - gt) ** 2).mean(0)
    if c.gt_alpha_mask is not None:
      m = m[c.gt_alpha_mask.squeeze().to(m.device) > 0.5]
    se.append(m.reshape(-1))
  return float(-10 * torch.log10(torch.cat(se).mean()))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--train-tag', default='left')
  parser.add_argument('--eval-tag', default='right')
  parser.add_argument('--samples', type=int, default=40)
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  allc = scene.getTrainCameras()
  pick = lambda tag: sorted([c for c in allc if tag in c.image_name],  # noqa: E731
                            key=lambda c: c.image_name)[::max(len(allc) // 2 // args.samples, 1)]
  train, held = pick(args.train_tag), pick(args.eval_tag)
  full = gaussians.active_sh_degree
  print(f'{len(train)} train / {len(held)} held-out views, model SH degree {full}\n')

  print(f'{"SH degree":>10} {"train PSNR":>12} {"held-out PSNR":>15} {"gap":>8}')
  for deg in range(full + 1):
    gaussians.active_sh_degree = deg
    a, b = score(train, gaussians, pipe, bg), score(held, gaussians, pipe, bg)
    print(f'{deg:>10} {a:9.2f} dB {b:12.2f} dB {a - b:6.2f} dB')
  gaussians.active_sh_degree = full

  # A flat disc (s0 ~ s1 >> s2) is the healthy shape for a surface element, so
  # longest/shortest alone cannot diagnose anything. Needles are the pathology, and they
  # are the ones with a large s0/s1.
  srt = gaussians.get_scaling.detach().sort(dim=-1, descending=True).values
  needle = (srt[:, 0] / srt[:, 1].clamp(min=1e-12)).cpu().numpy()
  disc = (srt[:, 1] / srt[:, 2].clamp(min=1e-12)).cpu().numpy()
  print(f'\n{len(needle)} Gaussians')
  for name, r in (('needle-ness (s0/s1)', needle), ('disc-ness  (s1/s2)', disc)):
    q = np.percentile(r, [50, 90, 99])
    print(f'  {name}: median {q[0]:8.1f}  p90 {q[1]:10.1f}  p99 {q[2]:12.1f}')
  print(f'  needles (s0/s1 > 10): {(needle > 10).mean():.2%} of Gaussians')
  print(f'  discs   (s1/s2 > 10): {(disc > 10).mean():.2%} of Gaussians')


if __name__ == '__main__':
  main()
