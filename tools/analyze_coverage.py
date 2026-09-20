"""Decide whether off-path render quality is a reconstruction failure or missing data.

A novel view can only be *rendered* as well as the capture *sampled*. For every pixel of
an off-path view this asks what the training cameras actually recorded about the surface
behind it, and sorts the answer into three buckets:

unobserved
    No training camera saw that surface at all -- it was outside every frustum, or hidden
    behind something. Nothing recoverable from this capture; filling it in is generation,
    not reconstruction.

undersampled
    Some camera saw it, but every one of them spread fewer pixels over it than the novel
    view now wants. Sampling density on a surface goes as ``f^2 |cos t| / z^2`` (``t`` the
    incidence angle), so a sidewalk viewed at a grazing angle from a metre off the ground
    is captured at a small fraction of the density you get looking down at it. Asking for
    more detail than was ever recorded yields blur, and the blur is irreducible.

well sampled
    Seen by some camera at this density or better. If these pixels look wrong, that is the
    reconstruction's fault and better priors, capacity or training would fix them.

The split is what matters. A large unobserved/undersampled share means the capture
geometry is the ceiling and no amount of method work moves it; a small share means the
data is there and the renderer is squandering it.

Depth comes from the trained splat rather than the mesh, so occlusion reasoning matches
what actually gets rendered. Surface normals are differenced from the depth map instead
of taken from the rasterizer, to keep this independent of the normals the model was
trained to predict.

    micromamba run -n vid2sim-recon python tools/analyze_coverage.py \
        -m output/urban_walk_20m_stereo --offsets 0 0.5 1.0 2.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from render_flythrough import shifted  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

STRIDE = 4          # subsample the novel view; coverage varies smoothly across pixels
MIN_DEPTH = 0.05    # SfM units; drop pixels the rasterizer left empty
OCCL_TOL = 0.02     # relative depth slack when deciding "occluded"


def cam_pose(cam):
  """(R_w2c, t) such that x_cam = R_w2c @ X + t."""
  wvt = cam.world_view_transform
  return wvt[:3, :3].T.contiguous(), wvt[3, :3].contiguous()


def backproject(depth, cam):
  """Depth map -> world points, plus a world normal per point from local differences."""
  H, W = depth.shape
  R, t = cam_pose(cam)
  ys, xs = torch.meshgrid(torch.arange(H, device=depth.device),
                          torch.arange(W, device=depth.device), indexing='ij')
  x = (xs - cam.Cx) / cam.Fx * depth
  y = (ys - cam.Cy) / cam.Fy * depth
  cam_pts = torch.stack([x, y, depth], -1)
  world = (cam_pts.reshape(-1, 3) - t) @ R
  world = world.reshape(H, W, 3)

  # Normals from neighbouring surface points. Sign is fixed to face the camera, since
  # only |cos incidence| enters the sampling density.
  du = world[1:-1, 2:] - world[1:-1, :-2]
  dv = world[2:, 1:-1] - world[:-2, 1:-1]
  n = torch.cross(du, dv, dim=-1)
  n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
  return world[1:-1, 1:-1], n, depth[1:-1, 1:-1]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument('--iteration', default=-1, type=int)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--offsets', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0],
                      help='Lateral offsets in metres.')
  parser.add_argument('--up', type=float, default=0.0)
  parser.add_argument('--samples', type=int, default=16, help='Novel views to probe.')
  parser.add_argument('--only', default='left')
  args = get_combined_args(parser)
  safe_state(args.quiet)

  dataset = model.extract(args)
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
  bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
  pipe = pipeline.extract(args)

  align = json.loads((pathlib.Path(dataset.source_path) / 'sfm_metric_alignment.json').read_text())
  m_per_unit = align['scale']

  cams = sorted([c for c in scene.getTrainCameras() if args.only in c.image_name],
                key=lambda c: c.image_name)
  print(f'{len(cams)} training cameras, {m_per_unit:.4f} m per SfM unit')

  # Every training camera's depth map, so occlusion can be tested against what the model
  # actually renders rather than against a proxy surface.
  with torch.no_grad():
    depths, Rs, ts, Ks = [], [], [], []
    for c in scene.getTrainCameras():
      d = render(c, gaussians, pipe, bg)['depth'].squeeze()
      depths.append(d.half())
      R, t = cam_pose(c)
      Rs.append(R)
      ts.append(t)
      Ks.append(torch.tensor([c.Fx, c.Fy, c.Cx, c.Cy], device='cuda'))
    depths = torch.stack(depths)
    Rs, ts, Ks = torch.stack(Rs), torch.stack(ts), torch.stack(Ks)
  N, H, W = depths.shape
  print(f'cached {N} training depth maps at {W}x{H}\n')

  # World up, borrowed from the LiDAR alignment: the Umeyama rotation takes SfM axes into
  # the LiDAR frame, where +z is up. Needed to tell a sidewalk from a wall.
  up = torch.tensor(np.array(align['R']).T @ np.array([0.0, 0.0, 1.0]),
                    dtype=torch.float32, device='cuda')

  probe = cams[::max(len(cams) // args.samples, 1)][:args.samples]
  rows, by_class = [], {}
  for off in args.offsets:
    d_cam = np.array([off, -args.up, 0.0]) / m_per_unit
    unobs, under, well, angles = [], [], [], []
    cls_acc = {'ground': [[], []], 'vertical': [[], []]}
    with torch.no_grad():
      for base in probe:
        view = base if (off == 0 and args.up == 0) else shifted(base, d_cam)
        depth = render(view, gaussians, pipe, bg)['depth'].squeeze()
        pts, nrm, z = backproject(depth, view)
        pts, nrm, z = pts[::STRIDE, ::STRIDE], nrm[::STRIDE, ::STRIDE], z[::STRIDE, ::STRIDE]
        keep = z > MIN_DEPTH
        pts, nrm, z = pts[keep], nrm[keep], z[keep]
        if not len(pts):
          continue

        Rv, tv = cam_pose(view)
        centre = -Rv.T @ tv
        ray = pts - centre
        ray = ray / ray.norm(dim=-1, keepdim=True)
        cos_novel = (ray * nrm).sum(-1).abs().clamp(min=1e-3)
        dens_novel = view.Fx * view.Fy * cos_novel / z ** 2

        # Project every point into every training camera at once.
        pc = torch.einsum('nij,pj->npi', Rs, pts) + ts[:, None, :]      # (N, P, 3)
        zc = pc[..., 2]
        u = pc[..., 0] / zc.clamp(min=1e-6) * Ks[:, None, 0] + Ks[:, None, 2]
        v = pc[..., 1] / zc.clamp(min=1e-6) * Ks[:, None, 1] + Ks[:, None, 3]
        inside = (zc > MIN_DEPTH) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        seen_z = torch.gather(depths.reshape(N, -1).float(), 1, vi * W + ui)
        visible = inside & (zc <= seen_z * (1 + OCCL_TOL)) & (seen_z > MIN_DEPTH)

        rayt = pts[None] - (-torch.einsum('nij,ni->nj', Rs, ts))[:, None, :]
        rayt = rayt / rayt.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cos_t = (rayt * nrm[None]).sum(-1).abs().clamp(min=1e-3)
        dens_t = Ks[:, None, 0] * Ks[:, None, 1] * cos_t / zc.clamp(min=1e-6) ** 2
        dens_t = torch.where(visible, dens_t, torch.zeros_like(dens_t))

        any_seen = visible.any(0)
        best_dens = dens_t.max(0).values
        enough = any_seen & (best_dens >= dens_novel)

        unobs.append((~any_seen).float().mean().item())
        under.append((any_seen & ~enough).float().mean().item())
        well.append(enough.float().mean().item())

        ang = torch.acos(((rayt * ray[None]).sum(-1)).clamp(-1, 1))
        ang = torch.where(visible, ang, torch.full_like(ang, np.pi))
        angles.append(np.degrees(ang.min(0).values[any_seen].median().item()))

        # Ground vs wall, by how far the surface normal tilts from world up.
        nz = (nrm * up).sum(-1).abs()
        for label, sel in (('ground', nz > np.cos(np.radians(30))),
                           ('vertical', nz < np.cos(np.radians(60)))):
          if sel.any():
            cls_acc[label][0].append((~enough[sel]).float().mean().item())
            # Incidence of the best observer: how squarely anything ever saw this surface.
            best = dens_t.argmax(0)
            inc = torch.gather(cos_t, 0, best[None]).squeeze(0)[sel & any_seen]
            if inc.numel():
              cls_acc[label][1].append(np.degrees(torch.acos(
                inc.clamp(0, 1)).median().item()))

    rows.append((off, np.mean(unobs), np.mean(under), np.mean(well), np.mean(angles)))
    by_class[off] = {k: (np.mean(v[0]) if v[0] else np.nan,
                         np.mean(v[1]) if v[1] else np.nan) for k, v in cls_acc.items()}

  print(f'{"offset":>8} {"unobserved":>11} {"undersampled":>13} {"well sampled":>13} '
        f'{"median view-angle change":>25}')
  for off, a, b, c, ang in rows:
    print(f'{off:7.2f}m {a:10.1%} {b:12.1%} {c:12.1%} {ang:22.1f} deg')

  print(f'\nshare of each surface type that is unobserved or undersampled:')
  print(f'{"offset":>8} {"ground":>10} {"vertical":>10}')
  for off in args.offsets:
    g, v = by_class[off]['ground'], by_class[off]['vertical']
    print(f'{off:7.2f}m {g[0]:9.1%} {v[0]:9.1%}')
  g, v = by_class[args.offsets[0]]['ground'], by_class[args.offsets[0]]['vertical']
  print(f'\nmedian incidence angle of the best observer '
        f'(90 deg = edge-on): ground {g[1]:.1f} deg, vertical {v[1]:.1f} deg')


if __name__ == '__main__':
  main()
