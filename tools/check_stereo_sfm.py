"""Score a stereo SfM solve, using the calibrated baseline as an independent ruler.

The stereo clip is handed to SfM as 2N unrelated images -- nothing tells the solver that
``00042_left`` and ``00042_right`` are 24.5 cm apart on a rigid bar. That omission is
what makes this check meaningful: the left/right separation SfM *recovers* is a
prediction, not an input, so comparing it to the calibration measures the solve rather
than echoing it back.

Two numbers come out of that comparison:

* **Scale.** The median recovered separation converts SfM units to metres, with no
  reliance on LiDAR. Cross-checking it against the Umeyama fit to the LiDAR trajectory
  gives two independent estimates that should agree to within their noise.
* **Local consistency.** The *spread* of the per-frame separation is a direct readout of
  local pose error. A global similarity transform cannot hide it -- scale cancels when
  you look at relative spread -- so unlike the Umeyama residual, which mixes drift and
  noise, this isolates how steady the solve is frame to frame.

The rectified pair should also come out very nearly parallel, so the residual rotation
between the two cameras is reported as a third, independent sanity check.

    micromamba run -n mcap python tools/check_stereo_sfm.py --seq data/<clip>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from check_sfm_vs_lidar import quat_to_R, umeyama  # noqa: E402
from tools.colmap_utils.loader import qvec2rotmat, read_extrinsics_binary  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True)
  ap.add_argument('--poses',
                  default='/robodata/arthurz/Datasets/lsmap_bags_processed/poses/legoloam/19.txt')
  ap.add_argument('--mission',
                  default='/robodata/arthurz/Research/playground/gauss_gym/mcap_scene_data/'
                          'mission_2025-01-17-10-37-51')
  args = ap.parse_args()

  seq = pathlib.Path(args.seq)
  meta = json.loads((seq / 'meta.json').read_text())
  t_of = {f['name']: f['timestamp'] for f in meta['frames']}
  baseline_m = meta['baseline_m']

  extr = read_extrinsics_binary(str(seq / 'sparse' / '0' / 'images.bin'))
  centre, rot = {}, {}
  for im in extr.values():
    R = qvec2rotmat(im.qvec)
    centre[im.name] = -R.T @ im.tvec
    rot[im.name] = R
  print(f'SfM registered {len(centre)}/{meta["num_frames"]} images '
        f'({meta["num_pairs"]} pairs)')

  pairs = [n[:-len('_left.jpg')] for n in centre if n.endswith('_left.jpg')
           and n.replace('_left', '_right') in centre]
  pairs.sort()
  sep = np.array([np.linalg.norm(centre[f'{p}_left.jpg'] - centre[f'{p}_right.jpg'])
                  for p in pairs])
  # Angle between the two optical axes; a rectified pair is parallel by construction.
  ang = np.array([np.degrees(np.arccos(np.clip(
    (np.trace(rot[f'{p}_left.jpg'] @ rot[f'{p}_right.jpg'].T) - 1) / 2, -1, 1)))
    for p in pairs])

  scale_stereo = baseline_m / np.median(sep)
  print(f'\ncomplete stereo pairs: {len(pairs)}')
  print(f'recovered separation spread: {sep.std() / sep.mean():.2%} of mean '
        f'(min {sep.min() / np.median(sep):.3f}x, max {sep.max() / np.median(sep):.3f}x)')
  print(f'left/right axis angle: median {np.median(ang):.3f} deg, max {ang.max():.3f} deg')
  print(f'scale from calibrated {baseline_m * 100:.2f} cm baseline: '
        f'{scale_stereo:.5f} m per SfM unit')

  # Independent estimate: fit the left-camera track to the LiDAR-SLAM trajectory.
  poses = np.loadtxt(args.poses)
  attrs = json.loads((pathlib.Path(args.mission) / 'data' / 'front' / '.zattrs').read_text())
  tr = attrs['transform']
  t_cb = np.array([tr['translation'][k] for k in 'xyz'])
  R_cb = quat_to_R(np.array([tr['rotation'][k] for k in 'wxyz']))

  names = [f'{p}_left.jpg' for p in pairs]
  lid = np.empty((len(names), 3))
  for i, n in enumerate(names):
    t = t_of[n]
    j = np.clip(np.searchsorted(poses[:, 0], t), 1, len(poses) - 1)
    a = (t - poses[j - 1, 0]) / max(poses[j, 0] - poses[j - 1, 0], 1e-9)
    p = (1 - a) * poses[j - 1, 1:4] + a * poses[j, 1:4]
    lid[i] = p + quat_to_R(poses[j, 4:8]) @ (-R_cb.T @ t_cb)

  src = np.array([centre[n] for n in names])
  scale_lidar, R, t = umeyama(src, lid)
  resid = np.linalg.norm((scale_lidar * (R @ src.T).T + t) - lid, axis=1)
  path = np.linalg.norm(np.diff(lid, axis=0), axis=1).sum()

  print(f'scale from LiDAR Umeyama fit:  {scale_lidar:.5f} m per SfM unit')
  print(f'  the two disagree by {abs(scale_stereo - scale_lidar) / scale_lidar:.2%}')
  print(f'\nLiDAR path length {path:.2f} m over {len(names)} pairs')
  print(f'residual after alignment: mean {resid.mean() * 100:.1f} cm  '
        f'median {np.median(resid) * 100:.1f} cm  max {resid.max() * 100:.1f} cm')
  print(f'drift as fraction of path: {resid.mean() / path * 100:.2f}%')

  out = dict(scale=scale_stereo, scale_from_lidar=scale_lidar,
             R=R.tolist(), t=t.tolist(),
             baseline_m=baseline_m, baseline_spread=float(sep.std() / sep.mean()),
             residual_mean_m=float(resid.mean()), residual_max_m=float(resid.max()),
             path_length_m=float(path), num_registered=len(centre), num_pairs=len(pairs))
  (seq / 'sfm_metric_alignment.json').write_text(json.dumps(out, indent=2))
  print(f'-> {seq / "sfm_metric_alignment.json"}')


if __name__ == '__main__':
  main()
