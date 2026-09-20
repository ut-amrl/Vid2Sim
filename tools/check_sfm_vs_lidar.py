"""Score Vid2Sim's SfM against the log's LiDAR-SLAM trajectory, and recover metric scale.

Vid2Sim reconstructs from pixels alone, so its world is fixed only up to a similarity
transform: rotation, translation, and an unknown global scale. Two things follow.

1. There is no way to state a novel-view offset in metres ("render this 30 cm to the
   left") without first pinning that scale down.
2. Nothing in the pipeline checks whether the camera track is actually *shaped* right.
   Drift that bends a straight sidewalk into an arc still fits the photometric loss.

The log carries an independent answer to both: LeGO-LOAM poses from the 3D LiDAR. This
fits the Umeyama similarity from SfM camera centres to LiDAR camera centres over the
matched frames. The scale is the metres-per-SfM-unit conversion; the residual after
alignment is the drift SfM could not have detected on its own.

Both tracks describe the same physical camera, so the rigid camera<-base extrinsic has
to be applied to the LiDAR poses before comparing -- otherwise a ~0.5 m lever arm leaks
into the residual and inflates it.

    micromamba run -n mcap python tools/check_sfm_vs_lidar.py --seq data/<clip>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))
from tools.colmap_utils.loader import read_extrinsics_binary, qvec2rotmat  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
  """Least-squares similarity (scale, R, t) with dst ~= scale * R @ src + t."""
  mu_s, mu_d = src.mean(0), dst.mean(0)
  s0, d0 = src - mu_s, dst - mu_d
  U, D, Vt = np.linalg.svd(d0.T @ s0 / len(src))
  S = np.eye(3)
  if np.linalg.det(U) * np.linalg.det(Vt) < 0:
    S[2, 2] = -1
  R = U @ S @ Vt
  scale = float((D * np.diag(S)).sum() / (s0 ** 2).sum() * len(src))
  return scale, R, dst.mean(0) - scale * R @ mu_s


def quat_to_R(q: np.ndarray) -> np.ndarray:
  """(w,x,y,z) -> rotation matrix."""
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True)
  ap.add_argument('--poses',
                  default='/robodata/arthurz/Datasets/lsmap_bags_processed/poses/legoloam/19.txt',
                  help='LeGO-LOAM poses: t x y z qw qx qy qz, describing os_sensor.')
  ap.add_argument('--mission',
                  default='/robodata/arthurz/Research/playground/gauss_gym/mcap_scene_data/'
                          'mission_2025-01-17-10-37-51',
                  help='Supplies the camera<-base extrinsic from the zarr attrs.')
  args = ap.parse_args()

  seq = pathlib.Path(args.seq)
  meta = json.loads((seq / 'meta.json').read_text())
  t_of = {f['name']: f['timestamp'] for f in meta['frames']}

  extr = read_extrinsics_binary(str(seq / 'sparse' / '0' / 'images.bin'))
  sfm_c, times = {}, {}
  for im in extr.values():
    R = qvec2rotmat(im.qvec)
    sfm_c[im.name] = -R.T @ im.tvec        # world-frame camera centre
    times[im.name] = t_of[im.name]
  names = sorted(sfm_c)
  print(f'SfM registered {len(names)}/{meta["num_frames"]} frames')

  # LiDAR poses describe os_sensor; hop to the camera so both tracks are the same point.
  poses = np.loadtxt(args.poses)
  attrs = json.loads((pathlib.Path(args.mission) / 'data' / 'front' / '.zattrs').read_text())
  tr = attrs['transform']
  t_cb = np.array([tr['translation'][k] for k in 'xyz'])
  R_cb = quat_to_R(np.array([tr['rotation'][k] for k in 'wxyz']))

  ts = np.array([times[n] for n in names])
  lid = np.empty((len(names), 3))
  for i, t in enumerate(ts):
    j = np.clip(np.searchsorted(poses[:, 0], t), 1, len(poses) - 1)
    t0, t1 = poses[j - 1, 0], poses[j, 0]
    a = (t - t0) / max(t1 - t0, 1e-9)
    p = (1 - a) * poses[j - 1, 1:4] + a * poses[j, 1:4]
    R_wl = quat_to_R(poses[j, 4:8])
    lid[i] = p + R_wl @ (-R_cb.T @ t_cb)   # base->camera offset, rotated into world

  src = np.array([sfm_c[n] for n in names])
  scale, R, t = umeyama(src, lid)
  resid = np.linalg.norm((scale * (R @ src.T).T + t) - lid, axis=1)
  path = np.linalg.norm(np.diff(lid, axis=0), axis=1).sum()

  print(f'metric scale      {scale:.5f} m per SfM unit')
  print(f'LiDAR path length {path:.2f} m over {len(names)} frames')
  print(f'residual after alignment: mean {resid.mean()*100:.1f} cm  '
        f'median {np.median(resid)*100:.1f} cm  max {resid.max()*100:.1f} cm')
  print(f'drift as fraction of path: {resid.mean()/path*100:.2f}%')

  out = dict(scale=scale, R=R.tolist(), t=t.tolist(),
             residual_mean_m=float(resid.mean()), residual_max_m=float(resid.max()),
             path_length_m=float(path), num_registered=len(names))
  (seq / 'sfm_metric_alignment.json').write_text(json.dumps(out, indent=2))
  print(f'-> {seq / "sfm_metric_alignment.json"}')


if __name__ == '__main__':
  main()
