"""Is the LiDAR projection accurate, or only plausible-looking?

A first pass comparing projected LiDAR against block-matched stereo gave a median
disagreement worse than the splat's own error, which would make LiDAR useless as a prior.
That comparison is not trustworthy as stated, for two reasons:

* **Edges dominate.** LiDAR returns are sparse, so a one-pixel registration slip at a
  depth discontinuity pairs a foreground return with a background match and produces an
  enormous relative error. Restricting to locally smooth neighbourhoods removes the
  effect without hiding real bias.
* **Stereo is unreliable on the sidewalk.** Block matching needs texture, and a large part
  of this scene is smooth concrete where SGBM quietly invents disparity.

So this reports *signed* error, which separates the two failure modes a plausible-looking
projection can still have: a constant relative bias means a scale or intrinsics problem,
while an error growing with range means a translation or timing problem. A timing sweep is
included because the LiDAR stamp marks the start of a sweep while the camera stamp marks
mid-exposure, and at walking pace that offset is worth centimetres.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lidar_depth import project_scan, read_static_tf, sgbm_depth  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True)
  ap.add_argument('--mcap', default='/scratch/zichaohu/real2sim/mission_2025-01-17-10-37-51.mcap')
  ap.add_argument('--frames', type=int, default=30)
  ap.add_argument('--against', choices=['stereo', 'sfm'], default='stereo',
                  help='Reference depth. "sfm" uses the triangulated points in '
                       'sfm_gt_depths, which need no texture to be reliable and whose '
                       'scale was already verified against the calibrated baseline.')
  ap.add_argument('--spread-tol', type=float, default=0.05)
  ap.add_argument('--points-frame', default='os_lidar')
  ap.add_argument('--smooth-tol', type=float, default=0.05,
                  help='Max relative spread of stereo depth in a 5x5 window to accept.')
  args = ap.parse_args()

  seq = pathlib.Path(args.seq)
  meta = json.loads((seq / 'meta.json').read_text())
  m_per_unit = json.loads((seq / 'sfm_metric_alignment.json').read_text())['scale']
  W, H = meta['width'], meta['height']
  fx, fy, cx, cy = meta['fx'], meta['fy'], meta['cx'], meta['cy']

  store = get_typestore(Stores.ROS2_HUMBLE)
  tf = read_static_tf(args.mcap, store)
  T_cam_lidar = np.linalg.inv(tf[(args.points_frame, 'left_optical')])
  with open(args.mcap, 'rb') as f:
    for schema, _ch, message in make_reader(f).iter_messages(
        topics=['/camera/left/camera_info']):
      R_rect = np.asarray(store.deserialize_cdr(message.data, schema.name).r,
                          np.float64).reshape(3, 3)
      break
  M = np.eye(4)
  M[:3, :3] = R_rect
  T = M @ T_cam_lidar

  frames = {f['timestamp']: f['name'] for f in meta['frames'] if f['side'] == 'left'}
  times = np.array(sorted(frames))
  probe = times[::max(len(times) // args.frames, 1)][:args.frames]

  scans = {}
  with open(args.mcap, 'rb') as f:
    it = make_reader(f).iter_messages(topics=['/points'],
                                      start_time=int((times[0] - 0.5) * 1e9),
                                      end_time=int((times[-1] + 0.5) * 1e9))
    for schema, _ch, message in it:
      m = store.deserialize_cdr(message.data, schema.name)
      t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
      j = int(np.argmin(np.abs(probe - t)))
      if abs(probe[j] - t) > 0.06 or probe[j] in scans:
        continue
      p = np.frombuffer(bytes(m.data), np.float32).reshape(-1, m.point_step // 4)[:, :3]
      scans[probe[j]] = p[np.isfinite(p).all(1)].astype(np.float64)

  rel_all, rel_smooth, signed, depths = [], [], [], []
  for t, pts in sorted(scans.items()):
    stem = frames[t][:-4]
    if args.against == 'sfm':
      z = np.load(seq / 'sfm_gt_depths' / f'{stem}.npz')
      uv = np.round(z['pts_cam']).astype(int)
      k = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
      ref = np.zeros((H, W), np.float32)
      # SfM depth is in SfM units; the metric scale came from the calibrated baseline.
      ref[uv[k, 1], uv[k, 0]] = z['depth'][k] * m_per_unit
      ok = ref > 0
      smooth = ok
    else:
      L = cv2.imread(str(seq / 'images' / f'{stem}.jpg'))
      R = cv2.imread(str(seq / 'images' / f'{stem.replace("_left", "_right")}.jpg'))
      ref, ok = sgbm_depth(L, R, fx, meta['baseline_m'])

      # Local spread of the stereo depth, used to drop edges and untextured guesses.
      m_ = ref.copy()
      m_[~ok] = np.nan
      mx = cv2.dilate(np.nan_to_num(m_, nan=-1e9), np.ones((5, 5), np.uint8))
      mn = -cv2.dilate(np.nan_to_num(-m_, nan=-1e9), np.ones((5, 5), np.uint8))
      smooth = ok & ((mx - mn) < args.smooth_tol * np.maximum(ref, 1e-6))

    d = project_scan(pts, T, W, H, fx, fy, cx, cy, spread_tol=args.spread_tol)

    base = (d > 0.5) & (d < 15) & (ref > 0.5) & (ref < 15)
    sel_all, sel_s = base & ok, base & smooth
    if sel_all.sum() > 100:
      rel_all.append(np.abs(d[sel_all] - ref[sel_all]) / ref[sel_all])
    if sel_s.sum() > 100:
      rel_smooth.append(np.abs(d[sel_s] - ref[sel_s]) / ref[sel_s])
      signed.append((d[sel_s] - ref[sel_s]) / ref[sel_s])
      depths.append(ref[sel_s])

  a, s, sg, zz = (np.concatenate(x) for x in (rel_all, rel_smooth, signed, depths))
  print(f'{len(scans)} frames')
  print(f'all overlapping pixels   ({len(a):7d}): median abs rel {np.median(a):6.2%}')
  print(f'locally smooth only      ({len(s):7d}): median abs rel {np.median(s):6.2%}  '
        f'p90 {np.percentile(s, 90):.2%}')
  print(f'signed (LiDAR - stereo)/stereo: median {np.median(sg):+.2%}  '
        f'mean {sg.mean():+.2%}')
  print('\nby range (smooth pixels):')
  for lo, hi in ((0.5, 2), (2, 4), (4, 7), (7, 15)):
    k = (zz >= lo) & (zz < hi)
    if k.sum() > 50:
      print(f'  {lo:4.1f}-{hi:4.1f} m ({k.sum():7d} px): median signed {np.median(sg[k]):+6.2%}'
            f'  abs {np.median(s[k]):6.2%}')


if __name__ == '__main__':
  main()
