"""Project LiDAR returns into each rectified camera frame as a sparse metric depth prior.

Depth-Anything gives a dense but only *relative* depth field, and the loss that consumes
it is patch-NCC, which is invariant to scale and offset by construction. Structure is
therefore supervised and absolute depth is not, which is how the reconstruction ends up
around 14% off in range while still fitting every training image.

LiDAR supplies exactly the missing piece. Each scan is transformed into the camera frame
through the fixed extrinsic and projected, so the result never depends on the SLAM poses
or on the SfM-to-LiDAR alignment -- only on calibration and time sync, both of which are
checkable. A per-frame pose error would bias the whole trajectory; an extrinsic error is a
single constant we can measure once against block-matched stereo (``--validate``).

Three transforms sit between the raw points and a pixel, and all three matter:

1. ``os_sensor -> os_lidar -> left_optical`` from ``/tf_static``, composed and inverted.
2. The rectification rotation ``R`` from ``CameraInfo``. The images were rectified, so the
   camera frame the points must land in is the *rectified* one. Skipping this looks
   harmless -- it is only a degree or two -- but at 2.4 m it displaces points by several
   centimetres, the same order as the error being chased.
3. The rectified projection ``P``, matching the intrinsics the frames were resampled to.

Output is inverse depth in SfM units with 0 marking "no return", matching what
``dataset_readers.py`` already expects, so the trainer needs no new file format.

    micromamba run -n mcap python tools/lidar_depth.py --seq data/<clip> --validate
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


def quat_to_R(q: np.ndarray) -> np.ndarray:
  """(w,x,y,z) -> rotation matrix."""
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def se3(t: np.ndarray, q: np.ndarray) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = quat_to_R(q)
  T[:3, 3] = t
  return T


def read_static_tf(mcap_path: str, store) -> dict[tuple[str, str], np.ndarray]:
  """Parent->child transforms, i.e. the pose of child expressed in parent."""
  out = {}
  with open(mcap_path, 'rb') as f:
    for schema, _ch, message in make_reader(f).iter_messages(topics=['/tf_static']):
      m = store.deserialize_cdr(message.data, schema.name)
      for tf in m.transforms:
        tr, r = tf.transform.translation, tf.transform.rotation
        key = (tf.header.frame_id.lstrip('/'), tf.child_frame_id.lstrip('/'))
        out[key] = se3(np.array([tr.x, tr.y, tr.z]),
                       np.array([r.w, r.x, r.y, r.z]))
      break
  return out


def read_sfm_depths(sparse_dir: pathlib.Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Per image, the triangulated keypoints as (pixel coords, depth in SfM units).

  Read straight from the reconstruction rather than from a cached ``sfm_gt_depths``
  directory, because those caches get copied between sequences while each reconstruction
  keeps its own arbitrary scale -- a copied cache silently mixes two unit systems.
  """
  import struct

  with open(sparse_dir / 'points3D.bin', 'rb') as f:
    n, = struct.unpack('<Q', f.read(8))
    xyz = {}
    for _ in range(n):
      pid, x, y, z = struct.unpack('<Qddd', f.read(32))
      f.read(3 + 8)  # rgb, reprojection error
      track, = struct.unpack('<Q', f.read(8))
      f.read(8 * track)
      xyz[pid] = (x, y, z)

  out = {}
  with open(sparse_dir / 'images.bin', 'rb') as f:
    n, = struct.unpack('<Q', f.read(8))
    for _ in range(n):
      _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack('<idddddddi', f.read(64))
      name = b''
      while (c := f.read(1)) != b'\x00':
        name += c
      npts, = struct.unpack('<Q', f.read(8))
      buf = np.frombuffer(f.read(24 * npts), dtype=np.dtype(
        [('x', '<f8'), ('y', '<f8'), ('pid', '<i8')]))
      k = buf['pid'] >= 0
      pts = np.array([xyz[p] for p in buf['pid'][k] if p in xyz])
      if not len(pts):
        continue
      R = quat_to_R(np.array([qw, qx, qy, qz]))
      out[name.decode()] = (np.stack([buf['x'][k], buf['y'][k]], 1),
                            (pts @ R.T + np.array([tx, ty, tz]))[:, 2])
  return out


def project_scan(pts, T, W, H, fx, fy, cx, cy, max_range=40.0, spread_tol=0.15):
  """Scan -> depth image, dropping pixels whose depth is ambiguous.

  A surface seen edge-on spans a huge depth range inside a single pixel, and near the
  horizon grazing ground returns land on the same pixels as distant structure. Taking the
  nearest return there assigns metres-close depth to pixels that actually show something
  far away -- which is how a projection that looks right in the near field ends up biased
  80% short at range.

  So a pixel is kept only when the returns around it agree: local spread, measured over a
  3x3 window, must be within ``spread_tol`` of the local minimum. Grazing and boundary
  pixels fail that test and are left empty, which is the honest answer for them.
  """
  p = pts @ T[:3, :3].T + T[:3, 3]
  z = p[:, 2]
  k = (z > 0.3) & (z < max_range)
  p, z = p[k], z[k]
  u = np.round(p[:, 0] / z * fx + cx).astype(int)
  v = np.round(p[:, 1] / z * fy + cy).astype(int)
  k = (u >= 0) & (u < W) & (v >= 0) & (v < H)
  u, v, z = u[k], v[k], z[k]

  near = np.full((H, W), np.inf)
  far = np.full((H, W), -np.inf)
  np.minimum.at(near, (v, u), z)
  np.maximum.at(far, (v, u), z)
  hit = np.isfinite(near)

  kern = np.ones((3, 3), np.uint8)
  loc_min = -cv2.dilate(np.where(hit, -near, -1e9).astype(np.float32), kern)
  loc_max = cv2.dilate(np.where(hit, far, -1e9).astype(np.float32), kern)

  depth = np.zeros((H, W), np.float64)
  good = hit & (loc_min > 0) & ((loc_max - loc_min) < spread_tol * loc_min)
  depth[good] = near[good]
  return depth


def sgbm_depth(left, right, fx, baseline):
  m = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8 * 3 * 7 ** 2, P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  disp = m.compute(left, right).astype(np.float32) / 16.0
  valid = disp > 2.0
  depth = np.zeros_like(disp)
  depth[valid] = fx * baseline / disp[valid]
  return depth, valid


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True)
  ap.add_argument('--mcap', default='/scratch/zichaohu/real2sim/mission_2025-01-17-10-37-51.mcap')
  ap.add_argument('--points-topic', default='/points')
  ap.add_argument('--points-frame', default='os_lidar',
                  help='Frame the cloud is really in, which is not what its header says.')
  ap.add_argument('--ros2-store', default='ROS2_HUMBLE')
  ap.add_argument('--max-dt', type=float, default=0.06,
                  help='Largest camera/LiDAR stamp gap to accept, seconds.')
  ap.add_argument('--max-range', type=float, default=40.0)
  ap.add_argument('--spread-tol', type=float, default=0.15,
                  help='Reject a pixel when returns in its 3x3 window disagree by more '
                       'than this fraction of the nearest one.')
  ap.add_argument('--validate', action='store_true',
                  help='Score the projection against block-matched stereo instead of '
                       'writing depth maps.')
  ap.add_argument('--sides', nargs='+', default=['left'], choices=['left', 'right'])
  ap.add_argument('--out', default='lidar_depths')
  args = ap.parse_args()

  seq = pathlib.Path(args.seq)
  meta = json.loads((seq / 'meta.json').read_text())
  W, H = meta['width'], meta['height']
  fx, fy, cx, cy = meta['fx'], meta['fy'], meta['cx'], meta['cy']

  store = get_typestore(getattr(Stores, args.ros2_store))
  tf = read_static_tf(args.mcap, store)

  frames = [f for f in meta['frames'] if f.get('side', 'left') in args.sides]
  sides = sorted({f.get('side', 'left') for f in frames})
  times = np.array(sorted({f['timestamp'] for f in frames}))
  lo, hi = times[0] - 0.5, times[-1] + 0.5

  # One transform per camera. The cloud's header says os_sensor, but the data is in
  # os_lidar -- the two differ by a 180 degree yaw, so trusting the header points the
  # camera backwards down the robot's -x axis. It still looks believable up close, because
  # objects a metre away project into roughly the right part of the frame either way, and
  # only falls apart at range: scored against the triangulated SfM points, honouring the
  # header gives 85% median depth error versus 8% for os_lidar. Hence the frame is chosen
  # here rather than read from the header. The rectification rotation is applied on top,
  # since the images were rectified and points must land in that frame -- it is only about
  # a degree, but that is several centimetres at the ranges this scene is made of.
  T_side = {}
  for side in sides:
    with open(args.mcap, 'rb') as f:
      for schema, _ch, message in make_reader(f).iter_messages(
          topics=[f'/camera/{side}/camera_info']):
        ci = store.deserialize_cdr(message.data, schema.name)
        break
    M = np.eye(4)
    M[:3, :3] = np.asarray(ci.r, np.float64).reshape(3, 3)
    T_side[side] = M @ np.linalg.inv(tf[(args.points_frame, f'{side}_optical')])
    P = np.asarray(ci.p, np.float64).reshape(3, 4)
    # Rectification is what makes a single intrinsic valid for both cameras; if that ever
    # stopped holding, every right-camera depth would be silently misplaced.
    assert np.allclose([P[0, 0], P[1, 1], P[0, 2], P[1, 2]], [fx, fy, cx, cy]), \
      f'{side} rectified intrinsics disagree with meta.json'
    print(f'{args.points_frame} -> rectified {side} camera:\n{np.round(T_side[side], 4)}')

  out_dir = seq / args.out
  if not args.validate:
    out_dir.mkdir(parents=True, exist_ok=True)

  # One LiDAR sweep per camera stamp, nearest in time. Stereo pairs are hardware synced,
  # so the two sides share a sweep rather than needing one each.
  scans: dict[float, np.ndarray] = {}
  with open(args.mcap, 'rb') as f:
    it = make_reader(f).iter_messages(topics=[args.points_topic],
                                      start_time=int(lo * 1e9), end_time=int(hi * 1e9))
    for schema, _ch, message in tqdm(it, desc='scans'):
      m = store.deserialize_cdr(message.data, schema.name)
      t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
      j = int(np.argmin(np.abs(times - t)))
      if abs(times[j] - t) > args.max_dt or times[j] in scans:
        continue
      pts = np.frombuffer(bytes(m.data), np.float32).reshape(-1, m.point_step // 4)[:, :3]
      scans[times[j]] = pts[np.isfinite(pts).all(1)].astype(np.float64)
  print(f'matched {len(scans)}/{len(times)} camera stamps to a LiDAR sweep')

  projected = {}
  for f in tqdm(frames, desc='projecting'):
    if f['timestamp'] not in scans:
      continue
    projected[f['name']] = project_scan(
      scans[f['timestamp']], T_side[f.get('side', 'left')], W, H, fx, fy, cx, cy,
      args.max_range, args.spread_tol)

  # Each reconstruction has its own arbitrary scale, so the metres-per-unit factor is fit
  # here against this sequence's own triangulated depths rather than read from a file that
  # may have been copied in from a different run. Matching depth directly also keeps the
  # factor tied to the quantity it is used for, instead of to trajectory length.
  sfm = read_sfm_depths(seq / 'sparse' / '0')
  ratios, bands = [], []
  for name, depth in projected.items():
    if name not in sfm:
      continue
    xy, zs = sfm[name]
    uv = np.round(xy).astype(int)
    k = ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    uv, zs = uv[k], zs[k]
    d = depth[uv[:, 1], uv[:, 0]]
    k = (d > 0.5) & (d < 25) & (zs > 0.05)
    ratios.append(d[k] / zs[k])
    bands.append(d[k])
  r, b = np.concatenate(ratios), np.concatenate(bands)
  m_per_unit = float(np.median(r))
  print(f'metres per SfM unit: {m_per_unit:.4f} from {len(r)} triangulated points')
  # A true scale is constant with range. Drift across bands would mean the LiDAR depth is
  # distorted rather than merely scaled, and no single factor would fix it.
  for blo, bhi in ((1, 3), (3, 6), (6, 12), (12, 25)):
    k = (b >= blo) & (b < bhi)
    if k.sum() > 100:
      print(f'    {blo:2d}-{bhi:2d} m: {np.median(r[k]):.4f}  (n={k.sum()})')

  rel, cov = [], []
  for name, depth in projected.items():
    cov.append((depth > 0).mean())
    if args.validate:
      stem = name[:-4]
      L = cv2.imread(str(seq / 'images' / f'{stem}.jpg'))
      R = cv2.imread(str(seq / 'images' / f'{stem.replace("_left", "_right")}.jpg'))
      ref, ok = sgbm_depth(L, R, fx, meta['baseline_m'])
      sel = ok & (depth > 0.5) & (depth < 15) & (ref > 0.5) & (ref < 15)
      if sel.sum() > 200:
        rel.append(np.abs(depth[sel] - ref[sel]) / ref[sel])
    else:
      inv = np.zeros((H, W), np.float32)
      nz = depth > 0
      # Inverse depth in SfM units: the renderer's depth is in SfM units too.
      inv[nz] = (m_per_unit / depth[nz]).astype(np.float32)
      np.save(out_dir / f'{name[:-4]}.npy', inv)

  print(f'wrote {len(projected)} maps for sides {sides}; '
        f'pixel coverage per frame: mean {np.mean(cov):.2%} max {np.max(cov):.2%}')
  if args.validate:
    r = np.concatenate(rel)
    print(f'LiDAR vs block-matched stereo over {len(r)} pixels:')
    print(f'  absolute relative difference: median {np.median(r):.2%}  '
          f'p90 {np.percentile(r, 90):.2%}')
  else:
    # Recorded so evaluation converts render depth to metres with the same factor the
    # priors were written with, instead of a scale copied in from another reconstruction.
    (seq / 'lidar_depth_scale.json').write_text(
      json.dumps({'m_per_unit': m_per_unit, 'num_points': len(r)}, indent=2))
    print(f'-> {out_dir}')


if __name__ == '__main__':
  main()
