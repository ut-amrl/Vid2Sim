"""Package a trained splat and its TSDF mesh as one metric, co-registered sim asset.

Both layers come out of training in SfM units, which are arbitrary, and the mesh covers
far more of the world than is worth colliding against: exported straight from
``export_mesh.py`` it reaches 121 m from a route only 27 m long, because the angle
threshold that has to be set to zero to retain the ground also stops rejecting distant
grazing surfaces. So this crops, cleans, scales, and reports what the caller is actually
getting.

The crop keeps geometry within a radius of the *nearest camera on the route* rather than
of the route's centroid, so the retained band follows the trajectory instead of being a
sphere around its middle -- on a 27 m walk those are very different shapes. The default
12.8 m radius gives a 25.6 m span, matching the SemanticKITTI BEV convention.

The two layers are treated differently on purpose. The mesh is for collision, so it wants
to be tight, clean, and ideally closed. The splat is for rendering, so distant background
is a feature, not clutter -- cropping it to the collision radius would leave a black void
past the crop. The splat is therefore left whole unless ``--splat-radius`` asks otherwise.

Scaling a splat is not just its positions. Gaussian extents are stored as logs -- the
model applies ``exp`` as the scaling activation -- so a factor ``s`` is ``+log(s)`` on
``scale_*``, not ``*s``. Getting that wrong leaves the centres correctly placed and every
Gaussian the wrong size, which looks like a mis-trained model rather than a unit bug.

Cropping necessarily opens the mesh along the cut, so watertightness is reported rather
than assumed, and the report separates the two reasons a mesh fails to be closed: holes,
which are honest boundary loops and fillable, and non-manifold edges, which are structural
damage that hole filling will not repair.

    micromamba run -n vid2sim-recon python tools/export_sim_asset.py \
        -m output/stereo_lidar_w0.5 --radius 12.8 --fill-holes 0.5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / 'src' / 'vid2sim_recon'))


def boundary_edges(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
  """Edges used by exactly one triangle -- the mesh's open rim."""
  tri = np.asarray(mesh.triangles)
  if not len(tri):
    return np.zeros((0, 2), int)
  e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
  uniq, cnt = np.unique(e, axis=0, return_counts=True)
  return uniq[cnt == 1]


def hole_sizes(mesh: o3d.geometry.TriangleMesh, be: np.ndarray) -> np.ndarray:
  """Edges per boundary loop, largest first.

  A bare hole count hides the distinction that matters: a hundred few-edge loops are
  reconstruction noise worth filling, whereas one loop with thousands of edges is the
  outer rim of an open scene, and capping that would dome the sky over the street.
  """
  if not len(be):
    return np.zeros(0, int)
  n = len(mesh.vertices)
  g = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n))
  _, lab = connected_components(g + g.T, directed=False)
  return np.sort(np.bincount(lab[be[:, 0]]))[::-1]


def report(mesh: o3d.geometry.TriangleMesh, note: str) -> dict:
  be = boundary_edges(mesh)
  nm = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True))
  sizes = hole_sizes(mesh, be)
  sizes = sizes[sizes > 0]
  info = {
    'vertices': len(mesh.vertices),
    'triangles': len(mesh.triangles),
    'watertight': bool(mesh.is_watertight()),
    'edge_manifold': bool(mesh.is_edge_manifold()),
    'vertex_manifold': bool(mesh.is_vertex_manifold()),
    'boundary_edges': int(len(be)),
    'holes': int(len(sizes)),
    'largest_hole_edges': int(sizes[0]) if len(sizes) else 0,
    'median_hole_edges': int(np.median(sizes)) if len(sizes) else 0,
    'non_manifold_edges': int(len(nm)),
  }
  print(f'  {note:22s} {info["vertices"]:7d} v {info["triangles"]:7d} f  '
        f'watertight={info["watertight"]!s:5s} holes={info["holes"]:<5d} '
        f'largest={info["largest_hole_edges"]:<6d} '
        f'median={info["median_hole_edges"]:<4d} '
        f'non_manifold={info["non_manifold_edges"]}')
  return info


def transform_splat(src: pathlib.Path, dst: pathlib.Path, s: float, T: np.ndarray,
                    centres: np.ndarray, radius: float) -> dict:
  """Write the splat scaled by ``s`` and then placed by the rigid metric transform ``T``."""
  ply = PlyData.read(str(src))
  v = ply['vertex']
  names = [p.name for p in v.properties]
  data = {n: np.asarray(v[n]).astype(np.float64) for n in names}
  n_before = len(data['x'])

  if radius > 0:
    xyz = np.stack([data['x'], data['y'], data['z']], 1)
    d = np.full(len(xyz), np.inf)
    for i in range(0, len(centres), 64):
      d = np.minimum(d, np.linalg.norm(
        xyz[:, None, :] - centres[None, i:i + 64, :], axis=2).min(1))
    keep = d <= radius
    data = {n: a[keep] for n, a in data.items()}

  R, t = T[:3, :3], T[:3, 3]
  xyz = np.stack([data['x'], data['y'], data['z']], 1) * s @ R.T + t
  data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

  # Extents are stored as logs because the model activates them with exp, so a uniform
  # scale is an offset here. Multiplying would leave centres right and sizes wrong.
  for n in names:
    if n.startswith('scale_'):
      data[n] += np.log(s)

  # Orientation lives in two places and both have to turn: the per-Gaussian quaternion,
  # and the SH that encodes view-dependent colour against world axes.
  q = Rotation.from_matrix(R)
  rot = np.stack([data[f'rot_{i}'] for i in range(4)], 1)          # stored w x y z
  rot = (q * Rotation.from_quat(rot[:, [1, 2, 3, 0]])).as_quat()   # scipy is x y z w
  for i, k in enumerate([3, 0, 1, 2]):
    data[f'rot_{i}'] = rot[:, k]

  n_rest = sum(n.startswith('f_rest_') for n in names) // 3
  deg = int(round(np.sqrt(n_rest + 1))) - 1
  M = sh_rotation(R, deg)
  for c in range(3):
    # f_rest is stored channel-major: channel c holds SH indices 1..n_rest.
    keys = [f'f_dc_{c}'] + [f'f_rest_{c * n_rest + k}' for k in range(n_rest)]
    coeff = np.stack([data[k] for k in keys], 1) @ M
    for i, k in enumerate(keys):
      data[k] = coeff[:, i]

  arr = np.empty(len(data['x']), dtype=[(n, 'f4') for n in names])
  for n in names:
    arr[n] = data[n]
  PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(dst))
  return {'gaussians_in': n_before, 'gaussians_out': int(len(arr)),
          'sh_degree_rotated': deg}


def ground_hits(mesh: o3d.geometry.TriangleMesh, origins: np.ndarray,
                up: np.ndarray, reach: float = 6.0):
  """Where a downward ray from each origin meets the mesh, and whether it did."""
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  rays = np.hstack([origins, np.tile(-up, (len(origins), 1))]).astype(np.float32)
  t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
  ok = np.isfinite(t) & (t < reach)
  return origins[ok] - up * t[ok, None], ok, t


def attitude_up(src: pathlib.Path, calib: pathlib.Path,
                poses_file: pathlib.Path) -> np.ndarray | None:
  """Gravity in SfM axes, from the LiDAR's per-frame orientation.

  Carries gravity down the calibration chain -- world to sensor by the pose quaternion,
  sensor to rectified camera by the extrinsic and rectification matrices, camera to SfM by
  the reconstructed pose -- and averages over every frame. Unlike a trajectory fit this
  reads a full 3D attitude per frame, so nothing about it is degenerate for a straight
  route. The os_sensor/os_lidar confusion that corrupted the depth projection is harmless
  here: those frames differ by a yaw about their shared z, which leaves gravity untouched.
  """
  import yaml
  from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

  cam0 = calib / 'calib_cam0_intrinsics.yaml'
  os2cam = calib / 'calib_os1_to_cam0.yaml'
  if not (cam0.exists() and os2cam.exists() and poses_file.exists()):
    return None
  R_rect = np.asarray(yaml.safe_load(cam0.read_text())
                      ['rectification_matrix']['data']).reshape(3, 3)
  E = np.asarray(yaml.safe_load(os2cam.read_text())
                 ['extrinsic_matrix']['data']).reshape(4, 4)[:3, :3]

  t_of = {f['name']: f['timestamp']
          for f in json.loads((src / 'meta.json').read_text())['frames']}
  poses = np.loadtxt(poses_file)
  ups = []
  for im in read_extrinsics_binary(str(src / 'sparse' / '0' / 'images.bin')).values():
    if '_right' in im.name or im.name not in t_of:
      continue
    j = int(np.clip(np.searchsorted(poses[:, 0], t_of[im.name]), 1, len(poses) - 1))
    w, x, y, z = poses[j, 4:8]
    L = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    ups.append(qvec2rotmat(im.qvec).T @ (R_rect @ E @ (L.T @ np.array([0, 0, 1.0]))))
  if len(ups) < 20:
    return None
  u = np.mean(np.asarray(ups) / np.linalg.norm(ups, axis=1, keepdims=True), axis=0)
  return u / np.linalg.norm(u)


def gravity_frame(src: pathlib.Path, mesh: o3d.geometry.TriangleMesh, route: np.ndarray,
                  calib: pathlib.Path, poses_file: pathlib.Path
                  ) -> tuple[np.ndarray, dict]:
  """Rigid 4x4 taking metric SfM coordinates to Z up, X forward, Y left.

  Three estimates of up are available and none is trustworthy on its own, so the frame
  takes each angle from whichever one actually constrains it.

  The trajectory fit in ``sfm_metric_alignment.json`` aligns SfM camera centres to LiDAR
  positions, which pins pitch tightly -- it reproduces the street's real 4.04 deg climb to
  within 0.1 deg -- but it is blind to roll. Rotation about the direction of travel is only
  constrained by how far the points spread sideways, and this route is 107:1 straight, so
  a 4 deg roll displaces the cameras by under 2 cm against an 8 cm alignment residual.

  The attitude estimate reads a full orientation per frame and so has no degenerate axis,
  making it the only real source of roll, but it inherits any pitch error in the
  sensor-to-camera extrinsic and lands 1.3 deg steep on the known grade.

  The ground plane is not gravity at all: the street genuinely climbs, so fitting the
  surface and calling it level would tilt the whole scene by the grade and put the robot on
  a flat road where there should be a hill. It is kept only as a cross-check, since the
  angle between it and gravity should equal the grade -- and it does.

  Forward is the route's net displacement flattened into the horizontal plane, and left
  follows from the other two, which is the usual robotics convention (REP 103). The origin
  sits on the ground beneath the first camera, so z=0 is the surface the robot starts on.
  """
  cands, n_inlier = {}, 0
  align = src / 'sfm_metric_alignment.json'
  if align.exists():
    # p_lidar = s R p_sfm + t, so LiDAR's +Z axis reads off as the third row of R.
    R = np.asarray(json.loads(align.read_text())['R'])
    cands['trajectory'] = R[2] / np.linalg.norm(R[2])
  att = attitude_up(src, calib, poses_file)
  if att is not None:
    cands['attitude'] = att

  # The ground the raycast already finds is a plane sample; its smallest-variance
  # direction is the surface normal. Sampling has to spread sideways as well as along the
  # route: a strip of hits directly beneath a near-straight path pins down the pitch but
  # leaves roll almost free, and the fit then reports a tilt it never actually measured.
  rough = np.array([0.0, -1.0, 0.0])
  tang = np.gradient(route, axis=0)
  tang /= np.linalg.norm(tang, axis=1, keepdims=True)
  side = np.cross(rough, tang)
  side /= np.linalg.norm(side, axis=1, keepdims=True)
  probe = np.concatenate([route + lat * side + rough
                          for lat in (-2.0, -1.0, 0.0, 1.0, 2.0)])
  hits, ok, _ = ground_hits(mesh, probe, rough)
  if len(hits) > 50:
    # Plain least squares would take the curbs, verges and wall bases that the wider
    # probes inevitably hit as if they were road, and a handful of those at 2 m lever the
    # normal over by degrees. Refit on inliers a few times so the road wins.
    keep = np.ones(len(hits), bool)
    for _ in range(5):
      pts = hits[keep]
      c = pts.mean(0)
      _, _, vt = np.linalg.svd(pts - c)
      n = vt[2] / np.linalg.norm(vt[2])
      resid = np.abs((hits - c) @ n)
      keep = resid < max(3.0 * np.median(resid), 0.05)
    ref = next(iter(cands.values()), -n)
    cands['ground_fit'] = n if n @ ref > 0 else -n
    n_inlier = int(keep.sum())

  if not cands:
    raise SystemExit('no gravity reference: need the alignment, calibration, or ground')

  up = cands.get('trajectory', cands.get('attitude', cands.get('ground_fit')))
  used = 'trajectory' if 'trajectory' in cands else next(iter(cands))
  if 'trajectory' in cands and 'attitude' in cands:
    # Keep the trajectory's pitch, take the attitude's roll: the axes each one can see.
    h = route[-1] - route[0]
    h -= up * (h @ up)
    lat = np.cross(up, h / np.linalg.norm(h))
    up = up + lat * ((cands['attitude'] - up) @ lat)
    up /= np.linalg.norm(up)
    used = 'trajectory pitch + attitude roll'

  fwd = route[-1] - route[0]
  fwd -= up * (fwd @ up)
  fwd /= np.linalg.norm(fwd)
  left = np.cross(up, fwd)

  T = np.eye(4)
  T[:3, :3] = np.stack([fwd, left, up])
  r = route @ T[:3, :3].T
  # Put z=0 on the ground at the route start. Taking the single ray cast straight down
  # from the first camera would be simpler but fails exactly where it is most likely to:
  # coverage is 96%, and the start of the route sits at the edge of the reconstruction.
  # Fitting the ground along the whole route and reading it off at the start survives
  # that, and follows the grade instead of assuming the surface is level.
  h = hits @ T[:3, :3].T if len(hits) > 50 else r + np.array([0, 0, -0.5])
  grade, z0 = np.polyfit(h[:, 0], h[:, 2], 1)
  T[:3, 3] = -np.array([r[0, 0], r[0, 1], grade * r[0, 0] + z0])

  info = {k: [round(float(x), 5) for x in v] for k, v in cands.items()}
  info['up'] = [round(float(x), 5) for x in up]
  info['used'] = used
  info['ground_inliers'] = n_inlier
  ang = lambda a, b: round(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))), 2)
  for k, v in cands.items():
    info[f'{k}_vs_used_deg'] = ang(v, up)
  # The check that actually validates the frame: in a plumb frame the route must climb
  # the street's real grade. Flat would mean the hill got rotated away.
  r = route @ T[:3, :3].T
  info['route_grade_deg'] = round(float(np.degrees(np.arctan2(
    r[-1, 2] - r[0, 2], r[-1, 0] - r[0, 0]))), 2)
  return T, info


def sh_rotation(R: np.ndarray, deg: int = 3, samples: int = 1024) -> np.ndarray:
  """Matrix taking spherical-harmonic coefficients from the old frame into the rotated one.

  Rotating a splat is not just its positions and quaternions. View-dependent colour is
  stored as SH in world axes, so turning the scene without turning the coefficients leaves
  every specular and shading cue pointing the old way -- worst here, where SH degree 3 was
  measured overfitting by 8 dB and is therefore carrying a lot of the appearance.

  The basis in ``sh_utils`` carries its own sign convention, so rather than hand-deriving
  Wigner-D matrices against it, this fits the band matrices numerically from the repo's own
  evaluator: sample directions, evaluate the basis at ``d`` and at ``R^T d``, and solve.
  SH bands are closed under rotation, so the fit is exact, and it cannot drift out of sync
  with the convention it is derived from.
  """
  from utils.sh_utils import eval_sh

  n = (deg + 1) ** 2
  d = np.random.default_rng(0).normal(size=(samples, 3))
  d /= np.linalg.norm(d, axis=1, keepdims=True)
  eye = np.tile(np.eye(n)[None], (samples, 1, 1))
  basis = eval_sh(deg, eye, d)                 # Y_j(d)
  rotated = eval_sh(deg, eye, d @ R)           # Y_j(R^T d)
  # Solving B c' = B_R c gives c' = M c; returned transposed so callers can keep
  # coefficients as rows and write ``coeffs @ M``.
  M, *_ = np.linalg.lstsq(basis, rotated, rcond=None)
  return M.T


def resample_route(route: np.ndarray, step: float) -> np.ndarray:
  d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
  s = np.arange(0.0, d[-1], step)
  return np.stack([np.interp(s, d, route[:, k]) for k in range(3)], 1)


def ground_ribbon(mesh: o3d.geometry.TriangleMesh, route: np.ndarray, half_width: float,
                  step: float = 0.25, lateral_step: float = 0.25, window: float = 2.5,
                  min_hits: int = 20) -> tuple[o3d.geometry.TriangleMesh, dict]:
  """A support surface that follows the route's own grade, fitted station by station.

  The reconstruction only has ground for about a metre either side of the path, so a
  physics scene needs something underneath that is defined everywhere the robot might go.
  Vid2Sim drops a horizontal plane in the Unity editor, which works because its scenes are
  flat; here the street climbs 4 degrees, so a flat plane would surface through the road
  partway along and bury it at the far end.

  One tilted plane is not the fix either. It only works while the whole clip shares a
  single grade, and it fails on exactly the case worth planning for -- a robot that runs
  down into a cul-de-sac and comes back up, where the route revisits ground at two
  different heights and no global plane can pass through both. So the surface is fitted
  locally: stations every ``step`` along the arc length, each taking its height and cross
  slope from the ground within ``window`` metres *measured along the route*, not in space.
  Arc length is what makes the doubling-back case work, since the outbound and return legs
  are far apart along the route even when they sit on top of each other in the world.

  Stations with too little ground beneath them inherit an interpolation from their
  neighbours rather than inventing a height, and the fit is trimmed so that curbs, verges
  and car roofs caught by the wider probes do not tilt a section.
  """
  up = np.array([0.0, 0.0, 1.0])
  c = resample_route(route, step)
  tang = np.gradient(c, axis=0)
  tang[:, 2] = 0.0
  tang /= np.linalg.norm(tang, axis=1, keepdims=True)
  lat = np.cross(up, tang)
  offs = np.arange(-half_width, half_width + 1e-9, lateral_step)
  n, m = len(c), len(offs)

  probe = (c[:, None, :] + offs[None, :, None] * lat[:, None, :] + up).reshape(-1, 3)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  rays = np.hstack([probe, np.tile(-up, (len(probe), 1))]).astype(np.float32)
  t_hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
  ok = (np.isfinite(t_hit) & (t_hit < 6.0)).reshape(n, m)
  z = (probe[:, 2] - t_hit).reshape(n, m)

  w = max(int(round(window / step)), 1)
  z0 = np.full(n, np.nan)
  slope = np.zeros(n)
  for i in range(n):
    lo, hi = max(i - w, 0), min(i + w + 1, n)
    sel = ok[lo:hi]
    if sel.sum() < min_hits:
      continue
    a = np.repeat((np.arange(lo, hi) - i) * step, m).reshape(-1, m)[sel]
    b = np.tile(offs, (hi - lo, 1))[sel]
    zz = z[lo:hi][sel]
    keep = np.ones(len(zz), bool)
    for _ in range(4):
      A = np.stack([np.ones(keep.sum()), a[keep], b[keep]], 1)
      coef, *_ = np.linalg.lstsq(A, zz[keep], rcond=None)
      resid = np.abs(zz - (coef[0] + coef[1] * a + coef[2] * b))
      keep = resid < max(3.0 * np.median(resid), 0.05)
      if keep.sum() < min_hits:
        break
    z0[i], slope[i] = coef[0], coef[2]

  valid = np.isfinite(z0)
  if not valid.any():
    raise SystemExit('no ground anywhere under the route; cannot fit a support surface')
  idx = np.arange(n)
  z0 = np.interp(idx, idx[valid], z0[valid])
  slope = np.interp(idx, idx[valid], slope[valid])
  # Station fits are independent, so smooth out the metre-scale wobble they leave behind.
  k = np.ones(max(int(round(1.0 / step)), 1)) / max(int(round(1.0 / step)), 1)
  pad = len(k) // 2
  sm = lambda v: np.convolve(np.pad(v, pad, mode='edge'), k, 'same')[pad:pad + n]
  z0, slope = sm(z0), sm(slope)

  V = c[:, None, :] + offs[None, :, None] * lat[:, None, :]
  V[:, :, 2] = z0[:, None] + slope[:, None] * offs[None, :]
  tri = []
  for i in range(n - 1):
    for j in range(m - 1):
      p = i * m + j
      tri += [[p, p + m, p + 1], [p + 1, p + m, p + m + 1]]
  surf = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(V.reshape(-1, 3)),
    o3d.utility.Vector3iVector(np.asarray(tri)))
  surf.compute_vertex_normals()

  grade = np.degrees(np.arctan(np.gradient(z0, step)))
  info = {'stations': n, 'station_step_m': step, 'half_width_m': half_width,
          'fitted_directly': int(valid.sum()), 'interpolated': int(n - valid.sum()),
          'grade_deg': [round(float(grade.min()), 2), round(float(grade.max()), 2)],
          'cross_slope_p95_deg': round(float(np.degrees(np.arctan(
            np.percentile(np.abs(slope), 95)))), 2)}
  fit = {'c': c, 'lat': lat, 'z0': z0, 'slope': slope, 'ok': ok, 'offs': offs}
  return surf, info, fit


def drivable_area(fit: dict, half_width: float) -> dict:
  """Centreline plus the band around it the reconstruction can actually back up.

  The bound is where a policy should be stopped, and the honest place for it is where the
  ground stops being measured rather than a round number. Per station this records how much
  of the band had real geometry under it, so a rollout can terminate on leaving the band
  and treat the thinly supported stations differently.
  """
  c, lat, z0, slope = fit['c'], fit['lat'], fit['z0'], fit['slope']
  band = np.abs(fit['offs']) <= half_width
  cover = fit['ok'][:, band].mean(1)
  edge = np.stack([c + half_width * lat, c - half_width * lat], 1)
  edge[:, 0, 2] = z0 + slope * half_width
  edge[:, 1, 2] = z0 - slope * half_width
  return {
    'half_width_m': half_width,
    'frame': 'Z up, X forward, Y left; metres',
    'centreline': np.stack([c[:, 0], c[:, 1], z0], 1).round(4).tolist(),
    'left_edge': edge[:, 0].round(4).tolist(),
    'right_edge': edge[:, 1].round(4).tolist(),
    'support': cover.round(3).tolist(),
    'support_median': round(float(np.median(cover)), 3),
    'stations_fully_supported': int((cover > 0.99).sum()),
    'stations_below_half': int((cover < 0.5).sum()),
  }


def ground_coverage(mesh: o3d.geometry.TriangleMesh, route: np.ndarray, up: np.ndarray,
                    offsets=(0.0, 1.0, 2.0, 4.0, 8.0)) -> dict:
  """Fraction of the route that has ground under it, at increasing lateral offsets.

  The crop radius is an upper bound on the asset, not a statement about what is in it.
  A forward-facing walk only ever observes a narrow corridor, so the TSDF has nothing to
  fuse a few metres to the side and the mesh simply stops -- which in a physics sim means
  the robot falls through the world rather than colliding with it. This is the number that
  decides how far a policy can be allowed to wander, so it is worth measuring rather than
  inferring from the bounding box.
  """
  tang = np.gradient(route, axis=0)
  tang /= np.linalg.norm(tang, axis=1, keepdims=True)
  left = np.cross(up, tang)
  left /= np.linalg.norm(left, axis=1, keepdims=True)

  out = {}
  for lat in offsets:
    hits = []
    for sgn in ((1, -1) if lat else (1,)):
      _, ok, t = ground_hits(mesh, route + sgn * lat * left + up, up)
      hits.append((ok, t))
    ok = np.concatenate([h[0] for h in hits])
    t = np.concatenate([h[1] for h in hits])
    out[f'{lat:g}m'] = round(float(ok.mean()), 3)
    drop = f'{np.median(t[ok]) - 1.0:+.2f} m vs camera' if ok.any() else '-'
    print(f'  {lat:4.1f} m off route: ground under {ok.mean():6.1%} of it   {drop}')
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('-m', '--model_path', required=True)
  ap.add_argument('-s', '--source_path', default='',
                  help='Defaults to the source_path recorded in the model cfg_args.')
  ap.add_argument('--mesh', default='mesh/tsdf_fusion_post.ply',
                  help='Relative to the model directory.')
  ap.add_argument('--iteration', type=int, default=-1)
  ap.add_argument('--radius', type=float, default=12.8,
                  help='Metres from the nearest camera to keep. 12.8 gives a 25.6 m '
                       'span, the SemanticKITTI BEV convention.')
  ap.add_argument('--splat-radius', type=float, default=0.0,
                  help='Metres, 0 keeps the whole splat. Distant background is wanted '
                       'for rendering even though it is useless for collision.')
  ap.add_argument('--min-cluster', type=float, default=0.002,
                  help='Drop connected components smaller than this fraction of the '
                       'largest, which is mostly debris the crop cut loose.')
  ap.add_argument('--fill-holes', type=float, default=0.0,
                  help='Metres. Close boundary loops up to this size; 0 disables.')
  ap.add_argument('--keep-fringe', action='store_true',
                  help='Keep the sliver triangles left hanging along the cut.')
  ap.add_argument('--keep-sfm-frame', action='store_true',
                  help='Skip gravity alignment and leave the asset in SfM axes.')
  ap.add_argument('--drivable-halfwidth', type=float, default=1.5,
                  help='Metres either side of the route a policy may use. The ground '
                       'coverage measurement supports 1-2 m.')
  ap.add_argument('--ground-halfwidth', type=float, default=4.0,
                  help='Metres either side for the support surface, wider than the '
                       'drivable bound so the robot never runs off its edge.')
  ap.add_argument('--obstacle-angle', type=float, default=15.0,
                  help='Degrees from vertical. Surfaces flatter than this are ground and '
                       'are dropped from the obstacle layer, as upstream Vid2Sim does.')
  ap.add_argument('--calib', type=pathlib.Path,
                  default=pathlib.Path('/robodata/arthurz/Datasets/lsmap_bags_processed'
                                       '/calibrations/19'))
  ap.add_argument('--poses', type=pathlib.Path,
                  default=pathlib.Path('/robodata/arthurz/Datasets/lsmap_bags_processed'
                                       '/poses/legoloam/19.txt'))
  ap.add_argument('--scale-from', choices=['auto', 'lidar', 'baseline'], default='auto',
                  help='auto prefers the stereo baseline when the scene has one.')
  ap.add_argument('--out', default='sim_asset')
  args = ap.parse_args()

  model = pathlib.Path(args.model_path)
  cfg = (model / 'cfg_args').read_text()
  src = pathlib.Path(args.source_path or cfg.split("source_path='")[1].split("'")[0])

  it = args.iteration
  if it < 0:
    it = max(int(p.name.split('_')[1])
             for p in (model / 'point_cloud').glob('iteration_*'))
  splat_in = model / 'point_cloud' / f'iteration_{it}' / 'point_cloud.ply'
  cams = json.loads((model / 'cameras.json').read_text())
  centres = np.array([c['position'] for c in cams])
  # A stereo scene stores both eyes as ordinary cameras, so walking every entry in order
  # zig-zags across the baseline and roughly quadruples the apparent route length.
  route = sorted([c for c in cams if 'right' not in c.get('img_name', '')],
                 key=lambda c: c.get('img_name', ''))
  route_xyz = np.array([c['position'] for c in route])

  # Two independent handles on scale. The LiDAR one is a median over projected returns,
  # so it carries whatever bias the projection has. The stereo baseline is a calibrated
  # constant measured directly between two reconstructed cameras, with no depth estimate
  # in the chain, so where both exist the baseline is the tighter reference.
  candidates = {}
  fitted = src / 'lidar_depth_scale.json'
  if fitted.exists():
    candidates['lidar'] = json.loads(fitted.read_text())['m_per_unit']
  elif (src / 'sfm_metric_alignment.json').exists():
    candidates['lidar'] = json.loads(
      (src / 'sfm_metric_alignment.json').read_text())['scale']

  pos = {c['img_name']: np.array(c['position']) for c in cams}
  pairs = np.array([np.linalg.norm(pos[k] - pos[k.replace('_left', '_right')])
                    for k in pos
                    if '_left' in k and k.replace('_left', '_right') in pos])
  meta = src / 'meta.json'
  if len(pairs) and meta.exists():
    b_m = json.loads(meta.read_text()).get('baseline_m')
    if b_m:
      candidates['baseline'] = b_m / float(np.median(pairs))

  pick = args.scale_from
  if pick == 'auto':
    pick = 'baseline' if 'baseline' in candidates else 'lidar'
  if pick not in candidates:
    raise SystemExit(f'no {pick} scale available; have {sorted(candidates)}')
  m_per_unit = candidates[pick]

  out = model / args.out
  out.mkdir(parents=True, exist_ok=True)
  for k, v in candidates.items():
    mark = '<-' if k == pick else '  '
    print(f'{mark} {v:.4f} m per SfM unit from {k}')
  if len(candidates) > 1:
    a, b = candidates['lidar'], candidates['baseline']
    print(f'   the two disagree by {abs(a - b) / b:.1%}')
  print(f'iteration {it}, {len(centres)} cameras, crop radius {args.radius} m\n')

  mesh = o3d.io.read_triangle_mesh(str(model / args.mesh))
  print('mesh:')
  stages = {'loaded': report(mesh, 'as exported')}

  # Crop in SfM units against the nearest camera on the route.
  v = np.asarray(mesh.vertices)
  d = np.full(len(v), np.inf)
  for i in range(0, len(centres), 64):
    d = np.minimum(d, np.linalg.norm(
      v[:, None, :] - centres[None, i:i + 64, :], axis=2).min(1))
  mesh.remove_vertices_by_mask(d * m_per_unit > args.radius)
  mesh.remove_unreferenced_vertices()
  stages['cropped'] = report(mesh, f'cropped to {args.radius} m')

  mesh.remove_degenerate_triangles()
  mesh.remove_duplicated_triangles()
  mesh.remove_duplicated_vertices()
  mesh.remove_unreferenced_vertices()

  if not args.keep_fringe:
    # A triangle with two open edges is attached to the body by a single edge: a spur
    # the cut left behind. Removing them can expose more, so iterate to a fixed point.
    for _ in range(8):
      be = boundary_edges(mesh)
      if not len(be):
        break
      rim = set(map(tuple, be))
      tri = np.asarray(mesh.triangles)
      open_edges = np.zeros(len(tri), int)
      for a, b in ((0, 1), (1, 2), (2, 0)):
        e = np.sort(tri[:, [a, b]], axis=1)
        open_edges += [(int(u), int(v)) in rim for u, v in e]
      spur = open_edges >= 2
      if not spur.any():
        break
      mesh.remove_triangles_by_mask(spur)
      mesh.remove_unreferenced_vertices()

  tri_lab, counts, _ = mesh.cluster_connected_triangles()
  counts = np.asarray(counts)
  if len(counts):
    small = counts < max(args.min_cluster * counts.max(), 1)
    mesh.remove_triangles_by_mask(small[np.asarray(tri_lab)])
    mesh.remove_unreferenced_vertices()
  stages['cleaned'] = report(mesh, 'cleaned')

  # Metres, from here on.
  mesh.scale(m_per_unit, center=np.zeros(3))
  route_m = route_xyz * m_per_unit

  T = np.eye(4)
  up = np.array([0.0, -1.0, 0.0])
  grav: dict = {}
  if not args.keep_sfm_frame:
    T, grav = gravity_frame(src, mesh, route_m, args.calib, args.poses)
    mesh.transform(T)
    route_m = route_m @ T[:3, :3].T + T[:3, 3]
    up = np.array([0.0, 0.0, 1.0])
    print('\ngravity alignment -> Z up, X forward, Y left, origin on the ground at '
          'the route start')
    for k, v in grav.items():
      print(f'  {k}: {v}')

  if args.fill_holes > 0:
    t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh = t.fill_holes(hole_size=args.fill_holes).to_legacy()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    stages['filled'] = report(mesh, f'holes <= {args.fill_holes} m filled')

  mesh.compute_vertex_normals()
  mesh_out = out / f'collision_{args.radius:g}m.ply'
  o3d.io.write_triangle_mesh(str(mesh_out), mesh)

  print('\nsplat:')
  splat_out = out / 'splat_metric.ply'
  sp = transform_splat(splat_in, splat_out, m_per_unit, T, centres,
                       args.splat_radius / m_per_unit if args.splat_radius > 0 else 0.0)
  print(f'  {sp["gaussians_in"]} -> {sp["gaussians_out"]} gaussians; positions, '
        f'quaternions and SH deg {sp["sh_degree_rotated"]} all carried into the new frame')

  print('\nground coverage (can a robot stand here?):')
  cover = ground_coverage(mesh, route_m, up)

  ground_out = drive_out = obst_out = None
  ground_info: dict = {}
  drive: dict = {}
  if not args.keep_sfm_frame:
    print('\nsupport surface, fitted locally along the route:')
    surf, ground_info, fit = ground_ribbon(mesh, route_m, args.ground_halfwidth)
    ground_out = out / 'ground.ply'
    o3d.io.write_triangle_mesh(str(ground_out), surf)
    print(f'  {ground_info["stations"]} stations every '
          f'{ground_info["station_step_m"]} m, {ground_info["fitted_directly"]} fitted '
          f'from ground and {ground_info["interpolated"]} interpolated')
    print(f'  grade along the route runs {ground_info["grade_deg"][0]:+.2f} to '
          f'{ground_info["grade_deg"][1]:+.2f} deg, cross slope p95 '
          f'{ground_info["cross_slope_p95_deg"]:.2f} deg')

    drive = drivable_area(fit, args.drivable_halfwidth)
    drive_out = out / 'drivable_area.json'
    drive_out.write_text(json.dumps(drive, indent=2))
    print(f'  drivable band +/-{args.drivable_halfwidth} m: real ground under '
          f'{drive["support_median"]:.0%} of it at the median station, '
          f'{drive["stations_below_half"]} of {ground_info["stations"]} below half')

    # Obstacle-only layer: the same recipe upstream uses, now that something else holds
    # the robot up. Keeping both layers is harmless but the reconstructed road is the part
    # that stops existing off-path, so a scene built on the primitive should not use it.
    obst = o3d.geometry.TriangleMesh(mesh)
    obst.compute_triangle_normals()
    flat = np.abs(np.asarray(obst.triangle_normals) @ up)
    obst.remove_triangles_by_mask(flat > np.cos(np.radians(args.obstacle_angle)))
    obst.remove_unreferenced_vertices()
    obst_out = out / f'obstacles_{args.radius:g}m.ply'
    o3d.io.write_triangle_mesh(str(obst_out), obst)
    print(f'  obstacle layer: {len(mesh.triangles)} -> {len(obst.triangles)} triangles '
          f'with surfaces within {args.obstacle_angle:g} deg of level removed')

  bb = mesh.get_axis_aligned_bounding_box()
  path_m = float(np.linalg.norm(np.diff(route_m, axis=0), axis=1).sum())
  manifest = {
    'model_path': str(model), 'source_path': str(src), 'iteration': it,
    'm_per_unit': m_per_unit, 'scale_source': pick,
    'scale_candidates': candidates, 'units': 'metres',
    'frame': ('SfM axes, scaled only' if args.keep_sfm_frame else
              'Z up, X forward, Y left; origin on the ground at the route start'),
    'T_metric_sfm_to_world': T.tolist(), 'gravity': grav,
    'crop_radius_m': args.radius, 'splat_radius_m': args.splat_radius,
    'route_length_m': path_m,
    'mesh': {'file': mesh_out.name,
             'extent_m': [round(float(x), 2) for x in bb.get_extent()],
             'min_m': [round(float(x), 2) for x in bb.get_min_bound()],
             'max_m': [round(float(x), 2) for x in bb.get_max_bound()],
             'stages': stages, 'ground_coverage': cover},
    'splat': {'file': splat_out.name, **sp},
  }
  if ground_out is not None:
    manifest['ground'] = {'file': ground_out.name, **ground_info}
    manifest['obstacles'] = {'file': obst_out.name,
                             'triangles': len(o3d.io.read_triangle_mesh(
                               str(obst_out)).triangles),
                             'angle_from_level_deg': args.obstacle_angle}
    manifest['drivable_area'] = {
      'file': drive_out.name,
      **{k: v for k, v in drive.items()
         if k not in ('centreline', 'left_edge', 'right_edge', 'support')}}
  (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

  final = stages.get('filled', stages['cleaned'])
  print(f'\nroute {path_m:.1f} m, mesh extent {manifest["mesh"]["extent_m"]} m')
  if final['watertight']:
    print('mesh is watertight')
  else:
    print(f'mesh is NOT watertight: {final["holes"]} holes over '
          f'{final["boundary_edges"]} boundary edges, largest loop '
          f'{final["largest_hole_edges"]} edges')
    print('  an outdoor capture is a surface, not a solid: the sky, the ground beyond '
          'the\n  crop, and everything behind the facades are open by construction, so '
          'the\n  largest loops are correct and should not be capped')
    if final['non_manifold_edges']:
      print(f'  {final["non_manifold_edges"]} non-manifold edges remain; these are '
            'structural, not holes')

  usable = [float(k[:-1]) for k, v in cover.items() if v >= 0.9]
  print(f'\nthe reconstructed ground is only real within {max(usable):g} m of the route. '
        f'Past that\nthe support surface carries the robot, so keep policies inside the '
        f'{args.drivable_halfwidth} m band.')
  print(f'\n-> {out}')


if __name__ == '__main__':
  main()
