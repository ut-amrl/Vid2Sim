"""Cut a *stereo* clip out of a ROS 2 ``.mcap`` into the layout Vid2Sim expects.

The monocular sibling of this script (``mcap_to_vid2sim.py``) produces a reconstruction
that is sharp along the walked path and falls apart as soon as the virtual camera steps
sideways. That is not a tuning problem: a camera carried straight down a sidewalk never
observes lateral parallax, so nothing in the data constrains what the scene looks like
from half a metre to the left. A stereo pair does observe it, at every single timestep.

So both cameras are emitted as ordinary frames into one flat ``images/`` directory, and
SfM is left to treat them as 2N independent views. Nothing tells it the two streams come
from a rigid rig. That is deliberate: the mapper here is GLOMAP, which does not consume
COLMAP's rig constraints, and an unconstrained solve gives us something better than a
speed-up -- the recovered left/right separation becomes an *independent* measurement we
can check against the calibrated baseline. ``check_stereo_sfm.py`` does exactly that.

Rectification differs from the monocular case in a way that matters:

* **The calibrated ``P`` is used verbatim as the new camera matrix**, rather than
  ``getOptimalNewCameraMatrix``. ``P`` is what makes the pair *rectified* -- identical
  focal lengths, identical principal points, rows aligned, and a pure horizontal
  baseline. Re-deriving per-camera intrinsics would destroy all of that, and with it the
  ability to pass ``--ImageReader.single_camera 1``.
* **Nothing is cropped**, for the same reason: a per-camera ROI crop shifts the two
  principal points differently. The black wedges that rectification leaves at the border
  are handled by masking them instead, which costs a few percent of the frame and keeps
  the two cameras geometrically identical.

Example
-------
    micromamba run -n mcap python tools/mcap_to_vid2sim_stereo.py \
        --mcap /scratch/zichaohu/real2sim/mission_2025-01-17-10-37-51.mcap \
        --out  data/urban_walk_20m_stereo \
        --start 1737132294.2 --end 1737132314.9
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

SIDES = ('left', 'right')


def _stamp_ns(stamp) -> int:
  return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _read_camera_infos(mcap_path: str, store, topics: dict[str, str]) -> dict[str, dict]:
  """First CameraInfo per side. Intrinsics are fixed for the whole mission."""
  want, out = set(topics.values()), {}
  with open(mcap_path, 'rb') as f:
    for schema, channel, message in make_reader(f).iter_messages(topics=list(want)):
      if channel.topic in {topics[s] for s in out}:
        continue
      m = store.deserialize_cdr(message.data, schema.name)
      side = next(s for s, t in topics.items() if t == channel.topic)
      out[side] = dict(K=np.asarray(m.k, np.float64).reshape(3, 3),
                       D=np.asarray(m.d, np.float64).ravel(),
                       R=np.asarray(m.r, np.float64).reshape(3, 3),
                       P=np.asarray(m.p, np.float64).reshape(3, 4),
                       width=int(m.width), height=int(m.height))
      if len(out) == len(want):
        return out
  raise SystemExit(f'missing CameraInfo, found only {sorted(out)}')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--mcap', required=True)
  ap.add_argument('--out', required=True, help='Destination seq_path.')
  ap.add_argument('--start', type=float, required=True, help='Unix seconds.')
  ap.add_argument('--end', type=float, required=True, help='Unix seconds.')
  ap.add_argument('--every', type=int, default=1, help='Keep every Nth stereo pair.')
  ap.add_argument('--ros2-store', default='ROS2_HUMBLE')
  ap.add_argument('--erode', type=int, default=5,
                  help='Shrink the valid region by this many px. Bilinear remap blends '
                       'the black border into neighbouring pixels, so the nominal '
                       'boundary is already slightly contaminated.')
  args = ap.parse_args()

  store = get_typestore(getattr(Stores, args.ros2_store))
  img_topics = {s: f'/camera/{s}/image_raw/compressed' for s in SIDES}
  info = _read_camera_infos(args.mcap, store, {s: f'/camera/{s}/camera_info' for s in SIDES})

  w, h = info['left']['width'], info['left']['height']
  P = info['left']['P']
  fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]
  # P_right = [fx 0 cx -fx*B; ...] by definition of a rectified pair.
  baseline = -info['right']['P'][0, 3] / fx
  if not np.allclose(info['left']['P'][:3, :3], info['right']['P'][:3, :3]):
    raise SystemExit('left/right rectified intrinsics differ; not a rectified pair')
  print(f'rectified {w}x{h}  f=({fx:.2f},{fy:.2f}) c=({cx:.2f},{cy:.2f})  '
        f'baseline={baseline * 100:.2f} cm')

  maps, valid = {}, {}
  for s in SIDES:
    m1, m2 = cv2.initUndistortRectifyMap(info[s]['K'], info[s]['D'], info[s]['R'],
                                         P[:3, :3], (w, h), cv2.CV_16SC2)
    maps[s] = (m1, m2)
    v = cv2.remap(np.full((h, w), 255, np.uint8), m1, m2, cv2.INTER_LINEAR, borderValue=0)
    v = (v == 255).astype(np.uint8)
    if args.erode > 0:
      v = cv2.erode(v, np.ones((args.erode, args.erode), np.uint8), 1)
    valid[s] = v * 255
    print(f'  {s:5s} valid pixels after rectification: {v.mean():.2%}')

  out = pathlib.Path(args.out)
  dirs = {n: out / n for n in ('images', 'inputs', 'masks', 'colmap_masks')}
  for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

  # Left and right are hardware-triggered and carry bit-identical header stamps, so the
  # stamp itself is the pairing key -- no nearest-neighbour matching needed.
  pending: dict[int, dict[str, np.ndarray]] = {}
  frames, kept, seen = [], 0, 0
  t_range = (int(args.start * 1e9), int(args.end * 1e9))
  with open(args.mcap, 'rb') as f:
    it = make_reader(f).iter_messages(topics=list(img_topics.values()),
                                      start_time=t_range[0], end_time=t_range[1])
    for schema, channel, message in tqdm(it, desc='decoding'):
      m = store.deserialize_cdr(message.data, schema.name)
      side = 'left' if '/left/' in channel.topic else 'right'
      key = _stamp_ns(m.header.stamp)
      pending.setdefault(key, {})[side] = m.data
      if len(pending[key]) < 2:
        continue

      raw = pending.pop(key)
      if seen % args.every:
        seen += 1
        continue
      seen += 1
      kept += 1
      for s in SIDES:
        img = cv2.imdecode(np.frombuffer(bytes(raw[s]), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.remap(img, *maps[s], cv2.INTER_LINEAR)
        name = f'{kept:05d}_{s}'
        for d in ('images', 'inputs'):
          cv2.imwrite(str(dirs[d] / f'{name}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(dirs['masks'] / f'{name}.jpg'), valid[s])
        cv2.imwrite(str(dirs['colmap_masks'] / f'{name}.jpg.png'), valid[s])
        frames.append(dict(name=f'{name}.jpg', side=s, timestamp=key * 1e-9))

  if not frames:
    raise SystemExit('no complete stereo pairs in window')

  ts = np.array([f['timestamp'] for f in frames if f['side'] == 'left'])
  meta = dict(
    mcap=str(args.mcap), topics=img_topics, stereo=True,
    start=args.start, end=args.end, every=args.every,
    num_pairs=kept, num_frames=len(frames), width=w, height=h,
    camera_model='PINHOLE',
    fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
    baseline_m=float(baseline),
    duration=float(ts[-1] - ts[0]), fps=float((len(ts) - 1) / (ts[-1] - ts[0])),
    frames=frames,
  )
  (out / 'meta.json').write_text(json.dumps(meta, indent=2))
  print(f'wrote {kept} pairs ({len(frames)} images) {w}x{h} '
        f'({meta["duration"]:.1f} s @ {meta["fps"]:.1f} Hz) -> {out}')


if __name__ == '__main__':
  main()
