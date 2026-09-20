"""Cut a monocular clip out of a ROS 2 ``.mcap`` into the layout Vid2Sim expects.

Vid2Sim's reconstruction stage wants a ``seq_path`` that looks like a hand-made video
folder: numbered frames in ``images/``, keep/ignore masks in ``masks/``, nothing else.
This script produces that from a robot log, for one camera topic over one time window.

Two things are done here rather than left to ``run_sfm.sh``:

* **Undistortion happens up front.** Upstream feeds distorted frames with
  ``--camera OPENCV`` and lets ``colmap image_undistorter`` rectify them at the end of
  SfM. That silently breaks the masks: the undistorter rewrites ``images/`` but knows
  nothing about ``masks/``, so training would pair rectified pixels with unrectified
  ignore regions. Rectifying both here with one shared map keeps them aligned, and
  makes the later ``image_undistorter`` pass a geometric no-op.
* **Frame timestamps are recorded** in ``meta.json``. SfM output is up to an arbitrary
  similarity transform; the timestamps are what lets us later align it to the metric
  LiDAR trajectory and recover real-world scale.

``inputs/`` is a duplicate of ``images/``. It is not redundant by choice --
``tools/run_sfm.py`` reads ``images/`` for feature extraction but ``inputs/`` for the
mapper and the undistorter, so both must exist with identical contents.

Example
-------
    micromamba run -n mcap python tools/mcap_to_vid2sim.py \
        --mcap /scratch/zichaohu/real2sim/mission_2025-01-17-10-37-51.mcap \
        --out  data/urban_walk_20m \
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


def _stamp_to_sec(stamp) -> float:
  return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _read_camera_info(mcap_path: str, store, info_topic: str) -> dict:
  """First CameraInfo on the topic. Intrinsics are fixed for the whole mission."""
  with open(mcap_path, 'rb') as f:
    for schema, _channel, message in make_reader(f).iter_messages(topics=[info_topic]):
      m = store.deserialize_cdr(message.data, schema.name)
      return dict(K=np.asarray(m.k, np.float64).reshape(3, 3),
                  D=np.asarray(m.d, np.float64).ravel(),
                  width=int(m.width), height=int(m.height),
                  model=str(m.distortion_model))
  raise SystemExit(f'no CameraInfo on {info_topic}')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--mcap', required=True)
  ap.add_argument('--out', required=True, help='Destination seq_path.')
  ap.add_argument('--topic', default='/camera/left/image_raw/compressed')
  ap.add_argument('--info-topic', default='/camera/left/camera_info')
  ap.add_argument('--start', type=float, required=True, help='Unix seconds.')
  ap.add_argument('--end', type=float, required=True, help='Unix seconds.')
  ap.add_argument('--every', type=int, default=1, help='Keep every Nth frame.')
  ap.add_argument('--ros2-store', default='ROS2_HUMBLE')
  ap.add_argument('--alpha', type=float, default=0.0,
                  help='getOptimalNewCameraMatrix alpha: 0 crops to valid pixels only, '
                       '1 keeps every source pixel and pads with black.')
  args = ap.parse_args()

  store = get_typestore(getattr(Stores, args.ros2_store))
  info = _read_camera_info(args.mcap, store, args.info_topic)
  w, h = info['width'], info['height']
  K, D = info['K'], info['D']
  print(f'camera {w}x{h} {info["model"]}  f=({K[0,0]:.1f},{K[1,1]:.1f}) '
        f'c=({K[0,2]:.1f},{K[1,2]:.1f})  D={np.round(D, 4)}')

  # alpha=0 keeps only pixels with valid source data, so no black border survives into
  # the splat as fake geometry.
  new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), args.alpha, (w, h))
  map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
  x, y, rw, rh = roi
  print(f'undistort -> K\'=({new_K[0,0]:.1f},{new_K[1,1]:.1f}) '
        f'c=({new_K[0,2]:.1f},{new_K[1,2]:.1f})  roi={roi}')

  out = pathlib.Path(args.out)
  dirs = {n: out / n for n in ('images', 'inputs', 'masks', 'colmap_masks')}
  for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

  t_ns = (int(args.start * 1e9), int(args.end * 1e9))
  frames, kept = [], 0
  with open(args.mcap, 'rb') as f:
    reader = make_reader(f)
    it = reader.iter_messages(topics=[args.topic],
                              start_time=t_ns[0], end_time=t_ns[1])
    for i, (schema, _channel, message) in enumerate(tqdm(it, desc='decoding')):
      if i % args.every:
        continue
      m = store.deserialize_cdr(message.data, schema.name)
      img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
      if img is None:
        continue
      img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)[y:y + rh, x:x + rw]

      kept += 1
      name = f'{kept:05d}'
      cv2.imwrite(str(dirs['images'] / f'{name}.jpg'), img,
                  [cv2.IMWRITE_JPEG_QUALITY, 98])
      cv2.imwrite(str(dirs['inputs'] / f'{name}.jpg'), img,
                  [cv2.IMWRITE_JPEG_QUALITY, 98])
      frames.append(dict(name=f'{name}.jpg', timestamp=_stamp_to_sec(m.header.stamp)))

  if not frames:
    raise SystemExit('no frames in window')

  # The ROI crop shifts the principal point; everything downstream reads these.
  K_final = new_K.copy()
  K_final[0, 2] -= x
  K_final[1, 2] -= y

  ts = np.array([f['timestamp'] for f in frames])
  meta = dict(
    mcap=str(args.mcap), topic=args.topic,
    start=args.start, end=args.end, every=args.every,
    num_frames=len(frames), width=int(rw), height=int(rh),
    # PINHOLE now that the frames are rectified; SfM must be told this, otherwise it
    # will try to re-estimate distortion that is no longer there.
    camera_model='PINHOLE',
    fx=float(K_final[0, 0]), fy=float(K_final[1, 1]),
    cx=float(K_final[0, 2]), cy=float(K_final[1, 2]),
    K_original=K.tolist(), D_original=D.tolist(), roi=[int(x), int(y), int(rw), int(rh)],
    duration=float(ts[-1] - ts[0]), fps=float((len(ts) - 1) / (ts[-1] - ts[0])),
    frames=frames,
  )
  (out / 'meta.json').write_text(json.dumps(meta, indent=2))
  print(f'wrote {len(frames)} frames {rw}x{rh} '
        f'({meta["duration"]:.1f} s @ {meta["fps"]:.1f} fps) -> {out}')


if __name__ == '__main__':
  main()
