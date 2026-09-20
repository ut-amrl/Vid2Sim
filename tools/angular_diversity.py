"""How many distinct directions did the capture actually see each surface point from?

Parallax and angular diversity are different things, and a walking capture supplies very
different amounts of each. Walking towards a wall gives plenty of parallax -- the wall
grows, and its depth is well determined -- while the direction you view it from barely
changes. Depth is therefore well constrained and *appearance as a function of direction*
is not, which is the ingredient a spherical-harmonic colour model needs in order to be
identifiable rather than free.

This measures the second quantity straight from the SfM tracks: for every triangulated
point, the angular spread of the rays that observed it. No rendering and no trained model
is involved, so it characterises the capture rather than any particular reconstruction.

    micromamba run -n mcap python tools/angular_diversity.py --seq data/<clip>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'vid2sim_recon'))

from tools.colmap_utils.loader import (qvec2rotmat, read_extrinsics_binary,  # noqa: E402
                                       read_points3d_binary)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seq', required=True, nargs='+', help='One or more seq_paths.')
  ap.add_argument('--min-track', type=int, default=3)
  args = ap.parse_args()

  for seq in args.seq:
    p = pathlib.Path(seq) / 'sparse' / '0'
    extr = read_extrinsics_binary(str(p / 'images.bin'))
    pts = read_points3d_binary(str(p / 'points3D.bin'))
    centre = {i: -qvec2rotmat(im.qvec).T @ im.tvec for i, im in extr.items()}

    spread, ntrack = [], []
    for pt in pts.values():
      ids = [i for i in pt.image_ids if i in centre]
      if len(ids) < args.min_track:
        continue
      d = np.array([centre[i] - pt.xyz for i in ids])
      d /= np.linalg.norm(d, axis=1, keepdims=True)
      # Widest angle between any two observing rays, via the extreme pair on the hull of
      # directions; with a narrow cone the pairwise max is cheap and exact enough.
      c = np.clip(d @ d.T, -1, 1)
      spread.append(np.degrees(np.arccos(c.min())))
      ntrack.append(len(ids))

    spread, ntrack = np.array(spread), np.array(ntrack)
    q = np.percentile(spread, [10, 50, 90])
    print(f'{pathlib.Path(seq).name}: {len(spread)} points, median track {np.median(ntrack):.0f} images')
    print(f'  observing-ray spread: p10 {q[0]:5.1f} deg  median {q[1]:5.1f} deg  p90 {q[2]:5.1f} deg')
    for thr in (5, 10, 30):
      print(f'    seen over a cone narrower than {thr:2d} deg: {(spread < thr).mean():6.1%}')


if __name__ == '__main__':
  main()
