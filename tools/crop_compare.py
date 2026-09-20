"""1:1 crops of ground truth against renders, for judging sharpness rather than layout.

Downscaled montages flatter a render: resampling hides exactly the high-frequency loss
that makes a splat look synthetic. These crops are pasted at native resolution so that
what reaches the eye is what the rasterizer produced.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--rows', nargs='+', required=True, help='"label=dir" entries.')
  ap.add_argument('--frames', nargs='+', required=True)
  ap.add_argument('--box', nargs=4, type=int, default=[150, 180, 400, 380],
                  help='x0 y0 x1 y1 crop in source pixels.')
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  rows = [(r.split('=', 1)[0], pathlib.Path(r.split('=', 1)[1])) for r in args.rows]
  x0, y0, x1, y1 = args.box
  w, h = x1 - x0, y1 - y0
  pad, label_w = 4, 210

  grid = Image.new('RGB', (label_w + len(args.frames) * (w + pad),
                           len(rows) * (h + pad)), 'white')
  draw = ImageDraw.Draw(grid)
  for r, (name, d) in enumerate(rows):
    y = r * (h + pad)
    draw.text((6, y + h // 2), name, fill='black')
    for c, f in enumerate(args.frames):
      hits = sorted(d.glob(f'{f}*'))
      if not hits:
        continue
      im = Image.open(hits[0]).convert('RGB').crop((x0, y0, x1, y1))
      grid.paste(im, (label_w + c * (w + pad), y))

  grid.save(args.out)

  # Laplacian variance on the luminance of each crop: a single number for "how much
  # high-frequency detail survived", comparable only between images of the same scene.
  print(f'-> {args.out}')
  for name, d in rows:
    v = []
    for f in args.frames:
      hits = sorted(d.glob(f'{f}*'))
      if not hits:
        continue
      g = np.asarray(Image.open(hits[0]).convert('L').crop((x0, y0, x1, y1)), np.float64)
      lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
             + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
      v.append(lap.var())
    print(f'  {name:34s} detail (laplacian var) {np.mean(v):8.1f}')


if __name__ == '__main__':
  main()
