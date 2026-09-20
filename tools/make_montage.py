"""Stack the flythrough variants into one grid so off-path degradation is visible.

Rows are offsets, columns are frames. Reading down a column shows what breaks as the
camera leaves the recorded path; the ground truth row is the reference for the on-rails
row only, since no camera ever observed the offset views.
"""

from __future__ import annotations

import argparse
import pathlib

from PIL import Image, ImageDraw


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('-m', '--model')
  ap.add_argument('-s', '--source')
  ap.add_argument('--rows', nargs='+', default=[],
                  help='Explicit "label=dir" rows, for comparing across runs. Overrides '
                       'the -m/-s auto-discovery.')
  ap.add_argument('--frames', nargs='+', default=['00030', '00080', '00130', '00180'])
  ap.add_argument('--width', type=int, default=430)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  if args.rows:
    rows = [(r.split('=', 1)[0], pathlib.Path(r.split('=', 1)[1])) for r in args.rows]
  else:
    fly = pathlib.Path(args.model) / 'flythrough'
    rows = [('ground truth', pathlib.Path(args.source) / 'images')]
    rows += [(d.name, d) for d in sorted(fly.iterdir()) if d.is_dir()]

  def find(d, frame):
    """Frame 00030 is '00030.jpg' in a mono clip and '00030_left.png' in a stereo one."""
    hits = sorted(d.glob(f'{frame}*'))
    return hits[0] if hits else None

  probe = Image.open(find(rows[0][1], args.frames[0]))
  w = args.width
  h = round(probe.height * w / probe.width)
  pad, label_w = 4, 190

  grid = Image.new('RGB', (label_w + len(args.frames) * (w + pad),
                           len(rows) * (h + pad)), 'white')
  draw = ImageDraw.Draw(grid)
  for r, (name, d) in enumerate(rows):
    y = r * (h + pad)
    draw.text((6, y + h // 2), name, fill='black')
    for c, f in enumerate(args.frames):
      p = find(d, f)
      if p is None:
        continue
      grid.paste(Image.open(p).convert('RGB').resize((w, h)),
                 (label_w + c * (w + pad), y))

  out = args.out or str(fly / 'comparison.png')
  grid.save(out)
  print(f'-> {out}  {grid.size[0]}x{grid.size[1]}  rows={[r[0] for r in rows]}')


if __name__ == '__main__':
  main()
