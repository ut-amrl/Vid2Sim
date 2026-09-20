"""Offscreen renders of an exported TSDF mesh: shaded top-down plus two oblique views.

Vertex colours make a mesh look fine even where the geometry is wrong, so the shaded
pass deliberately discards them -- holes, floaters and melted kerbs only show up under
flat lighting.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import open3d as o3d
from PIL import Image


def render(mesh, size, eye, up, out, shader):
  r = o3d.visualization.rendering.OffscreenRenderer(*size)
  r.scene.set_background([1, 1, 1, 1])
  mat = o3d.visualization.rendering.MaterialRecord()
  # 'normals' colours each face by its orientation and ignores scene lighting, so the
  # result does not depend on getting a light rig right, and flat ground reads as one
  # solid colour while TSDF noise reads as speckle.
  mat.shader = shader
  r.scene.add_geometry('m', mesh, mat)
  c = mesh.get_axis_aligned_bounding_box().get_center()
  r.setup_camera(60.0, c, c + eye, up)
  img = np.asarray(r.render_to_image())
  Image.fromarray(img).save(out)
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--mesh', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  mesh = o3d.io.read_triangle_mesh(args.mesh)
  mesh.compute_vertex_normals()
  d = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))

  tiles = []
  for name, eye, up, shader in [
      ('top', np.array([0.0, -0.55 * d, 0.0]), np.array([0.0, 0.0, 1.0]), 'normals'),
      ('oblique', np.array([0.35 * d, -0.35 * d, 0.35 * d]), np.array([0.0, -1.0, 0.0]), 'normals'),
      ('colour', np.array([0.35 * d, -0.35 * d, 0.35 * d]), np.array([0.0, -1.0, 0.0]),
       'defaultUnlit'),
  ]:
    p = pathlib.Path(args.out).with_suffix('').as_posix() + f'_{name}.png'
    tiles.append(render(mesh, (900, 700), eye, up, p, shader))
    print(f'-> {p}')

  imgs = [Image.open(t) for t in tiles]
  grid = Image.new('RGB', (sum(i.width for i in imgs), imgs[0].height), 'white')
  x = 0
  for i in imgs:
    grid.paste(i, (x, 0))
    x += i.width
  grid.save(args.out)
  print(f'-> {args.out}')


if __name__ == '__main__':
  main()
