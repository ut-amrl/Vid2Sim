"""Bind the exported layers into one USDZ for Isaac Lab.

Everything the export produces is already metric and in a common frame, so packaging is
mostly about saying which layer plays which role, in a way a physics engine understands.
Three roles:

- **appearance**, the Gaussian splat, visible and non-colliding;
- **collision**, the support surface and the obstacle mesh, colliding and *invisible* --
  the same split upstream Vid2Sim uses, where the mesh is physics-only and the splat
  supplies every pixel;
- **navigation**, the centreline and the drivable band, marked as guide geometry so they
  neither render nor collide but travel with the asset for episode setup.

The splat is written as a native ``UsdGeomPoints`` carrying the full Gaussian state in
primvars rather than as a packaged ``.ply`` alongside. A sidecar would be smaller to write
but would leave the asset only half described by USD, and the point positions and DC colour
double as a preview that opens anywhere. Nothing is lost: extents, quaternions, opacity and
all spherical-harmonic bands travel as primvars, so the splat can be reconstructed exactly.

USD's stage metadata is set rather than assumed -- Z up and one unit per metre -- because
the whole reason the export computes a gravity frame is lost if the consumer has to guess.

    micromamba run -n vid2sim-recon python tools/package_usdz.py -m output/stereo_lidar_w0.5
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import open3d as o3d
from plyfile import PlyData
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt

SH_C0 = 0.28209479177387814


def add_mesh(stage, path: str, mesh: o3d.geometry.TriangleMesh, collide: bool,
             visible: bool, purpose: str | None = None) -> UsdGeom.Mesh:
  v = np.asarray(mesh.vertices, np.float32)
  f = np.asarray(mesh.triangles, np.int32)
  m = UsdGeom.Mesh.Define(stage, path)
  m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
  m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.ravel()))
  m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
  m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
  if len(v):
    m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
      np.stack([v.min(0), v.max(0)]).astype(np.float32)))
  if mesh.has_vertex_colors():
    m.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(
      np.asarray(mesh.vertex_colors, np.float32)))
    UsdGeom.PrimvarsAPI(m).GetPrimvar('displayColor').SetInterpolation(
      UsdGeom.Tokens.vertex)
  if collide:
    # A reconstructed street is a static, open surface: it has no inside, so convex or
    # SDF approximations are meaningless and the triangles themselves are the collider.
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(
      UsdPhysics.Tokens.none)
  if purpose:
    m.CreatePurposeAttr(purpose)
  if not visible:
    m.MakeInvisible()
  return m


def add_splat(stage, path: str, ply: pathlib.Path, keep_sh: bool,
              stride: int) -> dict:
  v = PlyData.read(str(ply))['vertex']
  get = lambda n: np.asarray(v[n], np.float32)[::stride]
  xyz = np.stack([get('x'), get('y'), get('z')], 1)
  scale = np.exp(np.stack([get(f'scale_{i}') for i in range(3)], 1))
  rot = np.stack([get(f'rot_{i}') for i in range(4)], 1)
  rot /= np.linalg.norm(rot, axis=1, keepdims=True)
  opacity = 1.0 / (1.0 + np.exp(-get('opacity')))
  dc = np.stack([get(f'f_dc_{c}') for c in range(3)], 1)

  p = UsdGeom.Points.Define(stage, path)
  p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz))
  p.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
    np.stack([xyz.min(0), xyz.max(0)]).astype(np.float32)))
  # Widths and displayColor exist so the prim is legible in any USD viewer; the primvars
  # below are what actually define the splat.
  p.CreateWidthsAttr(Vt.FloatArray.FromNumpy((2.0 * scale.mean(1)).astype(np.float32)))
  p.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
  p.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(
    np.clip(0.5 + SH_C0 * dc, 0, 1).astype(np.float32)))

  api = UsdGeom.PrimvarsAPI(p)
  api.GetPrimvar('displayColor').SetInterpolation(UsdGeom.Tokens.vertex)

  def primvar(name, typ, data, element=1):
    pv = api.CreatePrimvar(name, typ, UsdGeom.Tokens.vertex)
    if element > 1:
      pv.SetElementSize(element)
    pv.Set(data)
    return pv

  primvar('gsplat:opacity', Sdf.ValueTypeNames.FloatArray,
          Vt.FloatArray.FromNumpy(opacity.astype(np.float32)))
  primvar('gsplat:scale', Sdf.ValueTypeNames.Float3Array,
          Vt.Vec3fArray.FromNumpy(scale.astype(np.float32)))
  primvar('gsplat:rotation', Sdf.ValueTypeNames.QuatfArray,
          Vt.QuatfArray.FromNumpy(rot[:, [1, 2, 3, 0]].astype(np.float32)))
  primvar('gsplat:shDC', Sdf.ValueTypeNames.Float3Array,
          Vt.Vec3fArray.FromNumpy(dc))

  n_rest = sum(n.name.startswith('f_rest_') for n in v.properties) // 3
  if keep_sh and n_rest:
    rest = np.stack([get(f'f_rest_{i}') for i in range(3 * n_rest)], 1)
    primvar('gsplat:shRest', Sdf.ValueTypeNames.FloatArray,
            Vt.FloatArray.FromNumpy(rest.ravel()), element=3 * n_rest)

  prim = p.GetPrim()
  prim.CreateAttribute('gsplat:count', Sdf.ValueTypeNames.Int).Set(len(xyz))
  prim.CreateAttribute('gsplat:shDegree', Sdf.ValueTypeNames.Int).Set(
    int(round(np.sqrt(n_rest + 1))) - 1 if keep_sh and n_rest else 0)
  prim.CreateAttribute('gsplat:opacityActivation', Sdf.ValueTypeNames.String).Set(
    'already applied (sigmoid)')
  prim.CreateAttribute('gsplat:scaleActivation', Sdf.ValueTypeNames.String).Set(
    'already applied (exp), metres')
  return {'gaussians': int(len(xyz)), 'sh_rest_per_channel': n_rest if keep_sh else 0}


def add_navigation(stage, drive: dict) -> None:
  centre = np.asarray(drive['centreline'], np.float32)
  left = np.asarray(drive['left_edge'], np.float32)
  right = np.asarray(drive['right_edge'], np.float32)

  c = UsdGeom.BasisCurves.Define(stage, '/World/Navigation/Centreline')
  c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(centre))
  c.CreateCurveVertexCountsAttr(Vt.IntArray([len(centre)]))
  c.CreateTypeAttr(UsdGeom.Tokens.linear)
  c.CreateWidthsAttr(Vt.FloatArray([0.05] * len(centre)))
  c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
  c.CreatePurposeAttr(UsdGeom.Tokens.guide)

  band = o3d.geometry.TriangleMesh()
  verts = np.empty((2 * len(centre), 3))
  verts[0::2], verts[1::2] = left, right
  tri = []
  for i in range(len(centre) - 1):
    a = 2 * i
    tri += [[a, a + 1, a + 2], [a + 1, a + 3, a + 2]]
  band.vertices = o3d.utility.Vector3dVector(verts)
  band.triangles = o3d.utility.Vector3iVector(np.asarray(tri))
  m = add_mesh(stage, '/World/Navigation/DrivableArea', band,
               collide=False, visible=True, purpose=UsdGeom.Tokens.guide)
  p = m.GetPrim()
  p.CreateAttribute('nav:halfWidthMeters', Sdf.ValueTypeNames.Float).Set(
    float(drive['half_width_m']))
  p.CreateAttribute('nav:support', Sdf.ValueTypeNames.FloatArray).Set(
    Vt.FloatArray.FromNumpy(np.asarray(drive['support'], np.float32)))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('-m', '--model_path', required=True)
  ap.add_argument('--asset', default='sim_asset')
  ap.add_argument('--name', default='')
  ap.add_argument('--stride', type=int, default=1,
                  help='Keep every Nth gaussian; 1 keeps the splat lossless.')
  ap.add_argument('--no-sh', action='store_true',
                  help='Drop the higher SH bands. Shrinks the file by about 3x and '
                       'throws away the view-dependent appearance with them.')
  ap.add_argument('--keep-usdc', action='store_true',
                  help='Keep the uncompressed stage the package is built from.')
  ap.add_argument('--full-mesh', action='store_true',
                  help='Collide against the whole cropped mesh instead of the obstacle '
                       'layer plus support surface.')
  args = ap.parse_args()

  out = pathlib.Path(args.model_path) / args.asset
  man = json.loads((out / 'manifest.json').read_text())
  name = args.name or pathlib.Path(args.model_path).name
  usdc, usdz = out / f'{name}.usdc', out / f'{name}.usdz'

  stage = Usd.Stage.CreateNew(str(usdc))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  world = UsdGeom.Xform.Define(stage, '/World')
  stage.SetDefaultPrim(world.GetPrim())
  UsdGeom.Scope.Define(stage, '/World/Collision')
  UsdGeom.Scope.Define(stage, '/World/Appearance')
  UsdGeom.Scope.Define(stage, '/World/Navigation')

  print(f'frame: {man["frame"]}')
  layers = []
  if args.full_mesh or 'ground' not in man:
    mesh = o3d.io.read_triangle_mesh(str(out / man['mesh']['file']))
    add_mesh(stage, '/World/Collision/Scene', mesh, collide=True, visible=False)
    layers.append(('Scene', len(mesh.triangles)))
  else:
    for prim, key in (('Ground', 'ground'), ('Obstacles', 'obstacles')):
      mesh = o3d.io.read_triangle_mesh(str(out / man[key]['file']))
      add_mesh(stage, f'/World/Collision/{prim}', mesh, collide=True, visible=False)
      layers.append((prim, len(mesh.triangles)))
  for prim, n in layers:
    print(f'  collision /World/Collision/{prim}: {n} triangles, invisible, '
          f'triangle-mesh collider')

  sp = add_splat(stage, '/World/Appearance/Splat', out / man['splat']['file'],
                 not args.no_sh, args.stride)
  print(f'  appearance /World/Appearance/Splat: {sp["gaussians"]} gaussians, '
        f'{sp["sh_rest_per_channel"]} higher SH coefficients per channel')

  drive_file = out / man.get('drivable_area', {}).get('file', '')
  if drive_file.is_file():
    drive = json.loads(drive_file.read_text())
    add_navigation(stage, drive)
    print(f'  navigation /World/Navigation: centreline of {len(drive["centreline"])} '
          f'stations, +/-{drive["half_width_m"]} m band, guide purpose')

  stage.SetMetadata('customLayerData', {
    'vid2sim': json.dumps({k: v for k, v in man.items()
                           if k not in ('mesh', 'splat')}, default=str),
    'units': 'metres', 'upAxis': 'Z', 'forwardAxis': 'X', 'leftAxis': 'Y'})
  stage.GetRootLayer().Save()

  UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(usdc)), str(usdz))
  if not args.keep_usdc:
    usdc.unlink()
  print(f'\n-> {usdz}  ({usdz.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
  main()
