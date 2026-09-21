# LiDAR-supervised Vid2Sim from ROS 2 `.mcap`

Turning a robot log into a Gaussian splat you can fly a *policy* through, not just replay.
Worked end to end on one clip: `mission_2025-01-17-10-37-51.mcap` (LSMap seq 19), a 20.6 s
walk down a sidewalk, 207 stereo pairs at 10 Hz.

## The finding

Off-path novel views were smeared and melted. That turned out to be geometry, not
appearance, and LiDAR fixes it.

The depth prior Vid2Sim ships is Depth-Anything-V2, consumed by a patch-NCC loss. NCC is
invariant to scale and offset *by construction*, so it constrains the shape of the depth
field and never its absolute value. A reconstruction can therefore fit every training
image perfectly while sitting metres from where the scene actually is. Nothing in the
monocular pipeline ever notices, because on-path rendering does not depend on getting
depth right — only on getting it consistently wrong.

Lateral motion is what exposes it. Step sideways by `b` and a surface at depth `z`
carrying depth error `dz` lands about `f·b·dz/z²` pixels from where it belongs. At this
clip's scale that is ~12 px for a half-metre step, which is exactly the smearing that was
visible.

Adding LiDAR as a **metric** depth prior — a plain L1 on inverse depth over pixels that
got a return, alongside the existing NCC term — moves every number that matters:

| | depth error | misreg. @0.5 m | held-out PSNR |
|---|---|---|---|
| mono | 15.8% | 12.1 px | 23.78 dB |
| stereo | 10.1% | 8.2 px | — |
| **mono + LiDAR** | **3.9%** | **3.3 px** | **25.27 dB** |
| **stereo + LiDAR** | **3.8%** | **3.2 px** | — |

Depth error is measured against block-matched stereo. Held-out PSNR trains on the left
camera only and scores the right camera — a 24.5 cm lateral novel view with a real
recorded image behind it, which is the one fully independent test available here.

Two consequences worth acting on:

**LiDAR substitutes for stereo, geometrically.** Stereo alone gets 15.8% → 10.1%; LiDAR
reaches ~3.8% from either starting point, and stereo adds essentially nothing on top of
it. If this holds on other clips, monocular streams are viable for asset generation and
you are not restricted to logs with a working stereo pair.

**3.8% is the measurement floor, not the model's.** LiDAR agrees with block-matched stereo
to only 3.0%, so the remaining error is dominated by the instrument. Resolving further
needs a better reference, not a better model.

Training PSNR drops ~0.34 dB with the prior on. That is the correct sign: the model can no
longer buy on-path photometric fit with wrong geometry, and on-path fit was never what was
failing.

### Two bugs found on the way, both silent

Worth knowing about because neither announces itself and both are easy to reproduce.

**The point cloud is not in the frame its header claims.** `/points` is stamped
`os_sensor`, but the data is in `os_lidar`. The two differ by a 180° yaw, so trusting the
header aims the camera backwards down the robot's −x axis. It still looks plausible up
close — objects a metre away project into roughly the right part of the frame either way —
and only falls apart at range. Scored against triangulated SfM points, the header-based
chain gives 85% median depth error versus 8% for `os_lidar`. `calib_os1_to_cam0.yaml`
independently confirms `os_lidar`.

**Rectification must be applied to the extrinsic.** The calibration files give the
LiDAR→camera transform in the *raw* camera frame, but the images are rectified. The
difference is only 1.074°, which displaces points ~4.5 cm at 2.4 m — the same order as the
error being chased.

After both fixes, projected LiDAR agrees with block-matched stereo to **3.0% median, flat
across the full depth range**.

### Off-path envelope

`perturbation_sweep.py` samples waypoints along the route and renders a translation ×
rotation grid at each. Scored by the share of frame where accumulated opacity is too low
to call the pixel explained — the failure a policy actually trips over.

Stereo + LiDAR, mean over 24 waypoints:

| | −1.0 m | −0.5 m | 0 | +0.5 m | +1.0 m |
|---|---|---|---|---|---|
| **−30°** | 6.9% | 5.1% | 4.2% | 3.5% | 3.1% |
| **−15°** | 3.6% | 2.1% | 1.4% | 0.9% | 0.7% |
| **0°** | 1.2% | 0.5% | 0.3% | 0.3% | 0.6% |
| **+15°** | 1.4% | 1.6% | 1.8% | 2.2% | 2.7% |
| **+30°** | 5.2% | 5.7% | 6.4% | 7.1% | 8.2% |

**Rotation costs far more than translation.** A full metre sideways leaves >99% of the
frame supported; 30° of yaw costs 4–6%. A lateral dolly keeps looking at surfaces the
capture saw head-on, whereas turning swings the frame toward things the camera only caught
at the edge of its field of view. Turning and stepping the *same* direction compounds
(+30° with +1.0 m is worst at 8.2%); turning back toward the path partially recovers.

Usable envelope: **within ±1 m and ±15°, at least 97% of every frame is supported.**

Against the control without LiDAR, at 0° yaw: 3.7% / 1.4% / 0.1% / 0.7% / 2.6% across the
same offsets, versus 1.2% / 0.5% / 0.3% / 0.3% / 0.6%. Roughly 3× fewer holes at ±1 m. The
control is marginally better exactly on the rails, same trade as its higher train PSNR.

## Limitations

**One clip, one scene.** Every number here is from a single 20 m sidewalk walk. The
mono-plus-LiDAR equivalence in particular is the kind of result that should be confirmed
on a second clip before being relied on.

**No held-out view for the stereo models.** Both cameras are in training, so for those the
evidence is depth accuracy, scored against a reference that is not fully independent of
LiDAR (they agree to 3.0%). The clean test is the left-only experiment.

**LiDAR covers ~7% of pixels.** One sweep per frame, after rejecting pixels whose
neighbouring returns disagree. The other 93% is still carried by Depth-Anything through
the NCC term. Denser supervision would need accumulation across sweeps, which reintroduces
a dependence on SLAM poses.

**Sweeps are not motion-compensated.** No per-point timestamps in `/points`, so points are
treated as simultaneous. The camera's angular sector spans ~22 ms of the 100 ms sweep, or
~3 cm at walking pace — under the 3.0% agreement floor, so currently invisible. It would
matter at higher speed.

**LiDAR reads ~2.3% short** of both block-matched stereo and SfM, which share a
baseline-derived scale. Cause not established — could be the baseline calibration or a
range bias. Handled by fitting metres-per-SfM-unit against each reconstruction's own
triangulated points, which keeps the prior consistent with the frame the Gaussians live in
rather than imposing a global scale shift the fixed camera poses would fight.

**SfM refines the intrinsics away from calibration.** COLMAP ends at
`fx=372.62, fy=384.27, cx=306.0, cy=256.0` against the calibrated
`375.04, 375.04, 302.27, 247.08`. The 3% `fx`/`fy` anisotropy is unphysical for a
rectified pair, so the refinement has absorbed something it should not have. The splat
therefore renders with slightly different rays than the prior was built with. Checked
rather than assumed: projecting LiDAR with the calibrated intrinsics agrees with the
triangulated points to 3.31% versus 4.23% for the refined ones, so the calibrated values
are kept. Cleaner would be to fix intrinsics during SfM via
`--ImageReader.camera_params`, which has not been tried.

**Weight 0.5 is empirical.** Chosen from a 4-point sweep (0.25/0.5/1.0/2.0) on one scene.
Higher weights drive heavy densification — 2.68 M Gaussians at 2.0 versus 1.08 M for the
control — while depth accuracy saturates by 0.25, so the extra capacity buys nothing.

**Route ends are degenerate.** The first two waypoints hit 44.6% and 40.8% empty at their
worst perturbation, because nothing has been observed behind or beside the camera yet.
Trim trajectory ends or avoid spawning there.

**Sharpness metrics mislead under rotation.** Laplacian variance scores turning right
*better* than on-path here, because it swings high-frequency foliage into view. That is a
fact about the scene, not the reconstruction. Use the alpha coverage numbers.

**Dynamic objects are masked, not reconstructed.** Pedestrians are removed via
Mask2Former. Their LiDAR returns are excluded from the depth loss by the same alpha mask.

## Environment

Four environments. The split is not fussiness: conda-forge's COLMAP/GLOMAP are built
against CUDA 12.9, which pins `cuda-version` and blocks the nvcc 12.1 that torch 2.1.1's
extensions must compile against. COLMAP is invoked as a binary and never imported, so the
two have no reason to share an env.

```bash
# vid2sim (SfM binaries) + vid2sim-recon (torch, rasterizer, open3d)
bash setup_micromamba.sh

# mcap: log reading and LiDAR projection. Deliberately minimal -- no torch.
micromamba create -y -n mcap -c conda-forge python=3.10 "numpy<2"
micromamba run -n mcap pip install mcap rosbags opencv-python pyyaml tqdm
```

Mask generation needs `transformers` + torch; any env with both works. Below uses `ns`,
a pre-existing nerfstudio env; `vid2sim-recon` also works.

`setup_micromamba.sh` documents its own pins inline. The ones that bite: `transformers`
must be 4.44.2 (4.45+ needs torch ≥2.2; upstream's 4.0 predates Depth-Anything-V2),
`setuptools<70` (torch 2.1.1 imports `pkg_resources.packaging`, dropped in 81), and gcc
held at 11 (≥12.3 hits a parser regression on the pybind11 2.11 that torch 2.1.1 vendors).

### One-time rasterizer fix

The bundled rasterizer collides with PyTorch's CUB, producing an out-of-bounds write in
`duplicateWithKeys` that corrupts memory and surfaces much later as
`linalg.inv: diagonal element 3 is zero` during training. Fixed by namespacing CUB:
`-DCUB_WRAPPED_NAMESPACE=diff_gauss_cub` in the nvcc flags of
`submodules/vid2sim-rasterizer/setup.py`, plus the matching alias in `rasterizer_impl.cu`.
Already applied in this tree; re-apply if the submodule is reset.

## Reproducing the example

Paths assume repo root. Source data: `/scratch/zichaohu/real2sim/`, calibrations
`/robodata/arthurz/Datasets/lsmap_bags_processed/calibrations/19`, LeGO-LOAM poses
`/robodata/arthurz/Datasets/lsmap_bags_processed/poses/legoloam/19.txt`.

```bash
SEQ=data/urban_walk_20m_stereo
MCAP=/scratch/zichaohu/real2sim/mission_2025-01-17-10-37-51.mcap
```

**1. Cut the clip.** Emits both cameras as ordinary frames into one flat `images/`,
rectified with the calibrated `P` verbatim and uncropped, so both share intrinsics.

```bash
micromamba run -n mcap python tools/mcap_to_vid2sim_stereo.py \
  --mcap $MCAP --out $SEQ --start 1737132294.2 --end 1737132314.9
```

**2. Mask dynamics.** Combines with the rectification-border mask already in `masks/`.

```bash
micromamba run -n ns python tools/generate_mask_mask2former.py --seq $SEQ --labels person
```

**3. SfM.** GLOMAP via COLMAP. `--overlap 20` because interleaving two cameras under one
numbering halves the time span a given overlap covers.

```bash
cd src/vid2sim_recon
micromamba run -n vid2sim python tools/run_sfm.py -s ../../$SEQ \
  -m ../../$SEQ/masks --camera PINHOLE --overlap 20
cd ../..
```

> If `image_undistorter` dies with "File exists", delete `$SEQ/images/` and re-run with
> `--skip_matching`. It refuses to write into a populated `images/`.

Check it against the calibrated baseline and the LiDAR trajectory — this is what makes a
metre a metre downstream:

```bash
micromamba run -n mcap python tools/check_stereo_sfm.py --seq $SEQ
```

**4. Depth priors.** Both are used: Depth-Anything is dense but scale-free, LiDAR is sparse
but metric.

```bash
cd src/vid2sim_recon
micromamba run -n vid2sim-recon python tools/generate_depth.py ../../$SEQ
cd ../..

micromamba run -n mcap python tools/lidar_depth.py --seq $SEQ --sides left right
```

Writes `lidar_depths/*.npy` (inverse depth in SfM units, 0 = no return) and
`lidar_depth_scale.json`. Verify before training — a wrong extrinsic is not obvious in the
rendered result:

```bash
micromamba run -n mcap python tools/validate_lidar_depth.py --seq $SEQ --against stereo
micromamba run -n mcap python tools/validate_lidar_depth.py --seq $SEQ --against sfm
```

Expect ~3% median absolute relative error against stereo, roughly flat across depth bands.
A bias that grows with range means the extrinsic or the frame is wrong.

**5. Train.**

```bash
cd src/vid2sim_recon
micromamba run -n vid2sim-recon python train.py \
  -s $PWD/../../$SEQ -m $PWD/../../output/stereo_lidar_w0.5 \
  --lidar_depth_weight 0.5 --iterations 30000 \
  --test_iterations 30000 --save_iterations 30000
cd ../..
```

`--lidar_depth_weight 0` reproduces stock Vid2Sim. ~20 min on an A6000.

**6. Mesh** (collision layer). `--angle_threshold 0.0` keeps the ground plane; the default
filters it out by surface normal, which is wrong for a collision mesh.

```bash
cd src/vid2sim_recon
micromamba run -n vid2sim-recon python export_mesh.py \
  -m $PWD/../../output/stereo_lidar_w0.5 --angle_threshold 0.0
cd ../..
```

## Flythroughs and evaluation

**Straight dolly**, constant lateral offset held for the whole route. `--only left`
restricts to one eye, otherwise the path hops between cameras every frame.

```bash
for OFF in 0.0 0.5 1.0 1.5; do
  micromamba run -n vid2sim-recon python tools/render_flythrough.py \
    -m output/stereo_lidar_w0.5 -s $SEQ --right $OFF --only left --tag off$OFF
done
```

Writes `output/<model>/flythrough/off<N>.mp4`.

**Perturbation sweep**, translation × rotation at waypoints along the route. Closer to
what a policy does, and yaw is the harsher axis.

```bash
micromamba run -n vid2sim-recon python tools/perturbation_sweep.py \
  -m output/stereo_lidar_w0.5 -s $SEQ --only left --waypoints 24 --tag perturb
```

Writes one grid PNG per waypoint plus `perturb.mp4`, and prints the empty-share table.
Grid is configurable: `--lateral -1 0 1 --yaw -30 0 30 --pitch 5 --forward 0.5`.

**Quantitative:**

```bash
# depth accuracy vs block-matched stereo, and the misregistration it implies
micromamba run -n vid2sim-recon python tools/check_depth_accuracy.py \
  -m output/stereo_lidar_w0.5 -s $SEQ

# held-out right-camera views -- only valid for a left-only model
micromamba run -n vid2sim-recon python tools/attribute_error.py \
  -m output/urban_walk_20m_leftonly -s $SEQ --fix-exposure
```

`--fix-exposure` removes a per-image, per-channel offset; the right camera runs 4.6%
darker, which otherwise lands on every pixel and masquerades as reconstruction error.

### Reproducing the left-only ablation

The clean held-out test. `urban_walk_20m_leftonly` is the stereo reconstruction with right
images filtered out — identical poses to machine precision, so the right camera stays a
valid held-out view in the same coordinate frame.

```bash
micromamba run -n mcap python tools/lidar_depth.py --seq data/urban_walk_20m_leftonly
cd src/vid2sim_recon
micromamba run -n vid2sim-recon python train.py \
  -s $PWD/../../data/urban_walk_20m_leftonly \
  -m $PWD/../../output/leftonly_lidar_w0.5 \
  --lidar_depth_weight 0.5 --iterations 30000 \
  --test_iterations 30000 --save_iterations 30000
```

## Exporting a sim asset

`tools/export_sim_asset.py` takes the trained splat and the TSDF mesh and emits both in
metres, co-registered, with the collision layer cropped and cleaned.

```bash
micromamba run -n vid2sim-recon python tools/export_sim_asset.py \
  -m output/stereo_lidar_w0.5 --radius 12.8 --fill-holes 0.5
```

Writes `output/<model>/sim_asset/`:

| file | what |
|---|---|
| `splat_metric.ply` | appearance layer, metric and gravity aligned |
| `collision_12.8m.ply` | full cropped mesh, ground included |
| `obstacles_12.8m.ply` | same mesh with level surfaces removed, for use with the primitive |
| `ground.ply` | support surface fitted locally along the route |
| `drivable_area.json` | centreline, ±1.5 m bound, per-station support |
| `manifest.json` | scale, frame, crop and every diagnostic below |

The asset is metric and gravity aligned; see the two sections below for the frame and for
how the splat's orientation-carrying fields are transformed.

**Scale.** Two independent handles exist and they disagree by 2.2%. The LiDAR fit
(`lidar_depth_scale.json`, 2.1790 m/unit) is a median over projected returns and carries
the projection's bias. The stereo baseline (2.2288 m/unit) is a calibrated 24.5 cm
constant measured directly between two reconstructed cameras with no depth estimate in the
chain, so `--scale-from auto` prefers it when the scene is stereo. Corroboration: after
scaling, the ground sits 0.46 m below the camera, which is the rig height.

Scaling the splat is not just its positions. Gaussian extents are stored as logs, because
the model applies `exp` as the scaling activation, so a factor `s` is `+log(s)` on
`scale_*`. Multiplying there leaves every centre correct and every Gaussian the wrong
size, which reads as a bad model rather than a unit bug.

**Crop.** Vertices within `--radius` of the *nearest camera on the route*, not of the
route's centroid, so the kept band is a tube following the trajectory rather than a sphere
around its middle. 12.8 m gives a 25.6 m span, the SemanticKITTI BEV convention. This
matters because the raw export reaches 121 m from a 27.9 m route: `--angle_threshold 0`
has to be set to keep the ground, and it also stops rejecting distant grazing surfaces.

The splat is deliberately left uncropped. Distant background is wanted for rendering even
though it is useless for collision, and cutting it at 12.8 m would leave a void past the
crop. `--splat-radius` overrides this.

**Watertightness.** Reported, not assumed, and never achieved — correctly so. An outdoor
capture is a surface, not a solid: sky, ground past the crop, and everything behind the
facades are open by construction. What the report separates is loop size. Filling holes up
to 0.5 m closes 234 of 287 loops, all of them few-edge reconstruction noise; the 53 that
remain are led by a 1672-edge loop that is the outer rim, and capping that would dome the
sky over the street. It also separates holes from non-manifold edges, since the latter are
structural damage that hole filling cannot repair.

### The crop radius is not the usable radius

The number that decides how far a policy may wander is ground coverage, so the script
raycasts down along the route and prints it:

| lateral offset | route with ground under it |
|---|---|
| 0 m | 95.7% |
| 1 m | 94.9% |
| 2 m | 76.1% |
| 4 m | 13.5% |
| 8 m | 6.3% |

Collision is trustworthy to roughly 1 m, and degrades hard past 2 m. A forward-facing walk
never observes the ground a few metres to the side, so the TSDF has nothing to fuse there
and the mesh simply stops — in a physics sim the robot falls through the world rather than
colliding with it. The crop radius bounds the asset; it does not fill it.

This is a tighter envelope than the appearance layer, which held up to ±1 m laterally with
>98.9% alpha coverage. Anything wider needs either a ground primitive underneath to catch
the robot, or a capture with lateral coverage.

### Gravity alignment

The asset comes out Z up, X forward, Y left (REP 103), origin on the ground beneath the
first camera. `--keep-sfm-frame` opts out. Three estimates of up are available and none is
usable alone, so the frame takes each angle from whichever one constrains it.

| source | what it is | verdict |
|---|---|---|
| trajectory | third row of the Umeyama `R` in `sfm_metric_alignment.json` | pitch only |
| attitude | gravity carried through LiDAR pose quaternions, extrinsics, and SfM poses | roll only |
| ground fit | robust plane through the mesh under the route | not gravity |

The trajectory fit aligns SfM camera centres to LiDAR positions, so pitch is pinned
tightly. It is blind to roll: rotation about the direction of travel is constrained only
by sideways spread, and this route is 107:1 straight (8.08 m of principal spread against
0.24 m and 0.08 m), so a 4° roll moves the cameras under 2 cm against an 8 cm alignment
residual. That axis is free.

The attitude estimate reads a full orientation per frame, so nothing about it is
degenerate, and 207 independent estimates scatter by a median of 0.59°. It is the only
source of roll. But it inherits any pitch error in the sensor-to-camera extrinsic and
lands 1.3° steep.

The ground plane is not gravity at all, and assuming it is would have been the easy
mistake. **The street genuinely climbs**: the LiDAR poses rise 2.02 m over 28.55 m
travelled, a 4.04° grade. Fitting the surface and declaring it level would have rotated
the hill out of the scene and put the robot on a flat road. It is kept only as a
cross-check, since the angle between it and gravity should equal the grade.

Validation: in the exported frame the route climbs **3.97°** against the LiDAR-measured
4.04°, the camera sits 0.48 m above z=0 at the start against a 0.46 m rig height, and the
ground spans −0.22 m to +12.12 m in z.

Because the grade is preserved, a ground primitive added underneath must follow the slope.
A horizontal plane at z=0 would cut through the road about 14 m along the route.

### Rotating a splat is more than its positions

Four things move and all four have to agree, or the asset is subtly wrong in a way that
still renders plausibly:

- **positions** scale then rotate;
- **extents** are logs, so scaling is `+log(s)` and rotation leaves them alone;
- **quaternions** compose with the rotation;
- **spherical harmonics** encode view-dependent colour against world axes and must rotate
  with them. Leaving them is the quiet failure: shading and speculars keep pointing the old
  way, which reads as a badly trained model rather than a frame bug. It matters most here,
  where SH degree 3 was measured overfitting by 8 dB and so carries a lot of the appearance.

The basis in `sh_utils.py` has its own sign convention, so instead of hand-deriving
Wigner-D matrices against it, `sh_rotation` fits the band matrices numerically from the
repo's own evaluator — sample directions, evaluate at `d` and at `Rᵀd`, solve. Bands are
closed under rotation so the fit is exact (verified at 8e-15, block diagonal, orthogonal
per band) and it cannot drift out of sync with the convention it derives from.

`tools/verify_sim_asset.py` checks the whole thing end to end by rendering the exported
file through the transformed camera and comparing pixels:

```bash
cd src/vid2sim_recon && micromamba run -n vid2sim-recon \
  python ../../tools/verify_sim_asset.py -m ../../output/stereo_lidar_w0.5 --frames 24
```

Median 84.9 dB against the original, mean absolute pixel difference 1.3e-3, and Gaussian
covariances matching `s²RCRᵀ` to 1.4e-7 — float32 storage precision. A few views differ in
dense canopy: the covariance check proves the geometry is exact, so what moves there is the
blend order of overlapping semi-transparent splats, which shifts when every position is
multiplied and rotated.

## How Vid2Sim keeps the agent inside the corridor

Worth reading before building on this, because the upstream answer is the opposite of what
the export does by default, and it is the better answer.

**Vid2Sim throws the reconstructed ground away.** `export_mesh.py` defaults to
`--angle_threshold 15`, which drops every surface whose normal is within 15° of vertical,
and `--use_ground_mask` additionally segments the ground out with SAM-HQ. What survives is
only the vertical structure. Unity then supplies its own horizontal walkable plane, and
both it and the scene mesh are tagged collidable but rendered invisible, with the splat
providing all the visuals.

That is exactly the right call given the coverage measured above. A forward-facing walk
never sees the ground a few metres to the side, so a reconstructed ground surface is
reliable for about 1 m and then stops existing. A synthetic plane is reliable everywhere,
and the reconstructed facades, poles and parked cars become invisible collision walls that
physically bound the corridor. The agent cannot leave because the geometry stops it.

On top of that the paper terminates an episode on leaving the drivable area, exceeding
3,000 steps, or exceeding three collisions, each worth −10, with start and goal randomised
per episode and success declared within 0.5 m. The released config is a little different
(`collision_limit: 5`, `max_episode_length: 60`), so the shipped numbers are not the
paper's. Note also that `max_depth` defaults to 999 in the argparse while the function
signature says 5.0 — the argparse wins, and that unbounded fusion is why the raw mesh
reaches 121 m from a 27.9 m route.

So there are four overlapping mechanisms, and the reconstruction only provides one:

| mechanism | where it lives |
|---|---|
| synthetic ground plane replacing unreliable reconstructed ground | Unity scene |
| invisible collision walls from reconstructed vertical geometry | TSDF mesh |
| termination on leaving the drivable area or on repeated collisions | Unity build |
| goals sampled along the captured route, short episodes | env config |

### The support surface, fitted locally

The export builds that layer. `ground.ply` is a ribbon following the route, plus
`obstacles_<r>m.ply` — the collision mesh with surfaces within `--obstacle-angle` (15°, as
upstream) of level removed — and `drivable_area.json` for the bound.

A single tilted plane would have been the obvious shortcut and it is wrong twice over.
It only holds while the whole clip shares one grade, and here the grade runs from **+0.38°
to +6.73°** along 28 m, so one plane fitted to the lot drifts ~20 cm from the road through
the middle of the route. Worse, it cannot represent the case actually worth planning for:
a robot that runs down into a cul-de-sac and back up revisits the same ground at two
heights, and no plane passes through both.

So the surface is fitted per station, every 0.25 m, each taking its height and cross slope
from ground within 2.5 m **measured along the route rather than through space**. Arc length
is what makes doubling back work — the outbound and return legs are far apart along the
route even when they sit on top of each other in the world. Stations with too little ground
interpolate from neighbours instead of inventing a height, and the fit is trimmed so curbs,
verges and car roofs caught by the wider probes cannot tilt a section.

Measured against the reconstructed ground inside the ±1.5 m band:

| | median | p90 abs | 
|---|---|---|
| local per-station fit | −0.4 cm | 7.3 cm |
| one global plane | +0.8 cm | 17.7 cm |

The medians are both fine — a global plane is unbiased on average, which is exactly why the
median hides the problem. The p90 is the honest number and the local fit is 2.4× tighter.
Largest step between neighbouring stations is 5.3 cm over 25 cm of travel, so nothing a
robot would trip on. All 112 stations fitted directly here; none needed interpolation.

**The drivable bound is ±1.5 m** (`--drivable-halfwidth`), inside the 1–2 m the coverage
table supports, with the support surface running wider at ±4 m so the robot never reaches
its edge. `drivable_area.json` carries the centreline, both edges, and a per-station
`support` fraction recording how much of the band had real geometry under it — median 1.00,
with 5 of 112 stations below half near the route ends. A rollout can terminate on leaving
the band and treat thinly supported stations differently rather than trusting a flat number.

The cross sections are the clearest argument for that bound: inside it the reconstruction
sits on the fitted surface, and past about 2 m the "ground" a downward ray finds is curbs,
walls and foliage standing 1.5–3 m proud.

## Packaging the USDZ

`tools/package_usdz.py` binds the layers into one file. It needs `usd-core`, which is not
part of the original environment:

```bash
micromamba run -n vid2sim-recon pip install usd-core
micromamba run -n vid2sim-recon python tools/package_usdz.py -m output/stereo_lidar_w0.5
```

The stage declares Z up and one unit per metre rather than leaving them to be inferred —
the whole point of computing a gravity frame is lost if the consumer has to guess. Three
roles, split the way upstream Vid2Sim splits them:

| prim | type | role |
|---|---|---|
| `/World/Collision/Ground` | Mesh | support surface, **invisible**, triangle-mesh collider |
| `/World/Collision/Obstacles` | Mesh | reconstructed obstacles, **invisible**, triangle-mesh collider |
| `/World/Appearance/Splat` | Points | the splat, visible, no collision |
| `/World/Navigation/Centreline` | BasisCurves | guide purpose, neither renders nor collides |
| `/World/Navigation/DrivableArea` | Mesh | guide purpose, carries `nav:halfWidthMeters` and per-station `nav:support` |

Collision approximation is `none`, i.e. the triangles themselves. Convex or SDF
approximations would be meaningless here: a reconstructed street is an open surface with no
inside, which is the same reason the watertightness report above never reaches "yes".

The splat is written as a native `UsdGeomPoints` with the full Gaussian state in primvars
(`gsplat:scale`, `gsplat:rotation`, `gsplat:opacity`, `gsplat:shDC`, `gsplat:shRest` with
`elementSize` 45) rather than as a `.ply` packaged alongside. A sidecar would write faster
but would leave the asset only half described by USD. Activations are applied on the way in
— `exp` on the extents so they are metres, `sigmoid` on opacity — and the prim says so in
`gsplat:scaleActivation` and `gsplat:opacityActivation`, because a consumer cannot tell by
looking. `points` and `displayColor` double as a preview that opens in any USD viewer.

Verified by reading the package back: positions, extents, quaternions, opacity and all 45
higher SH coefficients round-trip at **exactly zero error**, and raycasting the collision
prims extracted from the USDZ supports 99.1% of the ±1.5 m band.

`--stride N` subsamples the splat and `--no-sh` drops the higher bands for a roughly 3×
smaller file, at the cost of the view-dependent appearance the LiDAR work went to trouble
to get right. `--full-mesh` collides against the whole cropped mesh instead of the
obstacle-plus-support pair, which is the wrong default for training but useful for
inspecting what the reconstruction actually contains.

One caveat worth stating plainly: USD has no native Gaussian-splat schema, so nothing will
render this as splats out of the box. The geometry, physics and frame are standard USD and
will load anywhere; the appearance layer is a faithful container that a 3DGS renderer — or
a conversion to Isaac Sim's NuRec representation — has to consume.

## Code map

Added or changed for LiDAR supervision:

| path | what |
|---|---|
| `tools/lidar_depth.py` | sweeps → per-camera metric inverse depth; fits the metric scale |
| `tools/validate_lidar_depth.py` | scores that projection against stereo or SfM |
| `tools/perturbation_sweep.py` | translation × rotation grid at waypoints |
| `tools/export_sim_asset.py` | metric, gravity-aligned splat + cropped collision mesh |
| `tools/verify_sim_asset.py` | renders the export against the original to prove the transform |
| `tools/package_usdz.py` | binds splat, collision and navigation layers into one USDZ |
| `src/vid2sim_recon/train.py` | L1 inverse-depth term, gated on `--lidar_depth_weight` |
| `src/vid2sim_recon/arguments/__init__.py` | `lidar_depth_weight`, default 0 |
| `scene/dataset_readers.py`, `utils/camera_utils.py`, `scene/cameras.py` | `lidar_depth` field |
| `tools/check_depth_accuracy.py` | prefers the fitted scale over the copied one |

Default weight 0 means stock behaviour is unchanged when the flag is absent.

One trap: `sfm_metric_alignment.json` gets copied between sequence directories, and its
scale belongs to whichever reconstruction produced it. Anything needing metres should read
`lidar_depth_scale.json`, which is fit per reconstruction against its own triangulated
points.
