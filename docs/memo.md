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

## Code map

Added or changed for LiDAR supervision:

| path | what |
|---|---|
| `tools/lidar_depth.py` | sweeps → per-camera metric inverse depth; fits the metric scale |
| `tools/validate_lidar_depth.py` | scores that projection against stereo or SfM |
| `tools/perturbation_sweep.py` | translation × rotation grid at waypoints |
| `src/vid2sim_recon/train.py` | L1 inverse-depth term, gated on `--lidar_depth_weight` |
| `src/vid2sim_recon/arguments/__init__.py` | `lidar_depth_weight`, default 0 |
| `scene/dataset_readers.py`, `utils/camera_utils.py`, `scene/cameras.py` | `lidar_depth` field |
| `tools/check_depth_accuracy.py` | prefers the fitted scale over the copied one |

Default weight 0 means stock behaviour is unchanged when the flag is absent.

One trap: `sfm_metric_alignment.json` gets copied between sequence directories, and its
scale belongs to whichever reconstruction produced it. Anything needing metres should read
`lidar_depth_scale.json`, which is fit per reconstruction against its own triangulated
points.
