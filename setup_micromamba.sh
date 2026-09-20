#!/usr/bin/env bash
# Vid2Sim reconstruction environment, micromamba flavour.
#
# Upstream README assumes conda and puts everything in one env. That does not solve:
# conda-forge ships COLMAP/GLOMAP built against CUDA 12.9, which pins `cuda-version`
# and blocks the nvcc 12.1 that torch 2.1.1's extensions have to be compiled with. The
# two have no reason to share an env -- COLMAP is invoked as a binary, never imported --
# so they are split:
#
#   vid2sim        colmap + glomap + imagemagick   (stage 2 only)
#   vid2sim-recon  torch + rasterizer + open3d     (stages 3-5)
#
# Creates both from scratch:
#   bash setup_micromamba.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-vid2sim-recon}"
SFM_ENV="${SFM_ENV:-vid2sim}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="micromamba run -n ${ENV_NAME}"

echo "=== 0/5 envs ==="
micromamba create -y -n "${SFM_ENV}" -c conda-forge colmap glomap imagemagick ffmpeg
micromamba create -y -n "${ENV_NAME}" -c conda-forge python=3.10 ninja cmake ffmpeg

echo "=== 1/5 nvcc 12.1 (must match the torch cu121 wheels) ==="
# conda-forge only: mixing in the `nvidia` channel makes cuda-version=12.1 unsolvable.
#
# gcc is held at 11. cuda-nvcc would otherwise bring in 12.4, and GCC >=12.3 hits a
# parser regression on the `caster.operator typename make_caster<T>::template
# cast_op_type<T>()` line in pybind11 2.11 -- which is exactly the pybind11 torch 2.1.1
# vendors, so every CUDA extension here fails to compile against it.
micromamba install -y -n "${ENV_NAME}" -c conda-forge \
  cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 cuda-version=12.1 \
  gcc_linux-64=11 gxx_linux-64=11

echo "=== 2/5 torch ==="
$RUN pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== 3/5 python deps ==="
# transformers is pinned from both sides. Upstream's pyproject says 4.0, which predates
# Depth-Anything-V2 by five years and cannot load it. But 4.45+ calls
# torch.utils._pytree.register_pytree_node, which only exists from torch 2.2. 4.44.2 is
# the last release that both knows the model and still supports torch 2.1.1.
# setuptools is explicit, and pinned: the CUDA extensions below build with
# --no-build-isolation (they need the installed torch to compile against), so their
# setup.py resolves imports from this env. torch 2.1.1's cpp_extension.py does
# `from pkg_resources import packaging`, which setuptools dropped in 81.
$RUN pip install --no-cache-dir "setuptools<70" wheel \
  "numpy<2" opencv-python matplotlib pillow tqdm imageio plyfile aiofiles \
  "transformers==4.44.2" accelerate safetensors tensorboard scipy open3d==0.18.0

echo "=== 4/5 CUDA extensions ==="
CUDA_HOME="$(micromamba env list | awk -v e="${ENV_NAME}" '$1==e{print $NF}')"
export CUDA_HOME
echo "CUDA_HOME=${CUDA_HOME}"
# A6000 = sm_86. Pinning the arch list keeps nvcc from probing the (busy) GPU.
export TORCH_CUDA_ARCH_LIST="8.6"
# fused_ssim is listed as a plain dependency in pyproject.toml but has never been on
# PyPI; it only exists as source on GitHub.
#
# simple-knn is installed non-editable, unlike the README's `pip install -e`. Its
# source tree has no simple_knn/__init__.py -- the directory exists only to receive the
# compiled _C.so -- so the editable finder maps the name to a directory Python then
# refuses to treat as a package. A regular install copies the .so into site-packages
# and the import resolves.
$RUN env CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST=8.6 \
  pip install --no-cache-dir --no-build-isolation \
    -e "${REPO}/submodules/vid2sim-rasterizer" \
    "${REPO}/submodules/simple-knn" \
    "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim.git"

echo "=== 5/5 sanity ==="
$RUN python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
import diff_gaussian_rasterization as r; print("rasterizer", r.__file__)
import simple_knn._C  # noqa: F401
print("simple-knn ok")
from fused_ssim import fused_ssim  # noqa: F401
print("fused_ssim ok")
import open3d; print("open3d", open3d.__version__)
import transformers; print("transformers", transformers.__version__)
PY
echo "Done."
