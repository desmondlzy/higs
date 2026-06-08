#!/bin/bash
# Creates the conda environment and installs all dependencies.
# Usage: bash install.sh

set -e

ENV_NAME="higs"
CONDA_BASE=$(conda info --base)
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
conda create -n "$ENV_NAME" python=3.12 -y

echo "Installing PyTorch (CUDA 12.4 build)..."
# --index-url (NOT --extra-index-url): forces the CUDA 12.4 wheels. Using
# --extra-index-url here would let pip fall back to the default PyPI torch,
# which is built against a newer CUDA and fails to compile gsplat on a 12.x toolkit.
# PyTorch must also be installed *before* the package below, because gsplat's build
# imports torch at build time (combined with --no-build-isolation).
# Versions pinned: nerfstudio 1.1.5 is incompatible with newer torch (e.g. 2.7+ changed
# torch.load defaults and CUDA build requirements). 2.6.0 is the validated, working set.
"$PIP" install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

echo "Installing packages..."
# --no-build-isolation: so the gsplat CUDA extension can find the torch we just installed.
# This pulls nerfstudio, which depends on gsplat==1.4.0 (installed here as a transitive dep).
"$PIP" install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-build-isolation \
    -e .

echo "Installing gsplat 1.5.1 from extern/gsplat..."
# nerfstudio pins gsplat==1.4.0, but this project is developed/validated against gsplat 1.5.1.
# We install it from the extern/gsplat submodule (NOT PyPI: the PyPI 1.5.1 wheel ships no CUDA
# sources, so its backend can't build). nerfstudio runs fine with 1.5.1; --no-deps keeps pip from
# touching anything else, and the "nerfstudio requires gsplat==1.4.0" warning is expected.
"$PIP" install \
    --no-build-isolation \
    --no-deps \
    ./extern/gsplat

echo "Downloading freeimage binary for .exr support"
"$CONDA_BASE/envs/$ENV_NAME/bin/imageio_download_bin" freeimage

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
