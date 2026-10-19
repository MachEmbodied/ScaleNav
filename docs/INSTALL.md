# Installation

For navigation tests, complete [Environment](#environment),
[Simulator installation](#simulator-installation), [Scene converter](#scene-converter)
and [Checkpoints](#checkpoints) in order.

## Hardware

Use Linux and an NVIDIA GPU with BF16 support. We recommend **24 GB GPU memory**
and **64 GB system RAM** for inference and evaluation. See
[training requirements](TRAINING.md#installation-and-memory) for training.

## Environment

Create the Python 3.12 environment:

```bash
conda create -n scalenav --override-channels -c conda-forge python=3.12 pip ffmpeg git
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scalenav

git -c http.sslBackend=openssl clone https://github.com/MachEmbodied/ScaleNav.git
cd ScaleNav
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[inference,eval,training]'
awm doctor --require-cuda --data-tools
```

Install [Miniforge](https://github.com/conda-forge/miniforge#install) if Conda is
unavailable.

## Simulator installation

Required for navigation evaluation and simulation-data rendering.
After creating the environment, complete both steps below.

### Build dependencies

Install the compiler and CUDA toolkit for Habitat-GS and DeepSpeed in the same
`scalenav` environment:

```bash
conda activate scalenav
conda install --override-channels -c conda-forge \
  gcc_linux-64=12 gxx_linux-64=12 libegl-devel libgl-devel libjpeg-turbo zlib glm=0.9.9.8
conda install --override-channels -c nvidia/label/cuda-12.4.1 -c conda-forge cuda-toolkit=12.4.1
conda activate scalenav

python -m pip install cmake==3.31.10 ninja scikit-build-core pybind11 setuptools wheel
```

### Build Habitat-GS

Run from the repository root. Habitat-GS provides the `habitat_sim` module.

```bash
(
  set -e
  conda activate scalenav
  export PYTHONNOUSERSITE=1
  export CUDA_HOME="${CONDA_PREFIX:?Activate the scalenav Conda environment}"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export CUDAHOSTCXX="${CXX:?Install the build dependencies and reactivate scalenav}"
  export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  python -c 'import os, sys; assert sys.prefix == os.environ["CONDA_PREFIX"], "Python must come from the active Conda environment"; print(sys.executable)'
  python -m awm.cli doctor --require-cuda --data-tools
  "$CUDACXX" --version
  "$CUDAHOSTCXX" --version

  scalenav_git_slot=${GIT_CONFIG_COUNT:-0}
  export "GIT_CONFIG_KEY_${scalenav_git_slot}=http.sslBackend"
  export "GIT_CONFIG_VALUE_${scalenav_git_slot}=openssl"
  export GIT_CONFIG_COUNT=$((scalenav_git_slot + 1))

  if [ ! -d external/habitat-gs ]; then
    git clone https://github.com/zju3dv/habitat-gs.git external/habitat-gs
  fi
  git -C external/habitat-gs checkout cb01b2fdf850ce3cab28ad00461a68ac7719f23f
  git -C external/habitat-gs submodule update --init --recursive
  HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF \
    python -m pip install ./external/habitat-gs --no-build-isolation \
    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native
  python -m awm.cli doctor --require-cuda --data-tools --simulator
)
```

## Scene converter

Required for Habitat-GS-LN and simulation training data. Run from the repository root:

```bash
conda install --override-channels -c conda-forge nodejs=22
npm install --prefix external/splat-tools @playcanvas/splat-transform@3.10.1
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
```

## Checkpoints

| Model | Download | Size |
| --- | --- | ---: |
| ScaleNav | [Hugging Face](https://huggingface.co/Yangyihui/ScaleNav) | 10.02 GB |
| Wan VAE, text encoder and tokenizer | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | 14.20 GB |

Download both models and set the paths used by the commands below:

```bash
python scripts/download_checkpoint.py --output checkpoints/scalenav
awm check --checkpoint checkpoints/scalenav --sha256

python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    local_dir="checkpoints/Wan2.2-TI2V-5B",
    allow_patterns=["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"],
)
PYTHON

export CHECKPOINT=checkpoints/scalenav
export BASE_MODEL=checkpoints/Wan2.2-TI2V-5B
awm doctor --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --require-cuda
```

Continue with the [navigation quick test](../README.md#quick-test).

## Optional dependencies

### DeepSpeed

For training with `--deepspeed`, complete the
[build dependencies](#build-dependencies), then install:

```bash
python -m pip install -e '.[distributed]'
```

DeepSpeed builds optimizer extensions on first use. In a new terminal, activate
`scalenav` and set `CUDA_HOME="$CONDA_PREFIX"` before training.

### Go2 robot

Install the official
[Unitree SDK2 Python package](https://github.com/unitreerobotics/unitree_sdk2_python#installation)
in the environment running the robot adapter. Policy serving and the dry-run
adapter use the base environment above.

## Development

Install the test and package-building tools:

```bash
python -m pip install -e '.[dev]'
```

From the repository root, run:

```bash
python -m pytest
python -m build
python scripts/check_release.py
```
