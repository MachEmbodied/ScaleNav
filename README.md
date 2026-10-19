<h1 align="center">ScaleNav</h1>

<h3 align="center">Scalable Internet Video Pretraining<br>for Language-Conditioned Local Navigation</h3>

<p align="center">
  <a href="https://zju3dv.github.io/ScaleNav/"><img src="https://img.shields.io/badge/Project-Page-2563eb?logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper — coming soon" title="Paper link coming soon">
  <a href="https://huggingface.co/Yangyihui/ScaleNav"><img src="https://img.shields.io/badge/Hugging_Face-Model-ffcc4d?logo=huggingface&amp;logoColor=white" alt="Hugging Face model"></a>
  <a href="https://huggingface.co/datasets/Yangyihui/ScaleNav-Data"><img src="https://img.shields.io/badge/Hugging_Face-Data-57b894?logo=huggingface&amp;logoColor=white" alt="Hugging Face data"></a>
</p>

<p align="center">
  <a href="https://yangyihui0502.github.io/">Yihui Yang</a><sup>1</sup> ·
  <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1</sup> ·
  <a href="https://pluswave.top/">Jiawei Wang</a><sup>1</sup> ·
  Zhongrui Yu<sup>2</sup> ·
  Jia Shi<sup>2</sup><br>
  <a href="https://zhankunliauto.github.io/">Kun Zhan</a><sup>2</sup> ·
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a><sup>3</sup> ·
  <a href="https://xzhou.me/">Xiaowei Zhou</a><sup>1</sup> ·
  Yu&nbsp;Liu<sup>2</sup> ·
  <a href="https://pengsida.net/">Sida Peng</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;Zhejiang University &nbsp; <sup>2</sup>&nbsp;Li Auto Inc. &nbsp; <sup>3</sup>&nbsp;Shenzhen University
</p>

> **TL;DR:** ScaleNav learns navigation priors from Internet videos through **action-free pretraining**, then transfers them to **language-conditioned local navigation**.

<p align="center">
  <a href="https://zju3dv.github.io/ScaleNav/#demos"><img src="docs/assets/navigation-demo.gif" width="100%" alt="ScaleNav navigation: real robot on the left, simulation on the right"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#training">Training</a> ·
  <a href="#real-robot">Real robot</a>
</p>

## Quick start

Run the **Habitat-GS-LN test subset** and view navigation videos and trajectories:
[install](#installation) → [download weights](#checkpoints) →
[prepare data](#data) → [evaluate](#evaluation).

### Installation

#### Hardware

Use Linux and an NVIDIA GPU with BF16 support. We recommend **24 GB GPU memory**
and **64 GB system RAM** for inference and evaluation. See
[training requirements](docs/TRAINING.md#installation-and-memory) for training.

#### Environment

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

#### Simulator installation

Required for navigation evaluation and simulation-data rendering.
After creating the environment, expand and complete both steps below.

<details>
<summary><strong>Install the compiler, CUDA toolkit and Habitat-GS</strong></summary>

##### Build dependencies

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

##### Build Habitat-GS

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

</details>

#### Scene converter

Required for Habitat-GS-LN and simulation training data. Run from the repository root:

```bash
conda install --override-channels -c conda-forge nodejs=22
npm install --prefix external/splat-tools @playcanvas/splat-transform@3.10.1
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
```

#### Optional dependencies

<details>
<summary>Training with DeepSpeed, Go2 deployment and development tools</summary>

**DeepSpeed:** for training with `--deepspeed`, complete the
[build dependencies](#build-dependencies), then install:

```bash
python -m pip install -e '.[distributed]'
```

DeepSpeed builds optimizer extensions on first use. In a new terminal, activate
`scalenav` and set `CUDA_HOME="$CONDA_PREFIX"` before training.

**Go2 robot:** install the official
[Unitree SDK2 Python package](https://github.com/unitreerobotics/unitree_sdk2_python#installation)
in the environment running the robot adapter. Policy serving and the dry-run
adapter use the base environment above.

**Development:** install the test and package-building tools:

```bash
python -m pip install -e '.[dev]'
```

</details>

### Checkpoints

| Model | Download | Size |
| --- | --- | ---: |
| ScaleNav | [Hugging Face](https://huggingface.co/Yangyihui/ScaleNav) | 10.02 GB |
| Wan VAE, text encoder and tokenizer | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | 14.20 GB |

Download both models and set the paths used by the commands below:

<details>
<summary><strong>Download commands</strong></summary>

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

</details>

### Data

The test subset contains **16 tasks in 13 scenes**: **991.32 MB** of network
downloads and **2.96 GB** on disk after preparation.
Request [InteriorGS access](https://huggingface.co/datasets/spatialverse/InteriorGS)
and accept its [terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf),
then prepare the data:

```bash
hf auth login
export PATH="$PWD/external/splat-tools/node_modules/.bin:$PATH"
python scripts/prepare_data.py evaluation --subset test_subset \
  --source data/upstream/evaluation --output data/evaluation/test_subset --download
```

For other evaluation sets and training data, see the
[data catalog](docs/DATA.md#choose-a-download) or
[ScaleNav Data on Hugging Face](https://huggingface.co/datasets/Yangyihui/ScaleNav-Data#choose-a-dataset).

### Evaluation

Run the prepared test subset with visualization enabled:

```bash
export EVAL_SET=test_subset
python -m awm.scripts.eval_habitat_gs \
  --eval_config src/awm/configs/eval_habitatgs.yaml \
  --episode_manifest data/evaluation/"$EVAL_SET"/episodes.json \
  --gs_dataset_root data/upstream/evaluation \
  --scene_dataset_config data/evaluation/"$EVAL_SET"/scenes.scene_dataset_config.json \
  --model_path "$CHECKPOINT" --pretrained_model_path "$BASE_MODEL" \
  --visualize --output_dir outputs/evaluation/"$EVAL_SET"
```

Open `outputs/evaluation/test_subset/gallery/index.html` to view the videos and
trajectories. Metrics are saved in `outputs/evaluation/test_subset/summary.json`.
See [evaluation outputs](docs/EVALUATION.md#visualizations) for details.

The test subsets contain selected successful cases; use the full sets for benchmark scores.

| Benchmark | Test subset | Full evaluation |
| --- | --- | --- |
| Habitat-GS-LN | [16 tasks](docs/EVALUATION.md#test-subset) | [157 tasks](docs/EVALUATION.md#full-habitat-gs-ln) |
| Short-Horizon OVON | [7 tasks](docs/EVALUATION.md#ovon-test-subset) | [524 tasks](docs/EVALUATION.md#full-short-horizon-ovon) |

<a id="single-image-inference"></a>

For a standalone trajectory prediction from one image, see
[single-image inference](docs/INFERENCE.md#single-image-inference).

## Training

Prepare the [training data](docs/DATA.md#full-training-data) and
[latent caches](docs/DATA.md#build-training-caches), then follow the
[four-phase training recipe](docs/TRAINING.md#run-the-four-phases).
Use the [small training subset](docs/DATA.md#small-training-subset) to preview the pipeline.
The [training guide](docs/TRAINING.md) covers hardware, schedules and checkpoint resume.

## Real robot

Start the policy server on the GPU machine:

```bash
awm serve --checkpoint "$CHECKPOINT" --base-model "$BASE_MODEL" --decode-video
```

Open `http://127.0.0.1:8620/` for live camera, prediction and trajectory views.
Follow the [Go2 setup](docs/INFERENCE.md#robot-adapter) to connect the camera and
controller, or use the [HTTP API](docs/INFERENCE.md#http-api) with another platform.

<details>
<summary>Development</summary>

## Development

After installing the [development dependencies](#optional-dependencies):

```bash
python -m pytest
python -m build
python scripts/check_release.py
```

</details>

## License

Code: [Apache-2.0](LICENSE). Model weights:
[CC BY-NC 4.0](https://huggingface.co/Yangyihui/ScaleNav/blob/main/LICENSE).
Annotations and released Internet videos:
[CC BY-NC-SA 4.0](annotations/LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license scope.
